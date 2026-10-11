"""End-to-end referring detection on the failure-mode query benchmark (no ground-truth candidate boxes).

Input: data/comp_queries/comp_queries.json (one target box per query). Every method receives only the image and
the query sentence; the pipeline methods additionally receive the two class names (A, B) parsed from the query
template (the generator stores them as cls_A / cls_B).

Methods
  owlv2_native, owlvit_v1_native       detector run with the full query sentence; score = query logit
  gdino_tiny, gdino_base               Grounding DINO with the full query sentence; score = max token prob
  cands/owlv2_classname                 pipeline candidates ranked by OWLv2's class-name probability p(A) only
                                        (relation-blind reference on the same candidate set)
  cands/clip_ctx                        candidates re-ranked by CLIP EOS similarity on the context crop
  cands/dcsm_<ckpt>                     candidates re-ranked by a DCSM scoring network on the context crop

Pipeline (DCSM head on OWLv2 proposals)
  1. one OWLv2 forward with texts [query, "a photo of a {A}", "a photo of a {B}"]
  2. candidates = OWLv2 boxes with p(A) >= cand_thr after class-agnostic NMS, at most max_cands (at least 2)
  3. reference B̂ = OWLv2 boxes with p(B) >= cand_thr after NMS; spatial: the top-1 B̂; negation/affirm: the B̂
     nearest to each candidate; size/color: none -> plain crop
  4. context crop = union(candidate, B̂) padded 10 %; scorer batch = the candidates of one query
  5. final score = p(A) * softmax over candidates of the re-ranker score

Metrics (per subset): Acc@0.5 = top-1 box IoU >= 0.5 with the target (raw and size-balanced, same strata as
balanced_view.py); COCO-style AP (0.5:0.95) and AP50 with one category per subset, one GT per query item
(items are pseudo-images), detections pooled across items (so cross-item calibration matters). Candidate
recall = fraction of items whose candidate set contains a box with IoU >= 0.5 (upper bound for cands/*).
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPImageProcessor, CLIPModel,
                          CLIPTokenizer, Owlv2ForObjectDetection, OwlViTForObjectDetection)

from dcsm_surgery.boxes import area, centre, crop, iou, iou_matrix, nms_keep, pad_box, topk_dets, union_box
from dcsm_surgery.clip_readouts import text_token_features
from dcsm_surgery.constants import SUBSETS, TEXT_SEQ
from dcsm_surgery.detectors import gdino_dets, owl_forward
from dcsm_surgery.metrics import paired_bootstrap, referring_ap
from dcsm_surgery.scorer import DCSMScorer, clip_patch_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/comp_queries/comp_queries.json")
    ap.add_argument("--coco_root", default="data/coco")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--owlv2", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--owlv1", default="google/owlvit-base-patch32")
    ap.add_argument("--gdino", nargs="*", default=["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"])
    ap.add_argument("--dcsm_ckpts", nargs="*", default=[])
    ap.add_argument("--cand_thr", type=float, default=0.1, help="OWLv2 p(class) threshold for candidates / references")
    ap.add_argument("--max_cands", type=int, default=10)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--max_dets", type=int, default=100, help="native detections kept per item for AP")
    ap.add_argument("--limit", type=int, default=0, help="debug: items per subset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    items = json.load(open(args.queries))["items"]
    if args.limit:
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["subset"]] < args.limit:
                per[it["subset"]] += 1; keep.append(it)
        items = keep
    print("items", len(items), flush=True)

    clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float32).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(args.clip)
    img_proc = CLIPImageProcessor.from_pretrained(args.clip)
    owl2_proc, owl2 = AutoProcessor.from_pretrained(args.owlv2), Owlv2ForObjectDetection.from_pretrained(args.owlv2).to(device).eval()
    owl1_proc, owl1 = AutoProcessor.from_pretrained(args.owlv1), OwlViTForObjectDetection.from_pretrained(args.owlv1).to(device).eval()
    gdinos = {}
    for g in args.gdino:
        name = "gdino_" + g.split("-")[-1]
        gdinos[name] = (AutoProcessor.from_pretrained(g), AutoModelForZeroShotObjectDetection.from_pretrained(g).to(device).eval())
    scorers = {}
    for p in args.dcsm_ckpts:
        base = os.path.splitext(os.path.basename(p))[0].replace("_ckpt", "")
        m = re.search(r"seed_(\d+)", p)
        name = f"{base}_s{m.group(1)}" if m else base
        scorers[name] = DCSMScorer(p, clip, tok, img_proc, device)
    readouts = sorted({s.readout for s in scorers.values()})
    methods = ["owlv2_native", "owlvit_v1_native"] + list(gdinos) + ["cands/owlv2_classname", "cands/clip_ctx"] + \
              [f"cands/{k}" for k in scorers]
    print("methods", methods, flush=True)

    per_item, dets = [], {m: [] for m in methods}
    ap_items = []
    t0 = time.time()
    for idx, it in enumerate(items):
        img = Image.open(os.path.join(args.coco_root, "val2017", it["file_name"])).convert("RGB")
        W, H = img.size
        q, A, B = it["query"], it["cls_A"], it.get("cls_B")
        tgt = it["candidates"][it["target_idx"]]
        ap_items.append((idx, it["subset"], tgt, W, H))
        rec = {"idx": idx, "subset": it["subset"], "image_id": it["image_id"], "query": q, "n_gt_cands": len(it["candidates"]),
               "rel": it.get("rel") or it.get("attr"),
               "target_is_largest": it["target_idx"] == int(np.argmax([area(b) for b in it["candidates"]])),
               "hit": {}, "top1_iou": {}}

        # ---- OWLv2: native + class-name candidates/references (one forward)
        texts = [q, f"a photo of a {A}"] + ([f"a photo of a {B}"] if B else [])
        b2, logits = owl_forward(owl2, owl2_proc, img, texts, device, square_pad=True)
        prob = torch.sigmoid(logits)  # (K,T)
        native = {"owlv2_native": (b2, prob[:, 0])}
        pA = prob[:, 1]
        keepA = nms_keep(b2, pA, args.nms)
        keepA = keepA[pA[keepA] >= args.cand_thr][: args.max_cands]
        if len(keepA) < 2:
            keepA = nms_keep(b2, pA, args.nms)[:2]
        cands = b2[keepA]; cand_pA = pA[keepA]
        refs = None
        if B:
            pB = prob[:, 2]
            keepB = nms_keep(b2, pB, args.nms)
            keepB = keepB[pB[keepB] >= args.cand_thr][:5]
            if len(keepB) == 0:
                keepB = nms_keep(b2, pB, args.nms)[:1]
            refs = b2[keepB].cpu().tolist()
            if it["subset"] == "spatial":
                refs = refs[:1]

        # ---- OWL-ViT v1 native
        b1, l1 = owl_forward(owl1, owl1_proc, img, [q], device, square_pad=False)
        native["owlvit_v1_native"] = (b1, torch.sigmoid(l1[:, 0]))

        # ---- Grounding DINO native
        for name, (gp, gm) in gdinos.items():
            native[name] = gdino_dets(gm, gp, img, q, device)

        # ---- candidate re-rankers
        cl = cands.cpu().tolist()
        if refs is None:
            ctx_boxes = cl
        elif it["subset"] == "spatial":
            ctx_boxes = [union_box(c, refs[0]) for c in cl]
        else:
            ctx_boxes = []
            for c in cl:
                cc = centre(c)
                d = [(centre(r)[0] - cc[0]) ** 2 + (centre(r)[1] - cc[1]) ** 2 for r in refs]
                ctx_boxes.append(union_box(c, refs[int(np.argmin(d))]))
        ctx = [crop(img, pad_box(b, W, H)) for b in ctx_boxes]
        rerank = {"cands/owlv2_classname": torch.zeros(len(cl), device=device)}  # softmax(0) = uniform -> p(A) only
        with torch.no_grad():
            pv = img_proc(images=ctx, return_tensors="pt")["pixel_values"].to(device)
            emb = F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pv).pooler_output), dim=-1)
            tt, txt = text_token_features(clip, tok, q, device, TEXT_SEQ)
            t_eos = F.normalize(clip.text_projection(clip.text_model(**tt).pooler_output), dim=-1)
            rerank["cands/clip_ctx"] = 100.0 * (emb @ t_eos.T)[:, 0]
            if scorers:
                feats = clip_patch_feats(clip, pv, readouts)
                for k, s in scorers.items():
                    rerank[f"cands/{k}"] = s.score_feats(feats[s.readout], txt, tt, q)
        tgt_t = torch.tensor([tgt], dtype=torch.float32, device=device)
        rec["cand_recall"] = bool((iou_matrix(tgt_t, cands)[0] >= 0.5).any())
        rec["n_cands"] = len(cl)

        # ---- collect detections + top-1
        final = {}
        for m, (bx, sc) in native.items():
            final[m] = topk_dets(bx, sc, args.nms, args.max_dets)
        for m, r in rerank.items():
            sc = cand_pA * torch.softmax(r, dim=0)
            order = torch.argsort(sc, descending=True)
            final[m] = (cands[order].cpu().tolist(), sc[order].cpu().tolist())
        for m, (bx, sc) in final.items():
            dets[m] += [(idx, b, s) for b, s in zip(bx, sc)]
            top_iou = iou(bx[0], tgt) if bx else 0.0
            rec["top1_iou"][m] = round(top_iou, 4)
            rec["hit"][m] = top_iou >= 0.5
        per_item.append(rec)
        if (idx + 1) % 100 == 0:
            print(f"{idx + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    # ---- aggregate
    aps = referring_ap(ap_items, dets, SUBSETS)

    def strata(s):
        rs = [r for r in per_item if r["subset"] == s]
        if s == "size":
            return [r for r in rs if r["rel"] == "big"], [r for r in rs if r["rel"] == "small"]
        return [r for r in rs if r["target_is_largest"]], [r for r in rs if not r["target_is_largest"]]

    def acc(rs, m):
        return float(np.mean([r["hit"][m] for r in rs])) if rs else float("nan")

    summary = {"meta": {**vars(args), "n_items": len(items), "seconds": time.time() - t0, "methods": methods},
               "counts": {s: sum(r["subset"] == s for r in per_item) for s in SUBSETS},
               "cand_recall": {s: float(np.mean([r["cand_recall"] for r in per_item if r["subset"] == s])) for s in SUBSETS},
               "mean_n_cands": {s: float(np.mean([r["n_cands"] for r in per_item if r["subset"] == s])) for s in SUBSETS},
               "acc": {}, "acc_balanced": {}, "ap": aps, "spatial_by_rel": {}}
    rels = sorted({r["rel"] for r in per_item if r["subset"] == "spatial"})
    for m in methods:
        summary["acc"][m] = {s: acc([r for r in per_item if r["subset"] == s], m) for s in SUBSETS}
        summary["acc_balanced"][m] = {s: float(np.mean([acc(a, m), acc(b, m)])) for s in SUBSETS for a, b in [strata(s)]}
        summary["spatial_by_rel"][m] = {k: acc([r for r in per_item if r["subset"] == "spatial" and r["rel"] == k], m) for k in rels}

    # seed aggregation for retrained scorers
    groups = defaultdict(list)
    for m in methods:
        g = re.sub(r"_s\d+$", "", m)
        if g != m:
            groups[g].append(m)
    summary["seed_groups"] = {g: ms for g, ms in groups.items()}

    def cell(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "n/a"
        return f"{100 * np.mean(vals):.1f} ± {100 * np.std(vals):.1f}" if len(vals) > 1 else f"{100 * vals[0]:.1f}"

    lines = [f"# Referring detection on the COCO val2017 failure-mode query benchmark ({len(items)} items)", "",
             f"candidates: OWLv2 class-name boxes, p(A) ≥ {args.cand_thr}, NMS {args.nms}, ≤ {args.max_cands}; native methods: NMS {args.nms}, ≤ {args.max_dets} dets/item", ""]
    lines += ["| | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS),
              "| #items | " + " | ".join(str(summary["counts"][s]) for s in SUBSETS) + " |",
              "| candidate recall@0.5 (upper bound for cands/*) | " + " | ".join(f"{100 * summary['cand_recall'][s]:.1f}" for s in SUBSETS) + " |",
              "| mean #candidates | " + " | ".join(f"{summary['mean_n_cands'][s]:.1f}" for s in SUBSETS) + " |", ""]
    rows = [m for m in methods if m not in sum(groups.values(), [])] + list(groups)
    for title, key in [("Acc@0.5 (top-1 box IoU ≥ 0.5, %)", "acc"), ("Acc@0.5 size-balanced (%)", "acc_balanced")]:
        lines += [f"## {title}", "", "| method | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
        for m in rows:
            ms = groups.get(m, [m])
            lines.append(f"| {m}{' (3 seeds)' if len(ms) > 1 else ''} | " + " | ".join(cell([summary[key][x][s] for x in ms]) for s in SUBSETS) + " |")
        lines.append("")
    for title, key in [("AP (0.5:0.95)", "AP"), ("AP50", "AP50")]:
        lines += [f"## {title}", "", "| method | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
        for m in rows:
            ms = groups.get(m, [m])
            lines.append(f"| {m}{' (3 seeds)' if len(ms) > 1 else ''} | " + " | ".join(cell([aps[x][s][key] for x in ms]) for s in SUBSETS) + " |")
        lines.append("")
    lines += ["## spatial Acc@0.5 by relation (%)", "", "| method | " + " | ".join(f"{k} (n={sum(r['subset'] == 'spatial' and r['rel'] == k for r in per_item)})" for k in rels) + " |", "|---|" + "---|" * len(rels)]
    for m in rows:
        ms = groups.get(m, [m])
        lines.append(f"| {m} | " + " | ".join(cell([summary["spatial_by_rel"][x][k] for x in ms]) for k in rels) + " |")
    lines.append("")

    # paired bootstrap (Acc@0.5, spatial) for each cands/dcsm method vs the strongest native baselines
    sp = [r for r in per_item if r["subset"] == "spatial"]
    if sp:
        lines += ["## paired bootstrap, spatial Acc@0.5 difference (10k resamples, 95 % CI)", "", "| A | B | acc A | acc B | diff | CI |", "|---|---|---|---|---|---|"]
        summary["bootstrap_spatial"] = []
        refs_m = ["owlv2_native"] + [g for g in gdinos] + ["cands/owlv2_classname", "cands/clip_ctx"]
        for a in [m for m in methods if m.startswith("cands/dcsm")]:
            ha = [r["hit"][a] for r in sp]
            for b in refs_m:
                hb = [r["hit"][b] for r in sp]
                diff, lo, hi = paired_bootstrap(ha, hb)
                summary["bootstrap_spatial"].append({"A": a, "B": b, "accA": float(np.mean(ha)), "accB": float(np.mean(hb)), "diff": diff, "ci": [lo, hi]})
                lines.append(f"| {a} | {b} | {100 * np.mean(ha):.1f} | {100 * np.mean(hb):.1f} | {100 * diff:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        lines.append("")
    md = "\n".join(lines)
    print(md)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "per_item.json"), "w") as f:
        json.dump(per_item, f)
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump({m: [(i, [round(x, 2) for x in b], round(s, 6)) for i, b, s in d] for m, d in dets.items()}, f)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
