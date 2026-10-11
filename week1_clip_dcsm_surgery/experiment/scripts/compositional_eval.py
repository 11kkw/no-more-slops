"""
Candidate-selection evaluation on the failure-mode query benchmark (attribute binding / spatial relationship / negation).

Input: data/comp_queries/comp_queries.json from build_comp_queries.py. Each item has N same-class candidate
boxes and one target. A scorer assigns a score to every candidate; top-1 accuracy = argmax hits the target.
Chance = mean(1/N). All scorers are training-free w.r.t. this benchmark.

Scorers
  owlv2, owlvit_v1        query text -> detections; candidate score = max logit over detections with IoU>=0.5
  clip_crop               CLIP global embedding of the candidate crop (10% padding) . EOS text embedding
  clip_ctx                same, but crop = union(candidate, B box) for spatial / negation (context crop)
  dcsm_<ckpt>_ctx         the DCSM scoring CNN (Kang et al.) on the 197x30 DCSM of the context crop; the readout
                          and the Functional-Row column rule are read from the checkpoint (released = naive +
                          repo eval rule, retrained = recorded readout + BPE-aligned rule)
  readout/<r>/<t>         full-image heatmap for the query from last-block readout r in {naive,maskclip,clearclip}
                          and text readout t in {pooled (EOS), dcsm_tok (max over content tokens)};
                          candidate score = mean heatmap over patches whose centres fall in the box

Outputs <out>/summary.md, <out>/summary.json, <out>/per_item.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPImageProcessor, CLIPModel, CLIPTokenizer, Owlv2ForObjectDetection, OwlViTForObjectDetection

from dcsm_surgery.boxes import box_patch_mean, crop, pad_box, union_box
from dcsm_surgery.clip_readouts import dense_features, heatmaps, preprocess, text_features
from dcsm_surgery.constants import MODES, SUBSETS
from dcsm_surgery.detectors import owl_scores
from dcsm_surgery.scorer import DCSMScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/comp_queries/comp_queries.json")
    ap.add_argument("--coco_root", default="data/coco")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--owlv2", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--owlv1", default="google/owlvit-base-patch32")
    ap.add_argument("--dcsm_ckpts", nargs="+", default=["data/dcsm_ckpt/dcsm_obja_ckpt.pt", "data/dcsm_ckpt/dcsm_coco_ckpt.pt"])
    ap.add_argument("--size", type=int, default=224, help="full-image resolution for the readout heatmaps")
    ap.add_argument("--limit", type=int, default=0, help="debug: items per subset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    items = json.load(open(args.queries))["items"]
    if args.limit:
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["subset"]] < args.limit:
                per[it["subset"]] += 1; keep.append(it)
        items = keep
    print("items", len(items))

    clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float32).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(args.clip)
    img_proc = CLIPImageProcessor.from_pretrained(args.clip)
    patch = clip.config.vision_config.patch_size
    owl2_proc, owl2 = AutoProcessor.from_pretrained(args.owlv2), Owlv2ForObjectDetection.from_pretrained(args.owlv2).to(device).eval()
    owl1_proc, owl1 = AutoProcessor.from_pretrained(args.owlv1), OwlViTForObjectDetection.from_pretrained(args.owlv1).to(device).eval()
    dcsm = {os.path.splitext(os.path.basename(p))[0].replace("_ckpt", ""): DCSMScorer(p, clip, tok, img_proc, device) for p in args.dcsm_ckpts}

    scorers = ["owlv2", "owlvit_v1", "clip_crop", "clip_ctx"] + [f"{k}_ctx" for k in dcsm] + \
              [f"readout/{m}/{t}" for m in MODES for t in ["pooled", "dcsm_tok"]]
    per_item = []
    t0 = time.time()
    for n, it in enumerate(items):
        img = Image.open(os.path.join(args.coco_root, "val2017", it["file_name"])).convert("RGB")
        W, H = img.size
        cands, q = it["candidates"], it["query"]
        ref = it.get("B_box")
        sc = {}
        sc["owlv2"] = owl_scores(owl2, owl2_proc, img, q, cands, device, square_pad=True)
        sc["owlvit_v1"] = owl_scores(owl1, owl1_proc, img, q, cands, device, square_pad=False)

        # text features for the query (EOS + tokens)
        txt_pooled, txt_tokens, content = text_features(clip, tok, [q], device)

        # crops
        plain = [crop(img, pad_box(c, W, H)) for c in cands]
        if ref is not None:
            ctx = [crop(img, pad_box(union_box(c, ref), W, H)) for c in cands]
        elif it.get("B_boxes"):
            bb = it["B_boxes"]
            ctx = []
            for c in cands:  # nearest B by centre distance
                cc = ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
                d = [((b[0] + b[2]) / 2 - cc[0]) ** 2 + ((b[1] + b[3]) / 2 - cc[1]) ** 2 for b in bb]
                ctx.append(crop(img, pad_box(union_box(c, bb[int(np.argmin(d))]), W, H)))
        else:
            ctx = plain
        with torch.no_grad():
            for name, crops in [("clip_crop", plain), ("clip_ctx", ctx)]:
                pv = img_proc(images=crops, return_tensors="pt")["pixel_values"].to(device)
                emb = F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pv).pooler_output), dim=-1)
                sc[name] = (emb @ txt_pooled.T)[:, 0].float().cpu().numpy()
        for k, s in dcsm.items():
            sc[f"{k}_ctx"] = s.score(ctx, q)

        # full-image readout heatmaps
        feats, _ = dense_features(clip, preprocess(img, args.size).to(device), args.size)
        grid = args.size // patch
        for m in MODES:
            hm = heatmaps(feats[m][0, 1:], txt_pooled, txt_tokens, content, grid)
            for t in ["pooled", "dcsm_tok"]:
                sc[f"readout/{m}/{t}"] = box_patch_mean(hm[t][0], cands, args.size, W, H, patch)

        rec = {"subset": it["subset"], "image_id": it["image_id"], "query": q, "n": len(cands), "target": it["target_idx"],
               "rel": it.get("rel") or it.get("attr"), "pred": {k: int(np.argmax(v)) for k, v in sc.items()}}
        per_item.append(rec)
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    # ---- aggregate
    subsets = SUBSETS
    summary = {}
    lines = ["| scorer | " + " | ".join(subsets) + " |", "|---|" + "---|" * len(subsets)]
    chance = {s: float(np.mean([1 / r["n"] for r in per_item if r["subset"] == s] or [np.nan])) for s in subsets}
    counts = {s: sum(r["subset"] == s for r in per_item) for s in subsets}
    lines.append("| chance (mean 1/N) | " + " | ".join(f"{100 * chance[s]:.1f}" for s in subsets) + " |")
    lines.append("| #items | " + " | ".join(str(counts[s]) for s in subsets) + " |")
    for k in scorers:
        row = {}
        for s in subsets:
            rs = [r for r in per_item if r["subset"] == s]
            row[s] = float(np.mean([r["pred"][k] == r["target"] for r in rs])) if rs else np.nan
        summary[k] = row
        lines.append(f"| {k} | " + " | ".join(f"{100 * row[s]:.1f}" for s in subsets) + " |")
    # spatial breakdown by relation, size/color by attribute
    bd = {}
    for s in ["spatial", "size", "color"]:
        keys = sorted({r["rel"] for r in per_item if r["subset"] == s})
        if not keys:
            continue
        lines += ["", f"**{s} breakdown** (top-1 %, n per column)", "| scorer | " + " | ".join(f"{k} (n={sum(r['subset'] == s and r['rel'] == k for r in per_item)})" for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for k in scorers:
            vals = []
            for key in keys:
                rs = [r for r in per_item if r["subset"] == s and r["rel"] == key]
                vals.append(float(np.mean([r["pred"][k] == r["target"] for r in rs])) if rs else np.nan)
            bd.setdefault(s, {})[k] = dict(zip(keys, vals))
            lines.append(f"| {k} | " + " | ".join(f"{100 * v:.1f}" for v in vals) + " |")
    md = "\n".join(lines) + "\n"
    print(md)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "chance": chance, "counts": counts, "top1": summary, "breakdown": bd}, f, indent=1)
    with open(os.path.join(args.out, "per_item.json"), "w") as f:
        json.dump(per_item, f)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
