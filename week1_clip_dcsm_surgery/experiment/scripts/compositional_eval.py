"""
Compositional referring-detection evaluation (attribute binding / spatial relation / negation).

Input: data/comp_queries/comp_queries.json from build_comp_queries.py. Each item has N same-class candidate
boxes and one target. A scorer assigns a score to every candidate; top-1 accuracy = argmax hits the target.
Chance = mean(1/N). All scorers are training-free w.r.t. this benchmark.

Scorers
  owlv2, owlvit_v1        query text -> detections; candidate score = max logit over detections with IoU>=0.5
  clip_crop               CLIP global embedding of the candidate crop (10% padding) . EOS text embedding
  clip_ctx                same, but crop = union(candidate, B box) for spatial / negation (context crop)
  dcsm_<ckpt>_ctx         the DCSM scoring CNN (Kang et al.) on the naive 197x30 DCSM of the context crop,
                          special-word column insertion + batch standardisation exactly as in the repo's eval
  readout/<r>/<t>         full-image heatmap for the query from last-block readout r in {naive,maskclip,clearclip}
                          and text readout t in {pooled (EOS), dcsm_tok (max over content tokens)};
                          candidate score = mean heatmap over patches whose centres fall in the box

Outputs <out>/summary.md, <out>/summary.json, <out>/per_item.json
"""
import argparse
import json
import os
import pickle
import sys
import time
import types
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPImageProcessor, CLIPModel, CLIPTokenizer, Owlv2ForObjectDetection, OwlViTForObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcsm_detection_eval import cxcywh_to_xyxy  # noqa: E402
from probe_localization import MODES, dense_features, heatmaps, preprocess, text_features  # noqa: E402

from bpe_alignment import ALIGNMENT_VERSION, replace_special_bpe

SPECIAL_WORDS = ["above", "below", "left", "right", "small", "big", "not", "no", "without", "but", "absent"]
TEXT_SEQ, IMG_SEQ = 30, 14 * 14 + 1


# ----------------------------------------------------------------------------- DCSM scorer (repo-faithful)
class CrossModalConvNetwork(nn.Module):
    """Copied from baseline/src/util/models_and_dataloaders.py (dropout = 0 at eval)."""

    def __init__(self, img_seq, text_seq, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim // 32, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 32, hidden_dim // 32, 3, padding=1)
        self.fc1 = nn.Linear(hidden_dim // 32 * text_seq * img_seq, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (bi, bt, iseq, tseq)
        bi, bt = x.shape[:2]
        x = x.reshape(bi * bt, 1, x.shape[2], x.shape[3])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.reshape(bi * bt, -1)))
        return self.fc2(x).view(bi, bt, 1)


def tolerant_torch_load(path):
    """The checkpoints pickle an old transformers CLIPTokenizer; stub the classes that no longer exist."""
    class _Dummy:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except Exception:
                return _Dummy

    m = types.ModuleType("tolerant_pickle"); m.Unpickler = _U; m.load = pickle.load
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=m)


class DCSMScorer:
    """Released checkpoints: naive readout + the repo's eval-time special-word slicing (insert, drop last column).
    New checkpoints carry 'special_slicing' = 'bpe_replace_v1'; legacy slicing is preserved with a warning."""

    def __init__(self, ckpt_path, clip, tok, img_proc, device):
        st = tolerant_torch_load(ckpt_path)
        self.net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ).to(device).eval()
        self.net.load_state_dict(st["model_state_dict"])
        self.emb = st["external_logic_token_embeddings"].to(device)
        self.words = st.get("learned_lookup") or SPECIAL_WORDS
        self.readout = st.get("readout", "naive")
        self.slicing = st.get("special_slicing", "eval_shift")
        if self.slicing not in (ALIGNMENT_VERSION, "replace", "eval_shift"):
            raise ValueError(f"Unknown checkpoint alignment: {self.slicing}")
        if self.slicing != ALIGNMENT_VERSION:
            warnings.warn("Legacy checkpoint uses unaligned word-index slicing; retained for historical reproduction. Corrected BPE alignment requires retraining.", RuntimeWarning)
        self.clip, self.tok, self.img_proc, self.device = clip, tok, img_proc, device

    @torch.no_grad()
    def score(self, crops, text):
        """crops: list of PIL -> (len(crops),) scores for one text (batch = the candidates of one query)."""
        pv = self.img_proc(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        if self.readout == "naive":
            vo = self.clip.vision_model(pixel_values=pv)
            img = self.clip.visual_projection(self.clip.vision_model.post_layernorm(vo.last_hidden_state))
            img = img / img.norm(dim=-1, keepdim=True)  # (B,197,512)
        else:
            img = dense_features(self.clip, pv, 224)[0][self.readout]
        tt = self.tok(text=[text], padding="max_length", max_length=TEXT_SEQ, truncation=True, return_tensors="pt").to(self.device)
        txt = self.clip.text_projection(self.clip.text_model(**tt).last_hidden_state)
        txt = txt / txt.norm(dim=-1, keepdim=True)  # (1,30,512)
        cos = torch.einsum("bie,te->bit", img, txt[0])  # (B,197,30)
        if self.slicing == ALIGNMENT_VERSION:
            cos = replace_special_bpe(cos.unsqueeze(1), [text], self.emb,
                                      self.tok, self.words, tt["input_ids"])
        else:
            sent = text.split(" ")
            for w in self.words:  # same order / same (eval_simple.py) slicing as the repo
                if w in sent:
                    loc = sent.index(w)
                    mid = self.emb(torch.tensor([self.words.index(w)], device=self.device)).unsqueeze(2).repeat(len(crops), 1, 1)
                    tail = cos[:, :, loc + 1:] if self.slicing == "replace" else cos[:, :, loc:-1]
                    cos = torch.cat([cos[:, :, :loc], mid, tail], dim=-1)
            cos = cos.unsqueeze(1)
        cos = (cos - cos.mean()) / cos.std()  # repo's standardize() over the whole batch tensor
        return self.net(cos).squeeze(-1).squeeze(-1).float().cpu().numpy()


# ----------------------------------------------------------------------------- helpers
def iou_matrix(a, b):
    """a (N,4), b (M,4) xyxy tensors -> (N,M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def pad_box(b, W, H, frac=0.1):
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0, b[0] - frac * w), max(0, b[1] - frac * h), min(W, b[2] + frac * w), min(H, b[3] + frac * h)]


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def crop(img, b):
    return img.crop((int(b[0]), int(b[1]), max(int(b[2]), int(b[0]) + 2), max(int(b[3]), int(b[1]) + 2)))


def box_patch_mean(hm, boxes, size, W, H, patch):
    """hm (g,g) tensor in size-space; boxes xyxy image coords -> mean over patches whose centres fall inside."""
    g = hm.shape[-1]
    c = (torch.arange(g, device=hm.device).float() + 0.5) * patch
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b[0] * size / W, b[1] * size / H, b[2] * size / W, b[3] * size / H
        m = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        if m.sum() == 0:  # tiny box: nearest patch
            d = (cx - (x1 + x2) / 2) ** 2 + (cy - (y1 + y2) / 2) ** 2
            m = d == d.min()
        out.append(float(hm[m].mean()))
    return np.array(out)


@torch.no_grad()
def owl_scores(model, proc, img, query, cands, device, square_pad):
    inp = proc(text=[[query]], images=img, return_tensors="pt").to(device)
    o = model(**inp)
    W, H = img.size
    S = max(W, H) if square_pad else None
    boxes = cxcywh_to_xyxy(o.pred_boxes[0], S or W, S or H, W, H)
    logit = o.logits[0, :, 0]
    ious = iou_matrix(torch.tensor(cands, device=device, dtype=torch.float32), boxes)  # (N,P)
    masked = torch.where(ious >= 0.5, logit[None, :], torch.full_like(ious, -1e4))
    s = masked.max(dim=1).values
    # fallback for candidates no detection overlaps: use the best-IoU detection's logit minus a penalty
    nomatch = s <= -1e3
    if nomatch.any():
        s[nomatch] = logit[ious[nomatch].argmax(dim=1)] - 10.0
    return s.float().cpu().numpy()


# ----------------------------------------------------------------------------- main
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
        if ref is None and it.get("B_boxes"):  # negation/affirm: context = union with the nearest B box
            ref = None
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
    subsets = ["size", "color", "spatial", "negation", "affirm"]
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
