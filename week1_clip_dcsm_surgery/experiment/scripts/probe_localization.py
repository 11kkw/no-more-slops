"""
Localization probe: how well do CLIP ViT patch tokens (as consumed by DCSM) localize objects?

All variants share ONE frozen CLIP ViT; they differ only in how the last transformer
block is read out:
  naive     : last_hidden_state -> post_layernorm -> visual_projection      (exact DCSM code path)
  maskclip  : last block value embeddings, no q-k attention/residual/FFN     (MaskCLIP, ECCV'22)
  clearclip : last block q-q self-self attention, no residual/FFN             (ClearCLIP, ECCV'24)

Text side, two readouts:
  pooled    : EOS token (standard CLIP text embedding)
  dcsm_tok  : max over content-token columns of the DCSM (patches x tokens) matrix

Metrics on a COCO val2017 subset:
  pointing  : argmax patch falls inside a GT box of that class (oracle class)
  rec50     : box from thresholded heatmap has IoU>=0.5 with a GT box (oracle class)
  img_mAP   : image-level multi-label mAP from max-patch score (class discrimination)
  det AP/AP50: crude training-free detector (softmax over classes per patch -> components -> boxes)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage
from transformers import CLIPModel, CLIPTokenizer

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
MODES = ["naive", "maskclip", "clearclip"]
TEXT_READOUTS = ["pooled", "dcsm_tok"]


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def dense_features(model, pixel_values, size):
    """Return dict mode -> (B, N+1, D) L2-normalised features in CLIP joint space, plus pooled."""
    vm = model.vision_model
    kwargs = {"output_hidden_states": True}
    if size != 224:
        kwargs["interpolate_pos_encoding"] = True
    out = vm(pixel_values=pixel_values, **kwargs)
    feats = {}

    x_in = out.hidden_states[-2]  # input to the last block
    layer = vm.encoder.layers[-1]
    attn = layer.self_attn
    h = layer.layer_norm1(x_in)
    B, N, D = h.shape
    nh = attn.num_heads
    hd = D // nh
    scale = getattr(attn, "scale", None) or getattr(attn, "scaling", hd ** -0.5)
    q = attn.q_proj(h).view(B, N, nh, hd).transpose(1, 2)
    v = attn.v_proj(h).view(B, N, nh, hd).transpose(1, 2)

    variants = {
        "naive": out.last_hidden_state,
        "maskclip": attn.out_proj(v.transpose(1, 2).reshape(B, N, D)),
        "clearclip": attn.out_proj(
            (torch.softmax((q @ q.transpose(-1, -2)) * scale, dim=-1) @ v).transpose(1, 2).reshape(B, N, D)
        ),
    }
    for k, x in variants.items():
        x = model.visual_projection(vm.post_layernorm(x))
        feats[k] = F.normalize(x, dim=-1)
    pooled = F.normalize(model.visual_projection(out.pooler_output), dim=-1)
    return feats, pooled


@torch.no_grad()
def text_features(model, tokenizer, prompts, device, max_len=30):
    tok = tokenizer(prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt").to(device)
    tout = model.text_model(**tok)
    pooled = F.normalize(model.text_projection(tout.pooler_output), dim=-1)  # (C, D)
    tokens = F.normalize(model.text_projection(tout.last_hidden_state), dim=-1)  # (C, T, D)
    # content mask: exclude BOS (pos 0), EOS and padding
    ids = tok["input_ids"]
    eos_pos = (ids == tokenizer.eos_token_id).int().argmax(dim=1)
    content = torch.zeros_like(ids, dtype=torch.bool)
    for i, e in enumerate(eos_pos.tolist()):
        content[i, 1:e] = True
    return pooled, tokens, content


def heatmaps(patch_feats, txt_pooled, txt_tokens, content_mask, grid):
    """patch_feats: (N, D) without CLS. Returns dict readout -> (C, g, g)."""
    hm = {}
    hm["pooled"] = (patch_feats @ txt_pooled.T).T.reshape(-1, grid, grid)
    sim = torch.einsum("nd,ctd->cnt", patch_feats, txt_tokens)  # (C, N, T) == DCSM matrix per class
    sim = sim.masked_fill(~content_mask[:, None, :], -1e4)
    hm["dcsm_tok"] = sim.max(dim=-1).values.reshape(-1, grid, grid)
    return hm


def box_from_heatmap(h: np.ndarray, size: int, rel_thr=0.5):
    """h: (g,g) -> upsample to (size,size), threshold relative to max, take component containing argmax."""
    up = F.interpolate(torch.from_numpy(h)[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = up.min(), up.max()
    if hi - lo < 1e-8:
        return None, up
    norm = (up - lo) / (hi - lo)
    mask = norm >= rel_thr
    lab, _ = ndimage.label(mask)
    ay, ax = np.unravel_index(np.argmax(up), up.shape)
    comp = lab == lab[ay, ax]
    ys, xs = np.where(comp)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), up


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    if labels.sum() == 0:
        return np.nan
    tp = np.cumsum(labels)
    prec = tp / (np.arange(len(labels)) + 1)
    return float((prec * labels).sum() / labels.sum())


def crude_detections(hm_all: torch.Tensor, size: int, img_id: int, cat_ids, sx, sy, temp=100.0, prob_thr=0.3, min_area=4):
    """hm_all: (C,g,g) cosine maps -> list of COCO-format detections via per-patch softmax over classes."""
    C, g, _ = hm_all.shape
    up = F.interpolate(hm_all[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    prob = torch.softmax(temp * up, dim=0).cpu().numpy()  # (C,S,S)
    arg = prob.argmax(0)
    dets = []
    for c in range(C):
        mask = (arg == c) & (prob[c] > prob_thr)
        if mask.sum() < min_area:
            continue
        lab, n = ndimage.label(mask)
        for k in range(1, n + 1):
            ys, xs = np.where(lab == k)
            if len(xs) < min_area:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            score = float(prob[c][lab == k].max())
            dets.append({
                "image_id": img_id,
                "category_id": cat_ids[c],
                "bbox": [float(x1 * sx), float(y1 * sy), float((x2 - x1) * sx), float((y2 - y1) * sy)],
                "score": score,
            })
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a photo of a {}.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco = COCO(os.path.join(args.coco_root, "annotations", "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    cat_names = [coco.loadCats(c)[0]["name"] for c in cat_ids]
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    img_ids = sorted(coco.getImgIds())
    img_ids = [i for i in img_ids if len(coco.getAnnIds(imgIds=i, iscrowd=False)) > 0][: args.n_images]
    print(f"images: {len(img_ids)}  classes: {len(cat_ids)}  device: {device}")

    model = CLIPModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    patch = model.config.vision_config.patch_size
    txt_pooled, txt_tokens, content = text_features(model, tokenizer, [args.prompt.format(n) for n in cat_names], device)

    results = {}
    for size in args.sizes:
        grid = size // patch
        stats = {f"{m}/{r}": {"point_hit": 0, "point_tot": 0, "rec50_hit": 0, "ious": [], "img_scores": [], "dets": []}
                 for m in MODES for r in TEXT_READOUTS}
        stats["cls_pooled"] = {"img_scores": []}
        img_labels = []
        t0 = time.time()
        for n, img_id in enumerate(img_ids):
            info = coco.loadImgs(img_id)[0]
            img = Image.open(os.path.join(args.coco_root, "val2017", info["file_name"]))
            W, H = img.size
            sx, sy = W / size, H / size
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
            gt = {}
            for a in anns:
                x, y, w, h = a["bbox"]
                gt.setdefault(cat_index[a["category_id"]], []).append((x / sx, y / sy, (x + w) / sx, (y + h) / sy))
            lab = np.zeros(len(cat_ids), dtype=np.float32)
            lab[list(gt.keys())] = 1
            img_labels.append(lab)

            px = preprocess(img, size).to(device)
            feats, pooled = dense_features(model, px, size)
            stats["cls_pooled"]["img_scores"].append((pooled @ txt_pooled.T)[0].cpu().numpy())

            for m in MODES:
                hm = heatmaps(feats[m][0, 1:], txt_pooled, txt_tokens, content, grid)
                for r in TEXT_READOUTS:
                    key = f"{m}/{r}"
                    maps = hm[r]  # (C,g,g)
                    st = stats[key]
                    st["img_scores"].append(maps.flatten(1).max(1).values.cpu().numpy())
                    for c, boxes in gt.items():
                        h = maps[c].cpu().numpy()
                        gy, gx = np.unravel_index(np.argmax(h), h.shape)
                        py, pxx = (gy + 0.5) * patch, (gx + 0.5) * patch
                        st["point_tot"] += 1
                        if any(b[0] <= pxx <= b[2] and b[1] <= py <= b[3] for b in boxes):
                            st["point_hit"] += 1
                        box, _ = box_from_heatmap(h, size)
                        best = max((iou(box, b) for b in boxes), default=0.0) if box else 0.0
                        st["ious"].append(best)
                        st["rec50_hit"] += best >= 0.5
                    st["dets"] += crude_detections(maps, size, img_id, cat_ids, sx, sy)
            if (n + 1) % 100 == 0:
                print(f"  size {size}: {n + 1}/{len(img_ids)}  {time.time() - t0:.0f}s")

        img_labels = np.stack(img_labels)
        res = {}
        for key, st in stats.items():
            S = np.stack(st["img_scores"])
            aps = [average_precision(S[:, c], img_labels[:, c]) for c in range(len(cat_ids))]
            entry = {"img_mAP": float(np.nanmean(aps))}
            if "point_tot" in st:
                entry["pointing_acc"] = st["point_hit"] / max(1, st["point_tot"])
                entry["recall@0.5"] = st["rec50_hit"] / max(1, st["point_tot"])
                entry["mean_best_iou"] = float(np.mean(st["ious"]))
                if st["dets"]:
                    dt = coco.loadRes(st["dets"])
                    ev = COCOeval(coco, dt, "bbox")
                    ev.params.imgIds = img_ids
                    ev.evaluate(); ev.accumulate()
                    import contextlib, io
                    with contextlib.redirect_stdout(io.StringIO()):
                        ev.summarize()
                    entry["det_AP"] = float(ev.stats[0]); entry["det_AP50"] = float(ev.stats[1])
                    entry["det_APs"] = float(ev.stats[3]); entry["det_APl"] = float(ev.stats[5])
                    entry["n_dets"] = len(st["dets"])
            res[key] = entry
        results[f"size{size}"] = res
        print(json.dumps({f"size{size}": res}, indent=1))

    meta = {"model": args.model, "n_images": len(img_ids), "prompt": args.prompt, "sizes": args.sizes}
    with open(os.path.join(args.out, "localization_probe.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print("saved", os.path.join(args.out, "localization_probe.json"))


if __name__ == "__main__":
    main()
