"""
Localization probe: how well do CLIP ViT patch tokens (as consumed by DCSM) localize objects?

All variants share ONE frozen CLIP ViT; they differ only in how the last transformer block is read out
(naive = DCSM code path, maskclip, clearclip; see dcsm_surgery.clip_readouts) and in the text readout
(pooled = EOS, dcsm_tok = max over content-token columns of the DCSM matrix).

Metrics on a COCO val2017 subset (oracle class):
  pointing  : argmax patch falls inside a GT box of that class
  rec50     : box from thresholded heatmap has IoU>=0.5 with a GT box
  img_mAP   : image-level multi-label mAP from max-patch score (class discrimination)
  det AP/AP50: crude training-free detector (softmax over classes per patch -> components -> boxes)
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import CLIPModel, CLIPTokenizer

from dcsm_surgery.boxes import iou
from dcsm_surgery.clip_readouts import dense_features, heatmaps, preprocess, text_features
from dcsm_surgery.constants import MODES, TEXT_READOUTS
from dcsm_surgery.metrics import average_precision, box_from_heatmap, crude_detections


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
