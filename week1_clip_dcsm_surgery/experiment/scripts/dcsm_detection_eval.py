"""
Zero-shot COCO detection (single-class-name queries) with a FROZEN CLIP ViT and the DCSM (patches x text tokens)
matrix as the classification signal, under three last-block readouts, compared against open-vocabulary detectors.

Classification signal (per readout r in {naive, maskclip, clearclip}):
  class map  M_r[c, p] = max over content tokens t of cos(patch_p, token_t of prompt c)   ("dcsm_tok")
  reference  M_r[c, p] = cos(patch_p, EOS embedding of prompt c)                            ("pooled")

Two detection protocols, both training-free:
  P1 dense     : per-patch softmax over classes -> threshold -> connected components -> boxes
  P2 proposals : class-agnostic boxes from OWLv2's objectness head (top-K), each box classified by
                 mean-pooling M_r over the patches whose centres fall inside it; score = objectness * p(c)
Baselines (native, zero-shot):
  OWL-ViT  (google/owlvit-base-patch32)          text-queried detections
  OWLv2    (google/owlv2-base-patch16-ensemble)  text-queried detections  (same boxes as P2 -> controlled comparison)

Outputs <out>/summary.json and a markdown table.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoProcessor, CLIPModel, CLIPTokenizer, Owlv2ForObjectDetection, OwlViTForObjectDetection

from dcsm_surgery.boxes import cxcywh_to_xyxy, xyxy_to_coco
from dcsm_surgery.clip_readouts import dense_features, heatmaps, preprocess, text_features
from dcsm_surgery.constants import MODES
from dcsm_surgery.detectors import owl_native_dets, processor_boxes
from dcsm_surgery.metrics import coco_eval, crude_detections

TEXT_READOUTS = ["dcsm_tok", "pooled"]  # row order of the result table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--n_images", type=int, default=0, help="0 = all val2017 images with annotations")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448])
    ap.add_argument("--owlv2", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--owlv1", default="google/owlvit-base-patch32")
    ap.add_argument("--topk", type=int, default=100, help="class-agnostic proposals per image from OWLv2 objectness")
    ap.add_argument("--topc", type=int, default=3, help="classes kept per proposal")
    ap.add_argument("--temp", type=float, default=100.0)
    ap.add_argument("--prompt", default="a photo of a {}.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    coco = COCO(os.path.join(args.coco_root, "annotations", "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    cat_names = [coco.loadCats(c)[0]["name"] for c in cat_ids]
    img_ids = sorted(coco.getImgIds())
    img_ids = [i for i in img_ids if len(coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]
    if args.n_images:
        img_ids = img_ids[: args.n_images]
    print(f"images {len(img_ids)}  classes {len(cat_ids)}")

    # models
    clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float32).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(args.clip)
    patch = clip.config.vision_config.patch_size
    txt_pooled, txt_tokens, content = text_features(clip, tok, [args.prompt.format(n) for n in cat_names], device)

    owl2_proc = AutoProcessor.from_pretrained(args.owlv2)
    owl2 = Owlv2ForObjectDetection.from_pretrained(args.owlv2).to(device).eval()
    owl1_proc = AutoProcessor.from_pretrained(args.owlv1)
    owl1 = OwlViTForObjectDetection.from_pretrained(args.owlv1).to(device).eval()
    owl_queries = [[args.prompt.format(n).rstrip(".") for n in cat_names]]

    configs = ["owlvit_v1_native", "owlv2_native"]
    for size in args.sizes:
        for m in MODES:
            for r in TEXT_READOUTS:
                configs += [f"P1_dense/{m}/{r}/{size}", f"P2_owlv2boxes/{m}/{r}/{size}"]
    dets = {c: [] for c in configs}

    t0 = time.time()
    checked = False
    for n, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(args.coco_root, "val2017", info["file_name"])).convert("RGB")
        W, H = img.size

        # ---- OWLv2: native dets + class-agnostic proposals
        with torch.no_grad():
            inp = owl2_proc(text=owl_queries, images=img, return_tensors="pt").to(device)
            o2 = owl2(**inp)
        S = max(W, H)  # OWLv2 pads to square (bottom/right) before resizing
        b2 = cxcywh_to_xyxy(o2.pred_boxes[0], S, S, W, H)
        if not checked:  # sanity-check our box conversion against the processor's post-processing
            ref = processor_boxes(owl2_proc, o2, H, W, owl_queries)
            raw = cxcywh_to_xyxy(o2.pred_boxes[0], S, S, W, H, clamp=False)
            print("owlv2 box conversion max abs diff vs processor (unclamped):", float((ref - raw).abs().max()))
        dets["owlv2_native"] += owl_native_dets(o2.logits[0], b2, img_id, cat_ids)
        obj = torch.sigmoid(o2.objectness_logits[0])
        top = obj.topk(min(args.topk, obj.numel()))
        prop_boxes, prop_obj = b2[top.indices], top.values  # (K,4), (K,)

        # ---- OWL-ViT v1 native
        with torch.no_grad():
            inp1 = owl1_proc(text=owl_queries, images=img, return_tensors="pt").to(device)
            o1 = owl1(**inp1)
        b1 = cxcywh_to_xyxy(o1.pred_boxes[0], W, H, W, H)
        if not checked:
            ref1 = processor_boxes(owl1_proc, o1, H, W, owl_queries)
            raw1 = cxcywh_to_xyxy(o1.pred_boxes[0], W, H, W, H, clamp=False)
            print("owlvit v1 box conversion max abs diff vs processor (unclamped):", float((ref1 - raw1).abs().max()))
            checked = True
        dets["owlvit_v1_native"] += owl_native_dets(o1.logits[0], b1, img_id, cat_ids)

        # ---- frozen CLIP, three readouts
        for size in args.sizes:
            grid = size // patch
            feats, _ = dense_features(clip, preprocess(img, size).to(device), size)
            # proposal -> patch membership (K, g*g), normalised rows
            centres = (torch.arange(grid, device=device).float() + 0.5) * patch  # in size-space
            cy, cx = torch.meshgrid(centres, centres, indexing="ij")
            cx, cy = cx.flatten(), cy.flatten()
            pb = prop_boxes.clone()
            pb[:, 0::2] *= size / W
            pb[:, 1::2] *= size / H
            inside = (cx[None] >= pb[:, 0:1]) & (cx[None] <= pb[:, 2:3]) & (cy[None] >= pb[:, 1:2]) & (cy[None] <= pb[:, 3:4])
            empty = inside.sum(1) == 0
            if empty.any():  # tiny boxes: use the patch containing the box centre
                bc = ((pb[empty, 0:2] + pb[empty, 2:4]) / 2 / patch).long().clamp(0, grid - 1)
                inside[empty.nonzero()[:, 0], bc[:, 1] * grid + bc[:, 0]] = True
            member = inside.float() / inside.float().sum(1, keepdim=True)

            for m in MODES:
                hm = heatmaps(feats[m][0, 1:], txt_pooled, txt_tokens, content, grid)
                for r in TEXT_READOUTS:
                    maps = hm[r]  # (C,g,g)
                    dets[f"P1_dense/{m}/{r}/{size}"] += crude_detections(maps, size, img_id, cat_ids, W / size, H / size, temp=args.temp)
                    logits = member @ maps.flatten(1).T  # (K,C) mean cosine inside box
                    prob = torch.softmax(args.temp * logits, dim=-1)
                    sc = prob * prop_obj[:, None]
                    tc = sc.topk(args.topc, dim=-1)
                    key = f"P2_owlv2boxes/{m}/{r}/{size}"
                    for k in range(sc.shape[0]):
                        bb = xyxy_to_coco(prop_boxes[k])
                        for s, c in zip(tc.values[k].tolist(), tc.indices[k].tolist()):
                            dets[key].append({"image_id": img_id, "category_id": cat_ids[c], "bbox": bb, "score": s})
        if (n + 1) % 250 == 0:
            print(f"{n + 1}/{len(img_ids)}  {time.time() - t0:.0f}s", flush=True)

    summary = {c: coco_eval(coco, d, img_ids) for c, d in dets.items()}
    meta = {"n_images": len(img_ids), "clip": args.clip, "sizes": args.sizes, "owlv2": args.owlv2, "owlv1": args.owlv1,
            "topk": args.topk, "topc": args.topc, "temp": args.temp, "prompt": args.prompt, "seconds": time.time() - t0}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "results": summary}, f, indent=1)
    lines = ["| config | AP | AP50 | AP75 | APs | APm | APl | #dets |", "|---|---|---|---|---|---|---|---|"]
    for c, s in summary.items():
        lines.append(f"| {c} | {100*s['AP']:.1f} | {100*s['AP50']:.1f} | {100*s['AP75']:.1f} | {100*s['APs']:.1f} | {100*s['APm']:.1f} | {100*s['APl']:.1f} | {s['n_dets']} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print("saved", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
