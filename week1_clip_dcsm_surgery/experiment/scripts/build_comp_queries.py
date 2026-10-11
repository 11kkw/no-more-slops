"""
Build a failure-mode query benchmark (attribute binding / spatial relationship / negation; candidate selection and referring detection) from COCO val2017 annotations (no extra labels).

Every item = one image, one query, N candidate boxes of the SAME class A (2 <= N <= max_cands), exactly one
of which satisfies the query. Scorers must rank the target above same-class distractors, so the class name
alone cannot solve any item; the attribute / relation / negation has to be read.

Subsets
  size      "a picture of a big {A}" / "... small {A}"         target area >= 3x (or <= 1/3x) every other A
  color     "a picture of a {color} {A}"                        dominant HSV colour from the instance mask;
                                                                target colour must be absent from all distractors
  spatial   "a picture of a {A} to the left of a {B}" (+ right/above/below), B = unique instance of its class
  negation  "a picture of a {A} without a {B}"                  exactly one A has no attached B (box overlap)
  affirm    "a picture of a {A} with a {B}"                     control: exactly one A has an attached B

Writes <out>/comp_queries.json, <out>/stats.md and spot-check grids <out>/spotcheck_<subset>.png.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as maskutils
from pycocotools.coco import COCO

# objects with a reasonably uniform colour; sky-backed (airplane, kite, bird) and large cluttered furniture
# (dining table, bed, couch, bench) are excluded because the mask colour is dominated by background bleed
COLOR_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle", "boat", "train", "umbrella", "surfboard",
    "skateboard", "snowboard", "frisbee", "sports ball", "chair", "cup", "bottle", "bowl", "vase",
    "book", "suitcase", "backpack", "handbag", "tie", "toothbrush", "cell phone", "laptop", "remote", "toilet",
    "sink", "refrigerator", "tv", "fire hydrant", "stop sign", "cat", "dog", "horse", "cow", "sheep", "bear",
    "elephant", "zebra", "giraffe", "teddy bear", "clock", "scissors", "hair drier", "mouse", "keyboard", "oven",
    "microwave", "toaster", "wine glass", "fork", "knife", "spoon",
}
ANIMALS = {"cat", "dog", "horse", "cow", "sheep", "bear", "elephant", "zebra", "giraffe", "teddy bear"}
# negation / affirm: "a {A} with(out) a {B}" only for pairs where "with" reads as possession / carrying / riding
HOSTS = {"person", "dining table", "chair", "couch", "bed", "bench", "horse", "elephant", "bicycle", "motorcycle",
         "car", "truck", "bus", "boat", "train", "dog", "cat"}
PORTABLES = {"handbag", "backpack", "umbrella", "tie", "cell phone", "skateboard", "surfboard", "skis", "snowboard",
             "sports ball", "frisbee", "tennis racket", "baseball bat", "baseball glove", "kite", "bottle", "cup",
             "wine glass", "fork", "knife", "spoon", "bowl", "remote", "book", "laptop", "mouse", "keyboard", "dog",
             "cat", "person", "pizza", "cake", "sandwich", "donut", "banana", "apple", "orange", "vase", "potted plant",
             "teddy bear", "toothbrush", "hair drier", "scissors", "suitcase", "hot dog", "carrot", "broccoli"}
ANIMAL_COLORS = {"black", "white", "brown"}
TOO_CLOSE = {frozenset(p) for p in [("red", "orange"), ("orange", "yellow"), ("white", "gray"), ("gray", "black"),
                                     ("blue", "purple"), ("purple", "pink"), ("pink", "red"), ("orange", "brown"),
                                     ("red", "brown"), ("yellow", "brown"), ("gray", "brown"), ("black", "brown")]}


def color_bins(hsv: np.ndarray) -> np.ndarray:
    """hsv uint8 (N,3) from PIL (H 0..255 == 0..360deg). Returns bin name per pixel ('' = ambiguous)."""
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    out = np.full(len(h), "", dtype=object)
    out[v < 45] = "black"
    low_s = (s < 45) & (v >= 45)
    out[low_s & (v > 190)] = "white"
    out[low_s & (v <= 190)] = "gray"
    chroma = (s >= 70) & (v >= 70)
    hue = {"red": (h < 9) | (h >= 246), "orange": (h >= 9) & (h < 26), "yellow": (h >= 26) & (h < 46),
           "green": (h >= 46) & (h < 112), "blue": (h >= 130) & (h < 186), "purple": (h >= 186) & (h < 216),
           "pink": (h >= 216) & (h < 246)}
    for name, m in hue.items():
        out[chroma & m] = name
    brown = (s >= 45) & (v >= 45) & (v < 150) & ((h >= 6) & (h < 36))
    out[brown] = "brown"
    return out


def dominant_color(img_hsv: np.ndarray, ann, coco, H, W, min_pixels=600):
    from scipy import ndimage
    m = coco.annToMask(ann).astype(bool)
    # erode the mask so boundary pixels (background bleed, anti-aliasing) are ignored
    m = ndimage.binary_erosion(m, iterations=3)
    if m.sum() < min_pixels or min(ann["bbox"][2], ann["bbox"][3]) < 32:
        return None, {}
    ys, xs = np.where(m)
    px = img_hsv[ys, xs]
    bins = color_bins(px)
    cnt = Counter(bins.tolist())
    cnt.pop("", None)
    total = len(bins)
    shares = {k: v / total for k, v in cnt.items()}
    if not shares:
        return None, shares
    top = max(shares, key=shares.get)
    second = sorted(shares.values(), reverse=True)[1] if len(shares) > 1 else 0.0
    if shares[top] >= 0.55 and shares[top] >= 2.5 * second:
        return top, shares
    return None, shares


def box_xyxy(a):
    x, y, w, h = a["bbox"]
    return [x, y, x + w, y + h]


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inter_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def attached(a, b):
    """b is 'with' a: substantial overlap or either centre inside the other box."""
    ia = inter_area(a, b)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    if ia >= 0.3 * min(area_a, area_b):
        return True
    cx, cy = centre(b)
    if a[0] <= cx <= a[2] and a[1] <= cy <= a[3]:
        return True
    cx, cy = centre(a)
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_cands", type=int, default=6)
    ap.add_argument("--per_class_cap", type=int, default=60, help="max items per (subset, class A)")
    ap.add_argument("--per_image_cap", type=int, default=2, help="max items per (subset, image)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    coco = COCO(os.path.join(args.coco_root, "annotations", "instances_val2017.json"))
    name_of = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    items = {"size": [], "color": [], "spatial": [], "negation": [], "affirm": []}

    img_ids = sorted(coco.getImgIds())
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        W, H = info["width"], info["height"]
        img_area = W * H
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]
        ok = [a for a in anns if a["area"] >= 0.002 * img_area and min(a["bbox"][2], a["bbox"][3]) >= 24]
        by_cls = defaultdict(list)
        for a in ok:
            by_cls[a["category_id"]].append(a)
        all_by_cls = defaultdict(list)
        for a in anns:
            all_by_cls[a["category_id"]].append(a)

        hsv = None
        per_img = Counter()
        base = {"image_id": img_id, "file_name": info["file_name"], "W": W, "H": H}

        for cid, group in by_cls.items():
            # candidates must be ALL instances of class A in the image (otherwise an unlisted instance could
            # satisfy the query); require every instance to pass the size filter
            if len(all_by_cls[cid]) != len(group) or not (2 <= len(group) <= args.max_cands):
                continue
            A = name_of[cid]
            boxes = [box_xyxy(a) for a in group]
            areas = [a["area"] for a in group]

            # ---- size
            if per_img["size"] < args.per_image_cap:
                order = sorted(range(len(group)), key=lambda i: areas[i])
                big, small = order[-1], order[0]
                if areas[big] >= 3.0 * areas[order[-2]]:
                    items["size"].append({**base, "subset": "size", "cls_A": A, "attr": "big", "candidates": boxes,
                                          "target_idx": big, "query": f"a picture of a big {A}"})
                    per_img["size"] += 1
                elif areas[small] * 3.0 <= areas[order[1]] and areas[small] >= 0.004 * img_area:
                    items["size"].append({**base, "subset": "size", "cls_A": A, "attr": "small", "candidates": boxes,
                                          "target_idx": small, "query": f"a picture of a small {A}"})
                    per_img["size"] += 1

            # ---- color
            if A in COLOR_CLASSES and per_img["color"] < args.per_image_cap:
                if hsv is None:
                    hsv = np.asarray(Image.open(os.path.join(args.coco_root, "val2017", info["file_name"])).convert("HSV"))
                cols, shares = [], []
                for a in group:
                    c, s = dominant_color(hsv, a, coco, H, W)
                    cols.append(c); shares.append(s)
                allowed = ANIMAL_COLORS if A in ANIMALS else None
                for ti, c in enumerate(cols):
                    if c is None or (allowed and c not in allowed):
                        continue
                    # target colour must be (nearly) absent from every distractor and not a near-synonym of theirs
                    good = True
                    for di, s in enumerate(shares):
                        if di == ti:
                            continue
                        if s.get(c, 0.0) > 0.12:
                            good = False; break
                        dc = max(s, key=s.get) if s else None
                        if dc and frozenset((c, dc)) in TOO_CLOSE:
                            good = False; break
                    if good:
                        items["color"].append({**base, "subset": "color", "cls_A": A, "attr": c, "candidates": boxes,
                                               "target_idx": ti, "query": f"a picture of a {c} {A}",
                                               "color_shares": [{k: round(v, 2) for k, v in s.items()} for s in shares]})
                        per_img["color"] += 1
                        break

            # ---- spatial: B = unique instance of another class
            for bcid, bgroup in all_by_cls.items():
                if bcid == cid or len(bgroup) != 1:
                    continue
                b = bgroup[0]
                if b["area"] < 0.005 * img_area or b["area"] > 0.5 * img_area:
                    continue
                B = name_of[bcid]
                bb = box_xyxy(b)
                bcx, bcy = centre(bb)
                rels = {
                    "to the left of": (lambda c, bx: c[2] <= bcx and centre(c)[0] < bb[0] - 0.05 * W, lambda c: centre(c)[0] >= bcx),
                    "to the right of": (lambda c, bx: c[0] >= bcx and centre(c)[0] > bb[2] + 0.05 * W, lambda c: centre(c)[0] <= bcx),
                    "above": (lambda c, bx: c[3] <= bcy and centre(c)[1] < bb[1] - 0.05 * H, lambda c: centre(c)[1] >= bcy),
                    "below": (lambda c, bx: c[1] >= bcy and centre(c)[1] > bb[3] + 0.05 * H, lambda c: centre(c)[1] <= bcy),
                }
                for rel, (sat, distract_ok) in rels.items():
                    if per_img["spatial"] >= args.per_image_cap:
                        break
                    hits = [i for i, c in enumerate(boxes) if sat(c, bb)]
                    if len(hits) == 1 and all(distract_ok(boxes[i]) for i in range(len(boxes)) if i != hits[0]):
                        items["spatial"].append({**base, "subset": "spatial", "cls_A": A, "cls_B": B, "rel": rel,
                                                 "B_box": bb, "candidates": boxes, "target_idx": hits[0],
                                                 "query": f"a picture of a {A} {rel} a {B}"})
                        per_img["spatial"] += 1

            # ---- negation / affirm: B attached to exactly one / all-but-one A
            for bcid, bgroup in all_by_cls.items():
                B = name_of[bcid]
                if bcid == cid or A not in HOSTS or B not in PORTABLES:
                    continue
                bboxes = [box_xyxy(b) for b in bgroup if b["area"] <= 0.5 * img_area and b["area"] >= 0.001 * img_area]
                if not bboxes:
                    continue
                has = [any(attached(c, bb) for bb in bboxes) for c in boxes]
                if sum(has) == len(boxes) - 1 and per_img["negation"] < args.per_image_cap:
                    t = has.index(False)
                    items["negation"].append({**base, "subset": "negation", "cls_A": A, "cls_B": B, "candidates": boxes,
                                              "target_idx": t, "B_boxes": bboxes, "query": f"a picture of a {A} without a {B}"})
                    per_img["negation"] += 1
                elif sum(has) == 1 and per_img["affirm"] < args.per_image_cap:
                    t = has.index(True)
                    items["affirm"].append({**base, "subset": "affirm", "cls_A": A, "cls_B": B, "candidates": boxes,
                                            "target_idx": t, "B_boxes": bboxes, "query": f"a picture of a {A} with a {B}"})
                    per_img["affirm"] += 1

    # ---- per-class cap (person dominates otherwise), deterministic shuffle
    final = []
    stats_lines = ["| subset | items | images | classes A | mean #cands | chance top-1 |", "|---|---|---|---|---|---|"]
    for subset, lst in items.items():
        rng.shuffle(lst)
        per_cls = Counter()
        kept = []
        for it in lst:
            if per_cls[it["cls_A"]] < args.per_class_cap:
                per_cls[it["cls_A"]] += 1
                kept.append(it)
        kept.sort(key=lambda d: (d["image_id"], d["query"]))
        final += kept
        n_c = [len(it["candidates"]) for it in kept]
        stats_lines.append(f"| {subset} | {len(kept)} | {len({it['image_id'] for it in kept})} | {len(per_cls)} | "
                           f"{np.mean(n_c):.2f} | {np.mean([1 / n for n in n_c]):.3f} |")
        sub = Counter(it.get("rel") or it.get("attr") for it in kept)
        if subset in ("size", "color", "spatial"):
            stats_lines.append(f"|  ↳ breakdown | {dict(sub)} | | | | |")
        stats_lines.append(f"|  ↳ top classes | {per_cls.most_common(6)} | | | | |")

    with open(os.path.join(args.out, "comp_queries.json"), "w") as f:
        json.dump({"args": vars(args), "items": final}, f)
    with open(os.path.join(args.out, "stats.md"), "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    print("\n".join(stats_lines))
    print("saved", len(final), "items ->", os.path.join(args.out, "comp_queries.json"))

    # ---- spot-check grids
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for subset in items:
        sel = [it for it in final if it["subset"] == subset]
        rng.shuffle(sel)
        sel = sel[:20]
        if not sel:
            continue
        fig, axes = plt.subplots(4, 5, figsize=(22, 17))
        for ax, it in zip(axes.flat, sel):
            img = Image.open(os.path.join(args.coco_root, "val2017", it["file_name"])).convert("RGB")
            ax.imshow(img)
            for i, c in enumerate(it["candidates"]):
                col = "lime" if i == it["target_idx"] else "red"
                ax.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fill=False, ec=col, lw=2.5))
            for bb in ([it["B_box"]] if "B_box" in it else it.get("B_boxes", [])):
                ax.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="cyan", lw=2, ls="--"))
            ax.set_title(it["query"], fontsize=10)
            ax.axis("off")
        for ax in axes.flat[len(sel):]:
            ax.axis("off")
        fig.suptitle(f"{subset}: target=green, same-class distractors=red, B=cyan dashed", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"spotcheck_{subset}.png"), dpi=70)
        plt.close(fig)


if __name__ == "__main__":
    main()
