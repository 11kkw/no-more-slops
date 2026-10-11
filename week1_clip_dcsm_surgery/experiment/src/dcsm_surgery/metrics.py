"""Evaluation metrics: COCO AP, referring-detection AP, oracle-class probe metrics, paired bootstrap."""
import contextlib
import io

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage

from .boxes import area


def coco_eval(coco, dets, img_ids):
    """Standard COCO bbox AP (0-1 scale) for a list of COCO-format detections."""
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0, "n_dets": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "APs": float(s[3]), "APm": float(s[4]), "APl": float(s[5]), "n_dets": len(dets)}


def average_precision(scores, labels):
    """Image-level AP of one class from per-image scores and binary labels (NaN if no positives)."""
    order = np.argsort(-scores)
    labels = labels[order]
    if labels.sum() == 0:
        return np.nan
    tp = np.cumsum(labels)
    prec = tp / (np.arange(len(labels)) + 1)
    return float((prec * labels).sum() / labels.sum())


def box_from_heatmap(h: np.ndarray, size: int, rel_thr=0.5):
    """h: (g,g) -> upsample to (size,size), threshold relative to max, take the component containing the argmax."""
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


def crude_detections(hm_all: torch.Tensor, size: int, img_id: int, cat_ids, sx, sy, temp=100.0, prob_thr=0.3, min_area=4):
    """P1 dense protocol: (C,g,g) cosine maps -> per-pixel class softmax -> components -> COCO-format detections."""
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
            dets.append({"image_id": img_id, "category_id": cat_ids[c],
                         "bbox": [float(x1 * sx), float(y1 * sy), float((x2 - x1) * sx), float((y2 - y1) * sy)],
                         "score": score})
    return dets


def referring_ap(items, dets_by_method, subsets):
    """COCO-style AP for referring detection: one GT box per query item (items are pseudo-images), one category
    per subset, detections pooled across items so cross-item score calibration matters.

    items: [(item_idx, subset, target_xyxy, W, H)]; dets_by_method: {method: [(item_idx, box_xyxy, score)]}
    -> {method: {subset: {'AP', 'AP50'}}} on a 0-1 scale."""
    cat_id = {s: i + 1 for i, s in enumerate(subsets)}
    gt = {"images": [{"id": i, "width": W, "height": H} for i, _, _, W, H in items],
          "categories": [{"id": cat_id[s], "name": s} for s in subsets],
          "annotations": [{"id": i + 1, "image_id": i, "category_id": cat_id[s], "iscrowd": 0,
                           "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": area(b)}
                          for i, s, b, _, _ in items]}
    coco = COCO()
    coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    subset_of = {i: s for i, s, _, _, _ in items}
    out = {}
    for m, dets in dets_by_method.items():
        res = [{"image_id": i, "category_id": cat_id[subset_of[i]], "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                "score": float(sc)} for i, b, sc in dets]
        if not res:
            out[m] = {s: {"AP": 0.0, "AP50": 0.0} for s in subsets}
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            cd = coco.loadRes(res)
            ev = COCOeval(coco, cd, "bbox")
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate(); ev.accumulate()
        prec = ev.eval["precision"]  # (T,R,K,A,M)
        r = {}
        for s in subsets:
            k = cat_id[s] - 1
            p = prec[:, :, k, 0, 2]
            p50 = prec[0, :, k, 0, 2]
            r[s] = {"AP": float(p[p > -1].mean()) if (p > -1).any() else float("nan"),
                    "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")}
        out[m] = r
    return out


def paired_bootstrap(hit_a, hit_b, n_resamples=10000, seed=0):
    """Paired bootstrap of mean(hit_a - hit_b) over items -> (diff, lo95, hi95)."""
    ha, hb = np.asarray(hit_a, float), np.asarray(hit_b, float)
    n = len(ha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_resamples, n))
    d = (ha - hb)[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(ha.mean() - hb.mean()), float(lo), float(hi)
