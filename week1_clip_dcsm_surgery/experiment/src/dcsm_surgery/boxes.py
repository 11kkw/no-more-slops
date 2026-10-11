"""Box geometry (xyxy, image coordinates), crops, OWL box conversion and NMS helpers."""
import numpy as np
import torch
from torchvision.ops import nms


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def iou(a, b):
    """Scalar IoU of two xyxy boxes (lists/tuples)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0


def iou_matrix(a, b):
    """a (N,4), b (M,4) xyxy tensors -> (N,M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


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


def cxcywh_to_xyxy(b, sx, sy, W, H, clamp=True):
    """OWL relative (cx,cy,w,h) -> absolute xyxy; sx, sy = the size the relative coords refer to."""
    cx, cy, w, h = b.unbind(-1)
    out = torch.stack([(cx - w / 2) * sx, (cy - h / 2) * sy, (cx + w / 2) * sx, (cy + h / 2) * sy], -1)
    if clamp:  # the HF processor does not clamp; low-score boxes can extend far outside the image
        out[:, 0::2] = out[:, 0::2].clamp(0, W)
        out[:, 1::2] = out[:, 1::2].clamp(0, H)
    return out


def xyxy_to_coco(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def nms_keep(boxes, scores, thr, k=None):
    keep = nms(boxes, scores, thr)
    return keep[:k] if k else keep


def topk_dets(boxes, scores, nms_thr, k):
    keep = nms_keep(boxes, scores, nms_thr, k)
    return boxes[keep].cpu().tolist(), scores[keep].cpu().tolist()
