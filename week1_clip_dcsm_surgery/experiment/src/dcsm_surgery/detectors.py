"""Open-vocabulary detector helpers: OWL-ViT v1, OWLv2 and Grounding DINO (HF transformers)."""
import torch

from .boxes import cxcywh_to_xyxy, iou_matrix, xyxy_to_coco


def processor_boxes(proc, out, H, W, queries):
    """Reference boxes from the HF processor's own post-processing (API name differs across versions)."""
    ts = torch.tensor([[H, W]]).to(out.pred_boxes.device)
    if hasattr(proc, "post_process_grounded_object_detection"):
        return proc.post_process_grounded_object_detection(out, threshold=0.0, target_sizes=ts, text_labels=queries)[0]["boxes"]
    return proc.post_process_object_detection(out, threshold=0.0, target_sizes=ts)[0]["boxes"]


@torch.no_grad()
def owl_native_dets(logits, boxes_xyxy, img_id, cat_ids, max_dets=300):
    """logits: (N, C) raw; boxes (N,4) xyxy image coords -> COCO-format detections (top box-class pairs)."""
    scores = torch.sigmoid(logits)
    N, C = scores.shape
    top = scores.flatten().topk(min(max_dets, N * C))
    dets = []
    for s, idx in zip(top.values.tolist(), top.indices.tolist()):
        n, c = divmod(idx, C)
        dets.append({"image_id": img_id, "category_id": cat_ids[c], "bbox": xyxy_to_coco(boxes_xyxy[n]), "score": s})
    return dets


@torch.no_grad()
def owl_forward(model, proc, img, texts, device, square_pad):
    """One OWL forward with a list of text queries -> (boxes xyxy (K,4) image coords, logits (K,T))."""
    inp = proc(text=[texts], images=img, return_tensors="pt").to(device)
    o = model(**inp)
    W, H = img.size
    S = max(W, H) if square_pad else None  # OWLv2 pads to square (bottom/right) before resizing
    return cxcywh_to_xyxy(o.pred_boxes[0], S or W, S or H, W, H), o.logits[0]


@torch.no_grad()
def owl_scores(model, proc, img, query, cands, device, square_pad):
    """Candidate-selection score: max query logit over detections with IoU>=0.5 (best-IoU logit - 10 if none)."""
    boxes, logits = owl_forward(model, proc, img, [query], device, square_pad)
    logit = logits[:, 0]
    ious = iou_matrix(torch.tensor(cands, device=device, dtype=torch.float32), boxes)  # (N,P)
    masked = torch.where(ious >= 0.5, logit[None, :], torch.full_like(ious, -1e4))
    s = masked.max(dim=1).values
    nomatch = s <= -1e3
    if nomatch.any():
        s[nomatch] = logit[ious[nomatch].argmax(dim=1)] - 10.0
    return s.float().cpu().numpy()


@torch.no_grad()
def gdino_dets(model, proc, img, text, device):
    """Grounding DINO with one phrase -> (boxes xyxy (900,4) image coords, scores = max token prob)."""
    gi = proc(images=img, text=text.lower().rstrip(".") + ".", return_tensors="pt").to(device)
    go = model(**gi)
    W, H = img.size
    sc = torch.sigmoid(go.logits[0]).max(-1).values
    cb = go.pred_boxes[0]
    gb = torch.stack([(cb[:, 0] - cb[:, 2] / 2) * W, (cb[:, 1] - cb[:, 3] / 2) * H,
                      (cb[:, 0] + cb[:, 2] / 2) * W, (cb[:, 1] + cb[:, 3] / 2) * H], -1).clamp(min=0)
    return gb, sc
