import numpy as np
import pytest
import torch
from PIL import Image

from dcsm_surgery.boxes import (area, box_patch_mean, centre, crop, cxcywh_to_xyxy, iou, iou_matrix, nms_keep,
                                pad_box, topk_dets, union_box, xyxy_to_coco)


def test_iou_scalar_and_matrix_agree():
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert abs(iou(a, b) - 25 / 175) < 1e-9
    assert iou(a, [20, 20, 30, 30]) == 0.0
    m = iou_matrix(torch.tensor([a], dtype=torch.float32), torch.tensor([b, [0, 0, 10, 10]], dtype=torch.float32))
    assert torch.allclose(m, torch.tensor([[25 / 175, 1.0]]), atol=1e-6)


def test_area_centre_union_pad():
    b = [10, 20, 30, 60]
    assert area(b) == 800 and centre(b) == (20, 40)
    assert union_box(b, [0, 30, 15, 70]) == [0, 20, 30, 70]
    p = pad_box(b, W=32, H=100, frac=0.1)
    assert p == [8, 16, 32, 64]  # x2 clamped to W


def test_cxcywh_conversion_and_clamp():
    rel = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.95, 0.5, 0.5, 0.5]])
    out = cxcywh_to_xyxy(rel, 100, 50, 100, 50)
    assert out[0].tolist() == [25.0, 12.5, 75.0, 37.5]
    assert out[1, 2].item() == 100.0  # clamped to W
    raw = cxcywh_to_xyxy(rel, 100, 50, 100, 50, clamp=False)
    assert raw[1, 2].item() > 100.0
    assert xyxy_to_coco(out[0]) == [25.0, 12.5, 50.0, 25.0]


def test_crop_never_empty():
    img = Image.new("RGB", (40, 40))
    c = crop(img, [10, 10, 10.4, 10.2])
    assert c.size[0] >= 2 and c.size[1] >= 2


def test_box_patch_mean_and_tiny_box_fallback():
    g, patch, size = 4, 16, 64
    hm = torch.arange(g * g, dtype=torch.float32).reshape(g, g)  # row-major values 0..15
    W = H = 128  # image is 2x the heatmap space
    full = box_patch_mean(hm, [[0, 0, 128, 128]], size, W, H, patch)
    assert np.isclose(full[0], hm.mean().item())
    tiny = box_patch_mean(hm, [[2, 2, 3, 3]], size, W, H, patch)  # no patch centre inside -> nearest patch (0,0)
    assert tiny[0] == 0.0


def test_nms_keep_and_topk():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.5])
    keep = nms_keep(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]
    b, s = topk_dets(boxes, scores, 0.5, 1)
    assert b == [[0, 0, 10, 10]] and s == pytest.approx([0.9])
