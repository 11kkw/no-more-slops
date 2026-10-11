import numpy as np
import torch

from dcsm_surgery.metrics import average_precision, box_from_heatmap, crude_detections, paired_bootstrap, referring_ap


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1])) == (1.0 + 2 / 3) / 2
    assert np.isnan(average_precision(np.array([0.5]), np.array([0])))


def test_box_from_heatmap_component_contains_argmax():
    h = np.zeros((4, 4), dtype=np.float32); h[1, 1] = 1.0
    box, up = box_from_heatmap(h, 16)
    assert up.shape == (16, 16)
    x1, y1, x2, y2 = box
    assert x1 <= 6 <= x2 and y1 <= 6 <= y2  # patch (1,1) centre in 16-px space
    assert box_from_heatmap(np.zeros((4, 4), dtype=np.float32), 16)[0] is None


def test_crude_detections_one_box_per_dominant_class():
    hm = torch.zeros(2, 4, 4); hm[0, :2, :2] = 1.0; hm[1, 2:, 2:] = 1.0
    dets = crude_detections(hm, 16, img_id=7, cat_ids=[11, 22], sx=1.0, sy=1.0, temp=100.0)
    cats = {d["category_id"] for d in dets}
    assert cats == {11, 22} and all(d["image_id"] == 7 and d["score"] > 0.3 for d in dets)


def test_referring_ap_perfect_and_wrong():
    items = [(0, "spatial", [0, 0, 10, 10], 100, 100), (1, "spatial", [50, 50, 70, 70], 100, 100), (2, "color", [0, 0, 10, 10], 100, 100)]
    perfect = {"m": [(0, [0, 0, 10, 10], 0.9), (1, [50, 50, 70, 70], 0.8), (2, [0, 0, 10, 10], 0.7)]}
    r = referring_ap(items, perfect, ["spatial", "color"])
    assert np.isclose(r["m"]["spatial"]["AP"], 1.0) and np.isclose(r["m"]["color"]["AP50"], 1.0)
    wrong = {"m": [(0, [80, 80, 90, 90], 0.9), (1, [50, 50, 70, 70], 0.8), (2, [0, 0, 10, 10], 0.7)]}
    r = referring_ap(items, wrong, ["spatial", "color"])
    assert 0.0 < r["m"]["spatial"]["AP50"] < 1.0 and np.isclose(r["m"]["color"]["AP50"], 1.0)
    assert referring_ap(items, {"empty": []}, ["spatial", "color"])["empty"]["spatial"]["AP"] == 0.0


def test_paired_bootstrap_is_deterministic_and_brackets_diff():
    a = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1]); b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    d1 = paired_bootstrap(a, b, n_resamples=2000, seed=1); d2 = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert d1 == d2
    diff, lo, hi = d1
    assert np.isclose(diff, 0.3) and lo <= diff <= hi
