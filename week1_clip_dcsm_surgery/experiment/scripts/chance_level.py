import os, sys, numpy as np
from pycocotools.coco import COCO
root = sys.argv[1]; n = int(sys.argv[2])
coco = COCO(os.path.join(root, "annotations", "instances_val2017.json"))
ids = sorted(coco.getImgIds()); ids = [i for i in ids if len(coco.getAnnIds(imgIds=i, iscrowd=False)) > 0][:n]
fracs, prev = [], []
for i in ids:
    info = coco.loadImgs(i)[0]; W, H = info["width"], info["height"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
    by = {}
    for a in anns: by.setdefault(a["category_id"], []).append(a["bbox"])
    prev.append(len(by) / 80)
    for c, bxs in by.items():
        m = np.zeros((H, W), bool)
        for x, y, w, h in bxs: m[int(y):int(y+h)+1, int(x):int(x+w)+1] = True
        fracs.append(m.mean())
print(f"pairs={len(fracs)}  random-pointing chance (mean GT-box area fraction) = {np.mean(fracs):.3f}")
print(f"image-level multi-label chance mAP ~ mean class prevalence = {np.mean(prev):.3f}")
