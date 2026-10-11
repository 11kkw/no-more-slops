"""Training data for the DCSM scorer retraining recipe (ports of baseline/src/training/toydataset_dense.py).

  ObjaverseSpatial  A_rel_B.png + CUTMIX_ hard negative; captions "A picture of a {A} {rel} a {B}" (settype 0/1 swap)
  ObjaverseColor    c1_A_c2_B.png + MIRROR_ (colour composites + the small/big composites, dir derived by name)
  NegationPairs     negated COCO-train pairs (json); captions "An image of a {A} {and not|but no|without} a {B}"
  Paired / Multiple settype-0 and settype-1 copies zipped; one sample from each dataset, truncated to the smallest
  collate           runs in the DataLoader workers: augmentation + CLIP preprocessing
"""
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Subset
from transformers import CLIPImageProcessor

from .constants import CLIP_NAME

REL_TEXT = {"above": "above", "below": "below", "rightof": "to the right of", "leftof": "to the left of"}
REL_OPP = {"above": "below", "below": "above", "rightof": "leftof", "leftof": "rightof"}
NEG_WORDS = ["and not", "but no", "without"]
IMG_EXT = (".png", ".jpg", ".jpeg")


def _imgs(d, exclude):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and exclude not in f)


class ObjaverseSpatial(Dataset):
    def __init__(self, image_dir, settype):
        self.d, self.settype = image_dir, settype
        self.files = [f for f in _imgs(image_dir, "CUTMIX") if any(r in f for r in REL_TEXT)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        A, rel, B = os.path.splitext(f)[0].split("_")
        img = Image.open(os.path.join(self.d, f)).convert("RGB")
        opp = Image.open(os.path.join(self.d, "CUTMIX_" + f)).convert("RGB")
        if self.settype == 0:
            cap, ocap = f"A picture of a {A} {REL_TEXT[rel]} a {B}", f"A picture of a {B} {REL_TEXT[rel]} a {A}"
        else:
            r = REL_OPP[rel]
            cap, ocap = f"A picture of a {B} {REL_TEXT[r]} a {A}", f"A picture of a {A} {REL_TEXT[r]} a {B}"
        return img, opp, cap, ocap


class ObjaverseColor(Dataset):
    """colour composites + the small/big composites (second dir derived by name, as in the repo)."""

    def __init__(self, image_dir, settype):
        self.settype = settype
        self.d2 = image_dir.replace("_attributes_overlay_withbg", "_smallbig_withbg")
        self.items = [(image_dir, f) for f in _imgs(image_dir, "MIRROR")]
        if os.path.isdir(self.d2):
            self.items += [(self.d2, f) for f in _imgs(self.d2, "MIRROR")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, f = self.items[i]
        c1, A, c2, B = os.path.splitext(f)[0].split("_")[:4]
        img = Image.open(os.path.join(d, f)).convert("RGB")
        opp = Image.open(os.path.join(d, "MIRROR_" + f)).convert("RGB")
        if self.settype == 0:
            cap, ocap = f"A picture of a {c1} {A} and a {c2} {B}", f"A picture of a {c2} {A} and a {c1} {B}"
        else:
            cap, ocap = f"A picture of a {c2} {B} and a {c1} {A}", f"A picture of a {c1} {B} and a {c2} {A}"
        return img, opp, cap, ocap


class NegationPairs(Dataset):
    def __init__(self, json_file, data_dir, settype):
        self.data, self.root, self.settype = json.load(open(json_file)), data_dir, settype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        it = self.data[i]
        p1, p2 = it["img1_path"].replace("\\", "/"), it["img2_path"].replace("\\", "/")
        img = Image.open(os.path.join(self.root, p1)).convert("RGB")
        opp = Image.open(os.path.join(self.root, p2)).convert("RGB")
        o1, o2 = it["img1_obj"][self.settype], it["img2_obj"][self.settype]
        cap = f"An image of a {o1} {random.choice(NEG_WORDS)} a {o2}"
        ocap = f"An image of a {o2} {random.choice(NEG_WORDS)} a {o1}"
        return img, opp, cap, ocap


class Paired(Dataset):
    def __init__(self, a, b):
        assert len(a) == len(b)
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i]


class Multiple(Dataset):
    """One sample from each (truncated, shuffled) dataset, as the repo's MultipleDatasets."""

    def __init__(self, datasets):
        n = min(len(d) for d in datasets)
        self.ds = [Subset(d, torch.randperm(len(d))[:n].tolist()) for d in datasets]
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return [d[i] for d in self.ds]


class SaltPepper:
    def __init__(self, p=0.02):
        self.p = p

    def __call__(self, img):
        a = np.asarray(img).copy()
        a[np.random.rand(*a.shape) < self.p] = 255
        a[np.random.rand(*a.shape) < self.p] = 0
        return Image.fromarray(a)


# train.py also builds a RandomResizedCrop but never applies it
AUG = T.Compose([T.RandomPerspective(distortion_scale=0.05, p=0.2), T.RandomRotation(degrees=(-4, 4)), SaltPepper(0.02)])

_IMG_PROC = None
_CLIP_NAME = CLIP_NAME


def configure(clip_name):
    """Set the CLIP checkpoint whose image processor the collate workers use (call before building the DataLoader)."""
    global _CLIP_NAME, _IMG_PROC
    _CLIP_NAME, _IMG_PROC = clip_name, None


def collate(batch):
    """Runs in the DataLoader workers: augment + CLIP-preprocess every image so the main process only does GPU work.
    batch: list over items of list over datasets of (sample settype0, sample settype1); sample = (img, opp, cap, ocap).
    Returns {dataset_k: [(pixel_values (2B,3,224,224), texts (2B)) for version 0, 1]}."""
    global _IMG_PROC
    if _IMG_PROC is None:
        _IMG_PROC = CLIPImageProcessor.from_pretrained(_CLIP_NAME)
    out = {}
    for k in range(len(batch[0])):
        versions = []
        for version in (0, 1):
            smp = [item[k][version] for item in batch]
            images = [AUG(s[0]) for s in smp] + [AUG(s[1]) for s in smp]
            texts = [s[2] for s in smp] + [s[3] for s in smp]
            versions.append((_IMG_PROC(images=images, return_tensors="pt")["pixel_values"], texts))
        out[f"dataset_{k}"] = versions
    return out
