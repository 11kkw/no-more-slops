"""
Retrain the DCSM scoring CNN (Kang et al., ICCV 2025) on the ORIGINAL image-level recipe and data, with the
last-block readout of the frozen CLIP ViT-B/16 swapped: naive (= repo), maskclip, clearclip.

Faithful to baseline/src/training/{train.py,training_util.py,toydataset_dense.py}:
  data      Objaverse spatial composites (A_rel_B.png + CUTMIX_ hard negative), Objaverse colour / small-big
            composites (c1_A_c2_B.png + MIRROR_), negated COCO-train pairs (json); each dataset paired as
            settype 0 / 1 (caption vs. swapped caption); datasets truncated to the smallest one each epoch
  batch     per dataset, per version: 8 images + 8 opposite images  x  8 captions + 8 opposite captions
            -> 16x16 score matrix, rows/cols shuffled, symmetric cross-entropy against the identity
  matrix    197x30 cosine map, complete special-word BPE spans REPLACED by learned 197-d vectors (BOS-aware),
            standardised over the whole batch tensor
  augment   RandomPerspective(0.05, p=.2), RandomRotation(+-4 deg), salt&pepper 2% (the RandomResizedCrop in
            train.py is built but never applied)
  optim     Adam lr 1e-3, grads / 3 before each step, 27 epochs, seed 15, dropout 0

Difference: ONE CLIP forward per batch feeds all requested readouts, each with its own scorer + special-word
embedding + optimizer, so the three scorers see identical images, captions and augmentations.

Checkpoint format mirrors the released ones (model_state_dict, external_logic_token_embeddings, epoch, train_loss)
plus 'readout' and 'special_slicing' = 'bpe_replace_v1' so compositional_eval.py applies the matching rule.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compositional_eval import IMG_SEQ, SPECIAL_WORDS, TEXT_SEQ, CrossModalConvNetwork  # noqa: E402
from probe_localization import dense_features  # noqa: E402
from bpe_alignment import ALIGNMENT_VERSION, replace_special_bpe  # noqa: E402

REL_TEXT = {"above": "above", "below": "below", "rightof": "to the right of", "leftof": "to the left of"}
REL_OPP = {"above": "below", "below": "above", "rightof": "leftof", "leftof": "rightof"}
NEG_WORDS = ["and not", "but no", "without"]
IMG_EXT = (".png", ".jpg", ".jpeg")


# ----------------------------------------------------------------------------- datasets (ports of toydataset_dense.py)
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


_IMG_PROC = None


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


_CLIP_NAME = "openai/clip-vit-base-patch16"


class SaltPepper:
    def __init__(self, p=0.02):
        self.p = p

    def __call__(self, img):
        a = np.asarray(img).copy()
        a[np.random.rand(*a.shape) < self.p] = 255
        a[np.random.rand(*a.shape) < self.p] = 0
        return Image.fromarray(a)


AUG = T.Compose([T.RandomPerspective(distortion_scale=0.05, p=0.2), T.RandomRotation(degrees=(-4, 4)), SaltPepper(0.02)])


# ----------------------------------------------------------------------------- model pieces
def init_special_embedding(device):
    emb = nn.Embedding(len(SPECIAL_WORDS), IMG_SEQ).to(device)
    with torch.no_grad():
        emb.weight = nn.Parameter((F.normalize(emb.weight, p=2, dim=1) + 1) / 3)
    return emb


def initialize_shared(readouts, device):
    """Identical initial values with independent trainable storage per readout."""
    base_net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ).to(device)
    base_emb = init_special_embedding(device)
    return ({r: copy.deepcopy(base_net) for r in readouts},
            {r: copy.deepcopy(base_emb) for r in readouts})


def state_fingerprint(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def insert_special(cos, texts, emb, device, tokenizer=None, input_ids=None):
    """BPE-aligned replacement; use the tokenizer/IDs from the CLIP forward."""
    if tokenizer is None:
        tokenizer = _alignment_tokenizer(_CLIP_NAME)
    return replace_special_bpe(cos, texts, emb, tokenizer, SPECIAL_WORDS, input_ids)


@lru_cache(maxsize=4)
def _alignment_tokenizer(name):
    return CLIPTokenizer.from_pretrained(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir containing the unzipped DCSM training data")
    ap.add_argument("--spatial_dir", default="objaverse_train_composite_newrelations_withbg")
    ap.add_argument("--color_dir", default="objaverse_attributes_overlay_withbg")
    ap.add_argument("--neg_json", default="negated_pairs_cocotrain_v1.json")
    ap.add_argument("--readouts", nargs="+", default=["naive", "maskclip", "clearclip"])
    ap.add_argument("--epochs", type=int, default=27)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop an epoch after this many steps")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda"

    clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float32).to(device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(args.clip)
    global _CLIP_NAME
    _CLIP_NAME = args.clip

    R = args.data_root
    sp = os.path.join(R, args.spatial_dir); co = os.path.join(R, args.color_dir); ng = os.path.join(R, args.neg_json)
    datasets = [Paired(ObjaverseSpatial(sp, 0), ObjaverseSpatial(sp, 1)),
                Paired(ObjaverseColor(co, 0), ObjaverseColor(co, 1)),
                Paired(NegationPairs(ng, R, 0), NegationPairs(ng, R, 1))]
    print("dataset sizes (spatial, color+smallbig, negation):", [len(d) for d in datasets])

    nets, embs = initialize_shared(args.readouts, device)
    init_hashes = {r: {"scorer": state_fingerprint(nets[r]), "embedding": state_fingerprint(embs[r])} for r in args.readouts}
    assert len({x["scorer"] for x in init_hashes.values()}) == 1
    assert len({x["embedding"] for x in init_hashes.values()}) == 1
    json.dump({"seed": args.seed, "alignment": ALIGNMENT_VERSION, "args": vars(args), "initial_hashes": init_hashes},
              open(os.path.join(args.out, "initialization.json"), "w"), indent=2)
    print("shared initialization verified:", init_hashes, flush=True)
    opts = {r: torch.optim.Adam(list(nets[r].parameters()) + list(embs[r].parameters()), lr=args.lr) for r in args.readouts}
    crit = nn.CrossEntropyLoss()
    hist = {r: [] for r in args.readouts}

    for epoch in range(args.epochs):
        loader = DataLoader(Multiple(datasets), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=args.workers, drop_last=True, persistent_workers=False, pin_memory=True)
        json.dump({"epoch": epoch, "seed": args.seed, "dataset_order": ["spatial", "color_smallbig", "negation"],
                   "subset_indices": [d.indices for d in loader.dataset.ds]},
                  open(os.path.join(args.out, f"epoch_{epoch:02d}_sampling.json"), "w"))
        for r in args.readouts:
            nets[r].train()
        run = {r: 0.0 for r in args.readouts}; acc = {r: 0.0 for r in args.readouts}; nstep = 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            for o in opts.values():
                o.zero_grad()
            for key, data in batch.items():
                for pv, texts in data:  # version 0, 1 (augmented + preprocessed in the loader workers)
                    with torch.no_grad():
                        pv = pv.to(device, non_blocking=True)
                        feats, _ = dense_features(clip, pv, 224)  # dict readout -> (B,197,512) L2-normalised
                        tt = tok(text=texts, padding="max_length", max_length=TEXT_SEQ, truncation=True, return_tensors="pt").to(device)
                        txt = clip.text_projection(clip.text_model(**tt).last_hidden_state)
                        txt = txt / txt.norm(dim=-1, keepdim=True)
                    n = len(texts)
                    labels = torch.eye(n, device=device)
                    s1, s2 = torch.randperm(n), torch.randperm(n)
                    for r in args.readouts:
                        cos = torch.einsum("bie,lte->blit", feats[r], txt)  # (Bi,Bt,197,30)
                        cos = insert_special(cos, texts, embs[r], device, tokenizer=tok, input_ids=tt["input_ids"])
                        cos = (cos - cos.mean()) / cos.std()
                        cos = cos[s1][:, s2]
                        lab = labels[s1][:, s2]
                        out = nets[r](cos).squeeze(-1)
                        loss = crit(out, lab) + crit(out.t(), lab.t())
                        loss.backward()
                        run[r] += loss.item()
                        with torch.no_grad():
                            acc[r] += 0.5 * ((out.argmax(0) == lab.argmax(0)).float().mean() + (out.argmax(1) == lab.argmax(1)).float().mean()).item()
            for r in args.readouts:
                for p in list(nets[r].parameters()) + list(embs[r].parameters()):
                    if p.grad is not None:
                        p.grad.data /= 3
                opts[r].step()
            nstep += 1
            if nstep % 100 == 0:
                print(f"epoch {epoch} step {nstep}/{len(loader)}  " + "  ".join(f"{r}: loss {run[r] / (6 * nstep):.3f} acc {acc[r] / (6 * nstep):.3f}" for r in args.readouts) + f"  {time.time() - t0:.0f}s", flush=True)
        for r in args.readouts:
            hist[r].append({"epoch": epoch, "loss": run[r] / max(1, 6 * nstep), "acc": acc[r] / max(1, 6 * nstep)})
        print(f"== epoch {epoch} done ({time.time() - t0:.0f}s): " + "  ".join(f"{r} loss {hist[r][-1]['loss']:.3f} acc {hist[r][-1]['acc']:.3f}" for r in args.readouts), flush=True)
        if (epoch + 1) % 3 == 0 or epoch + 1 == args.epochs:
            for r in args.readouts:
                torch.save({"epoch": epoch, "model_state_dict": nets[r].state_dict(), "external_logic_token_embeddings": embs[r],
                            "readout": r, "special_slicing": ALIGNMENT_VERSION, "initial_hashes": init_hashes[r], "train_hist": hist[r], "args": vars(args)},
                           os.path.join(args.out, f"dcsm_{r}_retrain.pt"))
    json.dump(hist, open(os.path.join(args.out, "train_hist.json"), "w"), indent=1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
