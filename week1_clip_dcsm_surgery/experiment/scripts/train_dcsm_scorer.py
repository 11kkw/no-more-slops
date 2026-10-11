"""
Retrain the DCSM scoring CNN (Kang et al., ICCV 2025) on the ORIGINAL image-level recipe and data, with the
last-block readout of the frozen CLIP ViT-B/16 swapped: naive (= repo), maskclip, clearclip.

Faithful to baseline/src/training/{train.py,training_util.py,toydataset_dense.py} (see dcsm_surgery.data):
  data      Objaverse spatial composites (+ CUTMIX_ hard negative), Objaverse colour / small-big composites
            (+ MIRROR_), negated COCO-train pairs (json); each dataset paired as settype 0 / 1; datasets
            truncated to the smallest one each epoch
  batch     per dataset, per version: 8 images + 8 opposite images  x  8 captions + 8 opposite captions
            -> 16x16 score matrix, rows/cols shuffled, symmetric cross-entropy against the identity
  matrix    197x30 cosine map, complete Functional-Row BPE spans REPLACED by learned 197-d vectors (BOS-aware),
            standardised over the whole batch tensor
  augment   RandomPerspective(0.05, p=.2), RandomRotation(+-4 deg), salt&pepper 2%
  optim     Adam lr 1e-3, grads / 3 before each step, 27 epochs, seed 15, dropout 0

Difference: ONE CLIP forward per batch feeds all requested readouts, each with its own scorer + Functional-Row
embedding + optimizer, so the three scorers see identical images, captions and augmentations.

Checkpoint format mirrors the released ones (model_state_dict, external_logic_token_embeddings, epoch, train_loss)
plus 'readout' and 'special_slicing' = 'bpe_replace_v1' so the evaluation scripts apply the matching rule.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer

from dcsm_surgery import data as D
from dcsm_surgery.bpe_alignment import ALIGNMENT_VERSION
from dcsm_surgery.clip_readouts import dense_features
from dcsm_surgery.constants import TEXT_SEQ
from dcsm_surgery.scorer import initialize_shared, insert_special, state_fingerprint


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
    D.configure(args.clip)

    R = args.data_root
    sp = os.path.join(R, args.spatial_dir); co = os.path.join(R, args.color_dir); ng = os.path.join(R, args.neg_json)
    datasets = [D.Paired(D.ObjaverseSpatial(sp, 0), D.ObjaverseSpatial(sp, 1)),
                D.Paired(D.ObjaverseColor(co, 0), D.ObjaverseColor(co, 1)),
                D.Paired(D.NegationPairs(ng, R, 0), D.NegationPairs(ng, R, 1))]
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
        loader = DataLoader(D.Multiple(datasets), batch_size=args.batch_size, shuffle=True, collate_fn=D.collate,
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
