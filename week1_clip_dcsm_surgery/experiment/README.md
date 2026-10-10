# Experiment: DCSM readout surgery for zero-shot detection

Frozen CLIP ViT-B/16. The DCSM matrix (image patches × text tokens) is used as the **classification signal**
for zero-shot COCO detection under three ways of reading out the last transformer block, and compared with
open-vocabulary detectors (OWL-ViT v1, OWLv2).

| readout | what changes in the last ViT block | source |
|---|---|---|
| `naive` | nothing: `last_hidden_state → post_layernorm → visual_projection` (the DCSM repo's code path) | Kang et al. 2025 |
| `maskclip` | value embeddings only (no q-k attention, no residual, no FFN) | Zhou et al. ECCV 2022 |
| `clearclip` | q-q self-self attention, no residual, no FFN | Lan et al. ECCV 2024 |

Text readouts: `dcsm_tok` = max over content-token columns of the DCSM matrix (the DCSM representation),
`pooled` = CLIP EOS embedding (reference).

Protocols (all training-free):

- **P1 dense**: per-patch softmax over the 80 class maps → threshold → connected components → boxes.
- **P2 proposals**: top-100 class-agnostic boxes from OWLv2's objectness head, each classified by mean-pooling the
  class map inside the box; score = objectness × p(class). Same boxes as `owlv2_native`, so the gap isolates the
  classification signal.

## Run

```bash
uv sync                                   # env (torch cu128 for Blackwell GPUs, see pyproject.toml)
uv run scripts/setup_data.py              # COCO val2017 + annotations + DCSM checkpoints (~2 GB)
export HF_HOME=$PWD/hf_cache              # optional: keep HF weights local

uv run scripts/dcsm_detection_eval.py --coco_root data/coco --out results/full_val2017          # all 5000 images
uv run scripts/probe_localization.py   --coco_root data/coco --n_images 500 --out results/probe   # localization probe
uv run scripts/probe_qualitative.py    --image ../baseline/data/test_image2.jpg --queries car "school bus" --out results/qual.png
uv run scripts/chance_level.py data/coco 500                                                     # chance baselines
```

## Layout

```
scripts/   dcsm_detection_eval.py  probe_localization.py  probe_qualitative.py  chance_level.py  setup_data.py
results/   probe_500/ (localization probe, 2026-10-09)   smoke_50/   full_val2017/
data/      (git-ignored) coco/val2017, coco/annotations, dcsm_ckpt/
```
