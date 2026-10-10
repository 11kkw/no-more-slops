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

## Compositional queries (attribute binding / spatial relation / negation)

`scripts/build_comp_queries.py` derives a referring-detection benchmark from COCO val2017 annotations only.
Every item = one image, one query, N same-class candidate boxes (2–6), exactly one target. The class name alone
cannot solve an item; the attribute, relation or negation must be read. Metric = top-1 accuracy among candidates
(chance = mean 1/N).

| subset | query template | target definition |
|---|---|---|
| size | `a picture of a big/small {A}` | area ≥ 3× (≤ ⅓×) every other A |
| color | `a picture of a {color} {A}` | dominant HSV colour of the eroded instance mask (≥55% of pixels); colour absent (<12%) from all distractors |
| spatial | `a picture of a {A} to the left of / right of / above / below a {B}` | B unique in image; exactly one A on that side of B, all others on the other side |
| negation | `a picture of a {A} without a {B}` | exactly one A with no attached B (box overlap ≥30% of smaller box or centre containment); A∈hosts, B∈portables |
| affirm | `a picture of a {A} with a {B}` | control: exactly one A with an attached B |

`scripts/compositional_eval.py` scores every candidate with: OWLv2 / OWL-ViT v1 (query-text detections matched to
candidates by IoU≥0.5), CLIP crop / context-crop embeddings, the pretrained DCSM scoring CNN (both released
checkpoints, context crops, repo-faithful special-word insertion + standardisation), and the six full-image
readout heatmaps (3 last-block readouts × {EOS, DCSM token-max}) mean-pooled inside the candidate box.

## Run

```bash
uv sync                                   # env (torch cu128 for Blackwell GPUs, see pyproject.toml)
uv run scripts/setup_data.py              # COCO val2017 + annotations + DCSM checkpoints (~2 GB)
export HF_HOME=$PWD/hf_cache              # optional: keep HF weights local

uv run scripts/dcsm_detection_eval.py --coco_root data/coco --out results/full_val2017          # all 5000 images
uv run scripts/probe_localization.py   --coco_root data/coco --n_images 500 --out results/probe   # localization probe
uv run scripts/probe_qualitative.py    --image ../baseline/data/test_image2.jpg --queries car "school bus" --out results/qual.png
uv run scripts/chance_level.py data/coco 500                                                     # chance baselines

uv run scripts/build_comp_queries.py --coco_root data/coco --out data/comp_queries                # compositional benchmark (+ spot-check grids)
uv run scripts/compositional_eval.py --out results/comp_full                                     # all scorers, all subsets (~10 min)
uv run scripts/heuristic_baselines.py                                                           # largest/smallest/central-box baselines
uv run scripts/balanced_view.py results/comp_full                                               # size-balanced accuracy table
```

## Layout

```
scripts/   dcsm_detection_eval.py  probe_localization.py  probe_qualitative.py  chance_level.py  setup_data.py
           build_comp_queries.py  compositional_eval.py  heuristic_baselines.py  balanced_view.py
results/   probe_500/ (localization probe)   smoke_50/   full_val2017/ (COCO zero-shot AP)
           comp_smoke/   comp_full/ (compositional top-1)
data/      (git-ignored) coco/val2017, coco/annotations, dcsm_ckpt/
           comp_queries/ (comp_queries.json, stats.md, spotcheck_*.png — regenerate with build_comp_queries.py)
```
