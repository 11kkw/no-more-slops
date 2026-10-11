# Experiment: readout surgery turns DCSM into a spatial-relationship detection head for CLIP

Frozen CLIP ViT-B/16. The DCSM matrix (image patches × text tokens) is read out three ways from the last
transformer block, used (a) as a training-free classification signal for zero-shot COCO detection and (b) as the
input of the DCSM scoring network, retrained with the original recipe on each readout, which re-ranks OWLv2
proposals for queries that target CLIP's three documented failure modes. Baselines: OWL-ViT v1, OWLv2,
Grounding DINO (tiny, base).

| readout | what changes in the last ViT block | source |
|---|---|---|
| `naive` | nothing: `last_hidden_state → post_layernorm → visual_projection` (the DCSM repo's code path) | Kang et al. 2025 |
| `maskclip` | value embeddings only (no q-k attention, no residual, no FFN) | Zhou et al. ECCV 2022 |
| `clearclip` | q-q self-self attention, no residual, no FFN | Lan et al. ECCV 2024 |

Text readouts: `dcsm_tok` = max over content-token columns of the DCSM matrix (the DCSM representation),
`pooled` = CLIP EOS embedding (reference).

## Terminology (follows Kang et al.)

- **failure modes**: attribute binding, spatial relationship (the absolute location of one object or the relationship
  between two objects), negation. Earlier drafts said "compositional queries"; that term is retired.
- **failure-mode query benchmark**: queries built from COCO val2017 annotations (`scripts/build_comp_queries.py`);
  subsets `size`, `color` (attribute binding), `spatial` (spatial relationship), `negation` + control `affirm`.
- **candidate selection**: protocol with ground-truth candidate boxes; metric = top-1 accuracy (`compositional_eval.py`).
- **referring detection**: protocol with image + query only; metrics = Acc@0.5 of the top-1 box and COCO-style AP
  (`referring_detection_eval.py`).
- **single-class-name detection**: COCO 80-class zero-shot detection (`dcsm_detection_eval.py`); protocols P1 dense
  (per-patch softmax → components → boxes) and P2 proposals (OWLv2 objectness boxes classified by the pooled class map).
- **Functional Rows**: the 11 learned 197-d column vectors (above/below/left/right/small/big/not/no/without/but/absent)
  that replace those words' cosine columns in the DCSM scorer; the code calls them `SPECIAL_WORDS`.

## Failure-mode query benchmark

Every item = one image, one query, N same-class candidate boxes (2–6), exactly one target. The class name alone
cannot solve an item; the attribute, relation or negation must be read.

| subset | query template | target definition |
|---|---|---|
| size | `a picture of a big/small {A}` | area ≥ 3× (≤ ⅓×) every other A |
| color | `a picture of a {color} {A}` | dominant HSV colour of the eroded instance mask (≥55% of pixels); colour absent (<12%) from all distractors |
| spatial | `a picture of a {A} to the left of / right of / above / below a {B}` | B unique in image; exactly one A on that side of B, all others on the other side |
| negation | `a picture of a {A} without a {B}` | exactly one A with no attached B (box overlap ≥30% of smaller box or centre containment); A∈hosts, B∈portables |
| affirm | `a picture of a {A} with a {B}` | control: exactly one A with an attached B |

Candidate-selection scorers: OWLv2 / OWL-ViT v1 (query-text detections matched to candidates by IoU≥0.5), CLIP crop /
context-crop embeddings, the DCSM scoring network (released checkpoints with the repo's column rule; retrained
checkpoints with the BPE-aligned rule), and the six full-image readout heatmaps mean-pooled inside the box.

Referring-detection pipeline (`cands/*` rows): one OWLv2 forward with `[query, "a photo of a {A}", "a photo of a {B}"]`;
candidates = boxes with p(A) ≥ 0.1 after NMS (≤ 10); reference B̂ = top p(B) box (spatial) or nearest B box
(negation/affirm); each candidate's context crop = union(candidate, B̂) is scored by the re-ranker; final score =
p(A) × softmax(re-ranker). Native baselines receive the full query sentence.

## Retraining the DCSM scorer on surgery readouts

`scripts/train_dcsm_scorer.py` ports the original image-level recipe (Objaverse spatial / colour / small-big
composites + negated COCO-train pairs, 16×16 symmetric CE, learned Functional Rows, Adam 1e-3, 27 epochs) and trains
one scorer per last-block readout from a single shared CLIP forward (`dcsm_surgery.data`, `dcsm_surgery.scorer`).
Checkpoints record `readout` and `special_slicing` so the evaluation scripts apply the matching column rule.

## Setup and tests

```bash
uv sync                                   # env + editable install of src/dcsm_surgery (torch cu128 for Blackwell GPUs)
uv run pytest                             # unit tests (CPU; CLIP-dependent tests use the local hf_cache)
uv run scripts/setup_data.py              # COCO val2017 + annotations + DCSM checkpoints (~2 GB)
export HF_HOME=$PWD/hf_cache              # optional: keep HF weights local
```

## Run

```bash
uv run scripts/dcsm_detection_eval.py --coco_root data/coco --out results/full_val2017          # single-class-name zero-shot detection, 4,952 images
uv run scripts/probe_localization.py   --coco_root data/coco --n_images 500 --out results/probe   # localization probe
uv run scripts/probe_qualitative.py    --image ../baseline/data/test_image2.jpg --queries car "school bus" --out results/qual.png
uv run scripts/chance_level.py data/coco 500                                                     # chance baselines

uv run scripts/build_comp_queries.py --coco_root data/coco --out data/comp_queries                # failure-mode query benchmark (+ spot-check grids)
uv run scripts/compositional_eval.py --out results/comp_full                                     # candidate selection, all scorers (~10 min)
uv run scripts/heuristic_baselines.py                                                           # largest/smallest/central-box baselines
uv run scripts/balanced_view.py results/comp_full                                               # size-balanced accuracy table

# DCSM training data: download the two Drive zips (link in ../baseline/README.md) into data/dcsm_train/dcsm_training/, unzip into data/dcsm_train/;
# the negation json references COCO train2017 originals -> fetch them into data/dcsm_train/train2017/train2017/ (see findings.md §4.4)
uv run scripts/train_dcsm_scorer.py --data_root data/dcsm_train --spatial_dir objaverse_train_cutmix_newrelations_withbg --out results/retrain  # ~77 min
uv run scripts/run_bpe_multiseed.py                                                             # seeds 15/16/17: train + candidate selection + adaptation controls
uv run scripts/compositional_eval.py --dcsm_ckpts data/dcsm_ckpt/*.pt results/bpe_multiseed/seed_15/train/*.pt --out results/comp_retrain
uv run scripts/balanced_view.py results/comp_retrain && uv run scripts/paired_bootstrap.py results/comp_retrain spatial dcsm_maskclip_retrain_ctx dcsm_naive_retrain_ctx

# referring detection (no GT candidates): OWLv2 class-name proposals + re-rankers vs OWLv2 / OWL-ViT / Grounding DINO native (~10 min)
uv run scripts/referring_detection_eval.py --dcsm_ckpts data/dcsm_ckpt/*.pt results/bpe_multiseed/seed_1*/train/*.pt --out results/refdet_full
uv run scripts/paired_bootstrap.py results/refdet_full spatial cands/dcsm_maskclip_retrain_s15 gdino_tiny
```

## Layout

```
src/dcsm_surgery/   library code (installed by uv sync):
                    constants.py  clip_readouts.py (preprocess, dense_features, text_features, heatmaps)
                    bpe_alignment.py (Functional-Row column replacement)  scorer.py (CrossModalConvNetwork, DCSMScorer, init helpers)
                    boxes.py  detectors.py (OWL / Grounding DINO)  metrics.py (COCO AP, referring AP, probe metrics, bootstrap)
                    data.py (retraining datasets, augmentation, collate)
scripts/            experiment entry points (thin CLIs over the package):
                    probe_localization.py  probe_qualitative.py  dcsm_detection_eval.py  chance_level.py  setup_data.py
                    build_comp_queries.py  compositional_eval.py  referring_detection_eval.py  heuristic_baselines.py
                    balanced_view.py  paired_bootstrap.py  train_dcsm_scorer.py  run_bpe_multiseed.py
                    evaluate_bpe_adaptation.py  audit_alignment_split.py  audit_transfer_controls.py (validation audits)
tests/              pytest: boxes / metrics / readouts + scorer (CPU CLIP) / BPE alignment / shared initialisation
results/            probe_500/  smoke_50/  full_val2017/ (single-class-name detection AP)
                    comp_smoke/  comp_full/  comp_retrain/ (candidate selection)   retrain/ (first retrain, pre-BPE-fix)
                    validation_audit/  bpe_alignment/  bpe_multiseed/ (3-seed retrain + evaluation)
                    refdet_smoke/  refdet_full/ (referring detection: Acc@0.5 + AP, OWLv2 / OWL-ViT / Grounding DINO baselines)
data/               (git-ignored) coco/val2017, coco/annotations, dcsm_ckpt/, dcsm_train/, comp_queries/
```
