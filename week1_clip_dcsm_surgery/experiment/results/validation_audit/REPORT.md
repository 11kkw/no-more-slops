# Validation audit report

## Scope
UTC: 2026-10-10T14:36:18.162413+00:00
Git baseline: fe64250802b03b9324514ebffe88489134ed06bd
Production training/evaluation files and existing checkpoints were NOT modified. Only audit scripts and audit outputs were added.

## 1. BPE replacement — FAIL
The production implementation uses whitespace word indices without BOS/BPE alignment and replaces only the first repeated word.
Regression command: HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python scripts/test_alignment_regression.py
Exit code: 1; five failing subcases.
training_caption_alignment: {"special_word_occurrences": 71240, "unique_text_word_cases": 71240, "misaligned": 71240}
evaluation_caption_alignment: {"special_word_occurrences": 651, "unique_text_word_cases": 651, "misaligned": 651}

## 2. Dataset reconstruction
Reconstructed current available training pool, NOT historical epoch-level sampled membership.
Training images: 37944; evaluation images: 1025; missing: 0
COCO ID overlaps: 0; identical file overlaps: 0
No claim about near-duplicate/reencoded copies or released-checkpoint provenance.
Subset item counts: {"size": 655, "color": 142, "spatial": 527, "negation": 315, "affirm": 194}

## 3. Historical checkpoint adaptation diagnostic
Existing 27-epoch, seed-15 checkpoints with misaligned replacement. These results are NOT corrected retraining evidence.
| configuration | correct / n | top1 % |
|---|---|---|
| retrained_naive | 295 / 527 | 55.98 |
| retrained_maskclip | 376 / 527 | 71.35 |
| ordinary_scorer_on_maskclip | 169 / 527 | 32.07 |
| retrained_clearclip | 349 / 527 | 66.22 |
| ordinary_scorer_on_clearclip | 135 / 527 | 25.62 |

## 4. Seed stability — BLOCKED
Existing checkpoints use seed 15 only. Reproduced initialization recipe shows naive/maskclip/clearclip networks and learned embeddings have different initial weights.
Do not repeat known-misaligned training to claim robustness. Correct alignment + shared initial state is required before >=3 full seeds.

## 5. Controls — 527 spatial items
| readout | original % | relation removed % | reversed prediction change % | fixed train normalization % | per-candidate normalization % |
|---|---|---|---|---|---|
| naive | 55.98 | 37.38 | 60.34 | 56.36 | 57.50 |
| maskclip | 71.35 | 43.07 | 66.98 | 68.88 | 69.64 |
| clearclip | 66.22 | 41.56 | 56.74 | 66.79 | 69.26 |
All common-crop scores tie: 100% for every readout; current scorer lacks candidate-specific input under a shared crop. Its argmax accuracy is therefore an index/tie-breaking artifact and NOT recognition performance.
Relation reversal measures sensitivity, not correctness against reversed labels. Fixed normalization is calibrated on 16 Objaverse training images, not eval items.
Relation removal changes token positions/sequence length, including the known wrong replacement mechanism; decline is NOT proof of grounded relational reasoning.

## 6. Independent human-labeled test — NOT AVAILABLE
Current configured evaluation uses COCO-val annotation-derived queries. No unused, independently human-verified relation test set has been identified. No human-label claim made.

## 7. Evidence artifacts
success_failure_examples.json: five successes and five failures per readout, historical checkpoint diagnostics only.
data_manifest.json: image lists, dataset arguments, per-subset item/image counts.
No manuscript edit, external upload, or submission performed.

## Execution
audit_alignment_split.py: exit 0 (audit execution success; alignment assertions NOT passed)
audit_transfer_controls.py: exit 0 (completed diagnostics)
test_alignment_regression.py: exit 1 (expected failure exposing bug)
