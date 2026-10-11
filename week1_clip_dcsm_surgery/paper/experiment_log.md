# Experiment and drafting evidence log

Recorded on 10 October 2026. Original execution timestamps/revisions are not reconstructed from the drafting date. The repository snapshot revision is `c4c560a8a8d4fb0ceb78d5e8363e087ff113a292`; `evidence/manifest.json` hashes saved inputs and current evaluation code. The original execution revision is unknown.

## E1. Oracle-class localization probe — existing experiment

**Question.** Do different final-block readouts produce useful object-category spatial responses?

**Setup.** Frozen `openai/clip-vit-base-patch16`; first 500 images with non-crowd annotations in sorted val2017 IDs; 224/448 square resizing; naive, value-only, and query–query readouts; EOS or token-max text. Pointing trials are image–category pairs and accept any annotated box of that category. Threshold-box hits use IoU≥0.5 against any such box.

**Code and outputs.** [probe_localization.py](../experiment/scripts/probe_localization.py), [saved JSON](../experiment/results/probe_500/localization_probe.json), [log](../experiment/results/probe_500/run_log.txt). Exported under `evidence/measurements.json#/probe`.

**Result.** Pooled pointing accuracy at 224: 12.0% naive, 51.4% MaskCLIP-style, 62.4% ClearCLIP-style. At 448: 11.8%, 57.1%, 67.5%. Pooled threshold-box hit rates at 224: 5.1%, 21.1%, 26.3%.

**Evidence boundary.** Saved aggregate readouts and code inspected; no new inference or random image sampling. Aggregate output lacks per-image trial records needed for an image-cluster confidence interval. Probe results are not full-instance detection or relation-conditioned localization.

## E2. Category-conditioned detection — existing experiment

**Question.** How do these dense features perform in component-box detection (P1) and on OWLv2 objectness proposals (P2)?

**Setup.** 4,952 nonempty non-crowd val2017 images; COCO 80-category prompts; standard bbox AP0.50:0.95; final readout/text/resolution grid. P1 upsample→class softmax×100→winning class probability>0.3→components≥4pixels. P2 top100 objectness boxes→patch mean→softmax×100→top3classes→objectness×class probability. OWL native references use different box/class ranking.

**Code and outputs.** [dcsm_detection_eval.py](../experiment/scripts/dcsm_detection_eval.py), [summary](../experiment/results/full_val2017/summary.json), [log](../experiment/results/full_val2017/run_log.txt), [complete AP table](tables/detection.md).

**Result.** P2 pooled AP at 224: 2.6 naive, 26.0 MaskCLIP-style, 27.3 ClearCLIP-style. P1 maximum AP rounds to 4.6. Native OWLv1=28.8, native OWLv2=46.8. Resolution effects differ by configuration. Total saved elapsed time is 4441.449 seconds, not a per-model efficiency comparison.

**Evidence boundary.** Saved summary inspected and table units checked; no detections regenerated. Native/P2 AP gap is not a classification-only experiment. No detection confidence intervals were computed from aggregates.

## E3. Candidate selection on the failure-mode query benchmark (ground-truth candidates) — existing experiment

**Question.** Can a scorer select one annotated same-category object from a templated description of size, color, image-plane relation or attachment?

**Setup.** Builder seed0, max6/min2candidates, per-class60/per-image2 caps per subset. 1,833 queries from 1,025 distinct images, five subsets; 12 scorers. Ground-truth candidate boxes and reference annotations supplied to crop/map arms. Native OWL detection scores matched at IoU≥0.5 with a fallback penalty. Frozen evaluation includes already-trained DCSM and detector components.

**Code and outputs.** [builder](../experiment/scripts/build_comp_queries.py), [scorer adapter](../experiment/scripts/compositional_eval.py), [retained queries](../experiment/results/comp_full/benchmark/comp_queries.json), [per-item predictions](../experiment/results/comp_full/per_item.json), [summary](../experiment/results/comp_full/summary.json), [log](../experiment/results/comp_full/run_log.txt).

**Result.** Raw spatial DCSM ObjA=57.9%, COCO=56.9%; OWLv2=42.5%; naive EOS=49.3%; ClearCLIP-style EOS=39.7%. Color OWLv2=74.6%; size OWLv2=64.7%. See full [raw table](tables/compositional_raw.md).

**Evidence boundary.** Recomputed accuracy/chance matches original saved summary. This is conditional selection; label quality and training overlap are unresolved. DCSM scorer uses inherited whitespace-index row insertion and joint candidate-batch standardization. Surgery+learned-DCSM combinations are absent.

## E4. Shortcut audit and stratum balancing — recomputed from existing artifacts

**Question.** How sensitive are raw scores to target size and query imbalance?

**Setup.** Recomputed largest/smallest/most-central heuristics from saved boxes, compared with existing heuristic output. Balanced big/small for size, target-largest/not-largest by bbox area otherwise, with equally weighted strata and equally reweighted expected chance. Compared reaggregated values with the saved balanced table.

**Code and outputs.** [original balancing code](../experiment/scripts/balanced_view.py), [original heuristic code](../experiment/scripts/heuristic_baselines.py), repository skill [evidence exporter](../../skills/research/experiment-to-paper/scripts/export_dcsm_evidence.py); [balanced table](tables/compositional_balanced.md), [heuristics](tables/heuristics.md), [measurements](evidence/measurements.json).

**Result.** Largest-box raw size87.3%; smallest-box raw without72.7%; size OWLv2 raw64.7→balanced41.7%. Balanced spatial DCSM58.1/57.3 vschance45.7; naive EOS50.3 means DCSM is not uniquely above chance. Balanced without naive token-max55.7 means the results do not show universal below-chance performance.

**Code-audit findings.** Size labels use COCO annotation area while heuristics useboxarea; big branch is preferred. With/without branch priority yields different populations, including273/315 versus44/194two-candidate items. These are protocol findings, not new model experiments.

**Evidence boundary.** Balancing changes the estimand; it does not remove all shortcuts or label errors. Original result/code files remain intact.

## E5. Image-cluster uncertainty and scientific figures — new saved-prediction analysis

**Question.** What sample uncertainty surrounds balanced scores and paired method differences?

**Setup.** CPU-only analysis of E3 predictions. Within each subset, sample its observed image IDs with replacement; all items in an image share multiplicity across methods. Recompute both stratum means and equally weight them. 10,000 replicates; seed 20261010; percentile 95% intervals, no multiplicity correction. No zero-stratum replicates occurred in this run. Package versions and hashes are recorded in JSON.

**Code and outputs.** [analyze_saved_results.py](scripts/analyze_saved_results.py), [cluster bootstrap](evidence/cluster_bootstrap.json), [spatial interval table](tables/spatial_uncertainty.md), [detection plot](figures/detection_pooled.png), [spatial plot](figures/spatial_balanced.png); both plots also in SVG/PDF.

**Result.** Spatial ObjA58.1%[53.7,62.4], COCO57.3%[53.2,61.2]. ObjA−chance+12.4pp[8.0,16.7]; ObjA−CLIPcontext+19.1pp[14.3,23.9]. All point estimates match E4.

**Evidence boundary.** Intervals condition on this query collection and released models. They do not cover dataset labeling uncertainty, unseen categories, unseen task populations, or training overlap. Figure 1has no error bars because detection trial outputs were unavailable. Plots were visually inspected for legibility.

## E6. DCSM scorer retraining on surgery readouts — new experiment (2026-10-10/11)

**Question.** Does retraining the DCSM scoring network with the original recipe on MaskCLIP / ClearCLIP features improve failure-mode candidate selection over the same recipe on the ordinary readout?

**Setup.** Original image-level recipe and Drive data (Objaverse spatial / colour / small-big composites, negated COCO-train pairs); one shared CLIP forward per batch feeds three scorers with identical initial weights; BPE-aligned Functional-Row replacement; seeds 15, 16, 17; 27 epochs each (~77 min on one RTX PRO 6000).

**Code and outputs.** [train_dcsm_scorer.py](../experiment/scripts/train_dcsm_scorer.py), [run_bpe_multiseed.py](../experiment/scripts/run_bpe_multiseed.py), package modules `dcsm_surgery.data`, `dcsm_surgery.scorer`, `dcsm_surgery.bpe_alignment`; results `../experiment/results/bpe_multiseed/seed_*/{train,evaluation,adaptation}`, aggregate `aggregate.json`; audit `../experiment/results/validation_audit/`.

**Result.** Size-balanced spatial top-1: naive 59.0 ± 3.9, MaskCLIP 69.8 ± 2.2, ClearCLIP 69.3 ± 3.4. Colour / negation / size unchanged within noise. Naive-readout scorer on surgery features: 33.7 ± 1.3 / 28.5 ± 1.0 raw.

**Evidence boundary.** Three seeds; the released COCO-model recipe is not reproducible from the public code; the released checkpoints are evaluated with the repository's misaligned column rule.

## E7. Referring detection on OWLv2 proposals — new experiment (2026-10-11)

**Question.** Without ground-truth candidates, does the retrained scorer over OWLv2 class-name proposals detect the referred object better than OWLv2, OWL-ViT and Grounding DINO given the same sentence?

**Setup.** 1,833 items; one OWLv2 forward with [query, "a photo of a {A}", "a photo of a {B}"]; candidates p(A) ≥ 0.1, NMS 0.5, ≤ 10; reference B̂ = top p(B) (spatial) or nearest B (negation/affirm); final score p(A) × softmax(scorer). Baselines: OWLv2 / OWL-ViT native, Grounding DINO-T and -B native (NMS 0.5, ≤ 100 dets), p(A)-only and CLIP-context re-rankers on the same candidates. Metrics: Acc@0.5 (raw, balanced), COCO-style AP / AP50 per subset, candidate recall, paired bootstrap on spatial Acc@0.5.

**Code and outputs.** [referring_detection_eval.py](../experiment/scripts/referring_detection_eval.py), `dcsm_surgery.detectors`, `dcsm_surgery.metrics.referring_ap`; results `../experiment/results/refdet_full/{summary.md,summary.json,per_item.json,run_log.txt}` (per-detection dump `detections.json` is git-ignored, ~41 MB).

**Result.** Spatial Acc@0.5 / AP: MaskCLIP head 49.7 ± 0.6 / 34.4 ± 1.0; ClearCLIP 48.3 ± 1.3 / 34.4 ± 1.2; naive 40.7 ± 1.3 / 28.1 ± 0.6; p(A) only 37.8 / 27.5; OWLv2 7.8 / 6.7; OWL-ViT 12.0 / 5.5; GDINO-T 17.1 / 14.0; GDINO-B 63.0 / 52.0. Candidate recall 89.2. No benefit on size / color / negation / affirm.

**Evidence boundary.** Grounding DINO-B is pre-trained with COCO and RefCOCO labels (not zero-shot). Reference-box errors and candidate misses are not decomposed. ~9 min on one GPU; the pre- and post-refactor smoke runs give identical numbers.

## Code organisation (2026-10-11)

Shared code was moved into the package `experiment/src/dcsm_surgery` (installed by `uv sync`); the scripts are thin entry points and `uv run pytest` runs 29 unit tests (boxes, metrics, readouts and scorer with the cached CLIP, BPE alignment, shared initialisation). Script names referenced above are unchanged; the function `owl_native_dets` referenced by earlier claim maps now lives in `dcsm_surgery.detectors`.

## Draft preparation and verification

Primary sources for 8 references were retrieved; citation IDs and short excerpts were recorded in a project-specific ledger. A complete English manuscript and Korean explanation were written and revised through sequential author-side review. The source verifier checks citation/evidence formatting, and a separate numeric verifier compares manuscript tables to artifacts. Neither substitutes for scientific review. Exact verification results are recorded in `evidence/draft_validation.json` after those checks run.
