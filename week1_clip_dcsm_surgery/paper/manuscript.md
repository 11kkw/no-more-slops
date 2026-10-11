# Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP

*Working manuscript, 11 October 2026. Author names, affiliations, and a target venue have not been assigned. All numbers come from the experiment artifacts under `../experiment/results/`; the evaluation code is the `dcsm_surgery` package and the scripts in `../experiment/scripts/`.*

## Abstract

Dense Cosine Similarity Maps (DCSMs) keep the full matrix of CLIP image-patch × text-token similarities and score it with a small convolutional network, which lets a frozen CLIP read spatial relationships that its global embedding cannot. We show that the patch features consumed by the released DCSM code are anti-localized: with the ordinary last-block readout, the patch that best matches a category name lies inside a box of that category only 12% of the time (chance 17%), and the matrix classifies OWLv2 proposals at 1–3 COCO AP. Replacing only the last-block readout by the MaskCLIP (value-only) or ClearCLIP (query–query attention) variant, with the encoder frozen, raises that training-free classification to 26–27 AP. On these repaired features we retrain the DCSM scoring network with the original image-level recipe and data. On a failure-mode query benchmark built from COCO val2017 annotations (attribute binding, spatial relationship, negation), the retrained scorer selects the correct same-category instance for spatial-relationship queries with 69.8 ± 2.2% size-balanced accuracy (three seeds), versus 59.0 ± 3.9% for the same recipe on the ordinary readout and 43% for OWLv2. Used as a re-ranker over OWLv2 class-name proposals, it becomes a referring detector that outputs boxes: on spatial-relationship queries it reaches 49.7 ± 0.6% Acc@0.5 and 34.4 ± 1.0 AP, against 7.8% / 6.7 for OWLv2, 12.0% / 5.5 for OWL-ViT and 17.1% / 14.0 for Grounding DINO-T given the same sentences. Grounding DINO-B, which was pre-trained with COCO and RefCOCO supervision, remains stronger (63.0% / 52.0). The gain is specific to spatial relationships: on single-class-name COCO detection the DCSM signal has no advantage over the EOS embedding and trails OWLv2 by 20 AP, and neither readout surgery nor retraining helps attribute or negation queries.

## 1. Introduction

Open-vocabulary detectors such as OWL-ViT, OWLv2 and Grounding DINO accept free-form text, but a query like "a dog to the left of a person" is, for most of them, a query for dogs and people: the relation word is not used to pick the instance. Kang et al. identified attribute binding, spatial relationships and negation as three failure modes of CLIP's global embedding, proved that no single embedding can satisfy all of them, and proposed to keep the dense cosine similarity map between image patches and text tokens instead, scoring it with a trained CNN whose Functional Rows replace the columns of relation and negation words.[2] Their benchmarks are image–caption matching tasks; the question we ask is whether the same representation can select and localize the right object in a scene.

Two observations shape the answer. First, the DCSM code reads CLIP's last block in the ordinary way, and those patch features do not localize: the residual and feed-forward paths of the last block pull every patch toward the global representation, so the patch most similar to "bus" tends to lie outside the bus. MaskCLIP and ClearCLIP repair this for segmentation by reading only the value path or a query–query attention of the last block.[3][4] Second, a scorer trained on inverted features cannot simply be moved to repaired ones; it must be retrained on the representation it will see.

This paper therefore makes a method out of three existing parts and measures what the combination buys. The contributions are:

- **A quantified diagnosis.** With the ordinary readout, oracle-category pointing accuracy is below chance and proposal classification is 1–3 AP; readout surgery alone lifts them to 62% pointing and 27 AP, with the encoder frozen (§5.1).
- **A retrained DCSM scorer on surgery features.** Using the original recipe and data, the scorer on MaskCLIP or ClearCLIP features gains about ten points of size-balanced accuracy on spatial-relationship candidate selection over the same recipe on the ordinary readout, with the largest gains on above/below, which the training data covers only through caption swaps (§5.2).
- **A spatial-relationship detection head.** Re-ranking OWLv2 class-name proposals with the retrained scorer yields a referring detector that beats OWLv2, OWL-ViT and Grounding DINO-T on spatial-relationship queries by 32–42 Acc@0.5 points, and the relation-blind proposal baseline by 11 points (§5.3).

We state the limits as plainly. Single-class-name detection, attribute binding and negation do not benefit; Grounding DINO-B, trained with COCO and RefCOCO labels, is stronger than the head; the benchmark is automatically generated from COCO annotations, and candidate recall from OWLv2 caps the head at 89% on spatial queries.

## 2. Related Work

**Dense CLIP features.** MaskCLIP obtains dense labels from CLIP by using the value embeddings of the last attention layer;[3] ClearCLIP removes the residual connection and feed-forward network of the last block and uses query–query attention.[4] We use both as drop-in readouts of the frozen HF CLIP ViT-B/16 and do not reproduce their segmentation results.

**Dense similarity scoring.** DCSM keeps the patch-by-token similarity matrix, replaces the columns of functional words with learned vectors (Functional Rows), and trains a two-layer CNN scorer on synthetic Objaverse composites and COCO-train negation pairs.[2] We reuse the released scorers, the training recipe and the data, and change only the image-side readout.

**Open-vocabulary and grounded detection.** OWL-ViT fine-tunes a CLIP backbone for detection;[5] OWLv2 adds self-training at scale and exposes a class-agnostic objectness head that we use for proposals.[6] Grounding DINO fuses text and image features in the detector and is evaluated on referring expression comprehension; its Swin-T checkpoint is pre-trained on O365, GoldG and Cap4M, while the Swin-B checkpoint adds COCO, OpenImages, ODinW-35 and RefCOCO.[9] Fine-grained evaluation of open-vocabulary detectors has precedent in FG-OVD.[8] Our benchmark follows the three failure modes of Kang et al. rather than attribute taxonomies.

## 3. Method

### 3.1. Frozen encoder and last-block readouts

All arms share `openai/clip-vit-base-patch16` with no parameter updates. Let X be the input of the last vision block, H = LN₁(X), and Q = W_Q H, V = W_V H its head-wise projections. The three readouts are

Z_naive = Block(X), Z_mask = W_O V, Z_clear = W_O [softmax(QQᵀ/√d_h) V],

each followed by the same post-layer-norm, visual projection and L2 normalization. The naive readout is exactly the code path of the DCSM repository. For whole-image heatmaps the image is resized anisotropically to 224 or 448 pixels; for crops the standard CLIP resize-and-center-crop preprocessing is used.

### 3.2. DCSM matrix, Functional Rows and scorer

For a crop and a query, the DCSM matrix is the 197 × 30 cosine matrix between the projected patch tokens (CLS included) and the projected text tokens (BOS, content, EOS, padding to 30). The columns of the eleven functional words (above, below, left, right, small, big, not, no, without, but, absent) are replaced by learned 197-d vectors. The repository locates these columns by whitespace word index, which is off by the BOS token and by every multi-token word; we replace the complete BPE span of every standalone occurrence instead (`bpe_alignment.py`) and record the rule in the checkpoint. The released checkpoints are evaluated with the repository's original rule so that their numbers reproduce the published code. The matrix is standardized with the statistics of the whole batch (the repository's `standardize`) and scored by the two-conv, two-FC network of Kang et al. Because of the batch standardization, scores are comparable only within one query's batch.

### 3.3. Retraining recipe

We port the original training loop: Objaverse spatial composites with CutMix hard negatives, Objaverse colour and small/big composites with mirrored negatives, and negated COCO-train pairs; each dataset in two caption-order versions; per step and dataset, 8 images + 8 opposites × 8 captions + 8 opposite captions form a 16 × 16 score matrix trained with symmetric cross-entropy against the identity; Adam 1e-3, 27 epochs, the repository's augmentation. One CLIP forward feeds all three readouts, so the three scorers see identical images, captions and augmentations and start from identical initial weights. Each configuration is trained with seeds 15, 16 and 17; we report mean ± standard deviation. Retraining the naive readout reproduces the released checkpoint within noise (§5.4), which validates the port.

### 3.4. Detection head

Given a query with target class A and, where present, reference class B, the head runs OWLv2 once with the texts [query, "a photo of a A", "a photo of a B"]. Boxes are text-independent, so the same forward yields OWLv2's native score for the sentence and the class-name probabilities p(A), p(B) for every box. Candidates are the boxes with p(A) ≥ 0.1 after class-agnostic NMS at 0.5 (at most 10, at least 2). The reference B̂ is the highest-p(B) box for spatial queries and the nearest B box to each candidate for negation/affirm queries. Each candidate's context crop, the padded union of the candidate and B̂, is scored by the DCSM scorer with the full query; the head's final score is p(A) × softmax over candidates of the scorer output. The head therefore uses OWLv2 for localization and class recognition and the DCSM scorer only for the relation.

## 4. Evaluation Protocol

### 4.1. Terminology

We follow Kang et al.: the three *failure modes* are attribute binding, spatial relationship (the absolute location of one object or the relationship between two objects) and negation. Our *failure-mode query benchmark* has subsets size and color (attribute binding), spatial (spatial relationship) and negation with its control affirm. *Candidate selection* supplies ground-truth candidate boxes and measures top-1 accuracy; *referring detection* supplies only image and query and measures the top-1 box. *Single-class-name detection* is COCO 80-class zero-shot detection.

### 4.2. Single-class-name detection

COCO val2017, 4,952 images with non-crowd annotations, prompts "a photo of a {category}.", standard COCO AP. P1 (dense) thresholds the per-patch class softmax and takes connected components; P2 (proposals) classifies the top-100 OWLv2 objectness boxes by the mean class map inside each box, scoring by objectness × class probability. Native OWL-ViT v1 and OWLv2 detections are references with their own proposal and score rules.

### 4.3. Failure-mode query benchmark

Queries are derived from COCO val2017 annotations only (`build_comp_queries.py`, seed 0, ≤ 60 items per class and subset, ≤ 2 per image). Every item has one image, one templated query, 2–6 same-class candidates and exactly one target, so the class name alone cannot solve it.

| Subset | Failure mode | Query | Target | Items | Chance |
| --- | --- | --- | --- | --- | --- |
| size | attribute binding | a picture of a big/small {A} | area ≥ 3× (≤ ⅓×) every other A | 655 | 43.1 |
| color | attribute binding | a picture of a {color} {A} | dominant HSV colour of the eroded mask (≥ 55%), absent (< 12%) from distractors | 142 | 44.1 |
| spatial | spatial relationship | a picture of a {A} to the left of / right of / above / below a {B} | B unique; exactly one A on that side of B | 527 | 45.6 |
| negation | negation | a picture of a {A} without a {B} | exactly one A with no attached B | 315 | 47.4 |
| affirm | (control) | a picture of a {A} with a {B} | exactly one A with an attached B | 194 | 33.4 |

The size subset is dominated by "big" (575/80) and a largest-box heuristic scores 87% raw; negation targets tend to be small background instances and a smallest-box heuristic scores 73%. We therefore report *size-balanced* accuracy: the mean of accuracy over items whose target is the largest candidate and items where it is not (big and small for size). Any pure size heuristic scores 50 under this metric. A random spot-check of 20 items per subset found 20/20 correct for spatial and size, 19/20 for negation and about 16/20 for color; color is reported for reference only.

**Candidate selection.** Scorers receive the ground-truth candidates. OWL models are scored by the maximum query logit of a detection with IoU ≥ 0.5 to the candidate; CLIP crop and context crop use the global embedding of the candidate crop or of the union with the reference B box; DCSM scorers use the context crop; whole-image readout heatmaps are mean-pooled inside the box.

**Referring detection.** Methods receive the image and the query sentence only. We report Acc@0.5 (top-1 box IoU ≥ 0.5 with the target; raw and size-balanced) and COCO-style AP: each subset is a category, each query item a pseudo-image with one ground-truth box, detections are pooled across items, so cross-item calibration counts. All native detectors receive the same sentence, NMS 0.5 and at most 100 detections per item.

### 4.4. Baselines

OWL-ViT v1 (`google/owlvit-base-patch32`), OWLv2 (`google/owlv2-base-patch16-ensemble`), Grounding DINO-T (`IDEA-Research/grounding-dino-tiny`, Swin-T, pre-trained on O365, GoldG, Cap4M) and Grounding DINO-B (`IDEA-Research/grounding-dino-base`, Swin-B, pre-trained on COCO, O365, GoldG, Cap4M, OpenImages, ODinW-35, RefCOCO).[9] Only Grounding DINO-T is a zero-shot comparison; Grounding DINO-B has seen COCO detection labels and RefCOCO referring labels, although RefCOCO images come from COCO train2014 and do not overlap val2017. Two further references share the head's candidate set: `owlv2_classname` ranks the candidates by p(A) alone (relation-blind), and `clip_ctx` re-ranks them by CLIP's EOS similarity on the same context crops.

## 5. Results

### 5.1. Readout surgery repairs localization and proposal classification

**Table 1.** Oracle-category localization probe (first 500 eligible val2017 images, 224 px) and single-class-name detection on 4,952 images (COCO AP, 0–100). P2 rows share OWLv2's top-100 objectness boxes.

| Readout | Text | Pointing (%) | P2 AP | P2 AP50 | P1 AP |
| --- | --- | --- | --- | --- | --- |
| naive | DCSM token-max | 10.4 | 1.2 | 1.6 | 0.0 |
| naive | pooled (EOS) | 12.0 | 2.6 | 3.4 | 0.7 |
| MaskCLIP | DCSM token-max | 37.1 | 20.3 | 28.0 | 2.6 |
| MaskCLIP | pooled (EOS) | 51.4 | 26.0 | 35.6 | 3.4 |
| ClearCLIP | DCSM token-max | 52.5 | 22.8 | 31.0 | 3.5 |
| ClearCLIP | pooled (EOS) | 62.4 | 27.3 | 37.3 | 3.0 |
| chance / OWL-ViT v1 native / OWLv2 native | | 17.1 | 28.8 / 46.8 | 45.4 / 66.7 | |

The naive readout points below chance: the heatmaps invert the object region, which is the quantitative form of the "inverted similarity" remark in the DCSM paper. Changing only the readout lifts proposal classification from 1–3 to 20–27 AP. On class names the DCSM token-max is about 5 AP below the EOS embedding and the training-free signal trails OWLv2's own head by 20 AP; the dense protocol P1 never exceeds 4.6 AP. This is the first disclaimer: the DCSM matrix offers nothing on single-class-name detection.

### 5.2. Retrained scorers read spatial relationships from surgery features

**Table 2.** Candidate selection, size-balanced top-1 accuracy (%). Released scorers and baselines are deterministic single runs; retrained scorers are mean ± sd over seeds 15/16/17.

| Scorer | size | color | spatial | negation | affirm |
| --- | --- | --- | --- | --- | --- |
| balanced chance | 37.6 | 43.5 | 45.7 | 48.1 | 33.0 |
| OWLv2 | 41.7 | **73.5** | 43.2 | 39.5 | 45.3 |
| OWL-ViT v1 | 44.4 | 63.9 | 43.8 | 41.8 | 44.2 |
| CLIP crop | 36.8 | 69.2 | 44.2 | 53.0 | 38.3 |
| CLIP context crop | 36.8 | 69.2 | 39.0 | 34.6 | 46.2 |
| DCSM released (ObjA) | 38.2 | 65.7 | 58.1 | 38.7 | 47.8 |
| DCSM released (COCO) | 38.8 | 61.7 | 57.3 | 37.5 | 46.0 |
| DCSM retrained, naive readout | 38.9 ± 1.7 | 56.5 ± 2.6 | 59.0 ± 3.9 | 40.5 ± 3.3 | 47.2 ± 2.6 |
| DCSM retrained, MaskCLIP readout | 41.3 ± 1.7 | 61.2 ± 5.1 | **69.8 ± 2.2** | 45.7 ± 2.4 | 44.4 ± 4.8 |
| DCSM retrained, ClearCLIP readout | 42.1 ± 4.5 | 59.6 ± 6.8 | **69.3 ± 3.4** | 39.4 ± 2.7 | 41.6 ± 4.0 |
| MaskCLIP heatmap, token-max (no training) | 44.5 | 64.8 | 43.4 | 49.9 | 36.3 |

Only the DCSM scorers exceed chance on spatial queries; OWLv2, OWL-ViT, CLIP crops and every training-free heatmap stay at or below 45.7. Retraining on surgery features adds about ten points over the same recipe on the naive readout. Per relation (raw top-1, 3 seeds), MaskCLIP retraining reaches 60.8 / 56.4 / 73.6 / 70.8 on above / below / left of / right of versus 47.8 / 50.6 / 64.6 / 57.7 for naive retraining; the largest gains are on above/below, which the training images contain only through swapped captions. MaskCLIP and ClearCLIP are indistinguishable at three seeds. Color is best served by global embeddings (OWLv2 73.5, CLIP crop 69.2); nobody reads size; negation stays at or below chance for every method while the same methods score above chance on affirm, so the missing ingredient for "without" is training signal rather than representation. These are the second and third disclaimers.

### 5.3. The head beats zero-shot detectors on spatial-relationship referring detection

**Table 3.** Referring detection on spatial-relationship queries (n = 527). Native rows receive the sentence; `cands/*` rows share the OWLv2 class-name candidate set, whose recall@0.5 of the target is 89.2%. Retrained rows: mean ± sd over seeds.

| Method | Acc@0.5 | Acc@0.5 balanced | AP | AP50 |
| --- | --- | --- | --- | --- |
| OWLv2 native | 7.8 | 8.1 | 6.7 | 9.6 |
| OWL-ViT v1 native | 12.0 | 12.6 | 5.5 | 9.2 |
| Grounding DINO-T (zero-shot) | 17.1 | 18.2 | 14.0 | 19.4 |
| Grounding DINO-B (COCO + RefCOCO pre-training) | **63.0** | **64.3** | **52.0** | **66.3** |
| cands / OWLv2 class-name p(A) only | 37.8 | 39.2 | 27.5 | 37.5 |
| cands / CLIP context crop | 27.9 | 28.4 | 19.7 | 27.1 |
| cands / DCSM released (ObjA) | 41.6 | 42.1 | 30.7 | 41.0 |
| cands / DCSM released (COCO) | 43.1 | 43.9 | 32.3 | 43.8 |
| cands / DCSM retrained, naive | 40.7 ± 1.3 | 41.3 ± 1.4 | 28.1 ± 0.6 | 38.5 ± 0.9 |
| cands / DCSM retrained, MaskCLIP | **49.7 ± 0.6** | **50.6 ± 0.4** | **34.4 ± 1.0** | **47.5 ± 1.2** |
| cands / DCSM retrained, ClearCLIP | 48.3 ± 1.3 | 49.1 ± 1.2 | 34.4 ± 1.2 | 47.1 ± 1.5 |

Given the sentence, OWLv2, OWL-ViT and Grounding DINO-T return an A box or a B box regardless of the relation; their accuracy is that of ignoring the relation word. The relation-blind candidate baseline already scores 37.8% because OWLv2's class probability favours large, central instances. The head with MaskCLIP retraining adds 11.4 points over that baseline (paired bootstrap, seed 15: +11.4, 95% CI [+6.1, +16.7]), 21.3 over CLIP re-ranking of the same crops [+16.5, +25.8], 32.1 over Grounding DINO-T [+27.1, +37.0] and 41.4 over OWLv2 [+36.4, +45.9]; seeds 16 and 17 give the same signs with overlapping intervals. Naive retraining is not separable from the relation-blind baseline, so the gain comes from the readout surgery. Per relation, the MaskCLIP head reaches 37.6 / 40.4 / 52.9 / 52.5 Acc@0.5 on above / below / left of / right of, against 32.3 / 32.7 / 37.4 / 41.4 for the baseline. Grounding DINO-B remains 13.9 points ahead [−19.0, −8.9]; it was pre-trained with RefCOCO referring labels and COCO detection labels and is listed as an upper reference, not a zero-shot peer.

**Table 4.** Referring detection on the other subsets, Acc@0.5 raw / AP.

| Method | size | color | negation | affirm |
| --- | --- | --- | --- | --- |
| OWLv2 native | 58.9 / 28.1 | 69.7 / 40.4 | 13.0 / 5.9 | 4.1 / 1.9 |
| Grounding DINO-T | 71.6 / 55.2 | 69.0 / 51.9 | 16.2 / 6.2 | 38.7 / 20.3 |
| Grounding DINO-B | 74.8 / 57.4 | 72.5 / 55.7 | 30.2 / 18.2 | 57.7 / 38.4 |
| cands / p(A) only | 57.6 / 32.2 | 48.6 / 32.4 | 31.1 / 20.0 | 46.9 / 24.3 |
| cands / DCSM retrained, MaskCLIP | 42.8 ± 3.2 / 31.4 ± 2.6 | 54.2 ± 1.1 / 38.7 ± 0.7 | 28.1 ± 1.2 / 16.6 ± 0.5 | 38.7 ± 2.2 / 23.7 ± 1.7 |

The head does not help here. Raw size accuracy is largely the largest-box preference (balanced: Grounding DINO-B 44.8, OWLv2 38.9, head 31.9); color is read better by the detectors' own text matching; on negation no method beats the relation-blind baseline.

### 5.4. Controls

- **Port validity.** Retraining the naive readout reproduces the released ObjA scorer: 59.0 ± 3.9 versus 58.1 balanced spatial accuracy, with a paired difference whose interval includes zero.
- **Representation and scorer must match.** The naive-readout scorer applied unchanged to MaskCLIP or ClearCLIP features scores 33.7 ± 1.3 and 28.5 ± 1.0 raw spatial accuracy, below chance; the gain exists only when the scorer is retrained on the readout it sees.
- **The scorer uses the relation word.** In the validation audit (pre-fix checkpoints, seed 15), deleting the relation words from the query drops spatial accuracy from 71 to 43 (MaskCLIP) and from 66 to 42 (ClearCLIP), and reversing the relation changes 57–67% of the predictions; normalization variants (fixed training statistics, per-candidate standardization) change results by at most two points.
- **No train/test overlap.** The 37,944 training images and the 1,025 benchmark images share no COCO id and no SHA-256 hash.
- **Functional-Row alignment.** The repository's whitespace-index column rule misplaces the Functional Row in every training caption (BOS offset, multi-token words). Fixing it changed the single-seed MaskCLIP result from 71.7 to 69.8 ± 2.2 and removed the apparent MaskCLIP > ClearCLIP ordering; the released checkpoints are evaluated with the original rule.

## 6. Limitations and Disclaimers

1. **Scope of the claim.** The advantage is confined to spatial-relationship queries. Single-class-name detection (Table 1), attribute binding and negation (Tables 2 and 4) show no benefit from the DCSM matrix, readout surgery or retraining.
2. **Grounding DINO-B is stronger.** The Swin-B checkpoint, pre-trained with COCO detection and RefCOCO referring supervision, exceeds the head by 14 Acc@0.5 points on spatial queries. The head's comparison class is zero-shot CLIP-family detectors and Grounding DINO-T.
3. **Proposal dependence.** The head inherits OWLv2's candidate recall (89% on spatial, 82% on negation) and its reference-box errors; it adds a relation re-ranker, not a localizer.
4. **Benchmark construction.** Queries are templated and auto-labelled from COCO annotations; relations are image-plane box relations; size uses annotation area; attachment is a box-overlap proxy; the negation and affirm subsets are not matched polarity pairs; there is no human-validated independent test set. Balanced accuracy removes one size shortcut, not category, layout or candidate-count effects.
5. **Statistical precision.** Three training seeds give limited precision (sd up to 4 points); bootstrap intervals are per-item resamples conditional on the generated labels and do not cover label error.
6. **Training data.** The scorer is trained on Kang et al.'s Objaverse composites and COCO-train negation pairs; the released COCO-model recipe is not reproducible from the public code, so all retrained scorers follow the ObjA recipe.

## 7. Conclusion

Reading CLIP's last block the MaskCLIP or ClearCLIP way is what makes the DCSM matrix usable for localization: it turns an anti-localized patch map into a training-free proposal classifier at 27 COCO AP and, after retraining the DCSM scorer on the same features with the original recipe, into a re-ranker that resolves spatial relationships between objects. Placed over OWLv2 class-name proposals, this frozen-CLIP head detects the object a spatial-relationship query refers to far more often than OWLv2, OWL-ViT or Grounding DINO-T given the same sentence, while a detector trained with referring supervision remains ahead and attribute and negation queries remain unsolved. The natural next steps are region-level training signal for negation, a learned replacement for Functional Rows, evaluation on the official RefCOCO splits, and stronger proposal sources.

## Acknowledgements

This work was created as part of the No-More-Slops group activities at Pseudo Lab. AI assistance was used to write and run the evaluation code, organize evidence and draft prose; scientific claims and authorship require review by the human authors.

## Appendix A. Reproducibility

The evaluation code is a Python package (`experiment/src/dcsm_surgery`: readouts, scorer, Functional-Row alignment, boxes, detectors, metrics, training data) with pytest unit tests and thin scripts. From `week1_clip_dcsm_surgery/experiment`:

```bash
uv sync && uv run pytest
uv run scripts/dcsm_detection_eval.py --coco_root data/coco --out results/full_val2017
uv run scripts/build_comp_queries.py --coco_root data/coco --out data/comp_queries
uv run scripts/run_bpe_multiseed.py                       # 3-seed retraining + candidate selection
uv run scripts/referring_detection_eval.py --dcsm_ckpts data/dcsm_ckpt/*.pt results/bpe_multiseed/seed_1*/train/*.pt --out results/refdet_full
```

Result files: `results/probe_500/localization_probe.json` (Table 1 pointing), `results/full_val2017/summary.json` (Table 1 AP), `results/comp_retrain/summary_balanced.md` and `results/bpe_multiseed/seed_*/evaluation/summary_balanced.md` (Table 2), `results/refdet_full/summary.md` (Tables 3–4, bootstrap), `results/validation_audit/` and `results/bpe_multiseed/*/adaptation/` (§5.4). Hardware: one RTX PRO 6000 Blackwell; torch 2.11 cu128, transformers 5.19.

## Sources

[1] [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)

[2] [Is CLIP ideal? No. Can we fix it? Yes!](https://arxiv.org/abs/2503.08723)

[3] [Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071)

[4] [ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference](https://arxiv.org/abs/2407.12442)

[5] [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

[6] [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)

[7] [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

[8] [The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding](https://arxiv.org/abs/2311.17518)

[9] [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499) · checkpoint pre-training data: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
