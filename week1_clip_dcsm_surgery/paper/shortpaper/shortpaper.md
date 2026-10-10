# Dense CLIP Localization Gains Do Not Consistently Transfer to Relational Object Selection

*Venue-neutral working short paper. Authors and affiliations: to be determined.*

## Abstract

We report a descriptive separation between category-sensitive localization and relational object selection in evaluated CLIP pipelines. Category localization need not select the intended instance. We analyze saved COCO experiments with a frozen ViT-B/16, ordinary and modified final-block readouts, and local Dense Cosine Similarity Map (DCSM) adapters. At 224-pixel input resolution, a ClearCLIP-style readout increases proposal-based detection AP from 2.6 to 27.3 on 4,952 images, yet its pooled-text map scorer obtains 39.4% balanced spatial-selection accuracy, versus 50.3% for the ordinary readout. On 527 spatial queries from 375 images, DCSM ObjA and COCO adapters reach 58.1% [53.7, 62.4] and 57.3% [53.2, 61.2], respectively, against 45.7% expected chance (exploratory percentile 95% image-cluster bootstrap intervals). Release identities remain unverified. These conditional, annotation-derived results motivate separate validation of localization and relational selection; they neither isolate a causal mechanism nor test learned DCSM scoring on modified features.

## 1. Introduction and Related Work

CLIP compares image and text representations learned through natural-language supervision [@radford2021clip]. Applying these representations to objects involves two questions: whether a category produces a useful spatial response, and whether a description selects the intended instance. A heatmap can highlight every person while failing to select the person to the left of a bicycle. This distinction matters when dense-feature improvements are used to motivate stronger object-level language understanding.

Dense feature extraction and dense image--text scoring address different parts of this problem. MaskCLIP extracts dense predictions from pretrained CLIP [@zhou2021maskclip]. ClearCLIP decomposes the final block, removing residual and feed-forward paths and using self-self attention to improve segmentation [@lan2024clearclip]. DCSM instead retains token--patch similarities and applies a learned convolutional scorer [@kang2025dcsm]. Their original tasks and designs are not interchangeable: improvements in segmentation do not directly measure relation-conditioned instance selection. We evaluate readouts inspired by MaskCLIP and ClearCLIP, not full reproductions of their segmentation results.

Detection provides a second reference point. OWL-ViT adapts image--text pretraining through detection fine-tuning [@minderer2022owlvit], and OWLv2 scales detection supervision through self-training [@minderer2023owlv2]. Their learned box and score mechanisms differ from frozen-CLIP map pooling. Moreover, fine-grained detector evaluation has precedent in FG-OVD [@bianchi2023fgovd]. Our contribution is a cross-task analysis under a conditional object-selection protocol.

Our focused observation is that localization and detection gains from the evaluated CLIP readouts do not consistently transfer to relational candidate selection. We connect three existing measurements: oracle-category pointing, category-conditioned detection, and selection among supplied same-category boxes. The main comparison uses spatial descriptions; size and attachment diagnostics constrain its interpretation. The study contributes:

- A cross-task comparison separating category localization from relational selection.
- A protocol and construction audit bounding the observed transfer differences.

The evidence is a saved-artifact analysis, not newly rerun inference. Its practical implication is an evaluation requirement: validate the downstream selection rule independently, rather than infer relational capability from category heatmaps or detection AP.

## 2. Methods and Evaluation

### 2.1. Dense readouts and aggregation

All dense-map arms share a frozen CLIP ViT-B/16. The encoder is `openai/clip-vit-base-patch16`.

Whole images are resized anisotropically to 224 or 448 pixels with bicubic interpolation and CLIP normalization; these produce 14-by-14 or 28-by-28 patch grids. The larger input interpolates positional embeddings. Candidate crops instead use standard resize-and-center-crop preprocessing, a relevant input difference.

Let $X$ enter the last vision transformer block, and let $Q$ and $V$ be the head-wise query and value projections of its normalized input. We compare the ordinary block output, a MaskCLIP-style value-only output, and a ClearCLIP-style query--query output:

$$
\begin{aligned}
Z_{\mathrm{ordinary}}&=\mathrm{Block}(X),\quad Z_{\mathrm{value}}=W_O V,\\
Z_{\mathrm{QQ}}&=W_O\big[\mathrm{softmax}(QQ^\top/\sqrt{d_h})V\big].
\end{aligned}
$$

Here $d_h$ is the head dimension; head outputs are merged before $W_O$. Modified outputs omit the final residual and feed-forward paths. Unlike the ViT adaptation in MaskCLIP (Sec. 3.3), our value-only variant omits the residual and uses neither key smoothing nor prompt denoising [@zhou2021maskclip]. Every readout receives the same post-layer normalization, visual projection, and L2 normalization. Dense maps discard CLS. With normalized patch feature $z_p$ and pooled text embedding $t_{\mathrm{EOS}}$, the pooled-text map is $h(p,q)=z_p^\top t_{\mathrm{EOS}}(q)$.

A token-max variant replaces this scalar by $\max_{j\in\mathcal C(q)}z_p^\top t_j(q)$. Its content set excludes BOS, EOS, and padding, but includes ordinary prompt words and prepositions; text is padded or truncated to 30 positions. The code name `dcsm_tok` denotes this maximum, not the learned DCSM CNN. Aggregation removes explicit token-column order, although contextual token embeddings still depend on the sentence.

### 2.2. Localization and detection

The oracle-category probe uses the first 500 eligible COCO val2017 images in sorted ID order. A trial is an image--category pair: pointing succeeds if the maximum patch center lies in any non-crowd box of the known category. It does not ask which same-category instance a relational description intends. Pointing is micro-averaged over eligible image--category trials, not images; the trial count was not retained in the summary. The non-random probe does not support image-cluster intervals from the retained aggregates.

Detection covers 4,952 images with non-crowd annotations. We report COCO bbox AP averaged over IoU thresholds 0.50--0.95 on a 0--100 scale. Class prompts are `a photo of a {category}.` for the 80 COCO categories [@lin2014coco]. Two protocols connect the same maps to different box mechanisms:

**P1, dense components.** Bilinearly upsample maps to the input square (224 or 448 pixels, `align_corners=False`), apply an 80-way class softmax with multiplier 100, retain winning-class pixels with probability above 0.3, and extract four-connected components of at least four upsampled pixels. Maximum component probability is the box score.

**P2, supplied proposals.** Select the top 100 OWLv2 boxes by sigmoid objectness. Average class maps over patch centers inside each box, using the patch containing the box center, clamped to the grid, for empty memberships. Apply the same softmax multiplier, retain three classes per box, and multiply each class probability by objectness. Readout variants share proposals and scoring rules within P2.

Native OWL references instead retain the top 300 box--class pairs by sigmoid class logits across predicted boxes. Native OWLv2 and P2 therefore do not have identical final proposals or confidence rules. An AP difference between them is not a pure classification comparison. Reference models are `google/owlvit-base-patch32` and `google/owlv2-base-patch16-ensemble`; their training and backbones are also unmatched to frozen CLIP.

### 2.3. Conditional relational selection

Spatial items derive from COCO boxes. Each query supplies two to six same-category candidates $A$ and a single annotated reference instance $B$. One target lies on the requested side of $B$, with a center margin of 0.05 of the relevant image dimension and separation constraints; all distractor centers lie on the opposite side. Relations are left, right, above, and below in the image plane, not three-dimensional judgments. All non-crowd target-category instances must have annotation-area ratio at least 0.002 and box side lengths at least 24 pixels. Generation uses seed 0 and per-subset caps of 60 items per category and two per image. The spatial subset contains 527 queries from 375 images and 57 target categories. Its two box-area strata contain 233 target-largest and 294 target-not-largest items. Separation thresholds, reference disambiguation, duplicate/tie handling, candidate order and sentence templates remain unverified.

Whole-image map scorers average query similarities over patches inside each supplied candidate box at 224 resolution, using nearest-patch fallback when necessary, without a class softmax or learned CNN. CLIP crop scoring compares the crop CLS embedding with query EOS. Candidate crops add 10% padding per side; spatial context crops use the union of the candidate and the reference box. These context crops use privileged annotations and can change object scale and layout. The task is consequently conditional candidate selection, not end-to-end referring-expression detection.

Local DCSM adapters [@kang2025dcsm], labeled ObjA and COCO, score context crops with ordinary CLIP features and a pretrained CNN. Their image-position-by-text-position matrix (197-by-30) includes CLS and BOS/EOS/padding; the scorer has two convolutional layers and a 128-unit hidden layer. Checkpoint functional embeddings use inherited whitespace positions; alignment is unverified. Unlike the original text-token-row/image-patch-column description with EOS padding and z-score normalization, the adapter standardizes the full candidate-by-channel-by-image-by-text tensor per query, making scores batch dependent. Equivalence is unverified. No training or modified-feature CNN evaluation is performed here. Filenames and implementation provenance are recorded locally; release identities, hashes, versions and original execution revisions remain unverified.

OWL candidate scoring matches predicted boxes to each supplied candidate at IoU at least 0.5 and takes the maximum matching query logit. If no prediction matches, the best-IoU prediction supplies its logit minus 10. OWL selection thus includes localization and matching effects, unlike directly scoring supplied crops.

### 2.4. Balancing and uncertainty

Raw top-1 accuracy counts correct candidate choices. Balanced spatial accuracy equally weights the two target-box-area strata:

$$
A_{\mathrm{bal}}=\tfrac12\big(A_{\mathrm{largest}}+A_{\mathrm{not\ largest}}\big).
$$

Expected chance replaces each item's correctness indicator by $1/N_i$, then applies the same balancing. It is 45.7%, not 50%, because candidate counts vary. Balancing addresses one coarse size correlation, not every category, layout, relation, or candidate-count bias.

We reuse the recorded bootstrap of saved predictions: 10,000 paired image-cluster resamples within each subset, seed 20261010. Each resampled image contributes all its queries; stratum accuracies are recomputed and equally weighted. Percentile 95% intervals are exploratory, with no multiple-comparison correction. They condition on the retained labels, candidates, and checkpoints, and do not cover label errors or training overlap. Detection and pointing uncertainty is unavailable from retained summaries. Saved paired contrasts are located in `../evidence/cluster_bootstrap.json`, under `subsets.spatial.paired_differences`; their provenance still requires verification.

## 3. Results

### 3.1. Category-localization and proposal gains

Table 1 tests whether readout changes improve category-sensitive responses. At 224 pixels, pooled pointing rises from 12.0% to 62.4% with the ClearCLIP-style output, while P2 AP rises from 2.6 to 27.3. The shared P2 protocol supports a readout effect in this detector, but AP still mixes box quality, calibration, and ranking. In P1, the same ClearCLIP-style pooled configuration obtains only 3.0 AP. The highest P1 AP across the full evaluated grid rounds to 4.6, so this component rule does not convert useful maps into a strong detector.

**Table 1.** Pooled-text maps at 224 pixels. Pointing averages eligible image--category trials in a 500-image oracle-category probe; detection uses 4,952 images. AP uses a 0--100 scale. Native OWLv2 obtains 46.8 AP with different proposals and confidence scores and is not a P2 row. Intervals are unavailable.

| Readout | Pointing (%) ↑ | P1 AP ↑ | P2 AP ↑ |
| --- | --- | --- | --- |
| Ordinary | 12.0 | 0.7 | 2.6 |
| MaskCLIP-style | 51.4 | 3.4 | 26.0 |
| ClearCLIP-style | 62.4 | 3.0 | 27.3 |

Higher resolution does not ensure detection gains: ClearCLIP-style pooled pointing reaches 67.5% at 448 pixels, while P2 AP becomes 26.4. Native OWLv2 reaches 46.8 AP under its separate protocol. Neither comparison licenses the claim that localization alone determines detection or that the native/P2 gap measures classification alone.

### 3.2. Relational gains do not follow the same ranking

Table 2 and Figure 1 test whether the localization ranking transfers to spatial selection. Ordinary pooled maps obtain 50.3% balanced accuracy. Both modified pooled readouts obtain 39.4%, despite their much better pointing and P2 AP. This is a descriptive ranking reversal: a paired ordinary-versus-modified interval is unavailable in the retained bootstrap export. Token-max does not reverse the ordering: ordinary, MaskCLIP-style, and ClearCLIP-style spatial accuracies are 50.0%, 43.4%, and 42.5%. The observed transfer failure is specific to these scalar-map reductions and candidate rules, not evidence that dense representations cannot support relations.

**Table 2.** Balanced spatial-selection accuracy (%) on 527 queries/375 images; exploratory percentile 95% image-cluster bootstrap intervals. DCSM labels denote local adapters with unverified release identities. Candidates are supplied; crops, training and OWL matching differ. Chance is balanced expected uniform selection.

| Scorer | Accuracy ↑ | Exploratory 95% interval |
| --- | --- | --- |
| Ordinary pooled map | 50.3 | [46.6, 53.9] |
| MaskCLIP-style pooled map | 39.4 | [35.7, 43.0] |
| ClearCLIP-style pooled map | 39.4 | [35.8, 43.1] |
| CLIP context crop | 39.0 | [35.4, 42.7] |
| DCSM ObjA context | 58.1 | [53.7, 62.4] |
| DCSM COCO context | 57.3 | [53.2, 61.2] |
| OWLv2 matched scores | 43.2 | [39.5, 46.9] |
| Expected chance | 45.7 | — |

![Balanced spatial accuracy and image-cluster intervals](figures/spatial_balanced.pdf)

**Figure 1.** Balanced spatial-selection accuracy with exploratory percentile 95% intervals from 10,000 image-cluster resamples; dashed: balanced expected chance. MaskCLIP-style appears only in Table 2 because its rounded estimate matches ClearCLIP-style. DCSM denotes local adapters; inputs differ, so this is not a factorial ablation.

Local DCSM ObjA and COCO adapters reach 58.1% and 57.3%. ObjA exceeds balanced expected chance by 12.4 percentage points, with paired difference interval [8.0, 16.7], and ClearCLIP-style pooled-text maps by 18.7 points [13.1, 24.2]. These are sample-specific pretrained-pipeline advantages with unverified release identities. Ordinary pooled-text maps also exceed chance numerically. Context, training, functional embeddings and matrix scoring differ, preventing component-level causal attribution.

### 3.3. Construction diagnostics constrain the interpretation

The retained collection also contains size, color, without, and with queries; the detailed manuscript reports all 1,833 items and 12 scorers. We retain two diagnostics because they explain why selection accuracy alone can be misleading. In size queries, big descriptions outnumber small descriptions 575 to 80. The largest-box heuristic reaches 87.3% raw accuracy without reading language; OWLv2 falls from 64.7% raw to 41.7% after big/small reweighting. Size labels use annotation area rather than box area, so even a balanced box heuristic is not guaranteed to equal 50%.

Attachment-based without/with queries are not matched controls. Their builder prioritizes without when both rules are satisfied; two-candidate items account for 273/315 without items but only 44/194 with items. Overlap or center containment proxies attachment, and missing annotations do not prove visual absence. DCSM balanced without-selection accuracies are 38.7% and 37.5% against 48.1% balanced expected chance; ordinary token-max reaches 55.7%. These equally weight target-largest/not-largest strata within the without subset, unlike raw accuracy or size-query balancing. Thus neither universal negation failure nor a causal effect of changing polarity follows from this collection. These diagnostics are boundaries on the main spatial observation, not additional claims of general reasoning ability.

## 4. Discussion and Limitations

A category map and a relational selection rule require separate validation. One possible explanation is that scalar aggregation loses interactions useful for selection, whereas learned matrix scoring retains them. The experiments do not isolate that mechanism: DCSM uses different crops and prior supervision, and spatial context can change the relative scale of target and reference. The observation is therefore a cross-task diagnostic, not proof that final-block surgery harms learned DCSM scoring or that topology alone explains its advantage.

The spatial labels are filtered, templated, image-plane judgments. There are 404 two-candidate items, and left/right queries outnumber above/below queries. Area reweighting does not remove these imbalances. Human verification of target uniqueness and label quality is incomplete. Ground-truth candidates and reference boxes simplify the problem; unseen-category generalization and independent referring-expression evaluation are not measured. Reliable deployment would need candidate discovery and robustness checks beyond these annotations.

Training/evaluation overlap, exact checkpoint hashes, and original execution revisions remain incompletely audited. Frozen evaluation does not imply no prior training. Retained summaries and predictions support arithmetic checks, not a newly reproduced inference run. Original hardware and complete package inventories are unavailable, and no matched speed claim is made. Bootstrap intervals capture sample variability only; they cannot repair proxy labels or certify generalization.

A controlled follow-up would match crop geometry and supervision across readouts and learned scorers, use paired positive/negative descriptions, and validate labels on an independent collection. Those studies are not part of this report. 

## 5. Conclusion

The evaluated readouts improve category localization and detection, but their pooled-text maps show a descriptive ranking reversal in relational selection. Local DCSM adapters have a spatial advantage; unverified release identities and unmatched inputs and training prevent causal attribution. Separate task measurements support task-specific validation.

## Acknowledgements and Evidence Availability

This work was created as part of the No-More-Slops group activities at Pseudo Lab.

AI assistance supported evidence inspection, writing, and internal review; it is not independent human peer review. The non-public supplement contains the detailed draft (Appendices A--B), measurement and bootstrap JSON, claim map, and evaluation code. The local supplement and README provide evidence paths and build instructions, without establishing a public release, portable submission bundle or independently verified build.

## References

Citation keys correspond to the compiled bibliography in `references.bib`. Verified arXiv entries retain first-submission years; conference metadata is not inferred.

- **radford2021clip**. Alec Radford; Jong Wook Kim; Chris Hallacy; Aditya Ramesh; Gabriel Goh; Sandhini Agarwal; Girish Sastry; Amanda Askell; Pamela Mishkin; Jack Clark; Gretchen Krueger; Ilya Sutskever. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. https://proceedings.mlr.press/v139/radford21a.html

- **kang2025dcsm**. Kang, Raphi; Song, Yue; Gkioxari, Georgia; Perona, Pietro. (2025). *Is CLIP ideal? No. Can we fix it? Yes!*. https://arxiv.org/abs/2503.08723

- **zhou2021maskclip**. Zhou, Chong; Loy, Chen Change; Dai, Bo. (2021). *Extract Free Dense Labels from CLIP*. https://arxiv.org/abs/2112.01071

- **lan2024clearclip**. Lan, Mengcheng; Chen, Chaofeng; Ke, Yiping; Wang, Xinjiang; Feng, Litong; Zhang, Wayne. (2024). *ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference*. https://arxiv.org/abs/2407.12442

- **minderer2022owlvit**. Minderer, Matthias; Gritsenko, Alexey; Stone, Austin; Neumann, Maxim; Weissenborn, Dirk; Dosovitskiy, Alexey; Mahendran, Aravindh; Arnab, Anurag; Dehghani, Mostafa; Shen, Zhuoran; Wang, Xiao; Zhai, Xiaohua; Kipf, Thomas; Houlsby, Neil. (2022). *Simple Open-Vocabulary Object Detection with Vision Transformers*. https://arxiv.org/abs/2205.06230

- **minderer2023owlv2**. Minderer, Matthias; Gritsenko, Alexey; Houlsby, Neil. (2023). *Scaling Open-Vocabulary Object Detection*. https://arxiv.org/abs/2306.09683

- **lin2014coco**. Lin, Tsung-Yi; Maire, Michael; Belongie, Serge; Bourdev, Lubomir; Girshick, Ross; Hays, James; Perona, Pietro; Ramanan, Deva; Zitnick, C. Lawrence; Dollár, Piotr. (2014). *Microsoft COCO: Common Objects in Context*. https://arxiv.org/abs/1405.0312

- **bianchi2023fgovd**. Bianchi, Lorenzo; Carrara, Fabio; Messina, Nicola; Gennaro, Claudio; Falchi, Fabrizio. (2023). *The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding*. https://arxiv.org/abs/2311.17518
