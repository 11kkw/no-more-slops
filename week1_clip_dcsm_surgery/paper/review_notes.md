# Internal manuscript review and revisions

This is a sequential author-side review of this manuscript, result files, and evaluation code. It is not independent peer review, a venue decision, or a prediction of acceptance.

## Revision of 2026-10-11 — method framing, new experiments, terminology

- The manuscript was rewritten as a method paper with the decided title *Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP*. The earlier "localization gains do not transfer" framing (kept in `shortpaper/`, marked superseded) rested on applying training-free heatmaps and released scorers only; the new experiments E6 (scorer retraining on surgery features, three seeds) and E7 (referring detection on OWLv2 proposals vs OWLv2 / OWL-ViT / Grounding DINO) replace that gap, and claim C16 of the previous map ("no readout-by-trained-scorer experiment") is resolved.
- Terminology now follows Kang et al.: *failure modes* (attribute binding, spatial relationship, negation), *failure-mode query benchmark*, *candidate selection* (GT boxes given) vs *referring detection* (image + query only), *single-class-name detection*, *Functional Rows*. "Compositional" and "relational selection" are no longer used; subset names in code (`negation`, `affirm`, `dcsm_tok`) are unchanged.
- Explicit disclaimers were added (§6): no benefit on single-class-name detection, attribute binding or negation; Grounding DINO-B (COCO + RefCOCO pre-training) is stronger and not a zero-shot peer; the head is bounded by OWLv2 candidate recall; the benchmark is auto-generated.
- Items below from the 10 October review remain valid where they concern the benchmark construction (size/area definitions, negation/affirm asymmetry, label validation) and are carried into §4.3 and §6. The unresolved check "controlled readout-by-learned-scorer experiment" is now done (E6); "native/proposal protocol matching" is addressed for referring detection by sharing the candidate set across the `cands/*` rows and by the p(A)-only baseline; "comparison with established referring benchmarks" (RefCOCO splits) and human label validation remain open.

## Review of 10 October 2026 (previous framing)

## Summary

The draft now presents a bounded empirical study: dense readout changes improve the implemented category-localization and proposal-based detection pipelines, but do not consistently improve conditional compositional selection. Learned DCSM scoring has a spatial advantage on the retained query collection. The strongest interpretation concerns these evaluated pipelines, rather than universal visual reasoning or a new architecture.

## Strengths

- The manuscript distinguishes oracle localization, category-conditioned detection, and selection among supplied candidates.
- All candidate accuracies, balancing, heuristic scores, and expected-chance values have an artifact-level evidence chain.
- Shared image-cluster resampling preserves query dependence and method pairing; uncertainty is explicitly conditional on the constructed collection.
- The methods disclose inherited adapter behavior, privileged reference boxes, and unmatched native OWL scoring.
- The draft reports positive and negative findings without hiding above-chance baselines or query-construction effects.

## Major issues and revision actions

| Location | Observation and evidence | Impact | Action / status |
| --- | --- | --- | --- |
| Abstract; §3.2; §4.2 | Original `experiment/README.md` calls P2 and native OWLv2 the same boxes and claims their gap isolates classification. Actual `dcsm_detection_eval.py` applies different proposal and score selection. | Misstates the comparison's causal scope. | Corrected throughout. Native results are separate references; Table 3does not place native rows under a P2 column. |
| §3.3; §4.3 | Builder prioritizes big over small; counts575/80. Label area is `ann['area']`, whereas heuristics use bboxarea. | Inflated raw size scores and inaccurate claim that all balanced area heuristics must equal50. | Added generation rules, strata, raw/balanced pairs, and measured49.7largest-box balanced size result. |
| §3.3; §4.5 | Builder checks without before with; anN=2one-attached scene satisfies both rules. Cap-dependent priority changes query populations; N=2counts273/315and44/194. | With/without accuracy differences are not matched polarity effects. | Explicitly removed the matched-control interpretation and limited negation conclusions. |
| §3.4; §4.4 | Token-max pooling is not the trained DCSM CNN; trained scorer consumes ordinary features. There is no learned-scorer-by-surgery experiment. | A purported DCSM surgery ablation would be unsupported. | Separated scorer arms; stated missing factorial cells. |
| §4.4; §4.5 | Naive pooled spatial50.3exceeds chance45.7; naive token-max without55.7exceeds chance48.1. | Earlier blanket descriptions of unique above-chance DCSM or universal negation failure contradict measurements. | Corrected both statements and kept all 12 scorers in the appendix. |
| §3.5; Table 5 | Several queries share an image; aggregate detection logs do not permit analogous per-image analysis. | Item-wise resampling would understate dependence; fabricated AP intervals would overstate evidence. | Added paired image-cluster bootstrap for saved candidate predictions, and explicitly unavailable detection/localization intervals. |
| §3.4; §5 | DCSM insertion uses whitespace positions in a token matrix and batch-wide standardization; checkpoint training lists/hash inventory unavailable. | Reproduction quirks and possible domain/data overlap can affect model interpretation. | Disclosed behavior; did not silently alter inference or certify split disjointness. Further audit unresolved. |
| §2; §5 | Readouts, DCSM, and fine-grained detector evaluation have existing prior art. A few source checks do not settle novelty. | Claims of a new model, first benchmark, or SOTA are unsupported. | Scoped contribution to an empirical comparison and construction audit. Novelty remains provisional. |

## Minor issues and revisions

- A drafted DCSM COCO interval was corrected to the generated **[53.2,61.2]%**. Table 5is checked against the recorded bootstrap, not estimated from memory.
- Table 3separates native detector references from P2 configurations; very small nonzero AP may round to0.0.
- The known nonempty-image filter corrects the nominal 5,000-image description to 4,952.
- The source-retrieval script's initial CLIP excerpt matched page furniture. It was replaced with a verified 15-word excerpt from the paper abstract; citation IDs were preserved.
- Bibliography entries use verified arXiv first-submission years. Conference years and metadata are not guessed; this choice is documented in Appendix B.
- Figures use measured values, visible axes, explicit units, and a native detector reference labeled as different scoring. Both plots were visually inspected.

## Unresolved checks before submission

1. Human validation of colors, attachment proxies, target uniqueness, and ambiguous/missing annotations on a sufficiently large sample.
2. Paired with/without descriptions for the same candidates, with candidate-count and size/layout distributions controlled.
3. Exact released checkpoint hashes, training/evaluation example identity lists where available, and model/download revisions.
4. A controlled readout-by-learned-scorer experiment with matched crops, supervision, candidate sets, and score normalization. Existing CNN checkpoints may need adaptation rather than direct substitution.
5. Native/proposal protocol matching or explicit decomposition of box quality, classifier accuracy, confidence calibration, and proposal recall.
6. Independent evaluation collection and comparisons with established referring-expression/fine-grained benchmarks.
7. A fuller novelty audit, authorship confirmation, venue template, and an original runtime environment record. A manuscript PDF was not requested or compiled.

These unresolved matters limit submission readiness and generalization claims. They do not prevent completion of the present evidence-grounded working draft.

## Quality checklist

| Check | Status | Manuscript location / limit |
| --- | --- | --- |
| Claims bounded by observed results | supported | Abstract; §§1,4–6 |
| Frozen versus already-trained components | supported | §§3.1,3.4,5 |
| Actual counts and generation rules | supported | §3.3; Table1 |
| Comparator differences disclosed | supported | §§3.2,3.4; Figure2 |
| Table/figure units and source consistency | supported after recorded verification | Tables1–5; AppendixA; `draft_validation.json` |
| Uncertainty and sample dependence | supported for candidate analysis | §3.5; Table 5; AP/localization intervals unavailable |
| Primary references and metadata | supported within cited scope | §2; Sources; `references.bib`; novelty not verified |
| Limitations and alternative explanations | supported | §5 |
| Saved-result reproduction | supported | Appendix B; scripts and hashes; original inference not reproduced |
| Exact original environment/checkpoint inventory | not_available | §5; remaining checks 3 and 7 |
| Repository acknowledgement | supported | Acknowledgements |
| Independent peer review and venue requirements | not_available | Not performed; venue unspecified |
