# Claim and evidence example

| Claim | Saved evidence | Scope and status |
| --- | --- | --- |
| Adapted Mask-style and Clear-style scorers improve spatial selection over adapted ordinary CLIP in the three recorded seeds. | `evidence/measurements.json#/metrics/spatial`; seed means and per-seed values | Observed for 527 queries / 375 images with supplied candidate and reference boxes. This is conditional selection, not new end-to-end detection AP. |
| A scorer and its feature readout require adaptation together in this protocol. | `evidence/measurements.json#/adaptation`, `#/adaptation_gains` | Interpretation bounded by the measured ordinary-scorer transfer controls. CLIP is frozen; scorers and functional-word vectors are trained. |
| The modified readouts have no stable ordering across seeds. | `evidence/measurements.json#/metrics/spatial` | Rank changes are observed; three seeds do not establish statistical equivalence. |
| This establishes general compositional superiority or novelty. | No sufficient evidence | Unverified; not claimed. Negation, labels, overlap and independent review remain unresolved. |

The generic exporter transcribes 78 retained table values and hashes their source. It does not validate the aggregate calculations contained in this derived measurement file. Raw predictions and their original export manifests remain in the source project, outside this small example. The full saved measurement file also retains size/color diagnostics omitted from the revised main table.

BibTeX-key checks establish key existence, not support by the original literature. Source-text overlap is unavailable in this example because original article texts are not redistributed.
