Balanced top-1 accuracy (%). Size: big/small; others: target-largest/not-largest.

| scorer | size | color | spatial | negation | affirm |
| --- | --- | --- | --- | --- | --- |
| stratum counts | 575 / 80 | 90 / 52 | 233 / 294 | 74 / 241 | 117 / 77 |
| chance | 37.6 | 43.5 | 45.7 | 48.1 | 33.0 |
| owlv2 | 41.7 | 73.5 | 43.2 | 39.5 | 45.3 |
| owlvit_v1 | 44.4 | 63.9 | 43.8 | 41.8 | 44.2 |
| clip_crop | 36.8 | 69.2 | 44.2 | 53.0 | 38.3 |
| clip_ctx | 36.8 | 69.2 | 39.0 | 34.6 | 46.2 |
| dcsm_obja_ctx | 38.2 | 65.7 | 58.1 | 38.7 | 47.8 |
| dcsm_coco_ctx | 38.8 | 61.7 | 57.3 | 37.5 | 46.0 |
| readout/naive/pooled | 38.5 | 21.3 | 50.3 | 48.0 | 26.4 |
| readout/naive/dcsm_tok | 38.5 | 37.5 | 50.0 | 55.7 | 36.3 |
| readout/maskclip/pooled | 40.7 | 62.1 | 39.4 | 51.1 | 36.5 |
| readout/maskclip/dcsm_tok | 44.5 | 64.8 | 43.4 | 49.9 | 36.3 |
| readout/clearclip/pooled | 39.8 | 57.6 | 39.4 | 48.8 | 42.2 |
| readout/clearclip/dcsm_tok | 43.4 | 63.2 | 42.5 | 48.1 | 35.5 |
