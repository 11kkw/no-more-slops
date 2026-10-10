Raw top-1 accuracy (%); conditional candidate selection.

| scorer | size | color | spatial | negation | affirm |
| --- | --- | --- | --- | --- | --- |
| #items | 655 | 142 | 527 | 315 | 194 |
| chance | 43.1 | 44.1 | 45.6 | 47.4 | 33.4 |
| owlv2 | 64.7 | 74.6 | 42.5 | 36.8 | 46.9 |
| owlvit_v1 | 69.5 | 67.6 | 42.3 | 34.6 | 47.4 |
| clip_crop | 45.8 | 69.7 | 43.8 | 52.4 | 39.2 |
| clip_ctx | 45.8 | 69.7 | 39.3 | 33.7 | 45.9 |
| dcsm_obja_ctx | 43.5 | 64.8 | 57.9 | 38.4 | 47.4 |
| dcsm_coco_ctx | 49.3 | 61.3 | 56.9 | 38.1 | 46.4 |
| readout/naive/pooled | 42.1 | 23.9 | 49.3 | 44.1 | 27.3 |
| readout/naive/dcsm_tok | 47.8 | 38.7 | 49.5 | 53.7 | 37.1 |
| readout/maskclip/pooled | 46.9 | 61.3 | 39.3 | 52.4 | 37.1 |
| readout/maskclip/dcsm_tok | 51.6 | 64.1 | 43.3 | 49.2 | 37.6 |
| readout/clearclip/pooled | 42.4 | 57.0 | 39.7 | 54.0 | 41.8 |
| readout/clearclip/dcsm_tok | 44.1 | 62.7 | 42.5 | 51.4 | 35.1 |
