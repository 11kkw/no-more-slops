| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| chance (mean 1/N) | 41.7 | 42.2 | 50.0 | 45.8 | 36.1 |
| #items | 6 | 6 | 6 | 6 | 6 |
| owlv2 | 83.3 | 50.0 | 66.7 | 33.3 | 66.7 |
| owlvit_v1 | 83.3 | 66.7 | 66.7 | 16.7 | 66.7 |
| clip_crop | 33.3 | 50.0 | 33.3 | 16.7 | 50.0 |
| clip_ctx | 33.3 | 50.0 | 50.0 | 33.3 | 50.0 |
| dcsm_obja_ctx | 50.0 | 66.7 | 83.3 | 50.0 | 33.3 |
| dcsm_coco_ctx | 33.3 | 33.3 | 66.7 | 50.0 | 66.7 |
| readout/naive/pooled | 66.7 | 16.7 | 50.0 | 50.0 | 50.0 |
| readout/naive/dcsm_tok | 33.3 | 33.3 | 33.3 | 83.3 | 83.3 |
| readout/maskclip/pooled | 16.7 | 66.7 | 50.0 | 50.0 | 33.3 |
| readout/maskclip/dcsm_tok | 33.3 | 50.0 | 50.0 | 33.3 | 33.3 |
| readout/clearclip/pooled | 50.0 | 33.3 | 50.0 | 50.0 | 50.0 |
| readout/clearclip/dcsm_tok | 16.7 | 50.0 | 50.0 | 50.0 | 50.0 |

**spatial breakdown** (top-1 %, n per column)
| scorer | above (n=1) | to the left of (n=1) | to the right of (n=4) |
|---|---|---|---|
| owlv2 | 0.0 | 100.0 | 75.0 |
| owlvit_v1 | 0.0 | 100.0 | 75.0 |
| clip_crop | 0.0 | 0.0 | 50.0 |
| clip_ctx | 0.0 | 0.0 | 75.0 |
| dcsm_obja_ctx | 100.0 | 0.0 | 100.0 |
| dcsm_coco_ctx | 0.0 | 0.0 | 100.0 |
| readout/naive/pooled | 100.0 | 100.0 | 25.0 |
| readout/naive/dcsm_tok | 100.0 | 100.0 | 0.0 |
| readout/maskclip/pooled | 0.0 | 100.0 | 50.0 |
| readout/maskclip/dcsm_tok | 0.0 | 100.0 | 50.0 |
| readout/clearclip/pooled | 0.0 | 100.0 | 50.0 |
| readout/clearclip/dcsm_tok | 0.0 | 100.0 | 50.0 |

**size breakdown** (top-1 %, n per column)
| scorer | big (n=5) | small (n=1) |
|---|---|---|
| owlv2 | 100.0 | 0.0 |
| owlvit_v1 | 100.0 | 0.0 |
| clip_crop | 40.0 | 0.0 |
| clip_ctx | 40.0 | 0.0 |
| dcsm_obja_ctx | 60.0 | 0.0 |
| dcsm_coco_ctx | 40.0 | 0.0 |
| readout/naive/pooled | 60.0 | 100.0 |
| readout/naive/dcsm_tok | 40.0 | 0.0 |
| readout/maskclip/pooled | 20.0 | 0.0 |
| readout/maskclip/dcsm_tok | 40.0 | 0.0 |
| readout/clearclip/pooled | 40.0 | 100.0 |
| readout/clearclip/dcsm_tok | 0.0 | 100.0 |

**color breakdown** (top-1 %, n per column)
| scorer | black (n=2) | brown (n=1) | gray (n=1) | white (n=2) |
|---|---|---|---|---|
| owlv2 | 100.0 | 0.0 | 0.0 | 50.0 |
| owlvit_v1 | 100.0 | 0.0 | 100.0 | 50.0 |
| clip_crop | 50.0 | 100.0 | 100.0 | 0.0 |
| clip_ctx | 50.0 | 100.0 | 100.0 | 0.0 |
| dcsm_obja_ctx | 100.0 | 100.0 | 100.0 | 0.0 |
| dcsm_coco_ctx | 50.0 | 0.0 | 100.0 | 0.0 |
| readout/naive/pooled | 50.0 | 0.0 | 0.0 | 0.0 |
| readout/naive/dcsm_tok | 50.0 | 0.0 | 0.0 | 50.0 |
| readout/maskclip/pooled | 50.0 | 100.0 | 100.0 | 50.0 |
| readout/maskclip/dcsm_tok | 50.0 | 100.0 | 0.0 | 50.0 |
| readout/clearclip/pooled | 50.0 | 0.0 | 0.0 | 50.0 |
| readout/clearclip/dcsm_tok | 50.0 | 100.0 | 0.0 | 50.0 |
