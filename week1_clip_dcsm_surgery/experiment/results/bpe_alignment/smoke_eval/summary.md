| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| chance (mean 1/N) | 33.3 | 50.0 | 50.0 | 50.0 | 33.3 |
| #items | 1 | 1 | 1 | 1 | 1 |
| owlv2 | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 |
| owlvit_v1 | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 |
| clip_crop | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| clip_ctx | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| dcsm_naive_retrain_ctx | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| dcsm_maskclip_retrain_ctx | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| dcsm_clearclip_retrain_ctx | 0.0 | 100.0 | 0.0 | 100.0 | 0.0 |
| readout/naive/pooled | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| readout/naive/dcsm_tok | 100.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| readout/maskclip/pooled | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 |
| readout/maskclip/dcsm_tok | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 |
| readout/clearclip/pooled | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 |
| readout/clearclip/dcsm_tok | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 |

**spatial breakdown** (top-1 %, n per column)
| scorer | to the left of (n=1) |
|---|---|
| owlv2 | 100.0 |
| owlvit_v1 | 100.0 |
| clip_crop | 0.0 |
| clip_ctx | 0.0 |
| dcsm_naive_retrain_ctx | 0.0 |
| dcsm_maskclip_retrain_ctx | 0.0 |
| dcsm_clearclip_retrain_ctx | 0.0 |
| readout/naive/pooled | 100.0 |
| readout/naive/dcsm_tok | 100.0 |
| readout/maskclip/pooled | 100.0 |
| readout/maskclip/dcsm_tok | 100.0 |
| readout/clearclip/pooled | 100.0 |
| readout/clearclip/dcsm_tok | 100.0 |

**size breakdown** (top-1 %, n per column)
| scorer | big (n=1) |
|---|---|
| owlv2 | 100.0 |
| owlvit_v1 | 100.0 |
| clip_crop | 0.0 |
| clip_ctx | 0.0 |
| dcsm_naive_retrain_ctx | 0.0 |
| dcsm_maskclip_retrain_ctx | 0.0 |
| dcsm_clearclip_retrain_ctx | 0.0 |
| readout/naive/pooled | 100.0 |
| readout/naive/dcsm_tok | 100.0 |
| readout/maskclip/pooled | 0.0 |
| readout/maskclip/dcsm_tok | 0.0 |
| readout/clearclip/pooled | 0.0 |
| readout/clearclip/dcsm_tok | 0.0 |

**color breakdown** (top-1 %, n per column)
| scorer | black (n=1) |
|---|---|
| owlv2 | 100.0 |
| owlvit_v1 | 100.0 |
| clip_crop | 0.0 |
| clip_ctx | 0.0 |
| dcsm_naive_retrain_ctx | 0.0 |
| dcsm_maskclip_retrain_ctx | 0.0 |
| dcsm_clearclip_retrain_ctx | 100.0 |
| readout/naive/pooled | 100.0 |
| readout/naive/dcsm_tok | 0.0 |
| readout/maskclip/pooled | 0.0 |
| readout/maskclip/dcsm_tok | 0.0 |
| readout/clearclip/pooled | 0.0 |
| readout/clearclip/dcsm_tok | 0.0 |
