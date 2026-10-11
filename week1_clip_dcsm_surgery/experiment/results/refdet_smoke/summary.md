# Referring detection on COCO val2017 relational queries (15 items)

candidates: OWLv2 class-name boxes, p(A) ≥ 0.1, NMS 0.5, ≤ 10; native methods: NMS 0.5, ≤ 100 dets/item

| | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| #items | 3 | 3 | 3 | 3 | 3 |
| candidate recall@0.5 (upper bound for cands/*) | 66.7 | 66.7 | 100.0 | 100.0 | 100.0 |
| mean #candidates | 4.7 | 5.7 | 2.3 | 4.0 | 4.0 |

## Acc@0.5 (top-1 box IoU ≥ 0.5, %)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 66.7 | 33.3 | 33.3 | 0.0 | 0.0 |
| owlvit_v1_native | 66.7 | 66.7 | 0.0 | 0.0 | 33.3 |
| gdino_tiny | 66.7 | 66.7 | 33.3 | 33.3 | 66.7 |
| gdino_base | 66.7 | 66.7 | 66.7 | 33.3 | 100.0 |
| cands/owlv2_classname | 66.7 | 33.3 | 33.3 | 66.7 | 66.7 |
| cands/clip_ctx | 33.3 | 33.3 | 33.3 | 33.3 | 33.3 |
| cands/dcsm_obja | 33.3 | 66.7 | 66.7 | 66.7 | 100.0 |
| cands/dcsm_coco | 66.7 | 66.7 | 33.3 | 33.3 | 66.7 |
| cands/dcsm_clearclip_retrain | 66.7 | 0.0 | 100.0 | 33.3 | 66.7 |
| cands/dcsm_maskclip_retrain | 33.3 | 33.3 | 100.0 | 66.7 | 33.3 |
| cands/dcsm_naive_retrain | 33.3 | 33.3 | 100.0 | 33.3 | 33.3 |

## Acc@0.5 size-balanced (%)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 50.0 | n/a | 50.0 | 0.0 | 0.0 |
| owlvit_v1_native | 50.0 | n/a | 0.0 | 0.0 | 25.0 |
| gdino_tiny | 50.0 | n/a | 50.0 | 50.0 | 50.0 |
| gdino_base | 50.0 | n/a | 75.0 | 50.0 | 100.0 |
| cands/owlv2_classname | 50.0 | n/a | 50.0 | 75.0 | 75.0 |
| cands/clip_ctx | 25.0 | n/a | 25.0 | 25.0 | 50.0 |
| cands/dcsm_obja | 25.0 | n/a | 75.0 | 75.0 | 100.0 |
| cands/dcsm_coco | 50.0 | n/a | 25.0 | 25.0 | 75.0 |
| cands/dcsm_clearclip_retrain | 50.0 | n/a | 100.0 | 50.0 | 75.0 |
| cands/dcsm_maskclip_retrain | 25.0 | n/a | 100.0 | 75.0 | 50.0 |
| cands/dcsm_naive_retrain | 25.0 | n/a | 100.0 | 25.0 | 50.0 |

## AP (0.5:0.95)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 52.5 | 11.0 | 55.7 | 31.1 | 2.2 |
| owlvit_v1_native | 61.6 | 7.7 | 17.1 | 28.4 | 13.8 |
| gdino_tiny | 64.7 | 47.8 | 28.7 | 13.9 | 81.2 |
| gdino_base | 65.9 | 49.7 | 73.2 | 26.5 | 93.4 |
| cands/owlv2_classname | 61.4 | 40.8 | 58.5 | 64.4 | 53.0 |
| cands/clip_ctx | 40.9 | 27.4 | 63.4 | 53.0 | 35.6 |
| cands/dcsm_obja | 40.9 | 37.1 | 75.9 | 69.9 | 83.4 |
| cands/dcsm_coco | 53.3 | 56.4 | 63.4 | 62.1 | 73.9 |
| cands/dcsm_clearclip_retrain | 53.3 | 22.1 | 86.6 | 47.6 | 60.7 |
| cands/dcsm_maskclip_retrain | 30.7 | 37.8 | 86.6 | 58.4 | 30.8 |
| cands/dcsm_naive_retrain | 27.6 | 21.4 | 86.6 | 58.0 | 36.1 |

## AP50

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 55.9 | 13.3 | 62.7 | 31.1 | 2.5 |
| owlvit_v1_native | 75.5 | 43.0 | 27.6 | 35.2 | 14.8 |
| gdino_tiny | 68.7 | 56.7 | 32.2 | 30.7 | 86.5 |
| gdino_base | 70.4 | 57.7 | 79.0 | 37.2 | 100.0 |
| cands/owlv2_classname | 66.3 | 46.7 | 66.8 | 64.4 | 59.4 |
| cands/clip_ctx | 44.2 | 33.2 | 73.5 | 53.0 | 45.3 |
| cands/dcsm_obja | 44.2 | 44.2 | 86.5 | 69.9 | 100.0 |
| cands/dcsm_coco | 55.4 | 66.3 | 72.3 | 62.1 | 86.5 |
| cands/dcsm_clearclip_retrain | 55.4 | 26.5 | 100.0 | 47.6 | 75.6 |
| cands/dcsm_maskclip_retrain | 33.2 | 43.0 | 100.0 | 58.4 | 37.8 |
| cands/dcsm_naive_retrain | 29.9 | 26.2 | 100.0 | 58.0 | 44.4 |

## spatial Acc@0.5 by relation (%)

| method | above (n=1) | to the left of (n=1) | to the right of (n=1) |
|---|---|---|---|
| owlv2_native | 0.0 | 100.0 | 0.0 |
| owlvit_v1_native | 0.0 | 0.0 | 0.0 |
| gdino_tiny | 0.0 | 100.0 | 0.0 |
| gdino_base | 0.0 | 100.0 | 100.0 |
| cands/owlv2_classname | 0.0 | 100.0 | 0.0 |
| cands/clip_ctx | 0.0 | 0.0 | 100.0 |
| cands/dcsm_obja | 0.0 | 100.0 | 100.0 |
| cands/dcsm_coco | 0.0 | 0.0 | 100.0 |
| cands/dcsm_clearclip_retrain | 100.0 | 100.0 | 100.0 |
| cands/dcsm_maskclip_retrain | 100.0 | 100.0 | 100.0 |
| cands/dcsm_naive_retrain | 100.0 | 100.0 | 100.0 |

## paired bootstrap, spatial Acc@0.5 difference (10k resamples, 95 % CI)

| A | B | A | B | diff | CI |
|---|---|---|---|---|---|
| cands/dcsm_obja | owlv2_native | 66.7 | 33.3 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_obja | gdino_tiny | 66.7 | 33.3 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_obja | gdino_base | 66.7 | 66.7 | +0.0 | [+0.0, +0.0] |
| cands/dcsm_obja | cands/owlv2_classname | 66.7 | 33.3 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_obja | cands/clip_ctx | 66.7 | 33.3 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_coco | owlv2_native | 33.3 | 33.3 | +0.0 | [-100.0, +100.0] |
| cands/dcsm_coco | gdino_tiny | 33.3 | 33.3 | +0.0 | [-100.0, +100.0] |
| cands/dcsm_coco | gdino_base | 33.3 | 66.7 | -33.3 | [-100.0, +0.0] |
| cands/dcsm_coco | cands/owlv2_classname | 33.3 | 33.3 | +0.0 | [-100.0, +100.0] |
| cands/dcsm_coco | cands/clip_ctx | 33.3 | 33.3 | +0.0 | [+0.0, +0.0] |
| cands/dcsm_clearclip_retrain_s15 | owlv2_native | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_clearclip_retrain_s15 | gdino_tiny | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_clearclip_retrain_s15 | gdino_base | 100.0 | 66.7 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_clearclip_retrain_s15 | cands/owlv2_classname | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_clearclip_retrain_s15 | cands/clip_ctx | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_maskclip_retrain_s15 | owlv2_native | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_maskclip_retrain_s15 | gdino_tiny | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_maskclip_retrain_s15 | gdino_base | 100.0 | 66.7 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_maskclip_retrain_s15 | cands/owlv2_classname | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_maskclip_retrain_s15 | cands/clip_ctx | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_naive_retrain_s15 | owlv2_native | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_naive_retrain_s15 | gdino_tiny | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_naive_retrain_s15 | gdino_base | 100.0 | 66.7 | +33.3 | [+0.0, +100.0] |
| cands/dcsm_naive_retrain_s15 | cands/owlv2_classname | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |
| cands/dcsm_naive_retrain_s15 | cands/clip_ctx | 100.0 | 33.3 | +66.7 | [+0.0, +100.0] |

