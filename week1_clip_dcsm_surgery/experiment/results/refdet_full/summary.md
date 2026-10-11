# Referring detection on COCO val2017 relational queries (1833 items)

candidates: OWLv2 class-name boxes, p(A) ≥ 0.1, NMS 0.5, ≤ 10; native methods: NMS 0.5, ≤ 100 dets/item

| | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| #items | 655 | 142 | 527 | 315 | 194 |
| candidate recall@0.5 (upper bound for cands/*) | 91.9 | 94.4 | 89.2 | 81.9 | 92.3 |
| mean #candidates | 5.3 | 4.9 | 4.2 | 4.2 | 4.9 |

## Acc@0.5 (top-1 box IoU ≥ 0.5, %)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 58.9 | 69.7 | 7.8 | 13.0 | 4.1 |
| owlvit_v1_native | 62.6 | 59.9 | 12.0 | 21.9 | 19.1 |
| gdino_tiny | 71.6 | 69.0 | 17.1 | 16.2 | 38.7 |
| gdino_base | 74.8 | 72.5 | 63.0 | 30.2 | 57.7 |
| cands/owlv2_classname | 57.6 | 48.6 | 37.8 | 31.1 | 46.9 |
| cands/clip_ctx | 39.7 | 57.7 | 27.9 | 26.0 | 36.1 |
| cands/dcsm_obja | 38.0 | 57.7 | 41.6 | 27.3 | 46.4 |
| cands/dcsm_coco | 47.9 | 59.2 | 43.1 | 26.0 | 42.3 |
| cands/dcsm_clearclip_retrain (3 seeds) | 44.0 ± 1.6 | 51.4 ± 3.2 | 48.3 ± 1.3 | 26.7 ± 1.4 | 37.8 ± 3.9 |
| cands/dcsm_maskclip_retrain (3 seeds) | 42.8 ± 3.2 | 54.2 ± 1.1 | 49.7 ± 0.6 | 28.1 ± 1.2 | 38.7 ± 2.2 |
| cands/dcsm_naive_retrain (3 seeds) | 39.0 ± 4.4 | 48.6 ± 5.7 | 40.7 ± 1.3 | 26.8 ± 1.8 | 39.5 ± 2.1 |

## Acc@0.5 size-balanced (%)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 38.9 | 68.8 | 8.1 | 15.1 | 4.1 |
| owlvit_v1_native | 41.0 | 57.8 | 12.6 | 26.0 | 17.4 |
| gdino_tiny | 42.4 | 66.6 | 18.2 | 26.5 | 34.9 |
| gdino_base | 44.8 | 71.4 | 64.3 | 43.1 | 53.0 |
| cands/owlv2_classname | 37.1 | 43.6 | 39.2 | 40.5 | 42.9 |
| cands/clip_ctx | 30.1 | 54.9 | 28.4 | 27.8 | 34.6 |
| cands/dcsm_obja | 27.0 | 56.1 | 42.1 | 30.0 | 44.5 |
| cands/dcsm_coco | 34.3 | 57.6 | 43.9 | 29.2 | 39.9 |
| cands/dcsm_clearclip_retrain (3 seeds) | 29.0 ± 2.5 | 49.9 ± 3.3 | 49.1 ± 1.2 | 31.6 ± 0.7 | 36.5 ± 3.8 |
| cands/dcsm_maskclip_retrain (3 seeds) | 31.9 ± 2.5 | 52.0 ± 2.1 | 50.6 ± 0.4 | 34.2 ± 1.9 | 37.2 ± 2.7 |
| cands/dcsm_naive_retrain (3 seeds) | 30.8 ± 2.5 | 47.3 ± 5.1 | 41.3 ± 1.4 | 30.3 ± 1.8 | 38.8 ± 1.8 |

## AP (0.5:0.95)

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 28.1 | 40.4 | 6.7 | 5.9 | 1.9 |
| owlvit_v1_native | 29.5 | 24.6 | 5.5 | 8.3 | 4.7 |
| gdino_tiny | 55.2 | 51.9 | 14.0 | 6.2 | 20.3 |
| gdino_base | 57.4 | 55.7 | 52.0 | 18.2 | 38.4 |
| cands/owlv2_classname | 32.2 | 32.4 | 27.5 | 20.0 | 24.3 |
| cands/clip_ctx | 29.6 | 43.6 | 19.7 | 14.2 | 24.9 |
| cands/dcsm_obja | 28.3 | 42.1 | 30.7 | 16.2 | 26.7 |
| cands/dcsm_coco | 33.1 | 41.5 | 32.3 | 16.2 | 26.1 |
| cands/dcsm_clearclip_retrain (3 seeds) | 32.4 ± 2.1 | 38.5 ± 4.0 | 34.4 ± 1.2 | 15.8 ± 0.8 | 23.6 ± 1.8 |
| cands/dcsm_maskclip_retrain (3 seeds) | 31.4 ± 2.6 | 38.7 ± 0.7 | 34.4 ± 1.0 | 16.6 ± 0.5 | 23.7 ± 1.7 |
| cands/dcsm_naive_retrain (3 seeds) | 28.8 ± 3.1 | 35.5 ± 2.8 | 28.1 ± 0.6 | 15.4 ± 0.6 | 23.8 ± 1.8 |

## AP50

| method | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| owlv2_native | 37.5 | 53.3 | 9.6 | 9.7 | 3.6 |
| owlvit_v1_native | 44.1 | 43.6 | 9.2 | 15.5 | 9.2 |
| gdino_tiny | 70.9 | 72.4 | 19.4 | 10.4 | 30.8 |
| gdino_base | 73.1 | 73.1 | 66.3 | 26.3 | 54.9 |
| cands/owlv2_classname | 43.3 | 43.6 | 37.5 | 30.8 | 38.9 |
| cands/clip_ctx | 40.4 | 58.2 | 27.1 | 22.2 | 38.3 |
| cands/dcsm_obja | 38.4 | 56.4 | 41.0 | 25.2 | 42.7 |
| cands/dcsm_coco | 44.7 | 55.0 | 43.8 | 25.1 | 40.7 |
| cands/dcsm_clearclip_retrain (3 seeds) | 44.2 ± 3.1 | 50.9 ± 4.4 | 47.1 ± 1.5 | 24.8 ± 1.0 | 37.5 ± 3.0 |
| cands/dcsm_maskclip_retrain (3 seeds) | 42.7 ± 3.3 | 52.0 ± 0.5 | 47.5 ± 1.2 | 25.9 ± 0.7 | 38.0 ± 3.2 |
| cands/dcsm_naive_retrain (3 seeds) | 39.1 ± 4.0 | 48.0 ± 3.6 | 38.5 ± 0.9 | 23.8 ± 0.7 | 37.8 ± 1.8 |

## spatial Acc@0.5 by relation (%)

| method | above (n=62) | below (n=52) | to the left of (n=227) | to the right of (n=186) |
|---|---|---|---|---|
| owlv2_native | 6.5 | 3.8 | 7.9 | 9.1 |
| owlvit_v1_native | 19.4 | 19.2 | 9.3 | 10.8 |
| gdino_tiny | 17.7 | 30.8 | 18.1 | 11.8 |
| gdino_base | 32.3 | 48.1 | 65.2 | 74.7 |
| cands/owlv2_classname | 32.3 | 32.7 | 37.4 | 41.4 |
| cands/clip_ctx | 16.1 | 38.5 | 26.9 | 30.1 |
| cands/dcsm_obja | 25.8 | 36.5 | 44.5 | 44.6 |
| cands/dcsm_coco | 25.8 | 40.4 | 45.4 | 46.8 |
| cands/dcsm_clearclip_retrain | 37.1 ± 1.3 | 41.7 ± 6.3 | 50.5 ± 2.7 | 51.1 ± 0.4 |
| cands/dcsm_maskclip_retrain | 37.6 ± 4.0 | 40.4 ± 3.1 | 52.9 ± 1.8 | 52.5 ± 2.1 |
| cands/dcsm_naive_retrain | 28.0 ± 3.3 | 34.0 ± 3.3 | 43.5 ± 2.9 | 43.5 ± 1.9 |

## paired bootstrap, spatial Acc@0.5 difference (10k resamples, 95 % CI)

| A | B | acc A | acc B | diff | CI |
|---|---|---|---|---|---|
| cands/dcsm_obja | owlv2_native | 41.6 | 7.8 | +33.8 | [+29.2, +38.3] |
| cands/dcsm_obja | gdino_tiny | 41.6 | 17.1 | +24.5 | [+19.5, +29.4] |
| cands/dcsm_obja | gdino_base | 41.6 | 63.0 | -21.4 | [-26.6, -16.3] |
| cands/dcsm_obja | cands/owlv2_classname | 41.6 | 37.8 | +3.8 | [-1.5, +9.1] |
| cands/dcsm_obja | cands/clip_ctx | 41.6 | 27.9 | +13.7 | [+9.1, +18.4] |
| cands/dcsm_coco | owlv2_native | 43.1 | 7.8 | +35.3 | [+30.6, +39.9] |
| cands/dcsm_coco | gdino_tiny | 43.1 | 17.1 | +26.0 | [+21.1, +30.9] |
| cands/dcsm_coco | gdino_base | 43.1 | 63.0 | -19.9 | [-24.9, -14.8] |
| cands/dcsm_coco | cands/owlv2_classname | 43.1 | 37.8 | +5.3 | [+0.4, +10.2] |
| cands/dcsm_coco | cands/clip_ctx | 43.1 | 27.9 | +15.2 | [+11.0, +19.4] |
| cands/dcsm_clearclip_retrain_s15 | owlv2_native | 49.9 | 7.8 | +42.1 | [+37.6, +46.9] |
| cands/dcsm_clearclip_retrain_s15 | gdino_tiny | 49.9 | 17.1 | +32.8 | [+27.9, +38.0] |
| cands/dcsm_clearclip_retrain_s15 | gdino_base | 49.9 | 63.0 | -13.1 | [-18.0, -8.0] |
| cands/dcsm_clearclip_retrain_s15 | cands/owlv2_classname | 49.9 | 37.8 | +12.1 | [+6.8, +17.3] |
| cands/dcsm_clearclip_retrain_s15 | cands/clip_ctx | 49.9 | 27.9 | +22.0 | [+17.3, +26.8] |
| cands/dcsm_maskclip_retrain_s15 | owlv2_native | 49.1 | 7.8 | +41.4 | [+36.4, +45.9] |
| cands/dcsm_maskclip_retrain_s15 | gdino_tiny | 49.1 | 17.1 | +32.1 | [+27.1, +37.0] |
| cands/dcsm_maskclip_retrain_s15 | gdino_base | 49.1 | 63.0 | -13.9 | [-19.0, -8.9] |
| cands/dcsm_maskclip_retrain_s15 | cands/owlv2_classname | 49.1 | 37.8 | +11.4 | [+6.1, +16.7] |
| cands/dcsm_maskclip_retrain_s15 | cands/clip_ctx | 49.1 | 27.9 | +21.3 | [+16.5, +25.8] |
| cands/dcsm_naive_retrain_s15 | owlv2_native | 41.9 | 7.8 | +34.2 | [+29.4, +38.9] |
| cands/dcsm_naive_retrain_s15 | gdino_tiny | 41.9 | 17.1 | +24.9 | [+19.9, +29.8] |
| cands/dcsm_naive_retrain_s15 | gdino_base | 41.9 | 63.0 | -21.1 | [-26.4, -15.7] |
| cands/dcsm_naive_retrain_s15 | cands/owlv2_classname | 41.9 | 37.8 | +4.2 | [-1.3, +9.7] |
| cands/dcsm_naive_retrain_s15 | cands/clip_ctx | 41.9 | 27.9 | +14.0 | [+9.1, +19.0] |
| cands/dcsm_clearclip_retrain_s16 | owlv2_native | 48.2 | 7.8 | +40.4 | [+35.9, +45.0] |
| cands/dcsm_clearclip_retrain_s16 | gdino_tiny | 48.2 | 17.1 | +31.1 | [+25.8, +36.2] |
| cands/dcsm_clearclip_retrain_s16 | gdino_base | 48.2 | 63.0 | -14.8 | [-19.9, -9.7] |
| cands/dcsm_clearclip_retrain_s16 | cands/owlv2_classname | 48.2 | 37.8 | +10.4 | [+5.1, +15.7] |
| cands/dcsm_clearclip_retrain_s16 | cands/clip_ctx | 48.2 | 27.9 | +20.3 | [+15.4, +25.0] |
| cands/dcsm_maskclip_retrain_s16 | owlv2_native | 49.5 | 7.8 | +41.7 | [+37.0, +46.3] |
| cands/dcsm_maskclip_retrain_s16 | gdino_tiny | 49.5 | 17.1 | +32.4 | [+27.5, +37.4] |
| cands/dcsm_maskclip_retrain_s16 | gdino_base | 49.5 | 63.0 | -13.5 | [-18.8, -8.3] |
| cands/dcsm_maskclip_retrain_s16 | cands/owlv2_classname | 49.5 | 37.8 | +11.8 | [+6.5, +16.9] |
| cands/dcsm_maskclip_retrain_s16 | cands/clip_ctx | 49.5 | 27.9 | +21.6 | [+16.7, +26.4] |
| cands/dcsm_naive_retrain_s16 | owlv2_native | 41.4 | 7.8 | +33.6 | [+28.8, +38.1] |
| cands/dcsm_naive_retrain_s16 | gdino_tiny | 41.4 | 17.1 | +24.3 | [+19.4, +29.2] |
| cands/dcsm_naive_retrain_s16 | gdino_base | 41.4 | 63.0 | -21.6 | [-26.6, -16.5] |
| cands/dcsm_naive_retrain_s16 | cands/owlv2_classname | 41.4 | 37.8 | +3.6 | [-1.9, +9.1] |
| cands/dcsm_naive_retrain_s16 | cands/clip_ctx | 41.4 | 27.9 | +13.5 | [+8.9, +18.0] |
| cands/dcsm_clearclip_retrain_s17 | owlv2_native | 46.7 | 7.8 | +38.9 | [+34.3, +43.5] |
| cands/dcsm_clearclip_retrain_s17 | gdino_tiny | 46.7 | 17.1 | +29.6 | [+24.9, +34.5] |
| cands/dcsm_clearclip_retrain_s17 | gdino_base | 46.7 | 63.0 | -16.3 | [-21.3, -11.4] |
| cands/dcsm_clearclip_retrain_s17 | cands/owlv2_classname | 46.7 | 37.8 | +8.9 | [+3.8, +14.0] |
| cands/dcsm_clearclip_retrain_s17 | cands/clip_ctx | 46.7 | 27.9 | +18.8 | [+14.2, +23.5] |
| cands/dcsm_maskclip_retrain_s17 | owlv2_native | 50.5 | 7.8 | +42.7 | [+38.1, +47.2] |
| cands/dcsm_maskclip_retrain_s17 | gdino_tiny | 50.5 | 17.1 | +33.4 | [+28.3, +38.3] |
| cands/dcsm_maskclip_retrain_s17 | gdino_base | 50.5 | 63.0 | -12.5 | [-17.5, -7.4] |
| cands/dcsm_maskclip_retrain_s17 | cands/owlv2_classname | 50.5 | 37.8 | +12.7 | [+7.4, +17.8] |
| cands/dcsm_maskclip_retrain_s17 | cands/clip_ctx | 50.5 | 27.9 | +22.6 | [+17.8, +27.5] |
| cands/dcsm_naive_retrain_s17 | owlv2_native | 38.9 | 7.8 | +31.1 | [+26.4, +35.7] |
| cands/dcsm_naive_retrain_s17 | gdino_tiny | 38.9 | 17.1 | +21.8 | [+16.9, +26.8] |
| cands/dcsm_naive_retrain_s17 | gdino_base | 38.9 | 63.0 | -24.1 | [-29.4, -18.8] |
| cands/dcsm_naive_retrain_s17 | cands/owlv2_classname | 38.9 | 37.8 | +1.1 | [-4.4, +6.5] |
| cands/dcsm_naive_retrain_s17 | cands/clip_ctx | 38.9 | 27.9 | +11.0 | [+6.5, +15.6] |

