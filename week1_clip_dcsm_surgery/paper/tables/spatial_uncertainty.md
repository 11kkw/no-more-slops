Exploratory image-cluster bootstrap; spatial balanced accuracy, 527 items / 375 images.
10,000 replicates; seed 20261010. Intervals concern this constructed sample.

| quantity | estimate (%) or difference (pp) | 95% interval |
| --- | --- | --- |
| owlv2 | 43.2 | [39.5, 46.9] |
| owlvit_v1 | 43.8 | [40.1, 47.3] |
| clip_crop | 44.2 | [40.6, 48.0] |
| clip_ctx | 39.0 | [35.4, 42.7] |
| dcsm_obja_ctx | 58.1 | [53.7, 62.4] |
| dcsm_coco_ctx | 57.3 | [53.2, 61.2] |
| readout/naive/pooled | 50.3 | [46.6, 53.9] |
| readout/naive/dcsm_tok | 50.0 | [46.2, 53.9] |
| readout/maskclip/pooled | 39.4 | [35.7, 43.0] |
| readout/maskclip/dcsm_tok | 43.4 | [39.6, 47.1] |
| readout/clearclip/pooled | 39.4 | [35.8, 43.1] |
| readout/clearclip/dcsm_tok | 42.5 | [38.8, 46.3] |
| chance | 45.7 | [45.0, 46.5] |
| dcsm_obja_ctx - chance | 12.4 | [8.0, 16.7] |
| dcsm_coco_ctx - chance | 11.5 | [7.4, 15.5] |
| dcsm_obja_ctx - clip_ctx | 19.1 | [14.3, 23.9] |
| dcsm_coco_ctx - clip_ctx | 18.2 | [13.4, 23.0] |
| dcsm_obja_ctx - readout/clearclip/pooled | 18.7 | [13.1, 24.2] |
| dcsm_obja_ctx - owlv2 | 14.9 | [9.2, 20.4] |
