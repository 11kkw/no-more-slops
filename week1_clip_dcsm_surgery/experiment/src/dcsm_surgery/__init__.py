"""dcsm_surgery: frozen CLIP ViT-B/16 last-block readouts, the DCSM scoring network and the evaluation
primitives (boxes, detectors, metrics) shared by the experiment scripts in ../../scripts.

Modules
  constants      special-word list, sequence lengths, readout names, benchmark subsets
  clip_readouts  whole-image preprocessing, naive / MaskCLIP / ClearCLIP dense features, text features, heatmaps
  bpe_alignment  BPE-aligned Functional-Row (special-word) column replacement shared by training and evaluation
  scorer         CrossModalConvNetwork, checkpoint loading, DCSMScorer, scorer initialisation helpers
  boxes          box geometry, crops, OWL box conversion, NMS helpers
  detectors      OWL-ViT / OWLv2 / Grounding DINO helpers
  metrics        COCO AP, referring-detection AP, probe metrics, paired bootstrap
  data           training datasets, augmentation and collate for the DCSM scorer retraining recipe
"""
from .constants import CLIP_NAME, IMG_SEQ, MODES, SPECIAL_WORDS, SUBSETS, TEXT_READOUTS, TEXT_SEQ  # noqa: F401
