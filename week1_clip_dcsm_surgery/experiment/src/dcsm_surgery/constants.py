"""Shared constants."""
CLIP_NAME = "openai/clip-vit-base-patch16"

# Functional Rows (Kang et al.): the 11 words whose cosine columns are replaced by learned 197-d vectors.
# The code base calls them "special words"; the paper calls them Functional Rows.
SPECIAL_WORDS = ["above", "below", "left", "right", "small", "big", "not", "no", "without", "but", "absent"]
TEXT_SEQ = 30  # text tokens (BOS + content + EOS + padding)
IMG_SEQ = 14 * 14 + 1  # CLS + 14x14 patches at 224 px

MODES = ["naive", "maskclip", "clearclip"]  # last-block readouts
TEXT_READOUTS = ["pooled", "dcsm_tok"]  # EOS embedding vs max over content-token columns

# failure-mode query benchmark subsets (attribute binding = size, color; spatial relationship = spatial;
# negation = negation, with affirm as its control)
SUBSETS = ["size", "color", "spatial", "negation", "affirm"]
