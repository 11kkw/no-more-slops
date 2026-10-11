"""pytest setup: use the experiment-local Hugging Face cache and skip model-dependent tests when it is absent."""
import os
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
CACHE = EXPERIMENT / "hf_cache"
if CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CLIP_NAME = "openai/clip-vit-base-patch16"


def _clip_cached():
    return (CACHE / "hub" / ("models--" + CLIP_NAME.replace("/", "--"))).is_dir()


requires_clip = pytest.mark.skipif(not _clip_cached(), reason="CLIP ViT-B/16 weights not in experiment/hf_cache")


@pytest.fixture(scope="session")
def clip_tokenizer():
    from transformers import CLIPTokenizer
    if not _clip_cached():
        pytest.skip("CLIP tokenizer not cached")
    return CLIPTokenizer.from_pretrained(CLIP_NAME, local_files_only=True)


@pytest.fixture(scope="session")
def clip_model():
    import torch
    from transformers import CLIPModel
    if not _clip_cached():
        pytest.skip("CLIP weights not cached")
    torch.set_num_threads(4)
    return CLIPModel.from_pretrained(CLIP_NAME, local_files_only=True, torch_dtype=torch.float32).eval()
