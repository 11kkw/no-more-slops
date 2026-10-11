"""Readout + scorer tests with the real (CPU) CLIP ViT-B/16; skipped when the weights are not cached."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dcsm_surgery.bpe_alignment import ALIGNMENT_VERSION
from dcsm_surgery.clip_readouts import dense_features, heatmaps, preprocess, text_features, text_token_features
from dcsm_surgery.constants import IMG_SEQ, MODES, SPECIAL_WORDS, TEXT_SEQ
from dcsm_surgery.scorer import CrossModalConvNetwork, DCSMScorer, clip_patch_feats, initialize_shared
from tests.conftest import requires_clip


def test_cross_modal_conv_network_shape():
    net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ)
    out = net(torch.randn(3, 2, IMG_SEQ, TEXT_SEQ))
    assert out.shape == (3, 2, 1)


def test_initialize_shared_identical_values_independent_storage():
    nets, embs = initialize_shared(["naive", "maskclip"], "cpu")
    for k, v in nets["naive"].state_dict().items():
        assert torch.equal(v, nets["maskclip"].state_dict()[k])
    assert torch.equal(embs["naive"].weight, embs["maskclip"].weight)
    assert embs["naive"].weight.data_ptr() != embs["maskclip"].weight.data_ptr()
    assert embs["naive"].weight.shape == (len(SPECIAL_WORDS), IMG_SEQ)


def test_preprocess_shape_and_normalisation():
    x = preprocess(Image.new("RGB", (30, 50), (124, 116, 104)), 32)
    assert x.shape == (1, 3, 32, 32)
    assert torch.allclose(x, torch.zeros_like(x), atol=0.05)  # mean colour -> ~0 after CLIP normalisation


@requires_clip
def test_dense_features_three_readouts(clip_model, clip_tokenizer):
    img = Image.fromarray((np.random.default_rng(0).random((64, 48, 3)) * 255).astype(np.uint8))
    feats, pooled = dense_features(clip_model, preprocess(img, 224), 224)
    assert set(feats) == set(MODES) and pooled.shape == (1, 512)
    for m in MODES:
        assert feats[m].shape == (1, IMG_SEQ, 512)
        assert torch.allclose(feats[m].norm(dim=-1), torch.ones(1, IMG_SEQ), atol=1e-4)
    assert not torch.allclose(feats["naive"], feats["maskclip"], atol=1e-3)
    # naive readout == the DCSM repo's code path used by the scorer
    pf = clip_patch_feats(clip_model, preprocess(img, 224), ["naive"])["naive"]
    torch.testing.assert_close(pf, feats["naive"], atol=1e-5, rtol=1e-5)
    pooled_t, tokens, content = text_features(clip_model, clip_tokenizer, ["a photo of a cat."], "cpu")
    assert tokens.shape == (1, 30, 512) and not content[0, 0] and content.sum() == 6  # a photo of a cat .
    hm = heatmaps(feats["clearclip"][0, 1:], pooled_t, tokens, content, 14)
    assert hm["pooled"].shape == (1, 14, 14) and hm["dcsm_tok"].shape == (1, 14, 14)


@requires_clip
def test_dcsm_scorer_slicing_modes_and_batch_dependence(clip_model, clip_tokenizer):
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
    net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ)
    emb = torch.nn.Embedding(len(SPECIAL_WORDS), IMG_SEQ)
    crops = [Image.new("RGB", (64, 64), c) for c in ("red", "blue", "green")]
    text = "a picture of a cat above a dog"
    scores = {}
    with tempfile.TemporaryDirectory() as tmp:
        for slicing in (ALIGNMENT_VERSION, "replace", "eval_shift"):
            p = Path(tmp) / f"{slicing}.pt"
            torch.save({"model_state_dict": net.state_dict(), "external_logic_token_embeddings": emb,
                        "special_slicing": slicing, "readout": "maskclip"}, p)
            s = DCSMScorer(str(p), clip_model, clip_tokenizer, proc, "cpu")
            assert s.readout == "maskclip" and s.slicing == slicing
            out = s.score(crops, text)
            assert out.shape == (3,) and np.isfinite(out).all()
            scores[slicing] = out
            # score_feats is the path used by the end-to-end evaluation; it must match score()
            pv = proc(images=crops, return_tensors="pt")["pixel_values"]
            feats = clip_patch_feats(clip_model, pv, ["maskclip"])["maskclip"]
            tt, txt = text_token_features(clip_model, clip_tokenizer, text, "cpu", TEXT_SEQ)
            np.testing.assert_allclose(s.score_feats(feats, txt, tt, text).numpy(), out, atol=1e-5)
        # the misaligned legacy rules replace a different column than the BPE-aligned rule
        assert not np.allclose(scores[ALIGNMENT_VERSION], scores["eval_shift"])
        # batch standardisation makes scores depend on the candidate set
        sub = DCSMScorer(str(Path(tmp) / f"{ALIGNMENT_VERSION}.pt"), clip_model, clip_tokenizer, proc, "cpu").score(crops[:2], text)
        assert not np.allclose(sub, scores[ALIGNMENT_VERSION][:2])
