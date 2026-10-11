"""Frozen CLIP ViT dense features under three last-block readouts, plus text features and heatmaps.

  naive     : last_hidden_state -> post_layernorm -> visual_projection      (the DCSM repo's code path)
  maskclip  : last-block value embeddings, no q-k attention / residual / FFN (MaskCLIP, ECCV 2022)
  clearclip : last-block q-q self-self attention, no residual / FFN          (ClearCLIP, ECCV 2024)
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """Whole-image anisotropic resize to size x size (no centre crop), CLIP normalisation -> (1,3,size,size)."""
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def dense_features(model, pixel_values, size):
    """Return ({readout: (B, N+1, D) L2-normalised features in CLIP joint space}, pooled (B, D))."""
    vm = model.vision_model
    kwargs = {"output_hidden_states": True}
    if size != 224:
        kwargs["interpolate_pos_encoding"] = True
    out = vm(pixel_values=pixel_values, **kwargs)
    feats = {}

    x_in = out.hidden_states[-2]  # input to the last block
    layer = vm.encoder.layers[-1]
    attn = layer.self_attn
    h = layer.layer_norm1(x_in)
    B, N, D = h.shape
    nh = attn.num_heads
    hd = D // nh
    scale = getattr(attn, "scale", None) or getattr(attn, "scaling", hd ** -0.5)
    q = attn.q_proj(h).view(B, N, nh, hd).transpose(1, 2)
    v = attn.v_proj(h).view(B, N, nh, hd).transpose(1, 2)

    variants = {
        "naive": out.last_hidden_state,
        "maskclip": attn.out_proj(v.transpose(1, 2).reshape(B, N, D)),
        "clearclip": attn.out_proj(
            (torch.softmax((q @ q.transpose(-1, -2)) * scale, dim=-1) @ v).transpose(1, 2).reshape(B, N, D)
        ),
    }
    for k, x in variants.items():
        x = model.visual_projection(vm.post_layernorm(x))
        feats[k] = F.normalize(x, dim=-1)
    pooled = F.normalize(model.visual_projection(out.pooler_output), dim=-1)
    return feats, pooled


@torch.no_grad()
def text_features(model, tokenizer, prompts, device, max_len=30):
    """-> pooled (C, D), tokens (C, T, D) (both L2-normalised), content mask (C, T) excluding BOS/EOS/padding."""
    tok = tokenizer(prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt").to(device)
    tout = model.text_model(**tok)
    pooled = F.normalize(model.text_projection(tout.pooler_output), dim=-1)
    tokens = F.normalize(model.text_projection(tout.last_hidden_state), dim=-1)
    ids = tok["input_ids"]
    eos_pos = (ids == tokenizer.eos_token_id).int().argmax(dim=1)
    content = torch.zeros_like(ids, dtype=torch.bool)
    for i, e in enumerate(eos_pos.tolist()):
        content[i, 1:e] = True
    return pooled, tokens, content


@torch.no_grad()
def text_token_features(model, tokenizer, text, device, max_len=30):
    """Single query -> (tokenizer output, (1, T, D) L2-normalised projected token embeddings), as the DCSM scorer uses."""
    tt = tokenizer(text=[text], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt").to(device)
    txt = model.text_projection(model.text_model(**tt).last_hidden_state)
    return tt, txt / txt.norm(dim=-1, keepdim=True)


def heatmaps(patch_feats, txt_pooled, txt_tokens, content_mask, grid):
    """patch_feats: (N, D) without CLS. Returns {'pooled': (C, g, g), 'dcsm_tok': (C, g, g)}."""
    hm = {}
    hm["pooled"] = (patch_feats @ txt_pooled.T).T.reshape(-1, grid, grid)
    sim = torch.einsum("nd,ctd->cnt", patch_feats, txt_tokens)  # (C, N, T) == DCSM matrix per class
    sim = sim.masked_fill(~content_mask[:, None, :], -1e4)
    hm["dcsm_tok"] = sim.max(dim=-1).values.reshape(-1, grid, grid)
    return hm
