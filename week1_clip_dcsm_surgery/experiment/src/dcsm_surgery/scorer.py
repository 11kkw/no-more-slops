"""The DCSM scoring network (Kang et al., ICCV 2025) and helpers to load, initialise and apply it.

Checkpoint conventions
  released (data/dcsm_ckpt/*.pt)   naive readout; Functional-Row columns inserted with the repo's eval-time
                                   whitespace-index rule ("eval_shift": insert, drop last column)
  first retrain (results/retrain)  readout recorded; whitespace-index "replace" rule (misaligned, kept for history)
  BPE-aligned retrain              readout recorded; special_slicing == ALIGNMENT_VERSION (bpe_alignment.py)
"""
import copy
import hashlib
import pickle
import types
import warnings
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer

from .bpe_alignment import ALIGNMENT_VERSION, replace_special_bpe
from .clip_readouts import dense_features, text_token_features
from .constants import CLIP_NAME, IMG_SEQ, SPECIAL_WORDS, TEXT_SEQ


class CrossModalConvNetwork(nn.Module):
    """Copied from baseline/src/util/models_and_dataloaders.py (dropout = 0 at eval)."""

    def __init__(self, img_seq, text_seq, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim // 32, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 32, hidden_dim // 32, 3, padding=1)
        self.fc1 = nn.Linear(hidden_dim // 32 * text_seq * img_seq, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (bi, bt, iseq, tseq)
        bi, bt = x.shape[:2]
        x = x.reshape(bi * bt, 1, x.shape[2], x.shape[3])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.reshape(bi * bt, -1)))
        return self.fc2(x).view(bi, bt, 1)


def tolerant_torch_load(path):
    """The released checkpoints pickle an old transformers CLIPTokenizer; stub the classes that no longer exist."""
    class _Dummy:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except Exception:
                return _Dummy

    m = types.ModuleType("tolerant_pickle"); m.Unpickler = _U; m.load = pickle.load
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=m)


@torch.no_grad()
def clip_patch_feats(clip, pv, readouts):
    """pv (n,3,224,224) -> {readout: (n,197,512) L2-normalised}. 'naive' follows the repo's exact code path."""
    out = {}
    if "naive" in readouts:
        vo = clip.vision_model(pixel_values=pv)
        img = clip.visual_projection(clip.vision_model.post_layernorm(vo.last_hidden_state))
        out["naive"] = img / img.norm(dim=-1, keepdim=True)
    rest = [r for r in readouts if r != "naive"]
    if rest:
        feats = dense_features(clip, pv, 224)[0]
        for r in rest:
            out[r] = feats[r]
    return out


class DCSMScorer:
    """A loaded scoring network + its Functional-Row embeddings, applied to crops of one query at a time."""

    def __init__(self, ckpt_path, clip, tok, img_proc, device):
        st = tolerant_torch_load(ckpt_path)
        self.net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ).to(device).eval()
        self.net.load_state_dict(st["model_state_dict"])
        self.emb = st["external_logic_token_embeddings"].to(device)
        self.words = st.get("learned_lookup") or SPECIAL_WORDS
        self.readout = st.get("readout", "naive")
        self.slicing = st.get("special_slicing", "eval_shift")
        if self.slicing not in (ALIGNMENT_VERSION, "replace", "eval_shift"):
            raise ValueError(f"Unknown checkpoint alignment: {self.slicing}")
        if self.slicing != ALIGNMENT_VERSION:
            warnings.warn("Legacy checkpoint uses unaligned word-index slicing; retained for historical reproduction. "
                          "Corrected BPE alignment requires retraining.", RuntimeWarning)
        self.clip, self.tok, self.img_proc, self.device = clip, tok, img_proc, device

    @torch.no_grad()
    def score_feats(self, img, txt, tt, text):
        """img (n,197,512) patch feats for self.readout; txt (1,30,512); tt tokenizer output -> (n,) tensor."""
        n = img.shape[0]
        cos = torch.einsum("bie,te->bit", img, txt[0])  # (n,197,30)
        if self.slicing == ALIGNMENT_VERSION:
            cos = replace_special_bpe(cos.unsqueeze(1), [text], self.emb, self.tok, self.words, tt["input_ids"])
        else:
            sent = text.split(" ")
            for w in self.words:  # same order / same slicing as the repo (train.py 'replace', eval_simple.py 'eval_shift')
                if w in sent:
                    loc = sent.index(w)
                    mid = self.emb(torch.tensor([self.words.index(w)], device=self.device)).unsqueeze(2).repeat(n, 1, 1)
                    tail = cos[:, :, loc + 1:] if self.slicing == "replace" else cos[:, :, loc:-1]
                    cos = torch.cat([cos[:, :, :loc], mid, tail], dim=-1)
            cos = cos.unsqueeze(1)
        cos = (cos - cos.mean()) / cos.std()  # repo's standardize() over the whole batch tensor
        return self.net(cos).squeeze(-1).squeeze(-1).float()

    @torch.no_grad()
    def score(self, crops, text):
        """crops: list of PIL -> (len(crops),) numpy scores for one text (batch = the candidates of one query)."""
        pv = self.img_proc(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        img = clip_patch_feats(self.clip, pv, [self.readout])[self.readout]
        tt, txt = text_token_features(self.clip, self.tok, text, self.device, TEXT_SEQ)
        return self.score_feats(img, txt, tt, text).cpu().numpy()


# ----------------------------------------------------------------------------- training-side helpers
def init_special_embedding(device):
    emb = nn.Embedding(len(SPECIAL_WORDS), IMG_SEQ).to(device)
    with torch.no_grad():
        emb.weight = nn.Parameter((F.normalize(emb.weight, p=2, dim=1) + 1) / 3)
    return emb


def initialize_shared(readouts, device):
    """Identical initial values with independent trainable storage per readout."""
    base_net = CrossModalConvNetwork(IMG_SEQ, TEXT_SEQ).to(device)
    base_emb = init_special_embedding(device)
    return ({r: copy.deepcopy(base_net) for r in readouts},
            {r: copy.deepcopy(base_emb) for r in readouts})


def state_fingerprint(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=4)
def alignment_tokenizer(name=CLIP_NAME):
    return CLIPTokenizer.from_pretrained(name)


def insert_special(cos, texts, emb, device=None, tokenizer=None, input_ids=None, clip_name=CLIP_NAME):
    """BPE-aligned Functional-Row replacement; use the tokenizer / input IDs from the CLIP forward when available."""
    if tokenizer is None:
        tokenizer = alignment_tokenizer(clip_name)
    return replace_special_bpe(cos, texts, emb, tokenizer, SPECIAL_WORDS, input_ids)
