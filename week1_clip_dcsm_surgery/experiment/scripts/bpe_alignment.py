"""BPE-aligned logic-vector replacement shared by training and evaluation.

Each complete standalone lookup occurrence replaces ALL of its BPE columns.
BOS/EOS/padding are never touched; partially or fully truncated occurrences
are skipped as a unit. Input IDs must match the tokenizer's actual CLIP input.
Legacy checkpoints must retain their historical slicing, not be relabelled.
"""
import re
from functools import lru_cache
import torch

ALIGNMENT_VERSION = 'bpe_replace_v1'

@lru_cache(maxsize=8192)
def _spans(text, words, tokenizer):
    # Prefix tokenization uses the SAME tokenizer as the CLIP forward. A
    # standalone word boundary prevents matching 'not' inside 'nothing'.
    full = tokenizer(text, add_special_tokens=True)['input_ids']
    spans = []
    occupied = set()
    for wi, word in enumerate(words):
        if not word:
            raise ValueError('Empty special-word lookup')
        pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            prefix = tokenizer(text[:match.start()], add_special_tokens=False)['input_ids']
            piece = tokenizer(match.group(), add_special_tokens=False)['input_ids']
            start = 1 + len(prefix)  # CLIP BOS occupies column zero.
            end = start + len(piece)
            if full[0] != tokenizer.bos_token_id or full[1:start] != prefix or full[start:end] != piece:
                raise ValueError(f'Cannot reliably align lookup {word!r} in {text!r}')
            columns = set(range(start, end))
            if columns & occupied:
                raise ValueError('Overlapping special-word lookups are ambiguous')
            occupied.update(columns)
            spans.append((wi, start, end))
    return tuple(spans)


def replace_special_bpe(cos, texts, emb, tokenizer, words, input_ids=None):
    """Replace columns of (Bi,Bt,image_seq,text_seq), retaining autograd.

    A learned image_seq-vector is repeated over all BPE columns of a lookup.
    Optional input_ids are checked against the exact padded/truncated encoding
    so accidentally supplying a different tokenizer or sequence fails loudly.
    """
    if cos.ndim != 4 or cos.shape[1] != len(texts):
        raise ValueError('Expected (Bi,Bt,image_seq,text_seq) matching texts')
    length = cos.shape[-1]
    if length < 2:
        raise ValueError('CLIP input needs at least BOS and EOS')
    encoded = tokenizer(list(texts), padding='max_length', max_length=length,
                        truncation=True, return_tensors='pt')['input_ids']
    if input_ids is not None and not torch.equal(encoded, input_ids.detach().cpu()):
        raise ValueError('Supplied input_ids do not match the alignment tokenizer')
    if emb.weight.shape != (len(words), cos.shape[2]):
        raise ValueError('Logic embedding dimensions do not match lookup/image sequence')
    out = cos.clone()
    for k, text in enumerate(texts):
        ids = encoded[k].tolist()
        eos = ids.index(tokenizer.eos_token_id)
        for wi, start, end in _spans(text, tuple(words), tokenizer):
            if start >= eos or end > eos:
                continue  # Entire occurrence must survive truncation.
            value = emb(torch.tensor(wi, device=cos.device)).to(dtype=cos.dtype)
            out[:, k, :, start:end] = value.view(1, -1, 1)
    return out
