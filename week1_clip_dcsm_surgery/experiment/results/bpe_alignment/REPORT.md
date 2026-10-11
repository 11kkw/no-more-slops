# BPE alignment correction verification

Worktree: `/home/jovyan/no-more-slops-bpe-alignment`
Branch: `fix/bpe-alignment`
Original tree: `/home/jovyan/no-more-slops` (unchanged production code and existing checkpoints)

## Replacement contract
- Training and corrected evaluation share `scripts/bpe_alignment.py::replace_special_bpe`.
- Positions are derived with the actual CLIP tokenizer, including BOS and preceding words' BPE expansion.
- Every standalone occurrence is replaced, including repeated words, uppercase and adjacent punctuation.
- If a lookup spans several BPE columns, repeat its learned image-position vector across every column.
- Only complete retained occurrences are replaced. Fully/partially truncated occurrences are skipped as a whole. BOS/EOS/padding remain unchanged.
- The input matrix is not mutated and embedding gradients survive replacement.
- Supplied input IDs are validated against the padded/truncated encoding; ambiguous overlaps or inconsistent alignment fail loudly.

## Checkpoint compatibility
New training checkpoints use `special_slicing: bpe_replace_v1`. Old `replace` / `eval_shift` checkpoints retain historical behavior with an explicit warning. Changing old checkpoint metadata does NOT make it a correctly trained checkpoint. Unknown slicing versions are rejected.

## Executed verification
1. Baseline actual training-function regression: 5 failing subcases, exit 1 (`red.log`).
2. Corrected unit/integration suite: 11 tests passed, exit 0 (`green_full.log`). This includes actual `DCSMScorer.score()` on real CLIP and an independent training-path tensor comparison, plus backward-gradient checks.
3. Entire current evaluation query corpus: 1,833 queries, 1,497 retained functional-token occurrences, 0 mismatches (`validation.json`).
4. Actual training smoke: all three readouts, batch size 2, one optimizer step, exit 0 (`smoke_train.log`). Corrected checkpoint metadata verified on readback.
5. Actual evaluation CLI on the new smoke checkpoints: one item from each of five subsets, exit 0 (`smoke_eval.log`).
6. `git diff --check`: exit 0.

These smoke checkpoints are **not** full retraining, multi-seed evidence, or performance claims. No existing result was overwritten, no merge/commit was performed, and full research training has not been started.

## Repeat tests
From this worktree's `week1_clip_dcsm_surgery/experiment`:

```sh
HF_HOME=/home/jovyan/no-more-slops/week1_clip_dcsm_surgery/experiment/hf_cache \
HF_HUB_OFFLINE=1 PYTHONPATH=scripts \
/home/jovyan/no-more-slops/week1_clip_dcsm_surgery/experiment/.venv/bin/python \
-m unittest discover -s tests -v
```

The existing environment/data/cache are reused read-only where applicable; credentials and provider configuration were not copied.
