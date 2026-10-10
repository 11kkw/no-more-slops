# DCSM artifact adapter

The project input is `week1_clip_dcsm_surgery` inside the supplied repository. Read its `plan/02_discord_draft_handoff.md` before drafting. That file contains the task-specific measured findings and interpretation corrections.

Run the adapter using its actual installed skill path:

```bash
python3 /home/happy5700x/.hermes/profiles/paper/skills/research/experiment-to-paper/scripts/export_dcsm_evidence.py \
  --repo /home/happy5700x/work/paper-lab/repos/no-more-slops \
  --out /home/happy5700x/work/paper-lab/repos/no-more-slops/week1_clip_dcsm_surgery/paper
```

The command reads saved JSON, compares item-level aggregates and balanced tables, hashes result/code inputs, and generates `evidence/` and `tables/`. It uses Python's standard library; no GPU, model, network or raw image dataset is needed. Existing output files are preserved unless `--refresh` is explicitly passed for the generated files. It never writes the manuscript or authored claim/experiment logs.

## Source semantics

- `full_val2017/summary.json`: actual image count and detection settings, AP fractions and detection counts. Present AP on the 0–100 scale and identify AP separately from accuracy.
- `comp_full/summary.json`: raw top-1 fractions, counts, chance and inference arguments.
- `comp_full/per_item.json`: one target and one prediction per scorer for each evaluated query.
- `comp_full/benchmark/comp_queries.json`: candidate boxes, target and benchmark generation arguments. This tracked copy avoids dependence on ignored `data/` files.
- `comp_full/summary_balanced.md`: compare recomputed balanced values against this recorded table. Balance big/small for size; target-largest/target-not-largest for the other subsets.
- `probe_500/localization_probe.json`: location probe fractions. It is a separate subset from the detection run.

## Required interpretation checks

1. Use the metadata's actual image count (the current snapshot is 4,952), not the nominal dataset size 5,000.
2. P2 readout variants share objectness-selected proposals. OWLv2 native selects predictions differently and uses different scores. Its AP gap is not a clean classification-only ablation.
3. Token-max versus EOS, surgery versus trained scorer, and candidate selection versus full detection are separate comparisons. Avoid presenting them as a single improved model.
4. Ground-truth candidate boxes and reference-object context crops make the compositional task conditional candidate selection. They do not prove end-to-end referring detection.
5. OWL candidate scores use IoU matching and a fallback penalty; report this potential localization contribution to accuracy differences.
6. Balance controls a specific coarse size correlation. It does not remove all size, location, class or contextual shortcuts. The largest/smallest heuristic values are measured, not universally 50%.
7. Positive and negative queries have different items; their difference suggests a failure hypothesis, not a causal demonstration that the word `without` is ignored.
8. Color spot-checks are small-sample inspections, not an established label-quality ceiling.
9. The export confirms arithmetic and input consistency. It does not validate ground truth, implement significance tests, establish novelty, or reproduce the original inference.

Start `experiment_log.md` with three experiments: localization probe, category detection (P1/P2), and compositional candidate selection. In each, link the generated measurement block and identify the exact protocol and limitations. Then draft claims with their scope rather than promoting all observed differences to general model capabilities.
