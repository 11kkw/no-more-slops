# Paper draft package

Start with [manuscript.md](manuscript.md) or the [Korean explanation](summary.ko.md). Current draft (11 October 2026): *Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP*, a method paper built on the retraining (E6) and referring-detection (E7) experiments; terminology follows `../experiment/overview.md` §0. The 10 October analysis draft survives as `shortpaper/` (marked superseded) and in `tables/`, `figures/`, `evidence/`, which still describe that earlier draft's artifacts. `scripts/refresh_manuscript_tables.py` and `scripts/verify_draft.py` were written for the earlier draft's generated-table markers and have not been adapted to the new manuscript.

## Files

| File / directory | Purpose |
| --- | --- |
| `manuscript.md` | English paper draft with abstract through conclusion, appendices and sources |
| `summary.ko.md` | Korean interpretation and claim boundaries |
| `claim_evidence.md` | Claim-to-result/source mapping |
| `experiment_log.md` | Setup, outputs, findings and evidence limits for each experiment |
| `review_notes.md` | Sequential internal review, revisions and unresolved checks |
| `references.bib` | Publisher/arXiv metadata; arXiv entries use first-submission years |
| `tables/` | Checked saved-result tables and new spatial bootstrap table |
| `figures/` | Detection and spatial plots as PNG, SVG and PDF |
| `evidence/manifest.json` | Hashes of original result/code inputs; snapshot revision, not execution revision |
| `evidence/measurements.json` | Recomputed original aggregates; uncertainty field describes original artifacts |
| `evidence/cluster_bootstrap.json` | New uncertainty analysis of existing candidate predictions |
| `evidence/citations.json` | Project-specific numeric source ledger and short evidence excerpts |
| `evidence/literature/records.json` | Retrieved primary bibliographic metadata, timestamps and fetched page hashes |
| `evidence/draft_validation.json` | Actual completed checks of this draft |

## Refresh saved-result analysis

From the repository root:

```bash
python3 skills/research/experiment-to-paper/scripts/export_dcsm_evidence.py \
  --repo "$PWD" --out "$PWD/week1_clip_dcsm_surgery/paper" --refresh
uv run --no-project --python 3.10 --with numpy==2.2.6 --with matplotlib==3.10.7 \
  python week1_clip_dcsm_surgery/paper/scripts/analyze_saved_results.py
python3 week1_clip_dcsm_surgery/paper/scripts/refresh_manuscript_tables.py
python3 week1_clip_dcsm_surgery/paper/scripts/verify_draft.py
```

The exporter only refreshes its generated aggregate files. The manuscript-table refresh replaces marked Appendix A blocks and renders source links from the ledger; prose and main-table revisions remain authored and are checked by the verifier. The bootstrap uses a small isolated CPU environment via `uv`; it does not install or run the original GPU inference project.

To refresh primary records, use `scripts/prepare_sources.py`. Its default citation CLI path refers to the installed paper-profile `grounded-citations` skill on this host; adjust `CLI` if moving machines. Verify citation formatting and attached evidence with:

```bash
python3 /home/happy5700x/.hermes/profiles/paper/skills/research/grounded-citations/scripts/sources.py \
  --ledger "$PWD/week1_clip_dcsm_surgery/paper/evidence/citations.json" \
  render --replace-in "$PWD/week1_clip_dcsm_surgery/paper/manuscript.md"
python3 week1_clip_dcsm_surgery/paper/scripts/refresh_manuscript_tables.py
python3 /home/happy5700x/.hermes/profiles/paper/skills/research/grounded-citations/scripts/sources.py \
  --ledger "$PWD/week1_clip_dcsm_surgery/paper/evidence/citations.json" \
  verify "$PWD/week1_clip_dcsm_surgery/paper/manuscript.md" --evidence
```

Citation verification checks the ledger and literal excerpts. It does not mechanically establish that every scientific interpretation follows from its source. The manuscript and internal review supply those boundaries.

## Scope

Original inference was not rerun for this draft. The new work consists of code/protocol inspection, arithmetic checks, a CPU image-cluster bootstrap, plots and manuscript writing. Discord dispatch, training, publication and submission were not part of this drafting pass. Original experiment files and prior planning notes are retained.

Read the manuscript's protocol notes when consulting `../experiment/README.md`: evaluated detection images are 4,952; native OWLv2 and P2 differ in proposal selection/scoring; candidate selection supplies ground-truth candidates whereas referring detection does not; negation/affirm items are not matched controls. The scope paragraph above describes the 10 October drafting pass; the 11 October revision added new GPU experiments (see `experiment_log.md` E6–E7).

This work was created as part of the No-More-Slops group activities at Pseudo Lab.
