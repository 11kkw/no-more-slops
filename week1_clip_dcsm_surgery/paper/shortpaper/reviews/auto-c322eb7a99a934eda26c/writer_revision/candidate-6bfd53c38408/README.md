# DCSM-SHORT-20261010-001 — local short paper

## Deliverables

- `shortpaper.md`: English manuscript with keyed citations and reference records.
- `shortpaper.tex`, `references.bib`: venue-neutral two-column 10pt LaTeX source and bibliography.
- `shortpaper.pdf`: actual compiled PDF; **body pp.1–4, references p.5**.
- `format_review.ko.md`: original draft versus compiled short-paper format assessment.
- `discord_review.ko.md`: sequential Hermes internal review and recorded corrections; not independent peer review.
- `build_validation.json`: executed checks, actual page counts, versions, hashes, limits.
- `citation_audit.md`: original-source support locations and retrieval limits.
- `figures/`, `figure_data.json`: measured spatial data plot and exact figure inputs.
- `renders/page-1.png` through `page-5.png`, `pdf_text.txt`: actual PDF render/text evidence.
- `input_snapshot.json`: hashes of existing project files before short-paper writes (excludes pycache and new shortpaper folder).
- `evidence_export/`: fresh artifact-only recomputation; original evidence untouched.

## Build and validate

Run from this folder:

```bash
uv run --no-project --python 3.10 --with pypandoc_binary --with matplotlib==3.10.7 --with pymupdf python build.py
uv run --no-project --python 3.10 --with pypandoc_binary --with matplotlib==3.10.7 --with pymupdf python validate.py
```

`build.py` converts Markdown with Pandoc, makes a vector figure from saved bootstrap data, and executes `/home/happy5700x/.cache/paper-tools/tectonic-0.17.0/tectonic --keep-logs --keep-intermediates shortpaper.tex`. Markdown is the editable content source; regenerate LaTeX after editing. No model inference or training occurs. Rebuilding validation resets visual-review status to pending; inspect regenerated renders before marking a new version visually checked.

## Supporting material (preserved local files)

Paths relative to this folder:

- `../manuscript.md` — original full working manuscript, all candidate scorers and appendix.
- `../evidence/measurements.json`, `../evidence/cluster_bootstrap.json`, `../evidence/manifest.json`.
- `../claim_evidence.md`, `../experiment_log.md`, `../review_notes.md`.
- `../../experiment/results/full_val2017/summary.json` — detection.
- `../../experiment/results/probe_500/localization_probe.json` — pointing probe.
- `../../experiment/results/comp_full/per_item.json` and `benchmark/comp_queries.json` — candidate predictions and queries.
- `../../experiment/scripts/` — evaluation implementation, not executed for inference here.

## Latest author revision

The latest review was read and patched under the user request to resume review and apply corrections. See `reviews/author-revision-20261010-01/author_response.ko.md` for all 13 finding dispositions, `manuscript.patch` for edits, and `revision_validation.json` for executed results. The prior short-paper sources and PDF are preserved as `before-*` in that directory.

The revised PDF remains body pp.1–4 and references p.5. New actual renders were visually inspected. Numeric/layout validation passed 16/17 groups; the historical preservation test flags `plan/03_paper_pipeline.md`, which this revision did not modify. Do not interpret the retained `needs_revision` status as a numeric or PDF failure, or overwrite that historical snapshot. ChkTeX was rerun with 34 diagnostics, with triage in the response record; not warning-free.

DCSM adapter provenance: `../../experiment/scripts/compositional_eval.py:81–109,176`; local default checkpoint filenames are `data/dcsm_ckpt/dcsm_obja_ckpt.pt` and `data/dcsm_ckpt/dcsm_coco_ckpt.pt`. These filenames are not verified release identities or checkpoint hashes. Figure generation is in `build.py:17–35`, linked to the saved bootstrap path in `figure_data.json`.

A minimal local TeX archive is `reviews/author-revision-20261010-01/source-bundle.tar.gz` (TeX, bib, bbl, used PDF figure only). Its clean-directory Tectonic compilation succeeded and produced equal page text. Compile from the archive root with `tectonic --keep-logs --keep-intermediates shortpaper.tex`. It is not an arXiv-side compile and has not been uploaded. Current official arXiv guidance was inspected; author/license/category decisions and package-version compatibility remain unresolved.

## Scope and submission readiness

Request processed once as a bounded local writing/review job. No new experiments, delegation, server work, bot mentions, commits/pushes, external publication or submission. The evidence snapshot commit is not proof of the original execution revision. Authors, affiliations, venue template, novelty audit, human label verification, checkpoint overlap and independent evaluation remain unresolved. This PDF is not an official venue-compliant or anonymous submission: it intentionally retains the requested acknowledgement.
