---
name: experiment-to-paper
description: Draft papers from completed results with checked evidence.
license: MIT
metadata:
  version: 0.1.0
  hermes:
    tags: [Research, Paper Writing, Evidence, Results]
    related_skills: [research-paper-writing, grounded-citations]
---

# Experiment to Paper

## When to Use

Use when a research repository already contains measurements and the user wants a manuscript or a revision grounded in them. This skill supplies the bridge from result artifacts to claims, and a bounded internal review. It complements the installed writing and citation skills; it does not start new experiments.

## Establish the writing inputs

Read the user's current request, repository instructions, result metadata, and evaluation code. Record the repository revision and input files. An artifact snapshot is evidence of saved results, not proof that inference was reproduced or that the current revision generated them.

When the project is `week1_clip_dcsm_surgery`, read [references/dcsm.md](references/dcsm.md) and use its export command. For another project, create an equivalent evidence index from that project's actual schema; do not run the DCSM adapter against unrelated results.

Use current results to update earlier literature-only notes. Preserve their original scope; new measurements do not automatically resolve their novelty objections.

## Build the evidence bridge

Prepare:

- `evidence/measurements.json`: measured values, units, denominators and source locations.
- `evidence/manifest.json`: input file hashes, available execution metadata and snapshot revision.
- `tables/`: tables generated from raw summaries or item predictions, preserving missing values.
- `experiment_log.md`: question, setup, result, files, scope and unresolved checks for each experiment.
- `claim_evidence.md`: claim ID, proposed sentence, measurement or citation, comparator conditions, limitations and status (`observed`, `interpretation`, `unverified`).

For comparisons, check backbone, training, proposal source and selection, input/crop, resolution, scoring, calibration, matching and metric. Shared source boxes alone do not establish identical evaluated proposals. Do not call an AP gap pure classification quality when selection or score rules also change.

Distinguish no new training for this evaluation, a frozen encoder, an already trained scorer, and a trained detector. Check any claimed train/evaluation separation against the original training data when available.

Record uncertainty that is actually measured. Deterministic inference still has uncertainty about the evaluated sample and benchmark labels. Multiple queries from one image are dependent: an optional bootstrap should resample images as clusters, use paired method predictions, and retain the metric's strata. Never invent confidence intervals or interpret a nonsignificant test as equivalence.

## Verify literature and draft

Use the installed `research-paper-writing` Phase 5 drafting reference for narrative and section construction. Read it through that skill's `skill_view`, rather than resolving its reference relative to this skill. Use `grounded-citations` for literature evidence, with a **project-specific ledger** at `paper/evidence/citations.json`; do not reset a shared profile ledger.

In `grounded-citations`, `--ledger` goes before the subcommand. For example, using the installed script's actual path:

```text
python3 sources.py --ledger /absolute/project/paper/evidence/citations.json add SOURCE_URL --title TITLE
python3 sources.py --ledger /absolute/project/paper/evidence/citations.json verify manuscript.md --evidence
```

This verifies the supported numbered-citation format and evidence attachments, not scientific support or BibTeX metadata. Check the original paper and section/table for each substantive attribution. Fetch bibliographic metadata from an authoritative record. Mark unresolved references explicitly, and use the appropriate verifier for LaTeX/Pandoc if those formats are chosen.

Write a complete draft with bounded claims even when venue or novelty remains undecided. A venue-neutral Markdown draft is suitable until the user chooses a venue. Report unverified novelty separately from the observed result. Follow the repository's acknowledgement requirements.

Include `manuscript.md`, `summary.ko.md` when Korean discussion is requested, the evidence bridge, verified references and `review_notes.md`. Format choices and language follow the user; these names are defaults. Figures must come from measured data or identified qualitative inputs, with units, denominators, scope and uncertainty labels.

## Review and revise

Read [references/internal-review.md](references/internal-review.md). Review the draft against original artifacts and code, then revise the supported claims and wording. Distinguish internal self-review from independent peer review and an editorial decision. Use delegation only when already authorized; a sequential review is the default.

Fix resolvable transcription, citation and protocol issues. Record missing data, new experiments and unresolved novelty as follow-up work while completing the supported draft. Stop revising when identified fixes are addressed; do not create an unbounded scoring or agent loop.

## Scope and completion

Follow current user authorization. A local draft does not authorize a GPU run, paid provider setup, bot-to-bot assignment, publishing, submission, or unrelated commits. Existing explicit authorization for those actions continues to apply.

Before handoff, compare tables to their sources, verify claims and cited evidence, and report remaining gaps. Record actual compilation if PDF output was requested; an uncompiled source is not a verified PDF. Report installation/discovery, artifact checks, model-generated writing, and Discord execution as separate evidence layers.

Pipeline provenance and deliberately excluded components are in [references/pipeline-sources.md](references/pipeline-sources.md).


## Mandatory manuscript review gate

For this writing-only project, existing evidence is frozen: do not run new experiments. Every completed shortpaper draft must run the `paper-review-local` checks, including ChkTeX, before handoff. Missing or failed linter execution blocks a completed-review claim. The separate `paper-reviewer` profile automatically checks stable shortpaper changes. Read `paper/shortpaper/reviews/gate.json` and its matching `editorial_review.json`/`feedback.ko.md`; the input digest must match the current draft. If the review is pending, failed or has required findings, state that explicitly and do not label the manuscript submission-ready. Revise only within user-authorized writing scope; at most two revision checks per task. Do not invoke experiment or server bots to clear a writing gate. Similarity candidates require source/citation context; never invent a global plagiarism percentage.
