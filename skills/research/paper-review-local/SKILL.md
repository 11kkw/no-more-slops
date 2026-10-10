---
name: paper-review-local
description: Review a completed manuscript for academic style, source overlap and arXiv preparation using local files. Use for writing review; does not run experiments or submit papers.
metadata:
  version: 0.3.1
  hermes:
    tags: [Paper, Review, Integrity]
---

# Local manuscript review

Review the frozen manuscript, its cited sources, figure provenance and the requested format in a fresh reviewer session. Do not begin by reading the author's self-review or drafting conversation. Later revision reviews may read prior findings and the author's response.

## Local tool

Use `run_local_review` with the repository-relative package path. It records input hashes and checks sections, bibliography keys, local figure references, PDF text/page count, unresolved author placeholders, ChkTeX warnings, exact phrases and RapidFuzz sentence string similarity against the explicitly listed local source texts. `read_local_review` retrieves the report. Tool names may have the host's MCP prefix. It never contacts a provider or returns a global plagiarism percentage. These deterministic checks require no LLM inference.

Use `read_review_input` to read manuscript/source text in bounded pages and `read_rendered_page` to inspect existing PNG page renderings. These tools cannot edit the input manuscript. Read enough context around matching phrases before assessing attribution.

The review launcher attaches all existing PDF page images natively to the first model input. A MEDIA/path string returned by Hermes' MCP bridge is not itself visual evidence. Inspect the attached images, and report visual inspection as unverified if actual pixels are unavailable. Do not spend the review trying to read PNGs with the text-input tool. That tool accepts only .md, .txt, .tex, .bib and .json, with a maximum 30000-character page. Read original manuscript/source context before any writer self-review. Do not change or manage skills during review.

The tool's phrase matches are candidates for inspection, not plagiarism findings. No match does not establish originality. Examine matching source passages and citation/quotation context; short technical expressions and the author's own earlier drafts require different treatment. Record source coverage and every unavailable check. AI-generated-text scores are not plagiarism evidence.

## Reviewer output

Use `save_editorial_review` to save `editorial_review.json` under the returned review directory with the matching input digest. Keep originals unchanged. Findings need an ID, manuscript section/page or quote, evidence location, priority (`required`, `recommended`, `minor`), explanation, confidence and suggested correction. Classify each integrity finding as `attribution_issue`, `quotation_issue`, `candidate_overlap`, `acceptable_overlap` or `unverified`; do not declare misconduct from a score.

Use literal `high`, `medium` or `low` codes for confidence; put any qualification in explanation/evidence. All finding values must be strings, including evidence and location. Write the explanations and summary in Korean. Do not translate schema codes.

Inspect actual rendered PDF pages for table/figure spacing, clipping, units, caption length, typography and whitespace. Read the prose for narrative, claims, terminology, redundant bold emphasis and internal process notes that belong in review records. A successful compile is not a completed visual check.

For arXiv preparation, distinguish academic structure from upload readiness. Check author/affiliation placeholders, title/abstract metadata, source and figure completeness, bibliography and a reproducible source bundle. Refer to current official arXiv instructions; no universal arXiv two-column or four-page requirement is implied. Licenses, subject category, author decisions, moderation and arXiv-side compilation remain separate. Never report acceptance as verified.

The writing pipeline is: draft -> automatic local checks -> separate editorial/integrity review -> author revision -> bounded recheck -> PDF review -> author decision. Default to at most two revision rounds. Log unresolved issues instead of triggering experiments, other bots or indefinite revisions.

## External comparison

Read [provider-options.md](references/provider-options.md) only when connecting an external provider. Current deployment is local-only, with no external account or report. Do not install guessed integrations, scrape a report viewer, or send unpublished manuscripts to an external service. A provider-backed report requires a connected account and authorization for that upload; record provider, input hash, repository/exclusion settings, scan time and source matches. The user's prior authorization remains applicable.
