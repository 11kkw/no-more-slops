# Pipeline sources and integration decisions

Reviewed on 2026-10-10. The added instructions and exporter are local original work; no external execution engine or review prompt is vendored.

## Reused installed skills

The paper profile already contains `research-paper-writing`, `grounded-citations`, `arxiv` and `research-question-audit`. Its writing skill credits Orchestra Research and supplies drafting, citation guidance and templates. Reuse it instead of installing a second overlapping writing skill.

Verified upstream reference:
- [Orchestra Research ML paper writing](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/ml-paper-writing/SKILL.md), commit `773a52944ba4747a18bd4ae9ade53fff041adcbc`; inspected file blob `2090941c1d1e6b619e5ab32147741546c8dbdb09`.

Adopt its repository-first writing, coherent contribution framing and original-source citation checks. In this workflow, absent novelty evidence is marked unresolved while a bounded draft proceeds.

## Review pipeline reference

- [SakanaAI AI Scientist review implementation](https://github.com/SakanaAI/AI-Scientist/blob/1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb/ai_scientist/perform_review.py).

Use structured review categories and revision as design references. Do not import its executable code, model clients, automatic experiments, score aggregation or accept/reject decision. The original code has its own source-code license; this integration does not redistribute it. Separate internal critique from independent peer review and actual venue decisions.

## Official quality reference

- [NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist), retrieved 2026-10-10.

Use claim scope, limitations, reproducibility, experimental settings and uncertainty as quality checks. Recheck venue instructions at submission time. A missing statistic is reported as missing rather than manufactured.

## Reviewed but not installed

- [K-Dense peer-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/peer-review/SKILL.md), inspected blob `d4f3c3e3b5c19fec2518d569ded7e37ac96a964b`.

This is a formal review workflow with venue-policy/confidentiality intake and multiple review tools. The current task is author-side drafting from the user's own repository, so an internal review reference supplies the needed checks without adding that separate workflow.
