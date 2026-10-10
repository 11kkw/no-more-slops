# Citation support audit

Task: DCSM-SHORT-20261010-001. BibTeX keys are used for this academic Markdown/LaTeX pipeline; numeric IDs from the earlier ledger are not the PDF bibliography numbering. `plainnat` sorts references alphabetically. The original ledger and bibliography are preserved.

## Original-source locations actually inspected

| Key | Supported attribution in short paper | Primary-source locus | Verification boundary |
| --- | --- | --- | --- |
| radford2021clip | Natural-language supervision and image/text transfer | PMLR primary record, Abstract / supplied BibTeX abstract | Primary publisher record read in this task; no new claim about performance taken from CLIP |
| zhou2021maskclip | Dense prediction with pretrained CLIP | arXiv PDF 2112.01071, p.1 Abstract and §1 | PDF text retrieved after abs-page extraction omitted abstract; not a full reproduction |
| lan2024clearclip | Removal of residual/FFN and self-self attention for segmentation | arXiv PDF 2407.12442, p.1 Abstract, §1 | PDF text inspected; our QQ readout remains explicitly “style”, not full paper reproduction |
| kang2025dcsm | Token–patch matrix, learned CNN; surgery not tested jointly | arXiv HTML 2503.08723v2, §5.1, paragraphs “Instead of deriving”, “Model Pipeline”, “Training”; edited-feature discussion | §5.1 original read directly; ideal geometry claims are not elevated to universal claims here |
| minderer2022owlvit | Detection fine-tuning of image/text models | Primary arXiv record 2205.06230, Abstract | Abstract directly supports narrow attribution; metadata inherited from primary-record audit |
| minderer2023owlv2 | Self-training with pseudo-box annotations | arXiv PDF 2306.09683v3, p.1 Abstract, §1, p.2 pipeline | Firecrawl PDF retrieval returned 429; urllib fetched actual PDF and PyMuPDF extracted it (`sources/minderer2023owlv2.pdf`, `.txt`) |
| lin2014coco | Annotated object instances in scenes | Primary arXiv record 1405.0312, Abstract | Local val2017 counts/80-category protocol come from evaluation artifacts, not old COCO abstract counts |
| bianchi2023fgovd | Prior fine-grained detector evaluation | Primary arXiv record 2311.17518, Abstract | Dynamic-vocabulary fine-grained evaluation checked; novelty over broader literature remains unverified |

The sources establish these attributions, not acceptance or novelty. First-submission years and author/title metadata reuse `../evidence/literature/records.json` and `../references.bib`. Eight cited keys are checked against `.bib` and compiled `.bbl` by `validate.py`.

## Local source excerpts and provenance

`../evidence/citations.json` attaches a verified quote to each of the eight primary records. Current original PDF/HTML texts are archived under `sources/` where available. Table values are this project's measurements, not attributed to these publications. Consult `../claim_evidence.md` C01–C18 and the short-paper review for measurement/interpretation/unverified distinctions.

## What was not done

No exhaustive novelty search, citation graph review, independent human review, or new model experiment. Metadata consistency and successful compilation do not prove the scientific claim by themselves. Claim-support locations above were inspected separately.
