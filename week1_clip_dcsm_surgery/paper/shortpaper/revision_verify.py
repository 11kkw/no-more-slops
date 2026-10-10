"""Bounded post-review checks: local sources, clean source bundle, no inference."""
from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, tarfile, difflib
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
TOOLS=Path('/home/happy5700x/work/paper-lab/tools/chktex-local/extracted')
SCRIPTS=Path('/home/happy5700x/.hermes/profiles/paper/skills/research/paper-review-local/scripts')
OUT=ROOT/'reviews/author-revision-20261010-01'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
# Actual clean-room build from only the archive members.
files=['shortpaper.tex','references.bib','shortpaper.bbl','figures/spatial_balanced.pdf']
archive=OUT/'source-bundle.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for name in files: t.add(ROOT/name,arcname=name)
clean=OUT/'clean-build'
clean.mkdir(exist_ok=True)
for name in files:
    dest=clean/name; dest.parent.mkdir(exist_ok=True,parents=True); shutil.copy2(ROOT/name,dest)
cmd=['/home/happy5700x/.cache/paper-tools/tectonic-0.17.0/tectonic','--keep-logs','--keep-intermediates','shortpaper.tex']
p=subprocess.run(cmd,cwd=clean,capture_output=True,text=True)
(OUT/'clean-build.log.txt').write_text(p.stdout+p.stderr)
import pymupdf
pdf=pymupdf.open(clean/'shortpaper.pdf') if p.returncode==0 else None
original=pymupdf.open(ROOT/'shortpaper.pdf')
bundle={'archive':str(archive.relative_to(ROOT)),'archive_sha256':sha(archive),'members':files,'compilation_exit':p.returncode,'pages':len(pdf) if pdf else None,'pdf_text_equal':bool(pdf and [x.get_text() for x in pdf]==[x.get_text() for x in original]),'arxiv_side_compilation':'not_run','engine':'Tectonic 0.17.0, not arXiv TeX Live','package_dependencies':'Standard TeX packages resolved by local Tectonic; no custom style files; arXiv package-version compatibility not verified','absolute_input_paths':False}
(OUT/'source_bundle_validation.json').write_text(json.dumps(bundle,indent=2)+'\n')
# Refresh only the local task-state hash; do not retain old visual claims.
s=json.loads((ROOT/'task_state.json').read_text());s.update(status='author_revision_completed_with_open_review_items',pdf_sha256=sha(ROOT/'shortpaper.pdf'),revision='author-revision-20261010-01')
(ROOT/'task_state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
os.environ['LD_LIBRARY_PATH']=str(TOOLS/'usr/lib/x86_64-linux-gnu')
os.environ['PAPER_REVIEW_CHKTEX']=str(TOOLS/'usr/bin/chktex')
os.environ['PAPER_REVIEW_CHKTEX_RC']=str(TOOLS/'etc/chktexrc')
sys.path.insert(0,str(SCRIPTS))
from review_mcp import run_review
r=run_review('week1_clip_dcsm_surgery/paper/shortpaper','author-recheck-20261010-01',REPO)
(OUT/'manuscript.patch').write_text(''.join(difflib.unified_diff((OUT/'before-shortpaper.md').read_text().splitlines(True),(ROOT/'shortpaper.md').read_text().splitlines(True),fromfile='before-shortpaper.md',tofile='shortpaper.md')))
print(json.dumps({'bundle':bundle,'local_checks':r['checks'],'lint':r['latex_lint'],'exact_similarity':r['local_similarity']['status'],'near_similarity':r['near_similarity']['status']},ensure_ascii=False,indent=2))
