"""Stage narrowly scoped prose revisions; models cannot edit originals or run commands."""
from collections import Counter
import json
import re
from pathlib import Path
from review_mcp import ROOT, safe, sha

PACKAGE=ROOT/'week1_clip_dcsm_surgery/paper/shortpaper'


def context(review_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',review_id):raise ValueError('Invalid review id')
    gate=json.loads((PACKAGE/'reviews/gate.json').read_text())
    if gate.get('review_id')!=review_id or gate.get('status') not in ['author_revision_required','author_decision_pending']:
        raise ValueError('Review is not the current completed review')
    directory=safe(ROOT,gate['review_directory'])
    checks=json.loads((directory/'local_checks.json').read_text())
    editorial=json.loads((directory/'editorial_review.json').read_text())
    if checks['input_digest']!=gate['input_digest'] or editorial['input_digest']!=gate['input_digest']:raise ValueError('Review digest mismatch')
    for entry in checks['input_files']:
        if sha(safe(ROOT,entry['path']))!=entry['sha256']:raise ValueError('Manuscript changed after review')
    return directory,checks,editorial


def numeric_tokens(text):
    return re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',text)


def table_values(text):
    tables=re.findall(r'(?:^\|[^\n]+\n)+',text,re.M)
    return [[line.strip().strip('|').split('|')[1:] for line in table.splitlines()[2:]] for table in tables]


def validate_patches(original,patches,known_findings,bib):
    candidate=original
    if not isinstance(patches,list) or len(patches)>30:raise ValueError('At most 30 exact prose patches')
    for patch in patches:
        if set(patch)!={'finding_id','old','new','reason'} or not all(isinstance(v,str) for v in patch.values()):raise ValueError('Invalid patch fields')
        old,new=patch['old'],patch['new']
        if patch['finding_id'] not in known_findings or not old or len(old)>16000 or len(new)>20000 or candidate.count(old)!=1:
            raise ValueError('Patch must address a reviewed finding and match exactly once')
        if numeric_tokens(old)!=numeric_tokens(new):raise ValueError('Scientific numbers and numbered structure must remain unchanged')
        if re.search(r'\\(?:input|include|write|openin|openout|read|immediate|special|directlua|usepackage|documentclass|newcommand|def|catcode)\b',new):
            raise ValueError('Executable or external TeX input is not allowed')
        candidate=candidate.replace(old,new,1)
    if Counter(numeric_tokens(candidate))!=Counter(numeric_tokens(original)) or table_values(candidate)!=table_values(original):
        raise ValueError('Numerical evidence or table values changed')
    images=lambda text:re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    if images(candidate)!=images(original):raise ValueError('Figure paths must remain unchanged')
    cited=lambda text:set(k.strip() for group in re.findall(r'\\cite(?:p|t)?\{([^}]+)\}',text) for k in group.split(','))
    keys=set(re.findall(r'@\w+\s*\{([^,]+),',bib))
    if not cited(original).issubset(cited(candidate)) or not cited(candidate).issubset(keys):raise ValueError('Citations may only use verified bibliography entries; preserve original citations')
    if len(candidate)>500000:raise ValueError('Manuscript too large')
    return candidate


def submit(review_id,patches,responses,summary):
    directory,checks,editorial=context(review_id)
    known={f['id'] for f in editorial['findings']}
    if not isinstance(summary,str) or not summary or not isinstance(responses,list):raise ValueError('Writer response required')
    if {r.get('finding_id') for r in responses}!=known or len(responses)!=len(known):raise ValueError('Respond to each finding once')
    for response in responses:
        if set(response)!={'finding_id','status','explanation','evidence'} or not all(isinstance(v,str) for v in response.values()):raise ValueError('Invalid response schema')
        if response['status'] not in ['addressed','partial','unresolved','not_applicable']:raise ValueError('Invalid finding response status')
    original=(PACKAGE/'shortpaper.md').read_text()
    candidate=validate_patches(original,patches,known,(PACKAGE/'references.bib').read_text())
    job=directory/'writer_revision';job.mkdir(exist_ok=True)
    result={'review_id':review_id,'input_digest':checks['input_digest'],'status':'staged' if candidate!=original else 'no_safe_changes',
            'patch_count':len(patches),'patches':patches,'responses':responses,'summary':summary,'original_markdown_sha256':sha(PACKAGE/'shortpaper.md')}
    if (job/'revision_proposal.json').exists():raise ValueError('A revision proposal has already been submitted')
    (job/'proposed_shortpaper.md').write_text(candidate)
    (job/'revision_proposal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return {'status':result['status'],'patch_count':len(patches),'proposal_directory':str(job),'originals_modified':False}


def serve():
    from mcp.server.fastmcp import FastMCP
    server=FastMCP('paper_revision_local')

    @server.tool()
    def read_revision_context(review_id:str)->dict:
        """Read current review and original Markdown. No external access or edits."""
        directory,checks,editorial=context(review_id)
        return {'review_id':review_id,'input_digest':checks['input_digest'],'review':editorial,'manuscript':(PACKAGE/'shortpaper.md').read_text(),
                'bibliography':(PACKAGE/'references.bib').read_text(),'available_input_files':[str(safe(ROOT,e['path']).relative_to(PACKAGE)) for e in checks['input_files']],
                'mandatory_constraints':'Writing only. Preserve all numerical evidence, citations, figure paths and provenance. Do not invent author metadata, checkpoints, experiments, results, plagiarism rates or arXiv approval. Stage exact text patches and a response to every finding. Human decisions stay unresolved.'}

    @server.tool()
    def read_revision_input(review_id:str,relative_file:str,offset:int=0,limit:int=20000)->dict:
        """Read supporting text inside this paper package. Credentials and executables unavailable."""
        context(review_id)
        path=safe(PACKAGE,relative_file)
        if path.suffix not in ['.md','.txt','.json','.bib','.tex','.py'] or 'reviews' in path.relative_to(PACKAGE).parts:
            raise ValueError('Only original paper/supporting text is allowed')
        if offset<0 or not 1<=limit<=30000 or path.stat().st_size>10000000:raise ValueError('Invalid read bounds')
        text=path.read_text(errors='replace')
        return {'path':relative_file,'sha256':sha(path),'text':text[offset:offset+limit],'total_chars':len(text)}

    @server.tool()
    def submit_prose_revision(review_id:str,patches:list[dict],responses:list[dict],summary:str)->dict:
        """Stage one proposal without editing originals. patches: finding_id,old,new,reason (strings). Preserve the ordered numeric tokens in each patch exactly; table numeric cells unchanged. responses: finding_id,status,addressed/partial/unresolved/not_applicable,explanation,evidence (status is one string field). Include one response per finding. Safe CPU build/validation happens separately."""
        return submit(review_id,patches,responses,summary)

    server.run(transport='stdio')


if __name__=='__main__':serve()
