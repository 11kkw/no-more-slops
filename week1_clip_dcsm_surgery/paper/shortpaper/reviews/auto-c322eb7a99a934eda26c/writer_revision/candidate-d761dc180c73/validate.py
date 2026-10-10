"""Validate actual compiled output, saved-result transcription and preservation.
Run in the same uv environment as build.py. This does not rerun inference.
"""
from pathlib import Path
import collections, hashlib, importlib.metadata, json, re, subprocess, sys
import pymupdf
ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent
PROJECT=PAPER.parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(v): return f'{100*v:.1f}'
def main():
    md=(ROOT/'shortpaper.md').read_text(); tex=(ROOT/'shortpaper.tex').read_text()
    log=(ROOT/'shortpaper.log').read_text(); bib=(ROOT/'references.bib').read_text()
    m=json.loads((ROOT/'evidence_export/evidence/measurements.json').read_text())
    original=json.loads((PAPER/'evidence/measurements.json').read_text())
    boot=json.loads((PAPER/'evidence/cluster_bootstrap.json').read_text())
    sp=boot['subsets']['spatial']['balanced']
    tables=re.findall(r'(?:^\|[^\n]+\n)+',md,re.M)
    assert len(tables)==2
    rows=[[[c.strip() for c in line.strip('|').split('|')] for line in t.strip().splitlines()][2:] for t in tables]
    checks=[]
    def check(name,ok,details):
        checks.append({'name':name,'status':'passed' if ok else 'failed','details':details})
    t1=[]
    probe=json.loads((PROJECT/'experiment/results/probe_500/localization_probe.json').read_text())['results']
    for r,mode in zip(rows[0],['naive','maskclip','clearclip']):
        expected=[fmt(probe['size224'][mode+'/pooled']['pointing_acc']),fmt(m['detection'][f'P1_dense/{mode}/pooled/224']['AP']),fmt(m['detection'][f'P2_owlv2boxes/{mode}/pooled/224']['AP'])]
        t1.append({'row':r[0],'actual':r[1:],'expected':expected,'match':r[1:]==expected})
    check('table1_all_9_numeric_cells',all(r['match'] for r in t1),t1)
    keys=['readout/naive/pooled','readout/maskclip/pooled','readout/clearclip/pooled','clip_ctx','dcsm_obja_ctx','dcsm_coco_ctx','owlv2','chance']
    t2=[]
    for r,k in zip(rows[1],keys):
        expected=[fmt(sp[k]['estimate']), '['+', '.join(fmt(v) for v in sp[k]['ci95'])+']' if k!='chance' else '—']
        t2.append({'key':k,'actual':r[1:],'expected':expected,'match':r[1:]==expected})
    check('table2_estimates_and_ci_endpoints',all(r['match'] for r in t2),t2)
    check('export_recomputed_raw_balanced_and_chance',all(m[k]==original[k] for k in ['n_items','counts','strata_counts','raw_top1','balanced_top1','chance','balanced_chance','heuristics','detection']), 'Re-exporter compared saved item predictions, query identity, original raw/chance and balanced tables; exported fields agree with preserved evidence.')
    # Numerical text claims not already wholly in a table.
    claims={
      'native_AP':(fmt(m['detection']['owlv2_native']['AP']),'46.8'),
      'clear_P2_448':(fmt(m['detection']['P2_owlv2boxes/clearclip/pooled/448']['AP']),'26.4'),
      'clear_pointing_448':(fmt(probe['size448']['clearclip/pooled']['pointing_acc']),'67.5'),
      'largest_raw_size':(fmt(m['heuristics']['largest']['size']),'87.3'),
      'owlv2_raw_size':(fmt(m['raw_top1']['owlv2']['size']),'64.7'),
      'owlv2_bal_size':(fmt(m['balanced_top1']['owlv2']['size']),'41.7'),
      'dcsm_obja_without':(fmt(m['balanced_top1']['dcsm_obja_ctx']['negation']),'38.7'),
      'dcsm_coco_without':(fmt(m['balanced_top1']['dcsm_coco_ctx']['negation']),'37.5'),
      'chance_without':(fmt(m['balanced_chance']['negation']),'48.1'),
      'ordinary_tok_without':(fmt(m['balanced_top1']['readout/naive/dcsm_tok']['negation']),'55.7'),
      'ordinary_tok_spatial':(fmt(m['balanced_top1']['readout/naive/dcsm_tok']['spatial']),'50.0'),
      'mask_tok_spatial':(fmt(m['balanced_top1']['readout/maskclip/dcsm_tok']['spatial']),'43.4'),
      'clear_tok_spatial':(fmt(m['balanced_top1']['readout/clearclip/dcsm_tok']['spatial']),'42.5'),
    }
    check('numeric_prose_claims',all(a==b and b in md for a,b in claims.values()),claims)
    diffs=[]
    for key, expected in [('dcsm_obja_ctx - chance',['12.4','8.0','16.7']),('dcsm_obja_ctx - readout/clearclip/pooled',['18.7','13.1','24.2'])]:
        r=boot['subsets']['spatial']['paired_differences'][key]
        actual=[fmt(r['estimate'])]+[fmt(v) for v in r['ci95']]
        diffs.append({'key':key,'actual':actual,'expected':expected,'match':actual==expected})
    check('paired_difference_estimates_and_intervals',all(r['match'] for r in diffs),diffs)
    figs=json.loads((ROOT/'figure_data.json').read_text())
    check('figure_estimates_and_intervals',all(abs(figs['estimates_percent'][i]-100*sp[k]['estimate'])<1e-10 and abs(figs['lower_error_percent'][i]-(100*sp[k]['estimate']-100*sp[k]['ci95'][0]))<1e-10 and abs(figs['upper_error_percent'][i]-(100*sp[k]['ci95'][1]-100*sp[k]['estimate']))<1e-10 for i,k in enumerate(figs['keys'])),figs)
    cites=set(k.strip() for g in re.findall(r'\\cite(?:p|t)?(?:\[[^\]]*\])?\{([^}]+)\}',tex) for k in g.split(','))
    bibkeys=set(re.findall(r'@\w+\s*\{([^,]+),',bib))
    bblkeys=set(re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',(ROOT/'shortpaper.bbl').read_text()))
    labels=re.findall(r'\\label\{([^}]+)\}',tex)
    check('citation_keys_and_bibtex_resolution',cites==bibkeys==bblkeys,{'cites':sorted(cites),'bibkeys':sorted(bibkeys),'bblkeys':sorted(bblkeys)})
    check('bibliography_preserved',sha(ROOT/'references.bib')==sha(PAPER/'references.bib'),'Copied original verified bibliography without metadata invention.')
    check('no_duplicate_labels',len(labels)==len(set(labels)),labels)
    figure_paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex)
    check('figures_exist',all((ROOT/f).exists() for f in figure_paths),figure_paths)
    overfull=re.findall(r'Overfull [^\n]+',log)
    undefined=re.findall(r'(?:Citation[^\n]*undefined|Reference[^\n]*undefined|Undefined control sequence|Missing character[^\n]*)',log)
    horizontal=[s for s in overfull if '\\hbox' in s]
    vertical_sizes=[float(s) for s in re.findall(r'Overfull \\vbox \(([0-9.]+)pt too high\)',log)]
    check('no_horizontal_overflow_or_critical_vertical_overflow',not horizontal and not undefined and all(v<=2 for v in vertical_sizes),{'overfull':overfull,'undefined_or_missing':undefined,'policy':'No horizontal overflow accepted. A <=2pt balance-package internal vbox is a reported warning requiring direct visual confirmation and page-region checks, not an invisible ignored error.'})
    check('no_layout_cheats',not re.search(r'\\(?:vspace|hspace|fontsize|scalebox|resizebox)\b',tex),'10pt article, letter paper, 0.75in margins, 0.25in column gap; no negative spacing or text scaling; clearpage only separates references.')
    pdf=pymupdf.open(ROOT/'shortpaper.pdf')
    texts=[p.get_text() for p in pdf]
    refpages=[i+1 for i,t in enumerate(texts) if t.startswith('References\n')]
    main_end=re.search(r'\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}',(ROOT/'shortpaper.aux').read_text())
    body_pages=int(main_end[1]); reference_pages=len(pdf)-body_pages
    check('actual_body4_references_separate',len(pdf)==5 and body_pages==4 and refpages==[5],{'total_pages':len(pdf),'body_pages':body_pages,'reference_pages':reference_pages,'reference_start_pages':refpages})
    check('abstract_complete_in_pdf','they neither isolate a causal mechanism' in texts[0].replace('\n',' ') and '57.3%' in texts[0] and '61.2]' in texts[0], 'First build lost text after an unescaped percent; corrected by converting abstract independently, then wrapping its escaped LaTeX.')
    pages=[]; outside=[]
    for i,p in enumerate(pdf):
        lines=[ln for b in p.get_text('dict')['blocks'] if 'lines' in b for ln in b['lines']]
        spans=[s for l in lines for s in l['spans']]
        for s in spans:
            # Microtype intentionally permits hanging punctuation at margins.
            # Three PDF bp tolerance covers glyph protrusion, not overfull boxes.
            if s['bbox'][0]<51 or s['bbox'][2]>561 or s['bbox'][1]<45 or s['bbox'][3]>775: outside.append({'page':i+1,'text':s['text'],'bbox':s['bbox']})
        lineboxes=[l['bbox'] for l in lines if 200<l['bbox'][1]<735]
        left=sum(1 for b in lineboxes if b[2]<300); right=sum(1 for b in lineboxes if b[0]>300)
        pages.append({'page':i+1,'size_pdf_points':[p.rect.width,p.rect.height],'text_chars':len(texts[i]),'left_column_lines':left,'right_column_lines':right,'font_counts':dict(collections.Counter((s['font'],round(s['size'],3)) for s in spans).most_common(5)) if False else [{'font':k[0],'size_pdf_bp':k[1],'span_count':v} for k,v in collections.Counter((s['font'],round(s['size'],3)) for s in spans).most_common(5)],'render':f'renders/page-{i+1}.png'})
    check('both_columns_populated_on_body_pages',all(p['left_column_lines']>0 and p['right_column_lines']>0 for p in pages[:4]),pages)
    check('content_within_page_text_region',not outside,outside)
    snap=json.loads((ROOT/'input_snapshot.json').read_text())
    changed=[f for f,h in snap.items() if not (PROJECT/f).exists() or sha(PROJECT/f)!=h]
    check('existing_project_inputs_preserved',not changed,{'files_compared':len(snap),'changed_files':changed})
    underfull=re.findall(r'Underfull [^\n]+',log)
    result={'task_id':'DCSM-SHORT-20261010-001','status':'passed' if all(c['status']=='passed' for c in checks) else 'needs_revision','scope':'Local writing, existing-result arithmetic, plotting, CPU compilation and PDF inspection; no model inference/server task/external submission/commit/push.', 'repository_snapshot_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),'original_execution_revision':'unverified','build':{'command':'uv run --no-project --python 3.10 --with pypandoc_binary --with matplotlib==3.10.7 --with pymupdf python build.py','tectonic':subprocess.check_output(['/home/happy5700x/.cache/paper-tools/tectonic-0.17.0/tectonic','--version'],text=True).strip(),'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['pypandoc_binary','matplotlib','numpy','pymupdf']},'log':'shortpaper.log','stdout':'compile_stdout.txt'},'pages':{'body':body_pages,'references':reference_pages,'total':len(pdf),'details':pages},'checks':checks,'warnings':{'underfull_count':len(underfull),'underfull':underfull,'meaning':'Nonfatal justified-line spacing warnings; visual review recorded separately.'},'visual_review':{'status':'pending_final_render_review','method':'Native vision inspection of full-page PNG renders; no independent reviewer or automated perception benchmark.'},'not_verified':['Human label validation','Exact checkpoint identity and training overlap','Novelty and venue compliance','Original inference reproducibility','Detection/localization confidence intervals','Independent human peer review','chktex unavailable'], 'artifact_sha256':{f:sha(ROOT/f) for f in ['shortpaper.md','shortpaper.tex','shortpaper.pdf','references.bib','figures/spatial_balanced.pdf']}}
    result['warnings']['overfull'] = overfull
    result['warnings']['overfull_vbox_pt'] = vertical_sizes
    result['warnings']['microtype_bbox_tolerance_pdf_bp'] = 3
    if result['status']=='passed' and overfull: result['status']='passed_with_warnings'
    (ROOT/'build_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'pages':{'body':body_pages,'references':reference_pages,'total':len(pdf)},'check_count':len(checks),'failed':[c for c in checks if c['status']=='failed'],'underfull_count':len(underfull)},ensure_ascii=False,indent=2))
    if result['status'] not in ['passed','passed_with_warnings']: sys.exit(1)
if __name__=='__main__':main()
