"""Build the authorized short paper from Markdown and saved evidence only.
Run: uv run --no-project --python 3.10 --with pypandoc_binary --with matplotlib==3.10.7 --with pymupdf python build.py
No model inference; only manuscript conversion, plotting and compilation.
"""
from pathlib import Path
import hashlib, json, re, subprocess, sys
import pypandoc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
TASK = 'DCSM-SHORT-20261010-001'
TECTONIC = '/home/happy5700x/.cache/paper-tools/tectonic-0.17.0/tectonic'

def main():
    bootstrap = json.loads((PAPER/'evidence/cluster_bootstrap.json').read_text())
    spatial = bootstrap['subsets']['spatial']['balanced']
    keys = ['dcsm_obja_ctx', 'dcsm_coco_ctx', 'readout/naive/pooled', 'readout/clearclip/pooled', 'clip_ctx', 'owlv2']
    labels = ['DCSM ObjA', 'DCSM COCO', 'Ordinary map', 'ClearCLIP-style', 'CLIP context', 'OWLv2 matched']
    vals = [100*spatial[k]['estimate'] for k in keys]
    lo = [vals[i]-100*spatial[k]['ci95'][0] for i,k in enumerate(keys)]
    hi = [100*spatial[k]['ci95'][1]-vals[i] for i,k in enumerate(keys)]
    fig, ax = plt.subplots(figsize=(3.45,2.35))
    ax.errorbar(vals, range(len(keys)), xerr=[lo,hi], fmt='o', color='black', capsize=2.5, markersize=4)
    ax.axvline(100*spatial['chance']['estimate'], linestyle='--', color='#666666', linewidth=1)
    ax.set_yticks(range(len(keys)), labels, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(30,67); ax.set_xlabel('Balanced spatial accuracy (%)', fontsize=9)
    ax.tick_params(axis='x',labelsize=9); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.5)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/spatial_balanced.pdf')
    fig.savefig(ROOT/'figures/spatial_balanced.png', dpi=220)
    plt.close(fig)
    (ROOT/'figure_data.json').write_text(json.dumps({'source':'../evidence/cluster_bootstrap.json#/subsets/spatial/balanced', 'keys':keys,'estimates_percent':vals,'lower_error_percent':lo,'upper_error_percent':hi,'chance_percent':100*spatial['chance']['estimate']},indent=2)+'\n')
    (ROOT/'references.bib').write_bytes((PAPER/'references.bib').read_bytes())
    md = (ROOT/'shortpaper.md').read_text()
    title = md.splitlines()[0][2:]
    md = re.sub(r'^# .*?\n\n\*.*?\*\n\n', '', md, count=1, flags=re.S)
    md = md.split('## References')[0]
    # Replace the two Markdown tables by unsqueezed 10pt booktabs floats.
    table_n = 0
    def table(match):
        nonlocal table_n
        table_n += 1
        caption, block = match.group(1), match.group(2)
        rows = [[c.strip().replace('↑', '$\\uparrow$').replace('%', '\\%') for c in line.strip().strip('|').split('|')] for line in block.strip().splitlines()]
        rows = [r for r in rows if not all(re.fullmatch(r'[- :]+',c) for c in r)]
        if table_n == 1:
            rows[0] = ['Readout', 'Pointing', 'P1 AP', 'P2 AP']
        else:
            rows[0] = ['Scorer', 'Accuracy', '95\\% interval']
            rows[1][0]='Ordinary map'; rows[2][0]='MaskCLIP-style'; rows[3][0]='ClearCLIP-style'
            rows[4][0]='CLIP context'; rows[5][0]='DCSM ObjA'; rows[6][0]='DCSM COCO'; rows[7][0]='OWLv2 matched'
        cap = pypandoc.convert_text(caption,'latex',format='markdown',extra_args=['--natbib']).strip()
        tex = '\\begin{table}[tb]\n\\centering\n\\caption{'+cap+'}\n\\label{tab:'+str(table_n)+'}\n\\begin{tabular}{@{}l'+'r'*(len(rows[0])-1)+'@{}}\n\\toprule\n'
        for i,r in enumerate(rows):
            tex += ' & '.join(r)+' \\\\\n'
            if i == 0: tex += '\\midrule\n'
        tex += '\\bottomrule\n\\end{tabular}\n\\end{table}'
        return '\n```{=latex}\n'+tex+'\n```\n'
    md = re.sub(r'\*\*Table \d\.\*\* (.*?)\n\n(\|[^\n]+\n(?:\|[^\n]+\n)+)',table,md,flags=re.S)
    # Embed a vector plot in a single-column float.
    pattern = r'!\[.*?\]\(figures/spatial_balanced.pdf\)\n\n\*\*Figure 1\.\*\* ([^\n]+)'
    cap_match = re.search(pattern,md)
    cap=pypandoc.convert_text(cap_match[1],'latex',format='markdown').strip()
    md=re.sub(pattern,lambda m:'\n```{=latex}\n\\begin{figure}[tb]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/spatial_balanced.pdf}\n\\caption{'+cap+'}\n\\label{fig:spatial}\n\\end{figure}\n```\n',md)
    abstract_match = re.match(r'## Abstract\n\n(.*?)\n\n## 1\.', md, flags=re.S)
    assert abstract_match is not None
    abstract = pypandoc.convert_text(abstract_match.group(1), 'latex', format='markdown+tex_math_dollars').strip()
    md = md[md.index('## 1. Introduction and Related Work'):]
    md = md.replace('## 1. Introduction and Related Work', '## Introduction and Related Work', 1)
    md=re.sub(r'^(##|###) \d+\.(?:\d+\.)? ',r'\1 ',md,flags=re.M)
    body=pypandoc.convert_text(md, 'latex', format='markdown+raw_tex+tex_math_dollars', extra_args=['--natbib','--top-level-division=section','--shift-heading-level-by=-1'])
    body = re.sub(r'\\texttt\{((?:openai|google)/[^}]+)\}', lambda m: '\\path{'+m.group(1)+'}', body)
    body = '\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n\n'+body
    body = body.replace('\\section{Discussion and Limitations}', '\\balance\n\\section{Discussion and Limitations}')
    preamble=r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[margin=0.75in,columnsep=0.25in]{geometry}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype,booktabs,graphicx,amsmath,balance}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{1em}
\title{TITLE}
\author{Authors and affiliations to be determined}
\date{Venue-neutral working short paper}
\begin{document}
\raggedbottom
\maketitle
'''.replace('TITLE',title)
    ending=r'''
\label{main-text-end}
\clearpage
\nobalance
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
'''
    (ROOT/'shortpaper.tex').write_text(preamble+body+ending)
    result=subprocess.run([TECTONIC,'--keep-logs','--keep-intermediates','shortpaper.tex'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'compile_stdout.txt').write_text(result.stdout+result.stderr)
    warnings = [line for line in result.stderr.splitlines() if 'Overfull' in line or 'Underfull' in line or 'error:' in line]
    print('\n'.join(warnings))
    print('Compilation exit:',result.returncode, 'Pandoc:',pypandoc.get_pandoc_version())
    if result.returncode: sys.exit(result.returncode)
    import fitz
    pdf=fitz.open(ROOT/'shortpaper.pdf')
    print('ACTUAL PAGES',len(pdf))
    for n,p in enumerate(pdf):
        text=p.get_text()
        print('PAGE',n+1,'chars',len(text),'first',text[:140].replace('\n',' / '),'last',text[-160:].replace('\n',' / '))
    (ROOT/'pdf_text.txt').write_text('\n\n'.join('=== PAGE '+str(n+1)+' ===\n'+p.get_text() for n,p in enumerate(pdf)))
    (ROOT/'renders').mkdir(exist_ok=True)
    for n,p in enumerate(pdf):
        p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(ROOT/f'renders/page-{n+1}.png')
if __name__=='__main__': main()
