"""Rebuild this document in a new directory; no experiment execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, out = Path(__file__).resolve().parent, args.out.resolve()
    if out.exists(): parser.error('Choose a new output directory')
    tools = [os.environ.get('PDFLATEX', 'pdflatex'), os.environ.get('BIBTEX', 'bibtex')]
    if not all(shutil.which(tool) for tool in tools): parser.error('Install pdfLaTeX and BibTeX, or set PDFLATEX/BIBTEX')
    out.mkdir(parents=True)
    inputs = {}
    for name in ['shortpaper.tex', 'references.bib', 'tables/all_subsets.tex', 'figures/spatial_gain.pdf']:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        inputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    commands = [[tools[0], '-interaction=nonstopmode', '-halt-on-error', 'shortpaper.tex'],
                [tools[1], 'shortpaper'],
                [tools[0], '-interaction=nonstopmode', '-halt-on-error', 'shortpaper.tex'],
                [tools[0], '-interaction=nonstopmode', '-halt-on-error', 'shortpaper.tex']]
    records = []
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=120)
        (out / f'build-{i}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        records.append({'command': command, 'exit_code': result.returncode})
        (out / 'document_build.json').write_text(json.dumps({'scope': 'Document compilation only', 'input_hashes': inputs, 'commands': records}, indent=2) + '\n')
        result.check_returncode()
    print(json.dumps({'pdf': str(out / 'shortpaper.pdf'), 'commands': len(records), 'scope': 'Document compilation only'}))


if __name__ == '__main__':
    main()
