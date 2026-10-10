"""Check manuscript tables against source artifacts and record actual checks.

This verifies transcription, file links, snapshot integrity and citation-ledger
consistency. It cannot validate task labels, originality or scientific causality.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
RESULTS = PAPER.parent / 'experiment' / 'results'


def read(path):
    return json.loads(path.read_text())


def rows(text, heading, end):
    block = text.split(heading, 1)[1].split(end, 1)[0]
    return [[c.strip() for c in line.strip().strip('|').split('|')]
            for line in block.splitlines() if line.startswith('|') and '---' not in line][1:]


def pct(value):
    return f'{100 * value:.1f}'


def main():
    doc = (PAPER / 'manuscript.md').read_text()
    measured = read(PAPER / 'evidence' / 'measurements.json')
    original = read(RESULTS / 'comp_full' / 'summary.json')
    queries = read(RESULTS / 'comp_full' / 'benchmark' / 'comp_queries.json')['items']
    det = read(RESULTS / 'full_val2017' / 'summary.json')['results']
    probe = read(RESULTS / 'probe_500' / 'localization_probe.json')['results']
    boot = read(PAPER / 'evidence' / 'cluster_bootstrap.json')
    manifest = read(PAPER / 'evidence' / 'manifest.json')
    checks = []
    for path, expected in manifest['inputs'].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected['sha256'], path
    checks.append(f'{len(manifest["inputs"])} original input hashes unchanged')

    subsets = ['size', 'color', 'spatial', 'negation', 'affirm']
    for row, subset in zip(rows(doc, '**Table 1.**', '### 3.4.'), subsets):
        items = [q for q in queries if q['subset'] == subset]
        expected = [str(len(items)), str(len({q['image_id'] for q in items})),
                    str(len({q['cls_A'] for q in items})), pct(original['chance'][subset]),
                    ' / '.join(map(str, measured['strata_counts'][subset]))]
        assert row[1:] == expected, ('Table 1', row, expected)
    assert len({q['image_id'] for q in queries}) == boot['unique_images_overall'] == 1025
    checks.append('Table 1 counts, classes, chance, strata and overall distinct images')

    mode_names = {'Naive': 'naive', 'MaskCLIP-style': 'maskclip', 'ClearCLIP-style': 'clearclip'}
    for row in rows(doc, '**Table 2.**', '### 4.2.'):
        mode = mode_names[row[0]]
        expected = [pct(probe[f'size{size}'][f'{mode}/{text}']['pointing_acc'])
                    for size in (224, 448) for text in ('pooled', 'dcsm_tok')]
        assert row[1:] == expected, ('Table 2', row, expected)
    checks.append('Table 2 localization values against saved probe JSON')

    for row in rows(doc, '**Table 3.**', 'Native reference AP'):
        mode = mode_names[row[0]]
        text = 'pooled' if row[1] == 'Pooled' else 'dcsm_tok'
        expected = [pct(det[f'{protocol}/{mode}/{text}/{size}']['AP'])
                    for size in (224, 448) for protocol in ('P1_dense', 'P2_owlv2boxes')]
        assert row[2:] == expected, ('Table 3', row, expected)
    for name, key in [('OWL-ViT v1', 'owlvit_v1_native'), ('OWLv2', 'owlv2_native')]:
        assert pct(det[key]['AP']) in doc.split('Native reference AP', 1)[1].split('\n', 1)[0]
    checks.append('Table 3 AP values and native reference values against saved detection JSON')

    scorers = {'Chance': 'chance', 'OWLv2': 'owlv2', 'OWL-ViT v1': 'owlvit_v1',
               'CLIP crop': 'clip_crop', 'CLIP context crop': 'clip_ctx',
               'DCSM ObjA context': 'dcsm_obja_ctx', 'DCSM COCO context': 'dcsm_coco_ctx',
               'ClearCLIP-style pooled': 'readout/clearclip/pooled'}
    for row in rows(doc, '**Table 4.**', 'A model-free'):
        scorer = scorers[row[0]]
        raw = original['chance'] if scorer == 'chance' else original['top1'][scorer]
        balanced = measured['balanced_chance'] if scorer == 'chance' else measured['balanced_top1'][scorer]
        assert row[1:] == [pct(raw[s]) + ' / ' + pct(balanced[s]) for s in subsets], ('Table 4', row)
    checks.append('Table 4 raw/balanced pairs against original summaries and recomputed strata')

    spatial = boot['subsets']['spatial']
    table5 = rows(doc, '**Table 5.**', '![Balanced spatial')
    assert len(table5) == 6
    keys = [('balanced', 'dcsm_obja_ctx'), ('balanced', 'dcsm_coco_ctx'),
            ('paired_differences', 'dcsm_obja_ctx - chance'),
            ('paired_differences', 'dcsm_coco_ctx - chance'),
            ('paired_differences', 'dcsm_obja_ctx - clip_ctx'),
            ('paired_differences', 'dcsm_obja_ctx - readout/clearclip/pooled')]
    for row, (group, key) in zip(table5, keys):
        value = spatial[group][key]
        if group == 'balanced':
            expected = [pct(value['estimate']) + '%', f'[{pct(value["ci95"][0])}, {pct(value["ci95"][1])}]%']
        else:
            expected = ['+' + pct(value['estimate']) + ' pp',
                        f'[+{pct(value["ci95"][0])}, +{pct(value["ci95"][1])}] pp']
        assert row[1:] == expected, ('Table 5', row, expected)
    for subset, data in boot['subsets'].items():
        assert data['valid_replicates'] == boot['replicates'] == 10000
        for method, value in data['balanced'].items():
            expected = measured['balanced_chance'][subset] if method == 'chance' else measured['balanced_top1'][method][subset]
            assert abs(value['estimate'] - expected) < 1e-12
    checks.append('Table 5 intervals and all bootstrap point estimates; all resamples retained both strata')

    for marker, filename in [('RAW', 'compositional_raw.md'), ('BALANCED', 'compositional_balanced.md'), ('HEURISTIC', 'heuristics.md')]:
        block = doc.split(f'<!-- GENERATED_{marker}_TABLE -->', 1)[1].split(f'<!-- END_GENERATED_{marker}_TABLE -->', 1)[0]
        assert block.strip() == (PAPER / 'tables' / filename).read_text().strip()
    assert '2.6 to 27.3' in doc and '273/315' in doc and '44/194' in doc
    checks.append('Appendix A matches generated complete-result tables')

    link_count = 0
    for path in PAPER.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            assert resolved.exists(), (str(path), target)
            link_count += 1
    checks.append(f'{link_count} local document/figure links resolve')
    records = read(PAPER / 'evidence' / 'literature' / 'records.json')
    ledger = read(PAPER / 'evidence' / 'citations.json')
    bib = (PAPER / 'references.bib').read_text()
    assert len(records) == len(ledger['sources']) == 8
    for record, source in zip(records, ledger['sources']):
        assert record['citation_id'] == source['id'] and record['url'] == source['url']
        assert record['title'] in bib and all(author in bib for author in record['authors'])
        assert record['evidence'] in (PAPER / 'evidence' / 'literature' / (record['key'] + '.txt')).read_text()
        assert sum(len(q['text'].split()) for q in source['quotes']) <= 25
    checks.append('8 primary bibliographic records match ledger and BibTeX; short source excerpts')
    cli = Path('/home/happy5700x/.hermes/profiles/paper/skills/research/grounded-citations/scripts/sources.py')
    citation = subprocess.run(['python3', str(cli), '--ledger', str(PAPER / 'evidence' / 'citations.json'),
                               'verify', str(PAPER / 'manuscript.md'), '--evidence'], capture_output=True, text=True)
    assert citation.returncode == 0, citation.stdout + citation.stderr
    checks.append('grounded-citations verify --evidence passed')
    report = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='passed',
                  checks=checks, citation_verifier_output=citation.stdout.strip(),
                  scope='Saved-artifact transcription, local links, metadata and citation evidence formatting',
                  limitations=['No new model inference', 'No independent label validation', 'No novelty certification',
                               'No independent peer review', 'No original environment/checkpoint split audit',
                               'No manuscript PDF compilation or Discord dispatch'])
    (PAPER / 'evidence' / 'draft_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
