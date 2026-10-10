"""Refresh only the generated complete-result tables in Appendix A."""
from pathlib import Path
import json
import re

paper = Path(__file__).resolve().parents[1]
path = paper / 'manuscript.md'
text = path.read_text()
for marker, filename in [('RAW', 'compositional_raw.md'),
                         ('BALANCED', 'compositional_balanced.md'),
                         ('HEURISTIC', 'heuristics.md')]:
    start = f'<!-- GENERATED_{marker}_TABLE -->'
    end = f'<!-- END_GENERATED_{marker}_TABLE -->'
    generated = start + '\n\n' + (paper / 'tables' / filename).read_text().strip() + '\n\n' + end
    if end in text:
        text, count = re.subn(re.escape(start) + r'.*?' + re.escape(end), lambda _: generated, text, flags=re.S)
    else:
        count = text.count(start)
        text = text.replace(start, generated)
    if count != 1:
        raise ValueError(f'{marker}: expected exactly one marked block, got {count}')
ledger = json.loads((paper / 'evidence' / 'citations.json').read_text())
body = text.split('## Sources', 1)[0].rstrip()
cited = set(map(int, re.findall(r'\[(\d+)\]', body)))
sources = {s['id']: s for s in ledger['sources']}
if not cited <= sources.keys():
    raise ValueError('Unknown citation ID in manuscript')
links = [f'[{sid}] [{sources[sid]["title"]}]({sources[sid]["url"]})' for sid in sorted(cited)]
text = body + '\n\n## Sources\n\n' + '\n\n'.join(links) + '\n'
path.write_text(text)
print('Refreshed 3 Appendix A tables and ledger-based source links; prose preserved.')
