"""Fetch primary bibliographic records and register short citation evidence.

Uses the installed grounded-citations CLI; stores metadata and short excerpts,
not copies of full papers. Run from any directory with Python 3 (stdlib only).
"""
import concurrent.futures
import datetime
import hashlib
import html.parser
import json
from pathlib import Path
import re
import subprocess
import urllib.request

PAPER = Path(__file__).resolve().parents[1]
CLI = Path('/home/happy5700x/.hermes/profiles/paper/skills/research/grounded-citations/scripts/sources.py')
SOURCES = [
    ('radford2021clip', 'https://proceedings.mlr.press/v139/radford21a.html', 'After pre-training, natural language'),
    ('kang2025dcsm', 'https://arxiv.org/abs/2503.08723', 'Dense Cosine Similarity Maps'),
    ('zhou2021maskclip', 'https://arxiv.org/abs/2112.01071', 'with minimal modification'),
    ('lan2024clearclip', 'https://arxiv.org/abs/2407.12442', 'removing the residual connection'),
    ('minderer2022owlvit', 'https://arxiv.org/abs/2205.06230', 'end-to-end detection fine-tuning'),
    ('minderer2023owlv2', 'https://arxiv.org/abs/2306.09683', 'self-training'),
    ('lin2014coco', 'https://arxiv.org/abs/1405.0312', 'object recognition'),
    ('bianchi2023fgovd', 'https://arxiv.org/abs/2311.17518', 'fine-grained'),
]


class Page(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta, self.text = {}, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta' and 'name' in a and 'content' in a:
            self.meta.setdefault(a['name'], []).append(a['content'])

    def handle_data(self, data):
        self.text.append(data)


def fetch(spec):
    key, url, needle = spec
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'paper-draft-metadata/1.0'}), timeout=40) as response:
        raw = response.read()
    page = Page()
    page.feed(raw.decode('utf-8'))
    text = re.sub(r'\s+', ' ', ' '.join(page.text)).strip()
    abstract_start = text.find('Abstract:')
    pos = text.lower().find(needle.lower(), max(abstract_start, 0))
    if pos < 0:
        raise ValueError(f'{key}: evidence phrase not found: {needle}')
    # At most 15 words of verbatim supporting text per source.
    evidence = ' '.join(text[pos:].split()[:15])
    title = page.meta['citation_title'][0]
    authors = page.meta['citation_author']
    date = page.meta.get('citation_date', page.meta.get('citation_publication_date', ['']))[0]
    if not re.match(r'\d{4}', date):
        raise ValueError(f'{key}: missing publication date')
    return dict(key=key, url=url, title=title, authors=authors, date=date,
                evidence=evidence, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                fetched_sha256=hashlib.sha256(raw).hexdigest(), record_type='primary bibliographic page')


def main():
    target = PAPER / 'evidence' / 'literature'
    target.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES))
    ledger = PAPER / 'evidence' / 'citations.json'
    bibliography = []
    for record in records:
        args = ['python3', str(CLI), '--ledger', str(ledger)]
        registered = subprocess.check_output(args + ['add', record['url'], '--title', record['title']], text=True)
        number = int(re.search(r'\[(\d+)\]', registered).group(1))
        record['citation_id'] = number
        evidence_file = target / (record['key'] + '.txt')
        evidence_file.write_text(record['evidence'] + '\n')
        subprocess.run(args + ['quote', str(number), '--text', record['evidence'], '--from', str(evidence_file)], check=True, capture_output=True)
        author = ' and '.join(record['authors'])
        if 'arxiv.org/abs/' in record['url']:
            eid = record['url'].rsplit('/', 1)[-1]
            bibliography.append('@misc{' + record['key'] + ',\n' +
                f'  title = {{{{{record["title"]}}}}},\n  author = {{{author}}},\n' +
                f'  year = {{{record["date"][:4]}}},\n  eprint = {{{eid}}},\n' +
                '  archivePrefix = {arXiv},\n  primaryClass = {cs.CV},\n' +
                f'  url = {{{record["url"]}}}\n' + '}')
        else:
            bibliography.append('@inproceedings{' + record['key'] + ',\n' +
                f'  title = {{{{{record["title"]}}}}},\n  author = {{{author}}},\n' +
                f'  year = {{{record["date"][:4]}}},\n' +
                '  booktitle = {Proceedings of the 38th International Conference on Machine Learning},\n' +
                '  volume = {139},\n  pages = {8748--8763},\n  publisher = {PMLR},\n' +
                f'  url = {{{record["url"]}}}\n' + '}')
    (PAPER / 'references.bib').write_text('% arXiv entries use first-submission years; venue metadata is not inferred.\n\n' + '\n\n'.join(bibliography) + '\n')
    (target / 'records.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps([dict(id=r['citation_id'], key=r['key'], title=r['title'], authors=r['authors'], date=r['date']) for r in records], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
