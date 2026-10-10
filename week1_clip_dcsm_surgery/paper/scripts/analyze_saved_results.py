"""CPU-only, image-cluster bootstrap and figures from existing predictions.

Run: uv run --no-project --python 3.10 --with numpy==2.2.6 --with matplotlib==3.10.7 \
    python week1_clip_dcsm_surgery/paper/scripts/analyze_saved_results.py
No model inference or original-result writes. Percentile intervals are
exploratory, conditional on the observed query construction and checkpoints.
"""
import collections
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / 'experiment' / 'results'
SEED, REPLICATES = 20261010, 10000


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def main():
    pred_path = RESULTS / 'comp_full' / 'per_item.json'
    query_path = RESULTS / 'comp_full' / 'benchmark' / 'comp_queries.json'
    predictions = json.loads(pred_path.read_text())
    queries = json.loads(query_path.read_text())['items']
    index = {(q['image_id'], q['query']): q for q in queries}
    assert len(index) == len(predictions) == len(queries)
    measured = json.loads((PAPER / 'evidence' / 'measurements.json').read_text())
    methods = list(predictions[0]['pred']) + ['chance']
    rng = np.random.default_rng(SEED)
    report = dict(seed=SEED, replicates=REPLICATES, unit='image cluster within each subset',
                  interval='paired percentile 95%; exploratory, no multiplicity correction',
                  metric='mean of two stratum accuracies, recomputed per replicate',
                  python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                  unique_images_overall=len({q['image_id'] for q in queries}),
                  inputs={str(p.relative_to(PAPER.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (pred_path, query_path)}, subsets={})
    for subset in ('size', 'color', 'spatial', 'negation', 'affirm'):
        rows = [r for r in predictions if r['subset'] == subset]
        images = sorted({r['image_id'] for r in rows})
        image_index = {img: i for i, img in enumerate(images)}
        counts = np.zeros((len(images), 2))
        sums = np.zeros((len(images), 2, len(methods)))
        for row in rows:
            query = index[(row['image_id'], row['query'])]
            assert row['target'] == query['target_idx'] and row['n'] == len(query['candidates'])
            largest = int(np.argmax([area(b) for b in query['candidates']]))
            stratum = (0 if row['rel'] == 'big' else 1) if subset == 'size' else int(row['target'] != largest)
            img = image_index[row['image_id']]
            counts[img, stratum] += 1
            sums[img, stratum] += [float(row['pred'][m] == row['target']) for m in methods[:-1]] + [1 / row['n']]
        point = (sums.sum(axis=0) / counts.sum(axis=0)[:, None]).mean(axis=0)
        # Shared image weights preserve within-image dependence and pairing;
        # stratum sizes vary with the sampled images, but remain equally weighted.
        boot_weights = rng.multinomial(len(images), np.full(len(images), 1 / len(images)), size=REPLICATES)
        denominators = boot_weights @ counts
        valid = np.all(denominators > 0, axis=1)
        boot = (np.einsum('bg,gsm->bsm', boot_weights[valid], sums, optimize=True) /
                denominators[valid, :, None]).mean(axis=1)
        estimates = {}
        for j, method in enumerate(methods):
            expected = measured['balanced_chance'][subset] if method == 'chance' else measured['balanced_top1'][method][subset]
            assert np.isclose(point[j], expected, atol=1e-12)
            estimates[method] = dict(estimate=float(point[j]), ci95=np.quantile(boot[:, j], [.025, .975]).tolist())
        pairs = [('dcsm_obja_ctx', 'chance'), ('dcsm_coco_ctx', 'chance'),
                 ('dcsm_obja_ctx', 'clip_ctx'), ('dcsm_coco_ctx', 'clip_ctx'),
                 ('dcsm_obja_ctx', 'readout/clearclip/pooled'),
                 ('dcsm_obja_ctx', 'owlv2')]
        differences = {}
        for left, right in pairs:
            a, b = methods.index(left), methods.index(right)
            differences[left + ' - ' + right] = dict(estimate=float(point[a] - point[b]),
                ci95=np.quantile(boot[:, a] - boot[:, b], [.025, .975]).tolist())
        report['subsets'][subset] = dict(items=len(rows), images=len(images),
            strata_counts=counts.sum(axis=0).astype(int).tolist(), valid_replicates=int(valid.sum()),
            candidates=dict(collections.Counter(str(r['n']) for r in rows)),
            relations=dict(collections.Counter(str(r.get('rel', '')) for r in rows)),
            n_two_candidates=sum(r['n'] == 2 for r in rows),
            balanced=estimates, paired_differences=differences)
    (PAPER / 'evidence' / 'cluster_bootstrap.json').write_text(json.dumps(report, indent=2) + '\n')
    spatial = report['subsets']['spatial']
    lines = ['Exploratory image-cluster bootstrap; spatial balanced accuracy, 527 items / 375 images.',
             '10,000 replicates; seed 20261010. Intervals concern this constructed sample.', '',
             '| quantity | estimate (%) or difference (pp) | 95% interval |', '| --- | --- | --- |']
    for key, value in list(spatial['balanced'].items()) + list(spatial['paired_differences'].items()):
        lo, hi = value['ci95']
        lines.append(f'| {key} | {100 * value["estimate"]:.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] |')
    (PAPER / 'tables' / 'spatial_uncertainty.md').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    out = PAPER / 'figures'
    out.mkdir(exist_ok=True)
    det = measured['detection']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    labels = ['Naive', 'MaskCLIP-style', 'ClearCLIP-style']
    for ax, res in zip(axes, (224, 448)):
        x = np.arange(3)
        for delta, protocol, color in [(-.18, 'P1_dense', '#8da0cb'), (.18, 'P2_owlv2boxes', '#66c2a5')]:
            values = [100 * det[f'{protocol}/{m}/pooled/{res}']['AP'] for m in ('naive', 'maskclip', 'clearclip')]
            bars = ax.bar(x + delta, values, .36, label='P1: dense components' if protocol == 'P1_dense' else 'P2: OWLv2 proposals', color=color)
            ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=2)
        ax.axhline(100 * det['owlv2_native']['AP'], color='#555555', linestyle='--', linewidth=1, label='Native OWLv2 (different scoring)')
        ax.set(xticks=x, xticklabels=labels, ylabel='COCO bbox AP (0–100)', ylim=(0, 53), title=f'CLIP input: {res} × {res}')
    axes[0].legend(fontsize=8, loc='upper left')
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('detection_pooled.' + suffix), dpi=200)
    plt.close(fig)
    selected = ['owlv2', 'clip_ctx', 'dcsm_obja_ctx', 'dcsm_coco_ctx', 'readout/naive/pooled', 'readout/clearclip/pooled']
    labels = ['OWLv2', 'CLIP context crop', 'DCSM ObjA context', 'DCSM COCO context', 'Naive dense + EOS', 'ClearCLIP-style + EOS']
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    vals = [100 * spatial['balanced'][m]['estimate'] for m in selected]
    ci = np.array([spatial['balanced'][m]['ci95'] for m in selected]) * 100
    bars = ax.barh(labels, vals, color=['#b3b3b3', '#b3b3b3', '#66c2a5', '#66c2a5', '#8da0cb', '#8da0cb'],
                   xerr=np.stack([np.array(vals) - ci[:, 0], ci[:, 1] - vals]), capsize=3)
    ax.bar_label(bars, labels=[f'{v:.1f}' for v in vals], padding=26, fontsize=9)
    ax.axvline(100 * spatial['balanced']['chance']['estimate'], color='#555555', linestyle='--', label='Balanced expected chance: 45.7%')
    ax.set(xlim=(0, 77), xlabel='Balanced candidate-selection accuracy (%)', title='Spatial queries: 527 items / 375 image clusters')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='lower right')
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('spatial_balanced.' + suffix), dpi=200)
    plt.close(fig)
    print(json.dumps({'overall_images': report['unique_images_overall'], 'spatial': spatial['paired_differences'],
                      'figures': [str(p.name) for p in out.iterdir()]}, indent=2))


if __name__ == '__main__':
    main()
