"""Paired bootstrap CI (10k resamples) for the top-1 accuracy difference between two scorers on one subset.

Usage: uv run scripts/paired_bootstrap.py <results_dir> <subset> <scorerA> <scorerB>
Works on per_item.json from compositional_eval.py ('pred' vs 'target') and referring_detection_eval.py ('hit')."""
import json
import sys

import numpy as np

from dcsm_surgery.metrics import paired_bootstrap

res_dir, subset, a, b = sys.argv[1:5]
per = [r for r in json.load(open(f"{res_dir}/per_item.json")) if r["subset"] == subset]
hit = (lambda r, k: r["hit"][k]) if "hit" in per[0] else (lambda r, k: r["pred"][k] == r["target"])
ha = [hit(r, a) for r in per]; hb = [hit(r, b) for r in per]
diff, lo, hi = paired_bootstrap(ha, hb)
print(f"{subset} n={len(per)}: {a} {100*np.mean(ha):.1f} vs {b} {100*np.mean(hb):.1f}  diff {100*diff:+.1f}  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
