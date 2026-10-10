"""Paired bootstrap CI (10k resamples) for the top-1 accuracy difference between two scorers on one subset."""
import json, sys
import numpy as np
res_dir, subset, a, b = sys.argv[1:5]
per = [r for r in json.load(open(f"{res_dir}/per_item.json")) if r["subset"] == subset]
ha = np.array([r["pred"][a] == r["target"] for r in per], float); hb = np.array([r["pred"][b] == r["target"] for r in per], float)
rng = np.random.default_rng(0); n = len(per)
d = [(ha[i] - hb[i]).mean() for i in (rng.integers(0, n, n) for _ in range(10000))]
print(f"{subset} n={n}: {a} {100*ha.mean():.1f} vs {b} {100*hb.mean():.1f}  diff {100*(ha.mean()-hb.mean()):+.1f}  95% CI [{100*np.percentile(d,2.5):+.1f}, {100*np.percentile(d,97.5):+.1f}]")
