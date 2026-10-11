"""Size-balanced accuracy for the failure-mode query benchmark (candidate-selection protocol).

Most items have N=2 candidates, so a scorer that always prefers the largest (or smallest) box can score far from
chance on subsets where the target correlates with size (negation: targets tend to be small background instances;
color / affirm: targets tend to be large). Balanced accuracy = mean( acc | target is the largest candidate,
acc | target is not the largest ). A pure size heuristic scores exactly 50; a size-blind scorer is unaffected.
For 'size' the two strata are the 'big' and 'small' queries. Also writes the per-stratum table."""
import json, sys
import numpy as np
q = json.load(open("data/comp_queries/comp_queries.json"))["items"]
res_dir = sys.argv[1] if len(sys.argv) > 1 else "results/comp_full"
per = json.load(open(f"{res_dir}/per_item.json"))
cand = {(it["image_id"], it["query"]): it["candidates"] for it in q}
def area(b): return (b[2]-b[0])*(b[3]-b[1])
def is_largest(r):
    return r["target"] == int(np.argmax([area(b) for b in cand[(r["image_id"], r["query"])]]))
subs = ["size", "color", "spatial", "negation", "affirm"]
scorers = list(per[0]["pred"].keys())
heur = {"largest box": lambda r: int(np.argmax([area(b) for b in cand[(r["image_id"], r["query"])]])),
        "smallest box": lambda r: int(np.argmin([area(b) for b in cand[(r["image_id"], r["query"])]]))}
def strata(s):
    rs = [r for r in per if r["subset"] == s]
    if s == "size": return [r for r in rs if r["rel"] == "big"], [r for r in rs if r["rel"] == "small"]
    return [r for r in rs if is_largest(r)], [r for r in rs if not is_largest(r)]
def acc(rs, pred): return np.mean([pred(r) == r["target"] for r in rs]) if rs else np.nan
lines = ["| scorer | " + " | ".join(subs) + " |", "|---|" + "---|"*len(subs)]
lines.append("| #items (target largest / not) | " + " | ".join("{} / {}".format(*map(len, strata(s))) for s in subs) + " |")
lines.append("| chance (balanced) | " + " | ".join(f"{100*np.mean([np.mean([1/r['n'] for r in st]) for st in strata(s)]):.1f}" for s in subs) + " |")
for name, f in heur.items():
    lines.append(f"| {name} | " + " | ".join(f"{100*np.mean([acc(st, f) for st in strata(s)]):.1f}" for s in subs) + " |")
for k in scorers:
    f = lambda r, k=k: r["pred"][k]
    cells = []
    for s in subs:
        a, b = strata(s)
        cells.append(f"{100*np.mean([acc(a, f), acc(b, f)]):.1f} ({100*acc(a, f):.0f}/{100*acc(b, f):.0f})")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
md = "size-balanced top-1 accuracy, cells = balanced (target-largest / target-not-largest; for size: big / small)\n\n" + "\n".join(lines) + "\n"
print(md); open(f"{res_dir}/summary_balanced.md", "w").write(md)
