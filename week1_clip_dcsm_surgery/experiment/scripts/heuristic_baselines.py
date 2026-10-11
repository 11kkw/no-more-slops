"""Model-free baselines for the failure-mode query benchmark (candidate selection): largest box, smallest box, leftmost/rightmost, most central."""
import json, sys
from collections import defaultdict
import numpy as np
items = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/comp_queries/comp_queries.json"))["items"]
def area(b): return (b[2]-b[0])*(b[3]-b[1])
rules = {
    "largest": lambda it: int(np.argmax([area(b) for b in it["candidates"]])),
    "smallest": lambda it: int(np.argmin([area(b) for b in it["candidates"]])),
    "most_central": lambda it: int(np.argmin([((b[0]+b[2])/2-it["W"]/2)**2+((b[1]+b[3])/2-it["H"]/2)**2 for b in it["candidates"]])),
    "chance": None,
}
subs = ["size","color","spatial","negation","affirm"]
print("| heuristic | " + " | ".join(subs) + " |"); print("|---|" + "---|"*len(subs))
for name, f in rules.items():
    row = []
    for s in subs:
        rs = [it for it in items if it["subset"] == s]
        row.append(np.mean([1/len(it["candidates"]) for it in rs]) if f is None else np.mean([f(it) == it["target_idx"] for it in rs]))
    print(f"| {name} | " + " | ".join(f"{100*v:.1f}" for v in row) + " |")
