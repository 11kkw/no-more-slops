#!/usr/bin/env python3
"""Export checked DCSM result tables without inference or external services."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


SUBSETS = ("size", "color", "spatial", "negation", "affirm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
            f"Aggregate mismatch: {label}: {actual} != {expected}")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]) + "\n"


def md_numbers(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and all(re.match(r"^\d+\.\d+", c) for c in cells[1:]):
            rows[cells[0]] = [float(re.match(r"^\d+\.\d+", c)[0]) for c in cells[1:]]
    return rows


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def choose(item, rule):
    boxes = item["candidates"]
    if rule == "largest":
        return max(range(len(boxes)), key=lambda i: area(boxes[i]))
    if rule == "smallest":
        return min(range(len(boxes)), key=lambda i: area(boxes[i]))
    distances = [((b[0] + b[2] - item["W"]) / 2) ** 2
                 + ((b[1] + b[3] - item["H"]) / 2) ** 2 for b in boxes]
    return min(range(len(boxes)), key=lambda i: distances[i])


def export(repo, out, refresh=False):
    repo, out = repo.resolve(), out.resolve()
    experiment = repo / "week1_clip_dcsm_surgery/experiment"
    results = experiment / "results"
    require(not out.is_relative_to(results.resolve()), "Output must be outside original results")
    inputs = {}

    def read(relative):
        path = results / relative
        inputs[str(path.relative_to(repo))] = path
        return json.loads(path.read_text())

    detection = read("full_val2017/summary.json")
    summary = read("comp_full/summary.json")
    predictions = read("comp_full/per_item.json")
    queries = read("comp_full/benchmark/comp_queries.json")
    probe = read("probe_500/localization_probe.json")
    require(predictions, "No compositional predictions")
    counts = Counter(row["subset"] for row in predictions)
    require(dict(counts) == summary["counts"], "Item counts disagree with summary")
    require(set(counts) == set(SUBSETS), "Unexpected or missing subset")
    query_index = {}
    for item in queries["items"]:
        key = (item["image_id"], item["query"])
        require(key not in query_index, f"Duplicate benchmark item: {key}")
        query_index[key] = item
    require(len(query_index) == len(predictions), "Query/prediction count mismatch")
    scorers = tuple(summary["top1"])
    seen = set()
    for row in predictions:
        key = (row["image_id"], row["query"])
        require(key not in seen, f"Duplicate prediction: {key}")
        seen.add(key)
        require(key in query_index, f"Prediction has no query: {key}")
        item = query_index[key]
        require(row["subset"] == item["subset"], f"Subset mismatch: {key}")
        require(row["n"] == len(item["candidates"]), f"Candidate count mismatch: {key}")
        require(0 <= row["target"] < row["n"], f"Invalid target: {key}")
        require(row["target"] == item["target_idx"], f"Target mismatch: {key}")
        require(set(row["pred"]) == set(scorers), f"Scorer mismatch: {key}")
        require(all(isinstance(p, int) and 0 <= p < row["n"] for p in row["pred"].values()),
                f"Invalid candidate prediction: {key}")

    grouped = {s: [r for r in predictions if r["subset"] == s] for s in SUBSETS}
    raw, balanced, chance, balanced_chance, strata_counts = {}, {}, {}, {}, {}
    for subset, rows in grouped.items():
        def group_index(row):
            if subset == "size":
                require(row["rel"] in ("big", "small"), "Unknown size query")
                require(row["rel"] == query_index[(row["image_id"], row["query"])]["attr"],
                        "Size attribute mismatch")
                return 0 if row["rel"] == "big" else 1
            item = query_index[(row["image_id"], row["query"])]
            return 0 if row["target"] == choose(item, "largest") else 1

        strata = [[r for r in rows if group_index(r) == i] for i in (0, 1)]
        require(all(strata), f"Empty balanced stratum: {subset}")
        strata_counts[subset] = [len(st) for st in strata]
        chance[subset] = mean(1 / r["n"] for r in rows)
        close(chance[subset], summary["chance"][subset], f"chance/{subset}")
        balanced_chance[subset] = mean(mean(1 / r["n"] for r in st) for st in strata)
        for scorer in scorers:
            value = mean(r["pred"][scorer] == r["target"] for r in rows)
            close(value, summary["top1"][scorer][subset], f"{scorer}/{subset}")
            raw.setdefault(scorer, {})[subset] = value
            balanced.setdefault(scorer, {})[subset] = mean(
                mean(r["pred"][scorer] == r["target"] for r in st) for st in strata)

    balanced_path = results / "comp_full/summary_balanced.md"
    inputs[str(balanced_path.relative_to(repo))] = balanced_path
    recorded = md_numbers(balanced_path)
    for scorer, values in balanced.items():
        require(scorer in recorded, f"Missing balanced table row: {scorer}")
        expected = [float(f"{100 * values[s]:.1f}") for s in SUBSETS]
        require(expected == recorded[scorer], f"Recorded balanced table mismatch: {scorer}")
    expected_chance = [float(f"{100 * balanced_chance[s]:.1f}") for s in SUBSETS]
    require(recorded.get("chance (balanced)") == expected_chance, "Balanced chance mismatch")

    heuristics = {}
    for rule in ("largest", "smallest", "most_central"):
        heuristics[rule] = {
            s: mean(choose(query_index[(r["image_id"], r["query"])], rule) == r["target"]
                    for r in rows) for s, rows in grouped.items()}
    heuristics["chance"] = chance
    heuristics_path = results / "comp_full/heuristics.md"
    inputs[str(heuristics_path.relative_to(repo))] = heuristics_path
    recorded_heuristics = md_numbers(heuristics_path)
    for rule, values in heuristics.items():
        require(recorded_heuristics.get(rule) == [float(f"{100 * values[s]:.1f}") for s in SUBSETS],
                f"Recorded heuristic mismatch: {rule}")

    metrics = ("AP", "AP50", "AP75", "APs", "APm", "APl")
    detection_rows = []
    require(detection["meta"]["n_images"] > 0, "Invalid image count")
    for config, values in detection["results"].items():
        require(all(isinstance(values[m], (int, float)) and math.isfinite(values[m])
                    and 0 <= values[m] <= 1 for m in metrics), f"Invalid detection AP: {config}")
        detection_rows.append([config, *(f"{100 * values[m]:.1f}" for m in metrics), values["n_dets"]])

    for path in sorted((experiment / "scripts").glob("*.py")):
        inputs[str(path.relative_to(repo))] = path
    for relative in ("pyproject.toml", "README.md"):
        path = experiment / relative
        inputs[str(path.relative_to(repo))] = path
    for relative in ("full_val2017/run_log.txt", "comp_full/run_log.txt", "probe_500/run_log.txt"):
        path = results / relative
        if path.exists():
            inputs[str(path.relative_to(repo))] = path
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "schema_version": 1,
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo), "snapshot_revision": revision,
        "execution_revision": None,
        "scope": "Saved artifact consistency; no inference, label validation, significance or novelty proof",
        "inputs": {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size} for name, path in inputs.items()},
    }
    measurements = {
        "schema_version": 1, "units": {"AP": "fraction; table displays 100 * AP", "top1": "fraction"},
        "sources": {"detection": "experiment/results/full_val2017/summary.json#/results",
                    "raw_top1": "experiment/results/comp_full/summary.json#/top1",
                    "balanced_top1": "experiment/results/comp_full/per_item.json with benchmark/comp_queries.json"},
        "detection_meta": detection["meta"], "detection": detection["results"],
        "compositional_args": summary["args"], "benchmark_args": queries["args"],
        "n_items": len(predictions), "counts": dict(counts), "strata_counts": strata_counts,
        "raw_top1": raw, "balanced_top1": balanced, "chance": chance,
        "balanced_chance": balanced_chance, "heuristics": heuristics,
        "probe": probe, "uncertainty": "Not computed by this exporter",
    }
    percent_rows = lambda scores: [[k, *(f"{100 * v[s]:.1f}" for s in SUBSETS)] for k, v in scores.items()]
    payloads = {
        "evidence/manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        "evidence/measurements.json": json.dumps(measurements, indent=2, ensure_ascii=False) + "\n",
        "tables/detection.md": f"Evaluated images: {detection['meta']['n_images']}. AP scale: 0–100.\n\n"
                               + table(["config", *metrics, "#dets"], detection_rows),
        "tables/compositional_raw.md": "Raw top-1 accuracy (%); conditional candidate selection.\n\n"
                                     + table(["scorer", *SUBSETS], [["#items", *(counts[s] for s in SUBSETS)],
                                             *percent_rows({"chance": chance}), *percent_rows(raw)]),
        "tables/compositional_balanced.md": "Balanced top-1 accuracy (%). Size: big/small; others: target-largest/not-largest.\n\n"
                                          + table(["scorer", *SUBSETS], [["stratum counts", *(" / ".join(map(str, strata_counts[s])) for s in SUBSETS)],
                                                  *percent_rows({"chance": balanced_chance}), *percent_rows(balanced)]),
        "tables/heuristics.md": "Model-free raw top-1 accuracy (%).\n\n" + table(["heuristic", *SUBSETS], percent_rows(heuristics)),
    }
    existing = [str(out / name) for name in payloads if (out / name).exists()]
    require(refresh or not existing, "Generated files already exist; inspect them or use --refresh: " + ", ".join(existing))
    for name, content in payloads.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return {"images": detection["meta"]["n_images"], "items": len(predictions),
            "scorers": len(scorers), "input_files": len(inputs), "generated_files": list(payloads),
            "out": str(out), "checks": "raw aggregates, chance, candidate identity, balanced and heuristic tables"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Replace only this exporter's generated files")
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.repo, args.out, args.refresh), ensure_ascii=False))
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Evidence export failed: {error}\n")


if __name__ == "__main__":
    main()
