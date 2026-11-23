"""Per-cell metrics of a steering sweep on one benchmark item set, with paired differences against coefficient 0.

Metrics per row: ELEPHANT validation, indirectness, framing (1 = sycophantic) under --judge-name; Are You Sure flip =
1 - turn2_correct (every item's turn 1 was judged correct at selection); persona_sycophantic and persona_coherence from
the gpt-4.1-mini scores files. A metric that is None for a row (truncated, unparsed, not scored) leaves the row out of
that metric and is counted. Each cell reports the mean with a 95% percentile bootstrap interval over items, and the
paired mean difference to the unsteered base cell (persona, coefficient 0, no adapter) over items defined in both. The original benchmark labels
and scores of the same items (the run they were selected from) are summarised the same way, paired against coefficient 0.

Run from the repo root:
  python -m reproduction.persona_vector.steer_benchmarks_summary --items ... --judged-dir ... --scores-dir ... \
    --original-judged ... --original-scores ... --judge-name sonnet55 --n-boot 10000 --seed 0 --output ...
"""
import argparse
from pathlib import Path

import numpy as np

from utils.io import read_jsonl, write_json, write_meta

ELEPHANT_METRICS = ("validation", "indirectness", "framing")
SCORE_METRICS = {"persona_sycophantic": "sycophantic", "persona_coherence": "coherence"}


def label_metrics(row: dict, judge: str) -> dict:
    """Benchmark metrics of one judged row: ELEPHANT labels, or Are You Sure flip and the re-judged turn 1."""
    labels = {k.removesuffix(f"__{judge}"): v for k, v in row["labels"].items() if k.endswith(f"__{judge}")}
    if row["benchmark"] == "elephant":
        return {m: labels[m] for m in ELEPHANT_METRICS}
    if row["benchmark"] != "are_you_sure":
        raise ValueError(f"{row['id']}: unsupported benchmark {row['benchmark']!r}")
    t2 = labels["turn2_correct"]
    return {"flip": None if t2 is None else 1 - t2, "turn1_rejudged_correct": labels["turn1_correct"]}


def metric_table(judged: list[dict], scores: list[dict], ids: list[str], judge: str) -> dict[str, dict[str, float | None]]:
    """{metric: {item id: value or None}} over `ids`; score metrics absent from every scores row are omitted."""
    by_id = {r["id"]: r for r in judged}
    if sorted(by_id) != sorted(ids) or len(by_id) != len(judged):
        raise AssertionError("judged rows are not exactly the items")
    score_by_id = {s["id"]: s for s in scores}
    if len(score_by_id) != len(scores) or not set(score_by_id) <= set(ids):
        raise AssertionError("scores have duplicate ids or ids outside the items")
    table = {}
    for i in ids:
        for m, v in label_metrics(by_id[i], judge).items():
            table.setdefault(m, {})[i] = v
    for m, key in SCORE_METRICS.items():
        if any(key in s for s in scores):
            table[m] = {i: score_by_id[i][key] if i in score_by_id else None for i in ids}
    return table


def mean_ci(x: list[float], n_boot: int, seed: int) -> dict:
    """Mean and 95% percentile bootstrap interval over the values; undefined (None) when there are none."""
    if not x:
        return {"n": 0, "mean": None, "ci": None}
    a = np.asarray(x, dtype=np.float64)
    means = a[np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    return {"n": len(a), "mean": float(a.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def summarize(values: dict[str, float | None], base: dict[str, float | None] | None, n_boot: int, seed: int) -> dict:
    """Mean over defined values, the count of undefined ones, and the paired difference to `base` if given."""
    out = {**mean_ci([v for v in values.values() if v is not None], n_boot, seed),
           "n_undefined": sum(v is None for v in values.values())}
    if base is not None:
        pairs = [values[i] - base[i] for i in values if values[i] is not None and base[i] is not None]
        out["diff_vs_c0"] = mean_ci(pairs, n_boot, seed)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--judged-dir", type=Path, required=True, help="Holds <items stem>_L*_judged.jsonl.")
    parser.add_argument("--scores-dir", type=Path, required=True, help="Holds <items stem>_L*.scores.jsonl.")
    parser.add_argument("--original-judged", type=Path, required=True, help="Judged file the items were selected from.")
    parser.add_argument("--original-scores", type=Path, required=True, help="gpt-4.1-mini scores of those rows.")
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--n-boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ids = [r["id"] for r in read_jsonl(args.items)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{args.items}: duplicate ids")
    paths = sorted(args.judged_dir.glob(f"{args.items.stem}_L*_judged.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no judged cells for {args.items.stem} in {args.judged_dir}")
    cells, inputs = [], [args.items, args.original_judged, args.original_scores]
    for path in paths:
        judged = read_jsonl(path)
        steer = judged[0]["steer"]
        if any(r["steer"] != steer for r in judged):
            raise AssertionError(f"{path}: rows were steered differently")
        name = path.name.removesuffix("_judged.jsonl")
        scores_path = args.scores_dir / f"{name}.scores.jsonl"
        adapter = name.split("_ft-")[1] if "_ft-" in name else None  # fine-tuned cells carry _ft-<adapter dir>
        if any(r["generation"].get("adapter", None) != adapter for r in judged):  # files before --adapter have no key
            raise AssertionError(f"{path}: rows disagree with the file name about the adapter")
        cells.append({"name": name, "steer": steer, "adapter": adapter, "judged": judged, "table": metric_table(
            judged, read_jsonl(scores_path), ids, args.judge_name)})
        inputs += [path, scores_path]
    base = [c for c in cells if c["steer"]["coef"] == 0 and c["steer"]["direction"] == "persona" and c["adapter"] is None]
    if len(base) != 1:
        raise AssertionError(f"expected one persona coefficient-0 cell, found {len(base)}")
    base = base[0]["table"]

    original_rows = [r for r in read_jsonl(args.original_judged) if r["id"] in set(ids)]
    original_scores = [s for s in read_jsonl(args.original_scores) if s["id"] in set(ids)]
    original = metric_table(original_rows, original_scores, ids, args.judge_name)
    out = {"items": args.items.as_posix(), "n_items": len(ids), "judge_name": args.judge_name, "n_boot": args.n_boot,
           "metrics": sorted(base),
           "original": {m: summarize(v, base[m], args.n_boot, args.seed) for m, v in original.items()},
           "cells": []}
    for c in sorted(cells, key=lambda c: (c["adapter"] or "", c["steer"]["direction"], c["steer"]["coef"])):
        out["cells"].append({
            "name": c["name"], "adapter": c["adapter"], **{k: c["steer"][k] for k in ("direction", "layer", "block", "coef", "norm")},
            "n_rows": len(c["judged"]), "n_truncated": sum(r["truncated"] for r in c["judged"]),
            "mean_response_tokens": float(np.mean([r["n_response_tokens"] for r in c["judged"]])),
            "metrics": {m: summarize(v, base[m], args.n_boot, args.seed) for m, v in c["table"].items()}})
    write_json(args.output, out)
    write_meta(args.output, inputs, args, {"n_items": len(ids), "n_cells": len(cells)})
    for c in out["cells"]:
        print(c["name"], {m: (v["mean"] and round(v["mean"], 3), v["n"]) for m, v in c["metrics"].items()})


if __name__ == "__main__":
    main()
