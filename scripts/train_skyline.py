"""Fit probes on one benchmark label (position x layer x method) on the split's train groups; report held-out metrics.

Writes `<out-dir>/<benchmark>__<label>.probes.npz` and `.metrics.jsonl`.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import POSITIONS, act_key, load_index
from utils.io import check_counts, meta_path, read_json, read_jsonl, write_jsonl, write_meta
from utils.splits import split_masks


def select_labeled(rows: dict, index: list[dict], skips: list[dict], label: str) -> tuple[list[dict], dict]:
    """Index rows whose label is 0 or 1. Extraction skips and unlabeled (None) rows are excluded and counted."""
    excluded = Counter(f"extraction_{s['reason']}" for s in skips)
    kept = []
    for i in index:
        value = rows[i["id"]]["labels"][label]
        if value is None:
            excluded["unlabeled"] += 1
        elif value in (0, 1) and not isinstance(value, bool):
            kept.append(i)
        else:
            raise ValueError(f"{i['id']}: label {label}={value!r} is not 0, 1 or None")
    return kept, check_counts(len(rows), excluded, len(kept), label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Benchmark rows JSONL.")
    parser.add_argument("--label", required=True, help="Key in each row's labels dict.")
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=probes.METHODS, required=True)
    parser.add_argument("--C", type=float, required=True, help="Inverse L2 strength for logistic regression.")
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    stem = args.rows.stem
    rows = {r["id"]: r for r in read_jsonl(args.rows)}
    benchmarks = {r["benchmark"] for r in rows.values()}
    if len(benchmarks) != 1:
        raise ValueError(f"{args.rows}: expected one benchmark, got {sorted(benchmarks)}")
    benchmark = benchmarks.pop()
    generator_models = sorted({str(r["generator_model"]) for r in rows.values()})
    split = read_json(args.split)
    if split["group_field"] != "group_id":
        raise ValueError(f"{args.split} groups by {split['group_field']!r}, expected group_id")
    npz_path = args.activations_dir / f"{stem}.npz"
    act_meta = read_json(meta_path(npz_path))
    missing_layers = sorted(set(args.layers) - set(act_meta["layers"]))
    if missing_layers:
        raise ValueError(f"{npz_path} lacks layers {missing_layers}")

    index, skips = load_index(args.activations_dir, stem, list(rows))
    kept, counts = select_labeled(rows, index, skips, args.label)

    act_rows = np.array([i["row"] for i in kept])
    y = np.array([rows[i["id"]]["labels"][args.label] for i in kept], dtype=int)
    groups = [rows[i["id"]]["group_id"] for i in kept]
    tr, te = split_masks(groups, split)
    probes.check_binary(y[tr], f"{benchmark}/{args.label} train")
    counts.update(
        n_train=int(tr.sum()), n_train_pos=int(y[tr].sum()), n_test=int(te.sum()), n_test_pos=int(y[te].sum())
    )
    print(f"[{benchmark}/{args.label}] {counts}")

    fitted, metrics = {}, []
    with np.load(npz_path) as z:
        for position in args.positions:
            for layer in args.layers:
                X = z[act_key(position, layer)][act_rows]
                for method in args.methods:
                    probe = probes.fit(method, X[tr], y[tr], args.C, args.max_iter)
                    s = probes.score(method, probe, X[te])
                    fitted[probes.probe_key(method, position, layer)] = probe
                    metrics.append(
                        {
                            "source": "skyline",
                            "target": f"{benchmark}/{args.label}",
                            "method": method,
                            "position": position,
                            "layer": layer,
                            "split_seed": split["seed"],
                            "model": act_meta["model"],
                            "generator_models": generator_models,
                            "n_train": int(tr.sum()),
                            "n_test": int(te.sum()),
                            "n_test_pos": int(y[te].sum()),
                            "n_test_neg": int((y[te] == 0).sum()),
                            "holdout_auroc": probes.auroc(y[te], s),
                            "holdout_accuracy": probes.accuracy(y[te], s, method),
                        }
                    )
                    m = metrics[-1]
                    print(f"  {method:8s} {act_key(position, layer)}: auroc {m['holdout_auroc']} acc {m['holdout_accuracy']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = args.out_dir / f"{benchmark}__{args.label}.probes.npz"
    metrics_path = args.out_dir / f"{benchmark}__{args.label}.metrics.jsonl"
    probes.save_probes(probes_path, fitted)
    write_jsonl(metrics_path, metrics)
    extra = {"model": act_meta["model"], "generator_models": generator_models, "split_seed": split["seed"]}
    for out in (probes_path, metrics_path):
        write_meta(out, [args.rows, npz_path, args.split], args, counts, extra)


if __name__ == "__main__":
    main()
