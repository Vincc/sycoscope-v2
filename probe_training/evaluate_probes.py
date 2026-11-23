"""Score every probe of a sweep on judged activations; AUROC and balanced accuracy per label.

Activations come from a judged get_activations npz (--activations) or from the activation cache joined to a judged
JSONL (--cache and --rows). Rows excluded per label: in_probe_train_or_val (id in the sweep's train_ids or val_ids,
or its group in the sweep's train or val groups when the input has the sweep's group field) and label_none.
Balanced accuracy uses each probe's decision boundary (score 0), which may be miscalibrated on another distribution;
AUROC does not depend on it. Writes <sweep dir>/eval/<npz stem or JSONL stem>.jsonl, one row per probe x label.

Run from the repo root: python -m probe_training.evaluate_probes --sweep-dir ... --activations ... --labels all
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from interp.cache_activations import load_for_probes
from utils import probes
from utils.activations import NO_LABEL, act_key, read_labels
from utils.io import check_counts, meta_path, read_json, read_jsonl, sha256, write_jsonl, write_meta

PROBE_FIELDS = ("probe_id", "file", "probe_type", "pair_index", "cell", "pair_type", "label", "benchmarks",
                "method", "position", "layer", "C")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="probes/<model name>/<sweep>/")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--activations", type=Path, help="Judged *_activations.npz.")
    source_group.add_argument("--cache", type=Path, help="Activation-cache folder of the responses --rows were judged on.")
    parser.add_argument("--rows", type=Path, help="With --cache: the judged JSONL.")
    parser.add_argument("--labels", nargs="+", required=True, help="Judged label names, or 'all'.")
    args = parser.parse_args()
    if (args.cache is None) != (args.rows is None):
        parser.error("--cache and --rows go together")

    manifest_path, split_path = args.sweep_dir / "manifest.jsonl", args.sweep_dir / "split.json"
    manifest, split = read_jsonl(manifest_path), read_json(split_path)
    models = sorted({m["model"] for m in manifest})
    needed = {(act_key(m["position"], m["layer"]), m["layer"]) for m in manifest}
    cache_join = None
    if args.activations is not None:
        source, inputs = args.activations, [args.activations, manifest_path, split_path]
        stem = args.activations.name.removesuffix("_activations.npz")
        if stem == args.activations.name:
            raise ValueError(f"{args.activations} is not an *_activations.npz file")
    else:
        source, inputs = args.rows, [args.rows, manifest_path, split_path]
        stem = args.rows.stem
    out_path = args.sweep_dir / "eval" / f"{stem}.jsonl"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists")

    if args.activations is not None:
        act_meta = read_json(meta_path(args.activations))
        model, layers = act_meta["model"], act_meta["layers"]
        z = np.load(args.activations)
    else:
        positions = sorted({m["position"] for m in manifest})
        z, info = load_for_probes(args.cache, read_jsonl(args.rows), positions, sorted({m["layer"] for m in manifest}))
        model, layers = info["model"], info["layers"]
        inputs += info["part_metas"]
        cache_join = {k: v for k, v in info.items() if k != "part_metas"}
        print(f"cache joined to {args.rows}: {info['counts']}")
    if models != [model]:
        raise ValueError(f"probes are for {models}, activations {source} are from {model}")
    ids = z["id"]
    all_labels = read_labels(z)
    if not all_labels:
        raise ValueError(f"{source} has no labels__* arrays; it is not a judged activations file")
    selected = sorted(all_labels) if args.labels == ["all"] else args.labels
    unknown = sorted(set(selected) - set(all_labels))
    if unknown:
        raise ValueError(f"labels {unknown} not in {source} (has {sorted(all_labels)})")
    for key, layer in sorted(needed):
        if key not in z or layer not in layers:
            raise ValueError(f"{source} lacks required {key}")
        if z[key].shape[0] != len(ids):
            raise AssertionError(f"{key} has {z[key].shape[0]} rows, id has {len(ids)}")

    # No row used for fitting or hyperparameter choice may be scored.
    in_train = np.isin(ids, split["train_ids"] + split["val_ids"])
    if split["group_field"] in z:
        in_train |= np.isin(z[split["group_field"]], split["train"] + split["val"])
    eval_sha = sha256(source)  # with --cache, the JSONL's; the meta lists each cached part's meta file
    benchmarks = sorted(set(z["benchmark"].tolist()))

    masks, counts = {}, {}
    for name in selected:
        y_all = all_labels[name].astype(int)
        none = (y_all == NO_LABEL) & ~in_train
        masks[name] = ~in_train & ~none
        excluded = Counter({"in_probe_train": int(in_train.sum()), "label_none": int(none.sum())})
        counts[name] = check_counts(len(ids), excluded, int(masks[name].sum()), name)
        print(f"[{name}] {counts[name]}")

    rows, acts, probe_files = [], {}, {}
    for m in manifest:
        if m["file"] not in probe_files:
            probe_files[m["file"]] = np.load(args.sweep_dir / m["file"])
        probe = probes.load_probe(probe_files[m["file"]], m["probe_id"], m["method"])
        key = act_key(m["position"], m["layer"])
        if key not in acts:
            acts[key] = z[key]
        s_all = probes.score(m["method"], probe, acts[key])
        for name in selected:
            keep = masks[name]
            y, s = all_labels[name][keep].astype(int), s_all[keep]
            n = int(keep.sum())
            rows.append(
                {
                    **{f: m[f] for f in PROBE_FIELDS if f in m},
                    "eval_activations": source.as_posix() if args.cache is None else f"{args.cache.as_posix()} + {source.as_posix()}",
                    "eval_sha256": eval_sha,
                    "eval_benchmarks": benchmarks,
                    "eval_label": name,
                    "n": n,
                    "n_pos": int((y == 1).sum()),
                    "n_neg": int((y == 0).sum()),
                    "n_excluded_none": counts[name]["excluded"]["label_none"],
                    "n_excluded_train": counts[name]["excluded"]["in_probe_train"],
                    "auroc": probes.auroc(y, s),
                    "balanced_accuracy": probes.balanced_accuracy(y, s, m["method"]),
                }
            )

    write_jsonl(out_path, rows)
    extra = {"model": model, "eval_sha256": eval_sha, "labels": selected, "n_probes": len(manifest),
             "cache_join": cache_join}
    write_meta(out_path, inputs, args, counts, extra)
    print(f"{len(rows)} rows ({len(manifest)} probes x {len(selected)} labels) -> {out_path}")


if __name__ == "__main__":
    main()
