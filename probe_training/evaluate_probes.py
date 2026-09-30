"""Score every probe of a sweep on a judged activations npz; AUROC and balanced accuracy per label.

Rows excluded per label: in_probe_train_or_val (id in the sweep's train_ids or val_ids, or its group in the
sweep's train or val groups when the npz has the sweep's group field) and label_none. Balanced accuracy uses each probe's decision boundary
(score 0), which may be miscalibrated on another distribution; AUROC does not depend on it.
Writes <sweep dir>/eval/<npz stem>.jsonl, one row per probe x label.

Run from the repo root: python -m probe_training.evaluate_probes --sweep-dir ... --activations ... --labels all
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import NO_LABEL, act_key, read_labels
from utils.io import check_counts, meta_path, read_json, read_jsonl, sha256, write_jsonl, write_meta

PROBE_FIELDS = ("probe_id", "file", "probe_type", "pair_index", "cell", "pair_type", "label", "benchmarks",
                "method", "position", "layer", "C")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="probes/<model name>/<sweep>/")
    parser.add_argument("--activations", type=Path, required=True, help="Judged *_activations.npz.")
    parser.add_argument("--labels", nargs="+", required=True, help="Judged label names, or 'all'.")
    args = parser.parse_args()

    manifest_path, split_path = args.sweep_dir / "manifest.jsonl", args.sweep_dir / "split.json"
    manifest, split = read_jsonl(manifest_path), read_json(split_path)
    act_meta = read_json(meta_path(args.activations))
    models = sorted({m["model"] for m in manifest})
    if models != [act_meta["model"]]:
        raise ValueError(f"probes are for {models}, activations {args.activations} are from {act_meta['model']}")
    stem = args.activations.name.removesuffix("_activations.npz")
    if stem == args.activations.name:
        raise ValueError(f"{args.activations} is not an *_activations.npz file")
    out_path = args.sweep_dir / "eval" / f"{stem}.jsonl"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists")

    z = np.load(args.activations)
    ids = z["id"]
    all_labels = read_labels(z)
    if not all_labels:
        raise ValueError(f"{args.activations} has no labels__* arrays; it is not a judged activations file")
    selected = sorted(all_labels) if args.labels == ["all"] else args.labels
    unknown = sorted(set(selected) - set(all_labels))
    if unknown:
        raise ValueError(f"labels {unknown} not in {args.activations} (has {sorted(all_labels)})")
    needed = {(act_key(m["position"], m["layer"]), m["layer"]) for m in manifest}
    for key, layer in sorted(needed):
        if key not in z or layer not in act_meta["layers"]:
            raise ValueError(f"{args.activations} lacks required {key}")
        if z[key].shape[0] != len(ids):
            raise AssertionError(f"{key} has {z[key].shape[0]} rows, id has {len(ids)}")

    # No row used for fitting or hyperparameter choice may be scored.
    in_train = np.isin(ids, split["train_ids"] + split["val_ids"])
    if split["group_field"] in z:
        in_train |= np.isin(z[split["group_field"]], split["train"] + split["val"])
    eval_sha = sha256(args.activations)
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
                    "eval_activations": args.activations.as_posix(),
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
    extra = {"model": act_meta["model"], "eval_sha256": eval_sha, "labels": selected, "n_probes": len(manifest)}
    write_meta(out_path, [args.activations, manifest_path, split_path], args, counts, extra)
    print(f"{len(rows)} rows ({len(manifest)} probes x {len(selected)} labels) -> {out_path}")


if __name__ == "__main__":
    main()
