"""Fit one probe per unit x method x position x layer x C on train groups; score on held-out val and test groups.

A unit is a contrastive cell (--cells, contrastive activations npz, split by prompt_id so both responses to a
prompt share a side) or a skyline label (--labels, judged activations npz, split by group; rows whose label is
None are excluded). One split is shared by every unit; val is for choosing hyperparameters, test for reporting.
Writes to probes/<model name>/<sweep>/:
  split.json               train, val and test groups, and train_ids / val_ids: every row id in a train / val group
  <unit>.probes.npz        every probe for one unit (pairNN or the label name), arrays keyed <probe_id>__<field>
  manifest.jsonl           one row per probe: its parameters, file, key and train/val/test metrics (val None if
                           --val-frac 0)

Run from the repo root: python -m probe_training.train_probes --activations ... --sweep ... --cells all ...
"""
import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np

from utils import probes
from utils.activations import NO_LABEL, POSITIONS, act_key, read_labels
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, write_json, write_jsonl, write_meta
from utils.splits import group_split, split_masks


def complete_pairs(prompt_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mask of rows whose prompt has exactly one label-1 and one label-0 row."""
    by_prompt: dict[str, list[int]] = {}
    for pid, label in zip(prompt_ids, labels):
        by_prompt.setdefault(pid, []).append(int(label))
    ok = {pid for pid, ls in by_prompt.items() if sorted(ls) == [0, 1]}
    return np.array([pid in ok for pid in prompt_ids], dtype=bool)


def probe_id(unit: str, method: str, position: str, layer: int, C: float | None) -> str:
    base = f"{unit}__{probes.probe_key(method, position, layer)}"
    return base if C is None else f"{base}__C{C:g}"


def contrastive_units(z, cells: list[str], source: str) -> tuple[list[dict], dict]:
    """One unit per selected cell, keeping prompts with both a label-1 and a label-0 row."""
    labels, pair_index = z["label"].astype(int), z["pair_index"].astype(int)
    prompt_ids, cell_names, pair_types = z["prompt_id"], z["cell"], z["pair_type"]
    available = sorted(set(pair_index.tolist()))
    selected = available if cells == ["all"] else sorted(int(c) for c in cells)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"pair indices {unknown} not in {source} (has {available})")
    units, per_unit = [], {}
    for k in selected:
        in_cell = pair_index == k
        paired = complete_pairs(prompt_ids[in_cell], labels[in_cell])
        rows = np.flatnonzero(in_cell)[paired]
        name = f"pair{k:02d}"
        per_unit[name] = check_counts(int(in_cell.sum()), Counter({"pair_partner_missing": int((~paired).sum())}), len(rows), name)
        fields = {"probe_type": "contrastive", "pair_index": k, "cell": str(cell_names[rows[0]]), "pair_type": str(pair_types[rows[0]])}
        units.append({"name": name, "rows": rows, "y": labels[rows], "fields": fields, "paired": True})
    total = check_counts(
        len(labels),
        Counter({"cell_not_selected": int((~np.isin(pair_index, selected)).sum()),
                 "pair_partner_missing": sum(c["excluded"]["pair_partner_missing"] for c in per_unit.values())}),
        sum(c["n_out"] for c in per_unit.values()),
        "train_probes",
    )
    return units, {"total": total, "per_unit": per_unit}


def skyline_units(z, names: list[str], source: str) -> tuple[list[dict], dict]:
    """One unit per selected judged label, excluding rows whose label is None."""
    all_labels = read_labels(z)
    if not all_labels:
        raise ValueError(f"{source} has no labels__* arrays; it is not a judged activations file")
    selected = sorted(all_labels) if names == ["all"] else names
    unknown = sorted(set(selected) - set(all_labels))
    if unknown:
        raise ValueError(f"labels {unknown} not in {source} (has {sorted(all_labels)})")
    benchmarks = sorted(set(z["benchmark"].tolist()))
    units, per_unit = [], {}
    for name in selected:
        y_all = all_labels[name].astype(int)
        rows = np.flatnonzero(y_all != NO_LABEL)
        per_unit[name] = check_counts(len(y_all), Counter({"label_none": int((y_all == NO_LABEL).sum())}), len(rows), name)
        fields = {"probe_type": "skyline", "label": name, "benchmarks": benchmarks}
        units.append({"name": name, "rows": rows, "y": y_all[rows], "fields": fields, "paired": False})
    return units, {"per_unit": per_unit}  # units share rows, so there is no single total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True, help="Contrastive or judged *_activations.npz.")
    parser.add_argument("--sweep", required=True, help="Output directory name under probes/<model name>/.")
    unit_group = parser.add_mutually_exclusive_group(required=True)
    unit_group.add_argument("--cells", nargs="+", help="Contrastive: pair indices, or 'all'.")
    unit_group.add_argument("--labels", nargs="+", help="Skyline: judged label names, or 'all'.")
    parser.add_argument("--methods", nargs="+", choices=probes.METHODS, required=True)
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--C", type=float, nargs="+", required=True, help="Inverse L2 strengths (logistic only).")
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--test-frac", type=float, required=True)
    parser.add_argument("--val-frac", type=float, required=True, help="0 for no validation split.")
    parser.add_argument("--seed", type=int, required=True, help="Split seed.")
    args = parser.parse_args()

    act_meta = read_json(meta_path(args.activations))
    missing_layers = sorted(set(args.layers) - set(act_meta["layers"]))
    if missing_layers:
        raise ValueError(f"{args.activations} lacks layers {missing_layers}")
    out_dir = REPO_ROOT / "probes" / act_meta["model"].split("/")[-1] / args.sweep
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; pick a new --sweep name")

    z = np.load(args.activations)
    source = args.activations.as_posix()
    if args.cells is not None:
        if "label" not in z:
            raise ValueError(f"{args.activations} has no label array; it is not a contrastive activations file")
        units, counts = contrastive_units(z, args.cells, source)
        group_field = "prompt_id"
    else:
        units, counts = skyline_units(z, args.labels, source)
        group_field = "group"
    ids, groups = z["id"], z[group_field]
    if not len(ids) == len(groups) == z[act_key(args.positions[0], args.layers[0])].shape[0]:
        raise AssertionError("activation arrays and row fields differ in length")

    split = group_split(sorted(set(groups.tolist())), args.test_frac, args.val_frac, args.seed)
    split["group_field"] = group_field
    split["activations"] = source
    split["train_ids"] = sorted(ids[np.isin(groups, split["train"])].tolist())  # evaluate_probes excludes these
    split["val_ids"] = sorted(ids[np.isin(groups, split["val"])].tolist())  # and these
    out_dir.mkdir(parents=True)
    split_path = out_dir / "split.json"
    write_json(split_path, split)
    print(f"{out_dir}: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test "
          f"{group_field} groups, shared by all units")

    grid = []
    for method, position, layer in itertools.product(args.methods, args.positions, args.layers):
        for C in args.C if method == "logistic" else [None]:  # DIM has no regularisation
            grid.append((method, position, layer, C))

    manifest = []
    for unit in units:
        name, rows, y = unit["name"], unit["rows"], unit["y"]
        g = groups[rows].tolist()
        tr, va, te = split_masks(g, split)
        probes.check_binary(y[tr], f"{name} train")
        probes.check_binary(y[te], f"{name} test")
        if split["val"]:
            probes.check_binary(y[va], f"{name} val")
        print(f"\n[{name}] {counts['per_unit'][name]} | train {tr.sum()} / val {va.sum()} / test {te.sum()} rows")

        fitted = {}
        npz_name = f"{name}.probes.npz"
        for method, position, layer, C in grid:
            X = z[act_key(position, layer)][rows]
            probe = probes.fit(method, X[tr], y[tr], C, args.max_iter)
            held_out = {}
            for side, mask in (("val", va), ("test", te)):
                s = probes.score(method, probe, X[mask])
                sy, sg = y[mask], [x for x, m in zip(g, mask) if m]
                win, n_pairs = probes.paired_win_rate(s, sy, sg) if unit["paired"] and mask.any() else (None, None)
                held_out.update({
                    f"n_{side}": int(mask.sum()),
                    f"n_{side}_pos": int(sy.sum()),
                    f"{side}_auroc": probes.auroc(sy, s),
                    f"{side}_accuracy": probes.accuracy(sy, s, method) if mask.any() else None,
                    f"{side}_paired_win_rate": win,  # None for skyline: rows are not paired
                    f"n_{side}_pairs": n_pairs,
                })
            s_tr = probes.score(method, probe, X[tr])
            pid = probe_id(name, method, position, layer, C)
            fitted[pid] = probe
            manifest.append(
                {
                    "probe_id": pid,
                    "file": npz_name,
                    **unit["fields"],
                    "method": method,
                    "position": position,
                    "layer": layer,
                    "C": C,
                    "max_iter": args.max_iter,
                    "split_seed": args.seed,
                    "test_frac": args.test_frac,
                    "val_frac": args.val_frac,
                    "model": act_meta["model"],
                    "n_train": int(tr.sum()),
                    "n_train_pos": int(y[tr].sum()),
                    "train_auroc": probes.auroc(y[tr], s_tr),
                    "train_accuracy": probes.accuracy(y[tr], s_tr, method),
                    **held_out,
                }
            )
            m = manifest[-1]
            val = "" if m["val_auroc"] is None else f"val auroc {m['val_auroc']:.3f} | "
            paired = "" if m["test_paired_win_rate"] is None else f" paired {m['test_paired_win_rate']:.3f}"
            print(f"  {pid}: {val}test auroc {m['test_auroc']:.3f} acc {m['test_accuracy']:.3f}{paired}")
        probes.save_probes(out_dir / npz_name, fitted)

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    extra = {"model": act_meta["model"], "units": [u["name"] for u in units], "n_probes": len(manifest),
             "per_unit": counts["per_unit"]}
    for out in [split_path, manifest_path, *(out_dir / f"{u['name']}.probes.npz" for u in units)]:
        write_meta(out, [args.activations], args, counts.get("total", counts["per_unit"]), extra)
    print(f"\n{len(manifest)} probes -> {manifest_path}")


if __name__ == "__main__":
    main()
