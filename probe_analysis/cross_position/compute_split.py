"""Score every contrastive probe at all three positions on two group-disjoint halves of each OOD target.

Each OOD target's kept rows (as in probe_training.evaluate_probes) are split into folds 0 and 1 by group: groups are
shuffled with --seed, stably sorted by their fraction of label-1 rows, and assigned alternately, so the folds are
group-disjoint and near-balanced in label. AUROC is computed on each fold and on all rows; the all-row values must
reproduce cross_position.csv. Downstream analysis selects on one fold and measures on the other.

Writes <out-dir>/split_auroc.csv, one row per probe x eval position x dataset x fold.

Run from the repo root: python -m probe_analysis.cross_position.compute_split --sweep-dir ... --targets ... --eval-dir ... --cross-position ... --seed 0 --out-dir ...
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from probe_analysis.cross_position.compute_cross_position import POSITIONS, ood_rows, probe_forms, read_csv, score_all
from utils.activations import act_key
from utils.io import read_json, read_jsonl, write_meta

FIELDS = ("probe_id", "source_pair", "cell", "pair_type", "method", "C", "train_position", "layer", "eval_position",
          "dataset", "display", "fold", "n", "n_pos", "auroc")


def auroc_columns(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of S against binary y (Mann-Whitney with average ranks; equals roc_auc_score)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes")
    ranks = rankdata(S, axis=0, method="average")
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def group_folds(groups: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Fold 0/1 per row: whole groups, shuffled, sorted by label-1 fraction, assigned alternately."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    pos_frac = np.bincount(inverse, weights=y) / np.bincount(inverse)
    order = np.random.default_rng(seed).permutation(len(uniq))
    order = order[np.argsort(pos_frac[order], kind="stable")]
    group_fold = np.empty(len(uniq), dtype=int)
    group_fold[order] = np.arange(len(uniq)) % 2
    fold = group_fold[inverse]
    if set(groups[fold == 0].tolist()) & set(groups[fold == 1].tolist()):
        raise AssertionError("a group is in both folds")
    return fold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="CSV with eval_dataset, display, target_label, n.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory holding <eval_dataset>_activations.npz.")
    parser.add_argument("--cross-position", type=Path, required=True, help="cross_position.csv to check all-row AUROCs against.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path, split_path = args.sweep_dir / "manifest.jsonl", args.sweep_dir / "split.json"
    manifest, split, targets = read_jsonl(manifest_path), read_json(split_path), read_csv(args.targets)
    models = {m["model"] for m in manifest}
    if len(models) != 1:
        raise ValueError(f"manifest mixes models: {models}")
    model = next(iter(models))
    reference = {(r["probe_id"], r["eval_position"], r["dataset"]): float(r["auroc"]) for r in read_csv(args.cross_position)}
    layers, by_layer, forms, probe_files = probe_forms(manifest, args.sweep_dir)

    out_rows, counts, folds_info, inputs, max_err = [], {}, {}, [manifest_path, split_path, args.targets, args.cross_position], 0.0
    for t in targets:
        ds = t["eval_dataset"]
        path = args.eval_dir / f"{ds}_activations.npz"
        stored_path = args.sweep_dir / "eval" / f"{ds}.jsonl"
        inputs += [path, stored_path]
        stored_rows = [r for r in read_jsonl(stored_path) if r["eval_label"] == t["target_label"]]
        if len(stored_rows) != len(manifest):
            raise ValueError(f"{stored_path}: target-label rows do not match the manifest")
        z, keep, y, counts[ds] = ood_rows(path, t, split, stored_rows, model)
        fold = group_folds(z["group"][keep], y, args.seed)
        subsets = {"all": np.ones(len(y), dtype=bool), 0: fold == 0, 1: fold == 1}
        folds_info[ds] = {str(f): {"n": int(m.sum()), "n_pos": int(y[m].sum()), "n_groups": len(set(z["group"][keep][m].tolist()))}
                          for f, m in subsets.items() if f != "all"}
        for layer in layers:
            W, b = forms[layer]
            ms = by_layer[layer]
            for pos in POSITIONS:
                X = z[act_key(pos, layer)]
                if X.shape[0] != len(keep):
                    raise AssertionError(f"{path} {act_key(pos, layer)}: {X.shape[0]} rows, mask has {len(keep)}")
                S = score_all(X[keep], W, b)
                for f, mask in subsets.items():
                    aucs = auroc_columns(y[mask], S[mask])
                    for m, auc in zip(ms, aucs):
                        if f == "all":
                            err = abs(auc - reference[(m["probe_id"], pos, ds)])
                            max_err = max(max_err, err)
                            if err > 1e-9:
                                raise AssertionError(f"{m['probe_id']} {pos} {ds}: all-row AUROC {auc} != cross_position.csv")
                            continue
                        out_rows.append({
                            "probe_id": m["probe_id"], "source_pair": m["pair_index"], "cell": m["cell"],
                            "pair_type": m["pair_type"], "method": m["method"], "C": "" if m["C"] is None else m["C"],
                            "train_position": m["position"], "layer": m["layer"], "eval_position": pos, "dataset": ds,
                            "display": t["display"], "fold": f, "n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                            "auroc": float(auc),
                        })
        print(f"{ds}: folds {folds_info[ds]}", flush=True)

    expected = len(manifest) * len(POSITIONS) * len(targets) * 2
    if len(out_rows) != expected:
        raise AssertionError(f"{len(out_rows)} rows, expected {expected}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "split_auroc.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    write_meta(out, [*inputs, *(args.sweep_dir / f for f in probe_files)], args,
               {"n_in": len(manifest), "excluded": {}, "n_out": len(out_rows), "datasets": counts},
               {"model": model, "folds": folds_info, "all_rows_max_abs_diff_vs_cross_position": max_err})
    print(f"{len(out_rows)} rows -> {out}; all-row max |diff| vs cross_position.csv {max_err:.2e}")


if __name__ == "__main__":
    main()
