"""Stage C (CPU): ridge assumption probes and their per-dataset AUC (paper Section 3, Table A7).

For each labeller and dimension: split the train rows 80/20 (seed), fit Ridge(alpha) on the 80% at every
layer, keep the layer with the highest validation R^2 (upstream train_assumption_probe.py). Test AUC: score =
X @ w + b, positives = label > hi, negatives = label < lo, computed per dataset and macro-averaged over the
paper's six datasets. Writes results/probes/<labeller>_<dim>.npz (w, b, layer) and results/probe_results.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import DIMS, LABELLERS  # noqa: E402

PAPER_DATASETS = ("oeq", "aita", "ir", "factual", "cancer", "wildchat")


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def features(reps, row_index, rows, layer):
    idx = np.array([row_index[r["id"]] for r in rows])
    return reps[f"layer_{layer}"][idx]


def auc_by_dataset(y, s, datasets, lo, hi, quantile):
    """quantile=False: positives y > hi, negatives y < lo (paper text). quantile=True: lo and hi are per-dataset
    label quantiles, positives y >= q_hi, negatives y <= q_lo (upstream evaluate_auc_by_source.py)."""
    out = {}
    for ds in PAPER_DATASETS:
        m = datasets == ds
        if quantile:
            q_lo, q_hi = np.quantile(y[m], lo), np.quantile(y[m], hi)
            pos, neg = m & (y >= q_hi), m & (y <= q_lo)
        else:
            pos, neg = m & (y > hi), m & (y < lo)
        if pos.sum() == 0 or neg.sum() == 0:
            out[ds] = {"auc": None, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_total": int(m.sum())}
            continue
        yy = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
        ss = np.r_[s[pos], s[neg]]
        out[ds] = {"auc": float(roc_auc_score(yy, ss)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_total": int(m.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True, help="npz from extract_reps.py")
    ap.add_argument("--prepared", required=True, help="data/prepared directory")
    ap.add_argument("--labellers", default=",".join(LABELLERS))
    ap.add_argument("--alpha", type=float, required=True, help="ridge regularisation (upstream: 10)")
    ap.add_argument("--seed", type=int, required=True, help="inner train/validation split seed (upstream: 42)")
    ap.add_argument("--lo", type=float, required=True, help="negatives: score < lo (paper: 0.3)")
    ap.add_argument("--hi", type=float, required=True, help="positives: score > hi (paper: 0.7)")
    ap.add_argument("--quantile", action="store_true", help="treat lo/hi as per-dataset label quantiles (upstream code)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    reps = np.load(args.reps)
    row_index = {i: k for k, i in enumerate(reps["ids"].tolist())}
    layers = sorted(int(k.removeprefix("layer_")) for k in reps.files if k.startswith("layer_"))
    os.makedirs(os.path.join(args.out_dir, "probes"), exist_ok=True)
    results = {"reps": args.reps, "layers": layers, "alpha": args.alpha, "seed": args.seed, "lo": args.lo, "hi": args.hi, "quantile": args.quantile, "probes": {}}

    for lab in args.labellers.split(","):
        for group, dims in DIMS.items():
            train = load_rows(os.path.join(args.prepared, f"{lab}_{group}_train.jsonl"))
            test_all = load_rows(os.path.join(args.prepared, f"{lab}_{group}_test.jsonl"))
            assert not {r["id"] for r in train} & {r["id"] for r in test_all}, f"{lab} {group}: train/test prompt overlap"
            test = [r for r in test_all if r["dataset"] in PAPER_DATASETS]
            for dim in dims:
                assert all(dim in r["scores"] for r in train + test)
                y_train = np.array([r["scores"][dim] for r in train])
                tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.2, random_state=args.seed)
                per_layer, best = {}, None
                for L in layers:
                    X = features(reps, row_index, train, L)
                    probe = Ridge(alpha=args.alpha).fit(X[tr_idx], y_train[tr_idx])
                    r2 = float(r2_score(y_train[va_idx], probe.predict(X[va_idx])))
                    per_layer[L] = r2
                    if best is None or r2 > best[1]:
                        best = (L, r2, probe)
                L, r2, probe = best
                X_test = features(reps, row_index, test, L)
                s = X_test @ probe.coef_ + probe.intercept_
                y_test = np.array([r["scores"][dim] for r in test])
                datasets = np.array([r["dataset"] for r in test])
                by_ds = auc_by_dataset(y_test, s, datasets, args.lo, args.hi, args.quantile)
                defined = [v["auc"] for v in by_ds.values() if v["auc"] is not None]
                macro = float(np.mean(defined)) if len(defined) == len(PAPER_DATASETS) else None
                np.savez(os.path.join(args.out_dir, "probes", f"{lab}_{dim}.npz"), w=probe.coef_.astype(np.float32),
                         b=np.float32(probe.intercept_), layer=L)
                results["probes"][f"{lab}/{dim}"] = {
                    "layer": L, "val_r2": r2, "val_r2_by_layer": per_layer, "test_r2": float(r2_score(y_test, s)),
                    "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": len(test),
                    "auc_by_dataset": by_ds, "macro_auc": macro,
                }
                print(f"{lab:9s} {dim:28s} layer {L:2d}  valR2 {r2:.3f}  testR2 {results['probes'][f'{lab}/{dim}']['test_r2']:.3f}  "
                      f"macroAUC {macro if macro is None else f'{macro:.3f}'}  "
                      + " ".join(f"{ds}={v['auc']:.2f}" if v["auc"] is not None else f"{ds}=undef" for ds, v in by_ds.items()))

    json.dump(results, open(os.path.join(args.out_dir, "probe_results.json"), "w"), indent=1)
    print("\nMacro AUC averaged over dimensions (Table A7 analogue, probe model = this run):")
    for lab in args.labellers.split(","):
        vals = [v["macro_auc"] for k, v in results["probes"].items() if k.startswith(lab + "/")]
        und = sum(v is None for v in vals)
        print(f"  {lab:9s} {np.mean([v for v in vals if v is not None]):.3f}  ({und} undefined of {len(vals)})")


if __name__ == "__main__":
    main()
