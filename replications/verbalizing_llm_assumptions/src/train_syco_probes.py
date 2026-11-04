"""Sycophancy-probe baseline, training (paper Section 3.1, Table A8 "NR", Fig. A16; upstream
sycophancy_probes/train_probe_bin.py). Per (response model, sycophancy dimension): stratified 80/20 split of
the train rows (seed), LogisticRegression(C, class_weight="balanced") per layer, best layer by validation F1;
test AUC per dataset with sigmoid scores, macro-averaged. Writes <out_dir>/probes/<model>_<dim>.npz (w, b,
layer; same format as the assumption probes, so steer_generate.py can use them) and probe_results.json."""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_syco_data import MODELS, SYCO_DIMS  # noqa: E402


def load_reps(paths):
    """Concatenate several extract_reps.py outputs (same layers); returns ({layer: X}, {id: row})."""
    parts = [np.load(p) for p in paths]
    layers = sorted(int(k.removeprefix("layer_")) for k in parts[0].files if k.startswith("layer_"))
    assert all(sorted(int(k.removeprefix("layer_")) for k in p.files if k.startswith("layer_")) == layers for p in parts)
    ids = np.concatenate([p["ids"] for p in parts]).tolist()
    assert len(set(ids)) == len(ids), "duplicate ids across reps files"
    X = {L: np.concatenate([p[f"layer_{L}"] for p in parts]) for L in layers}
    return X, {i: k for k, i in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True, nargs="+")
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--C", type=float, required=True, help="upstream: 1.0")
    ap.add_argument("--seed", type=int, required=True, help="upstream: 0")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    X, row = load_reps(args.reps)
    layers = sorted(X)
    os.makedirs(os.path.join(args.out_dir, "probes"), exist_ok=True)
    results = {"reps": args.reps, "layers": layers, "C": args.C, "seed": args.seed, "probes": {}}
    for short in MODELS:
        train = [json.loads(l) for l in open(os.path.join(args.prepared, f"{short}_train.jsonl"), encoding="utf-8")]
        test_all = [json.loads(l) for l in open(os.path.join(args.prepared, f"{short}_test.jsonl"), encoding="utf-8")]
        train_ids = {r["id"] for r in train}
        test = [r for r in test_all if r["id"] not in train_ids]  # upstream keeps these few rows; we exclude them
        for dim in SYCO_DIMS:
            y = np.array([r["labels"][dim] for r in train])
            tr, va = train_test_split(np.arange(len(train)), test_size=0.2, random_state=args.seed, stratify=y)
            per_layer, best = {}, None
            for L in layers:
                F = X[L][[row[r["id"]] for r in train]]
                clf = LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced", random_state=args.seed).fit(F[tr], y[tr])
                f1 = float(f1_score(y[va], clf.predict(F[va])))
                auc = float(roc_auc_score(y[va], clf.decision_function(F[va])))
                per_layer[L] = {"f1": f1, "auc": auc}
                if best is None or f1 > best[1]:
                    best = (L, f1, clf)
            L, f1, clf = best
            Ft = X[L][[row[r["id"]] for r in test]]
            s = 1 / (1 + np.exp(-(Ft @ clf.coef_[0] + clf.intercept_[0])))
            yt = np.array([r["labels"][dim] for r in test])
            ds = np.array([r["dataset"] for r in test])
            by_ds = {}
            for d in sorted(set(ds)):
                m = ds == d
                by_ds[d] = {"n": int(m.sum()), "n_pos": int(yt[m].sum()),
                            "auc": float(roc_auc_score(yt[m], s[m])) if 0 < yt[m].sum() < m.sum() else None}
            defined = [v["auc"] for v in by_ds.values() if v["auc"] is not None]
            macro = float(np.mean(defined)) if len(defined) == len(by_ds) else None
            np.savez(os.path.join(args.out_dir, "probes", f"{short}_{dim}.npz"), w=clf.coef_[0].astype(np.float32),
                     b=np.float32(clf.intercept_[0]), layer=L)
            results["probes"][f"{short}/{dim}"] = {"layer": L, "val_f1": f1, "val_by_layer": per_layer, "n_train": int(len(tr)),
                                                   "n_val": int(len(va)), "n_test": len(test), "n_test_dropped_in_train": len(test_all) - len(test),
                                                   "test_auc_overall": float(roc_auc_score(yt, s)), "auc_by_dataset": by_ds, "macro_auc": macro}
            print(f"{short:9s} {dim:12s} layer {L:2d} valF1 {f1:.3f} overallAUC {results['probes'][f'{short}/{dim}']['test_auc_overall']:.3f} "
                  f"macroAUC {'undef' if macro is None else f'{macro:.3f}'}  " + " ".join(f"{d}={v['auc']:.2f}" if v["auc"] is not None else f"{d}=undef" for d, v in by_ds.items()))
    json.dump(results, open(os.path.join(args.out_dir, "probe_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
