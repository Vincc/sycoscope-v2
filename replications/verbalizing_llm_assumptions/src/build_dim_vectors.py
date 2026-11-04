"""Difference-in-means (DiM) steering vectors from verbalized assumptions, as an alternative to the ridge probes.

Two constructions, both at the layer the matching ridge probe selected (so steering is compared at the same layer):
  label: mean(train prompt reps with the labeller's score > hi) - mean(score < lo), per (labeller, dimension).
  cf:    label-free, per dimension: mean over base prompts of rep(prompt + Table A5 positive instruction)
         - rep(prompt + negative instruction), from the counterfactual-prompting reps. Same direction for every
         labeller of a dimension except for the layer, which follows that labeller's ridge probe.
Raw DiM on these reps is dominated by the mean activation direction (mean pooling without a chat template, so the
BOS token's share, i.e. prompt length, moves every rep along it): raw label DiM has cos ~ -0.7 with the mean rep and
macro AUC 0.56. So by default the mean direction and the top --remove_pcs principal components of the labeller's
train reps at that layer are projected out of both DiMs (k = 10: AUC 0.64, cos with ridge +0.46).
Diagnostics per probe: cosine with the ridge weights, and test macro AUC of the projection (upstream quantile
definition, same as results/probes_a10_quantile) next to the ridge AUC.
Writes <out_dir>/<variant>/probes/<labeller>_<dim>.npz with w (raw DiM, steer_generate.py unit-normalises it),
b = 0, layer, and <out_dir>/dim_results.json.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import DIMS, LABELLERS  # noqa: E402
from train_probes import PAPER_DATASETS, auc_by_dataset, features, load_rows  # noqa: E402


def macro_auc(reps, row_index, test, dim, L, w):
    s = features(reps, row_index, test, L) @ w
    y = np.array([r["scores"][dim] for r in test])
    by_ds = auc_by_dataset(y, s, np.array([r["dataset"] for r in test]), 0.3, 0.7, quantile=True)
    vals = [v["auc"] for v in by_ds.values() if v["auc"] is not None]
    return float(np.mean(vals)) if len(vals) == len(PAPER_DATASETS) else None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--cf_reps", required=True)
    ap.add_argument("--cf_prompts", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--probe_dir", required=True, help="ridge probes (layer source and comparison)")
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--remove_pcs", type=int, default=10, help="project out mean direction + top-k train PCs; -1 = raw DiM")
    ap.add_argument("--min_group", type=int, default=30, help="fail if a label group is smaller than this")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    reps = np.load(args.reps)
    row_index = {i: k for k, i in enumerate(reps["ids"].tolist())}
    cf = np.load(args.cf_reps)
    cf_index = {i: k for k, i in enumerate(cf["ids"].tolist())}
    pairs = defaultdict(dict)
    for r in map(json.loads, open(args.cf_prompts, encoding="utf-8")):
        pairs[(r["dim"], r["base_id"])][r["polarity"]] = cf_index[r["id"]]
    assert all(set(v) == {"pos", "neg"} for v in pairs.values())
    cf_cache, nuisance = {}, {}

    for v in ("label", "cf"):
        os.makedirs(os.path.join(args.out_dir, v, "probes"), exist_ok=True)
    res = {"args": vars(args), "probes": {}}
    print(f"{'probe':40s} L  n_hi  n_lo  cos(lab,ridge) cos(cf,ridge) cos(lab,cf)  AUC ridge/lab/cf")
    for lab in LABELLERS:
        for group, dims in DIMS.items():
            train = load_rows(os.path.join(args.prepared, f"{lab}_{group}_train.jsonl"))
            test = [r for r in load_rows(os.path.join(args.prepared, f"{lab}_{group}_test.jsonl")) if r["dataset"] in PAPER_DATASETS]
            for dim in dims:
                name = f"{lab}_{dim}"
                ridge = np.load(os.path.join(args.probe_dir, f"{name}.npz"))
                L, w_ridge = int(ridge["layer"]), ridge["w"].astype(np.float64)
                X = features(reps, row_index, train, L).astype(np.float64)
                y = np.array([r["scores"][dim] for r in train])
                hi, lo = y > args.hi, y < args.lo
                if min(hi.sum(), lo.sum()) < args.min_group:
                    raise ValueError(f"{name}: label groups too small (hi {hi.sum()}, lo {lo.sum()})")
                w_lab = X[hi].mean(0) - X[lo].mean(0)
                if args.remove_pcs >= 0 and (lab, group, L) not in nuisance:
                    mu = X.mean(0)
                    sub = np.random.default_rng(0).choice(len(X), min(4000, len(X)), replace=False)
                    Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)[2][:args.remove_pcs]
                    nuisance[(lab, group, L)] = np.linalg.qr(np.vstack([mu / np.linalg.norm(mu), Vt]).T)[0].T
                U = nuisance.get((lab, group, L))
                clean = (lambda w: w - U.T @ (U @ w)) if U is not None else (lambda w: w)
                w_lab = clean(w_lab)
                if (dim, L) not in cf_cache:
                    keys = sorted(k for k in pairs if k[0] == dim)
                    Xc = cf[f"layer_{L}"].astype(np.float64)
                    cf_cache[(dim, L)] = (np.mean([Xc[pairs[k]["pos"]] - Xc[pairs[k]["neg"]] for k in keys], 0), len(keys))
                w_cf, n_cf = cf_cache[(dim, L)]
                w_cf = clean(w_cf)
                for v, w in (("label", w_lab), ("cf", w_cf)):
                    np.savez(os.path.join(args.out_dir, v, "probes", f"{name}.npz"), w=w.astype(np.float32), b=np.float32(0), layer=L)
                auc = {k: macro_auc(reps, row_index, test, dim, L, w) for k, w in (("ridge", w_ridge), ("label", w_lab), ("cf", w_cf))}
                res["probes"][name] = {
                    "layer": L, "n_hi": int(hi.sum()), "n_lo": int(lo.sum()), "n_cf_pairs": n_cf,
                    "norm_label": float(np.linalg.norm(w_lab)), "norm_cf": float(np.linalg.norm(w_cf)),
                    "cos_label_ridge": cos(w_lab, w_ridge), "cos_cf_ridge": cos(w_cf, w_ridge), "cos_label_cf": cos(w_lab, w_cf),
                    "macro_auc_quantile": auc,
                }
                p = res["probes"][name]
                f = lambda a: "undef" if a is None else f"{a:.3f}"  # noqa: E731
                print(f"{name:40s} {L:2d} {p['n_hi']:5d} {p['n_lo']:5d}  {p['cos_label_ridge']:+.3f}         {p['cos_cf_ridge']:+.3f}        "
                      f"{p['cos_label_cf']:+.3f}      {f(auc['ridge'])}/{f(auc['label'])}/{f(auc['cf'])}")
    json.dump(res, open(os.path.join(args.out_dir, "dim_results.json"), "w"), indent=1)
    P = res["probes"].values()
    m = lambda k, s: np.mean([p["macro_auc_quantile"][s] for p in P if p["macro_auc_quantile"][s] is not None])  # noqa: E731
    print(f"\nmean macro AUC  ridge {m(0, 'ridge'):.3f}  label-DiM {m(0, 'label'):.3f}  cf-DiM {m(0, 'cf'):.3f}")
    print(f"mean cos  label/ridge {np.mean([p['cos_label_ridge'] for p in P]):+.3f}  cf/ridge {np.mean([p['cos_cf_ridge'] for p in P]):+.3f}  "
          f"label/cf {np.mean([p['cos_label_cf'] for p in P]):+.3f}")


if __name__ == "__main__":
    main()
