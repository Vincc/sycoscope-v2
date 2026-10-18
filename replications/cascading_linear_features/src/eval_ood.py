"""Table 1: detection accuracy on the Anthropic test split and the three OOD sets.

Probes are fit on the Anthropic training prompts only (levels -3/0/+3) and applied unchanged to every test set.
Rows: CLiF (all), CLiF level-k (features monotone and strictly increasing/decreasing through level k only),
Contrastive baseline (all SAE features, no filter), random subset, residual stream. LR and SVM each.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from select_and_probe import monotone_sets, split_ids


def level_k_mask(mean, k):
    """F+ restricted to levels 0..k (strict somewhere) union F- restricted to levels 0..-k."""
    pos = np.ones_like(mean[0], dtype=bool)
    neg = np.ones_like(mean[0], dtype=bool)
    for a in range(k):
        pos &= mean[a] <= mean[a + 1]
        neg &= mean[-a] <= mean[-a - 1]
    pos &= mean[k] > mean[0]
    neg &= mean[-k] > mean[0]
    return pos | neg


def fit(X, y):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X), y)
    svm = LinearSVC(C=1.0, max_iter=20000).fit(sc.transform(X), y)
    return sc, lr, svm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--ood_features", nargs="+", required=True, help="stage-B npz files for the OOD cascades")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.train_features)
    feats, resid, ids, levels = z["feats"], z["resid"], z["ids"], z["levels"]
    train_ids, test_ids = split_ids(ids, args.test_frac, args.seed)
    pos, neg, mean = monotone_sets(feats, levels, ids, train_ids)
    rng = np.random.RandomState(args.seed + 1)
    feature_sets = {"clif_all": pos | neg, "contrastive_all_sae": np.ones_like(pos),
                    "random_subset": np.isin(np.arange(len(pos)), rng.choice(len(pos), int((pos | neg).sum()), replace=False))}
    for k in (1, 2, 3):
        feature_sets[f"clif_level{k}"] = level_k_mask(mean, k)

    keep = np.isin(levels, [-3, 0, 3])
    tr = keep & np.isin(ids, train_ids)
    te = keep & np.isin(ids, test_ids)
    sets = {"anthropic_test": (feats[te].astype(np.float32), resid[te], levels[te])}
    for path in args.ood_features:
        o = np.load(path)
        k = o["levels"]
        k = np.isin(k, [-3, 0, 3])
        name = path.split("/")[-1].removeprefix("feats_L19_").removesuffix(".npz")
        for src in np.unique(o["sources"]):
            mm = k & (o["sources"] == src)
            sets[str(src)] = (o["feats"][mm].astype(np.float32), o["resid"][mm], o["levels"][mm])
    for name, (_, _, y) in sets.items():
        assert sorted(np.unique(y)) == [-3, 0, 3], name

    res = {"n_features": {k: int(v.sum()) for k, v in feature_sets.items()}, "n_rows": {k: int(len(v[2])) for k, v in sets.items()}}
    print("features:", res["n_features"]); print("test rows:", res["n_rows"])
    Xtr_all, ytr = feats[tr].astype(np.float32), levels[tr]
    header = f"{'probe':26s}" + "".join(f"{n:>22s}" for n in sets)
    print(header)
    for fname, mask in feature_sets.items():
        sc, lr, svm = fit(Xtr_all[:, mask], ytr)
        for cname, clf in (("LR", lr), ("SVM", svm)):
            row = {}
            for sname, (X, _, y) in sets.items():
                row[sname] = float((clf.predict(sc.transform(X[:, mask])) == y).mean())
            res[f"{fname}+{cname}"] = row
            print(f"{fname + '+' + cname:26s}" + "".join(f"{row[n]:22.3f}" for n in sets))
    sc, lr, svm = fit(resid[tr], ytr)
    for cname, clf in (("LR", lr), ("SVM", svm)):
        row = {s: float((clf.predict(sc.transform(R)) == y).mean()) for s, (_, R, y) in sets.items()}
        res[f"residual_L19+{cname}"] = row
        print(f"{'residual_L19+' + cname:26s}" + "".join(f"{row[n]:22.3f}" for n in sets))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
