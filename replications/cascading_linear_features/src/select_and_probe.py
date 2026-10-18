"""Stage C: CLiF feature selection (Section 3.2) and 3-class detection probes (Section 3.3, Table 1).

Selection uses only training prompts. Probes: logistic regression and linear SVM on levels {-3, 0, +3},
trained on CLiF features vs. all SAE features, plus a residual-stream probe (Appendix E).
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def split_ids(ids, test_frac, seed):
    """Prompt-level train/test split; the same call in every stage keeps the held-out prompts identical."""
    uniq = np.unique(ids)
    rng = np.random.RandomState(seed)
    test_ids = rng.choice(uniq, size=int(round(len(uniq) * test_frac)), replace=False)
    train_ids = np.setdiff1d(uniq, test_ids)
    assert not set(train_ids) & set(test_ids)
    return train_ids, test_ids


def monotone_sets(feats, levels, ids, train_ids):
    m = np.isin(ids, train_ids)
    mean = {l: feats[m & (levels == l)].astype(np.float32).mean(0) for l in range(-3, 4)}
    pos = (mean[0] <= mean[1]) & (mean[1] <= mean[2]) & (mean[2] <= mean[3])
    neg = (mean[0] <= mean[-1]) & (mean[-1] <= mean[-2]) & (mean[-2] <= mean[-3])
    # A feature that is zero at every level is trivially monotone; it carries nothing.
    nonzero = np.stack(list(mean.values())).max(0) > 0
    return pos & nonzero, neg & nonzero, mean


def probe(X, y, train, test, name):
    sc = StandardScaler().fit(X[train])
    Xtr, Xte = sc.transform(X[train]), sc.transform(X[test])
    out = {}
    for cname, clf in (("logreg", LogisticRegression(max_iter=5000, C=1.0)),
                       ("svm", LinearSVC(C=1.0, max_iter=20000))):
        clf.fit(Xtr, y[train])
        out[cname] = float((clf.predict(Xte) == y[test]).mean())
    print(f"{name:28s} dims={X.shape[1]:6d}  logreg={out['logreg']:.3f}  svm={out['svm']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.features)
    feats, resid, ids, levels = z["feats"], z["resid"], z["ids"], z["levels"]
    train_ids, test_ids = split_ids(ids, args.test_frac, args.seed)
    rng = np.random.RandomState(args.seed + 1)  # for the random-subset baseline only

    pos, neg, mean = monotone_sets(feats, levels, ids, train_ids)
    clif = pos | neg
    print(f"prompts train={len(train_ids)} test={len(test_ids)}; F+={pos.sum()} F-={neg.sum()} "
          f"both={int((pos & neg).sum())} CLiF={clif.sum()} of {feats.shape[1]}")

    keep = np.isin(levels, [-3, 0, 3])
    y = levels[keep]
    tr = keep & np.isin(ids, train_ids)
    te = keep & np.isin(ids, test_ids)
    tr, te = tr[keep], te[keep]
    assert sorted(np.unique(y[tr])) == [-3, 0, 3] and sorted(np.unique(y[te])) == [-3, 0, 3]
    X_all = feats[keep].astype(np.float32)
    res = {
        "n_train_prompts": int(len(train_ids)), "n_test_prompts": int(len(test_ids)),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_clif": int(clif.sum()),
        "clif_sae": probe(X_all[:, clif], y, tr, te, "CLiF SAE features"),
        "all_sae": probe(X_all, y, tr, te, "all SAE features"),
        "random_subset_sae": probe(X_all[:, rng.choice(feats.shape[1], clif.sum(), replace=False)],
                                   y, tr, te, "random SAE subset (same k)"),
        "residual_L19": probe(resid[keep], y, tr, te, "residual stream L19"),
    }
    # Spearman-style sanity check: mean CLiF score (sum of F+ minus sum of F-) per level.
    score = feats[:, pos].astype(np.float32).sum(1) - feats[:, neg].astype(np.float32).sum(1)
    res["mean_score_by_level"] = {int(l): float(score[levels == l].mean()) for l in range(-3, 4)}
    print("mean (F+ - F-) score by level:", {k: round(v, 2) for k, v in res["mean_score_by_level"].items()})
    np.save(args.out.replace(".json", "_clif_mask.npy"), np.stack([pos, neg]))
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
