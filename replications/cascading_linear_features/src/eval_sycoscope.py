"""Evaluate the original CLiF detector on binary SycoScope benchmarks."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from eval_ood import fit
from select_and_probe import monotone_sets, split_ids
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def metrics(y, score, pred):
    assert set(np.unique(y)) == {0, 1}
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_extreme_classes": float(balanced_accuracy_score(y, pred)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--git-dirty", action="store_true")
    args = ap.parse_args()

    train = np.load(args.train_features)
    feats, resid, ids, levels = (train[k] for k in ("feats", "resid", "ids", "levels"))
    train_ids, _ = split_ids(ids, args.test_frac, args.seed)
    pos, neg, _ = monotone_sets(feats, levels, ids, train_ids)
    mask = pos | neg
    keep = np.isin(levels, [-3, 0, 3]) & np.isin(ids, train_ids)
    classes = np.array([-3, 0, 3])
    X_train = feats[keep].astype(np.float32)
    y_train = levels[keep]
    assert np.array_equal(np.unique(y_train), classes)
    sae_scaler, sae_lr, sae_svm = fit(X_train[:, mask], y_train)
    resid_scaler, resid_lr, resid_svm = fit(resid[keep], y_train)

    results = {
        "method": "Original three-class CLiF probes trained on Anthropic levels {-3,0,+3}. Binary AUROC uses ordered class expectation for LR and the +3 minus -3 decision margin for SVM. Balanced accuracy maps predicted +3 to positive and -3/0 to negative.",
        "train": {
            "features": str(args.train_features),
            "n_prompts": int(len(train_ids)),
            "n_rows": int(keep.sum()),
            "n_clif_features": int(mask.sum()),
        },
        "datasets": {},
    }
    paths = sorted(p for p in args.eval_dir.glob("*.npz") if p.is_file())
    assert paths
    for path in paths:
        z = np.load(path)
        y = z["labels"]
        assert np.array_equal(z["feature_indices"], np.flatnonzero(mask)), path
        X = z["feats"].astype(np.float32)
        R = z["resid"]
        lr_prob = sae_lr.predict_proba(sae_scaler.transform(X))
        lr_score = lr_prob @ classes
        svm_decision = sae_svm.decision_function(sae_scaler.transform(X))
        svm_score = svm_decision[:, 2] - svm_decision[:, 0]
        resid_prob = resid_lr.predict_proba(resid_scaler.transform(R))
        resid_score = resid_prob @ classes
        resid_svm_decision = resid_svm.decision_function(resid_scaler.transform(R))
        resid_svm_score = resid_svm_decision[:, 2] - resid_svm_decision[:, 0]
        results["datasets"][path.stem] = {
            "clif_lr": metrics(y, lr_score, sae_lr.predict(sae_scaler.transform(X)) == 3),
            "clif_svm": metrics(y, svm_score, sae_svm.predict(sae_scaler.transform(X)) == 3),
            "residual_lr": metrics(y, resid_score, resid_lr.predict(resid_scaler.transform(R)) == 3),
            "residual_svm": metrics(y, resid_svm_score, resid_svm.predict(resid_scaler.transform(R)) == 3),
            "clif_lr_predicted_levels": {
                str(level): int((sae_lr.predict(sae_scaler.transform(X)) == level).sum()) for level in classes
            },
        }
        print(path.stem, "CLiF LR AUROC", f"{results['datasets'][path.stem]['clif_lr']['auroc']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    meta_dir = args.out.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    inputs = [{"path": str(args.train_features), "sha256": sha256(args.train_features)}]
    inputs.extend({"path": str(path), "sha256": sha256(path)} for path in paths)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
        "arguments": {
            "train_features": str(args.train_features),
            "eval_dir": str(args.eval_dir),
            "test_frac": args.test_frac,
            "seed": args.seed,
            "out": str(args.out),
        },
        "inputs": inputs,
        "output": {"path": str(args.out), "sha256": sha256(args.out)},
        "row_counts": {
            "datasets_in": len(paths),
            "datasets_excluded": 0,
            "datasets_out": len(results["datasets"]),
            "evaluation_rows": sum(v["clif_lr"]["n"] for v in results["datasets"].values()),
        },
    }
    assert meta["row_counts"]["datasets_in"] == meta["row_counts"]["datasets_out"]
    (meta_dir / f"{args.out.name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
