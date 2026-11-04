"""Synthetic activations whose labels are a known linear function of one layer: the trainer must recover
that layer with R^2 ~ 1 and AUC 1 on every dataset, which fails if ids and feature rows are misaligned."""
import json
import os
import subprocess
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def test_recovers_planted_direction(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 400, 16
    ids = [f"{i:016x}" for i in range(n)]
    reps = {f"layer_{L}": rng.normal(size=(n, d)).astype(np.float32) for L in (4, 8)}
    w = rng.normal(size=d)
    y = 1 / (1 + np.exp(-(reps["layer_8"] @ w)))  # planted in layer 8 only
    perm = rng.permutation(n)  # the npz row order must not matter
    np.savez(tmp_path / "reps.npz", ids=np.array(ids)[perm], **{k: v[perm] for k, v in reps.items()})
    datasets = ["oeq", "aita", "ir", "factual", "cancer", "wildchat"]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    from prepare_data import DIMS  # noqa: E402

    for group, dims in DIMS.items():
        rows = [{"id": ids[i], "occurrence": 0, "user_text": str(i), "dataset": datasets[i % 6],
                 "scores": {dim: float(y[i]) for dim in dims}} for i in range(n)]
        for split, part in (("train", rows[:300]), ("test", rows[300:])):
            with open(prepared / f"gemini_{group}_{split}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in part)
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(SRC, "train_probes.py"), "--reps", str(tmp_path / "reps.npz"),
                    "--prepared", str(prepared), "--labellers", "gemini", "--alpha", "0.1", "--seed", "42",
                    "--lo", "0.3", "--hi", "0.7", "--out_dir", str(out)], check=True, cwd=SRC)
    res = json.load(open(out / "probe_results.json"))
    for k, v in res["probes"].items():
        assert v["layer"] == 8, (k, v["layer"])
        assert v["val_r2"] > 0.75, (k, v["val_r2"])  # sigmoid target, linear fit
        assert v["macro_auc"] > 0.99, (k, v["auc_by_dataset"])
