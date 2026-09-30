import sys

import numpy as np

from audit import fit_directions, package_directions
from tests.test_eval_probes import evaluate, fake_judged
from utils import probes
from utils.io import read_json, write_json, write_jsonl


def fake_sums(tmp_path, n_layers=2, hidden=8):
    """extract_source-like npz: class sums over fit rows and cached val/test rows; label shifts dim 0."""
    rng = np.random.default_rng(0)
    n = 200
    y = np.arange(n) % 2
    X = rng.normal(size=(n, n_layers, hidden))
    X[:, :, 0] += 2.0 * y[:, None]
    split = np.array(["fit"] * 120 + ["val"] * 40 + ["test"] * 40)
    fit = split == "fit"
    arrays = {"sum__p__u__1": X[fit & (y == 1)].sum(0), "sum__p__u__0": X[fit & (y == 0)].sum(0),
              "count__p__u__1": np.int64((fit & (y == 1)).sum()), "count__p__u__0": np.int64((fit & (y == 0)).sum()),
              "X__p": X[~fit].astype(np.float32), "split": split, "cached": ~fit, "labels__u": y.astype(np.int8)}
    path = tmp_path / "src.npz"
    np.savez(path, **arrays)
    return path, X, y, fit


def test_directions_from_sums_match_fit_dim(tmp_path):
    path, X, y, fit = fake_sums(tmp_path)
    d, thr, diff, _ = fit_directions.directions_from_sums(np.load(path), "p", "u")
    for L in range(X.shape[1]):
        ref = probes.fit_dim(X[fit, L], y[fit])
        assert np.allclose(d[L], ref["direction"]) and np.isclose(thr[L], ref["threshold"])


def test_packaged_sweep_scores_through_evaluate_probes(tmp_path, monkeypatch):
    npz = tmp_path / "ays_activations.npz"
    Xj, syco = fake_judged(npz)  # response_L01, hidden 8, model org/fake-model
    rng = np.random.default_rng(3)
    d = rng.normal(size=(2, 8))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    directions = {"u__direction": d, "u__threshold": np.array([0.1, -0.2])}
    table = [{"unit": "u", "layer": L, "selected": L == 1, "select_rule": "val", "val_auroc": 0.9, "test_auroc": 0.8,
              "n_test": 40} for L in range(2)]
    info = {"model": "org/fake-model", "audit_method": "fake", "cell": "c", "native_position": "response mean",
            "native_matches_cached_position": "response", "source_repo": "r", "source_commit": "abc"}
    probe_sets, manifest = package_directions.build_sweep(directions, table, ["u"], [], ["response"], info)
    sweep = tmp_path / "audit_fake"
    sweep.mkdir()
    probes.save_probes(sweep / "u.probes.npz", probe_sets["u"])
    write_jsonl(sweep / "manifest.jsonl", manifest)
    write_json(sweep / "split.json", package_directions.empty_split())
    assert manifest[0]["position_matches_native"] and manifest[0]["C"] is None

    rows = evaluate(monkeypatch, sweep, npz, labels=("syco",))
    assert len(rows) == 1 and rows[0]["n_excluded_train"] == 0
    keep = syco != -1
    s = Xj @ d[1] - 0.2 * -1
    assert np.isclose(rows[0]["auroc"], probes.auroc(syco[keep].astype(int), s[keep]))
    assert rows[0]["probe_id"] == "u__dim__response_L01"
