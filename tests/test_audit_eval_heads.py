import sys

import numpy as np
import pytest

from audit import evaluate_heads, fit_heads
from audit.filter_control_rows import select_control_test
from utils import probes
from utils.io import meta_path, read_jsonl, write_json, write_jsonl


def test_evaluate_heads_scores_selected_head_slices(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n, width = 60, fit_heads.N_HEADS * 4
    y = np.arange(n) % 2
    X = {L: rng.normal(size=(n, width)).astype(np.float32) for L in (3, 7)}
    X[7][:, 5 * 4] += 3 * y  # layer 7 head 5, first dim carries the label
    labels = y.astype(np.int8)
    labels[::10] = -1
    npz = tmp_path / "bench_heads.npz"
    np.savez(npz, answer_mean_L03=X[3], answer_mean_L07=X[7], id=np.arange(n).astype(str),
             benchmark=np.array(["b"] * n), labels__syco=labels)
    write_json(meta_path(npz), {"model": "m"})
    det_dir = tmp_path / "audit_x"
    det_dir.mkdir()
    heads_list = [[7, 5], [3, 0]]
    feats = evaluate_heads.features(np.load(npz), "answer_mean", heads_list)
    assert feats.shape == (n, 8) and np.array_equal(feats[:, :4], X[7][:, 20:24])
    det = probes.fit_logistic(feats, y, C=1.0, max_iter=1000)
    probes.save_probes(det_dir / "detectors.npz", {"d": det})
    write_jsonl(det_dir / "manifest.jsonl", [{"probe_id": "d", "probe_type": "audit_heads", "method": "logistic",
                                               "layers": [3, 7], "heads": heads_list, "model": "m", "cell": "c"}])
    monkeypatch.setattr(evaluate_heads, "write_meta", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["x", "--detector-dir", str(det_dir), "--activations", str(npz), "--pooling", "answer_mean"])
    evaluate_heads.main()
    (row,) = read_jsonl(det_dir / "eval" / "bench__answer_mean.jsonl")
    keep = labels != -1
    assert row["n"] == keep.sum() and row["n_excluded_none"] == 6
    assert np.isclose(row["auroc"], probes.auroc(y[keep], probes.score_logistic(det, feats)[keep]))


def test_select_control_test_rows():
    split = {"train": ["g1"], "val": ["g2"], "test": ["g3"], "group_field": "prompt_id"}
    rows = [{"id": i, "pair_type": t, "prompt_id": g} for i, (t, g) in
            enumerate([("control", "g3"), ("control", "g1"), ("taxonomy", "g3"), ("control", "g2"), ("control", "g3")])]
    kept, excluded = select_control_test(rows, split)
    assert [r["id"] for r in kept] == [0, 4]
    assert excluded == {"group_in_train_or_val": 2, "pair_type_taxonomy": 1}
    with pytest.raises(ValueError):
        select_control_test([{"id": 9, "pair_type": "control", "prompt_id": "g9"}], split)
