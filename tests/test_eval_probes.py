import sys

import numpy as np
import pytest

from probe_training import evaluate_probes, train_probes
from probe_training.get_activations import judging_fields, skyline_group
from tests.test_train_probes import fake_activations
from tests.test_train_probes import run as run_contrastive
from utils import probes
from utils.activations import NO_LABEL, read_labels
from utils.io import meta_path, read_json, read_jsonl, write_json


def fake_judged(path, model="org/fake-model", n_groups=30, hidden=8):
    """AYS-like judged npz: 2 rows per question group; syco shifts dim 0, every 7th row has syco None."""
    rng = np.random.default_rng(1)
    n = 2 * n_groups
    syco = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, hidden)).astype(np.float32)
    X[:, 0] += 3.0 * syco
    syco_stored = syco.copy()
    syco_stored[::7] = NO_LABEL
    np.savez(
        path,
        response_L01=X,
        id=np.array([f"ays__{i:03d}" for i in range(n)]),
        benchmark=np.array(["are_you_sure"] * n),
        group=np.array([f"g{i // 2:02d}" for i in range(n)]),
        truncated=np.zeros(n, dtype=bool),
        labels__syco=syco_stored.astype(np.int8),
        labels__flipped=np.where(syco_stored == NO_LABEL, NO_LABEL, 1 - syco_stored).astype(np.int8),
    )
    write_json(meta_path(path), {"model": model, "layers": [1]})
    return X, syco_stored


def train_skyline(tmp_path, monkeypatch, npz):
    monkeypatch.setattr(train_probes, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_probes, "write_meta", lambda *a, **k: None)  # needs a git commit
    argv = ["train_probes", "--activations", str(npz), "--sweep", "sky", "--labels", "all", "--methods", "logistic", "dim",
            "--positions", "response", "--layers", "1", "--C", "1", "--max-iter", "1000",
            "--test-frac", "0.3", "--val-frac", "0", "--seed", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    train_probes.main()
    return tmp_path / "probes" / "fake-model" / "sky"


def evaluate(monkeypatch, sweep_dir, npz, labels=("all",)):
    monkeypatch.setattr(evaluate_probes, "write_meta", lambda *a, **k: None)
    argv = ["evaluate_probes", "--sweep-dir", str(sweep_dir), "--activations", str(npz), "--labels", *labels]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_probes.main()
    return read_jsonl(sweep_dir / "eval" / f"{npz.name.removesuffix('_activations.npz')}.jsonl")


def test_judging_fields_label_none_and_groups():
    base = {"benchmark": "are_you_sure", "truncated": False, "messages": [{"role": "user", "content": "x"}]}
    rows = [
        {**base, "id": "a", "question": "Q1", "labels": {"syco": 1, "turn1_correct": 1}},
        {**base, "id": "b", "question": "Q1", "labels": {"syco": None, "turn1_correct": 0}},
        {**base, "id": "c", "question": "Q2", "labels": {"syco": 0, "turn1_correct": 1}},
    ]
    f = judging_fields(rows)
    assert f["labels__syco"].tolist() == [1, NO_LABEL, 0]
    assert f["group"][0] == f["group"][1] != f["group"][2]
    with pytest.raises(ValueError):
        judging_fields([{**rows[0], "labels": {"syco": True, "turn1_correct": 1}}])
    with pytest.raises(ValueError):
        judging_fields([rows[0], {**rows[2], "labels": {"syco": 0}}])
    ele = {"id": "e", "benchmark": "elephant", "messages": [{"role": "user", "content": "Q1"}]}
    assert skyline_group(ele) == skyline_group({**ele, "id": "f"}) != skyline_group({**ele, "messages": [{"role": "user", "content": "Q2"}]})


def test_read_labels_rejects_other_values(tmp_path):
    np.savez(tmp_path / "x.npz", labels__a=np.array([0, 1, 2], dtype=np.int8))
    with pytest.raises(ValueError):
        read_labels(np.load(tmp_path / "x.npz"))


def test_skyline_excludes_none_and_never_splits_a_group(tmp_path, monkeypatch):
    npz = tmp_path / "ays_judged_activations.npz"
    _, syco = fake_judged(npz)
    out = train_skyline(tmp_path, monkeypatch, npz)
    split = read_json(out / "split.json")
    assert split["group_field"] == "group" and not set(split["train"]) & set(split["test"])
    groups = np.load(npz)["group"]
    ids = np.load(npz)["id"]
    assert set(split["train_ids"]) == set(ids[np.isin(groups, split["train"])].tolist())
    manifest = read_jsonl(out / "manifest.jsonl")
    assert {m["label"] for m in manifest} == {"syco", "flipped"}
    labeled = syco != NO_LABEL
    n_test = int((labeled & np.isin(groups, split["test"])).sum())
    assert all(m["n_train"] + m["n_train_balance_excluded"] + m["n_test"] == labeled.sum()
               and m["n_test"] == n_test and m["n_train_pos"] * 2 == m["n_train"] for m in manifest)
    for label in ("syco", "flipped"):
        fit_ids = split["fit_ids"][label]
        assert len(fit_ids) == len(set(fit_ids))
        assert set(fit_ids) <= set(split["train_ids"])
    assert all(m["test_paired_win_rate"] is None for m in manifest)


def test_skyline_on_own_file_scores_only_test_rows(tmp_path, monkeypatch):
    npz = tmp_path / "ays_judged_activations.npz"
    fake_judged(npz)
    out = train_skyline(tmp_path, monkeypatch, npz)
    manifest = {m["probe_id"]: m for m in read_jsonl(out / "manifest.jsonl")}
    rows = evaluate(monkeypatch, out, npz)
    assert len(rows) == len(manifest) * 2
    for r in rows:
        m = manifest[r["probe_id"]]
        if r["eval_label"] == m["label"]:  # same rows, same label: must reproduce the held-out metrics
            assert r["n"] == m["n_test"] and r["n_pos"] == m["n_test_pos"]
            assert r["auroc"] == pytest.approx(m["test_auroc"])
            assert r["balanced_accuracy"] == pytest.approx(m["test_balanced_accuracy"])


def test_contrastive_probe_on_judged_labels(tmp_path, monkeypatch):
    train_npz = tmp_path / "c_activations.npz"
    fake_activations(train_npz)
    sweep = run_contrastive(tmp_path, monkeypatch, train_npz, "--cells", "0")
    npz = tmp_path / "ays_judged_activations.npz"
    X, syco = fake_judged(npz)
    rows = evaluate(monkeypatch, sweep, npz, labels=("syco", "flipped"))
    keep = syco != NO_LABEL
    for r in rows:
        assert r["n"] == keep.sum() and r["n_excluded_none"] == (~keep).sum() and r["n_excluded_train"] == 0
        assert r["n_pos"] == (syco == (1 if r["eval_label"] == "syco" else 0)).sum()
        assert (r["auroc"] > 0.9) if r["eval_label"] == "syco" else (r["auroc"] < 0.1)
    m = read_jsonl(sweep / "manifest.jsonl")[0]
    probe = probes.load_probe(np.load(sweep / m["file"]), m["probe_id"], m["method"])
    expected = probes.auroc(syco[keep], probes.score(m["method"], probe, X[keep]))
    got = next(r for r in rows if r["probe_id"] == m["probe_id"] and r["eval_label"] == "syco")
    assert got["auroc"] == pytest.approx(expected)


def test_model_mismatch_and_missing_layer_raise(tmp_path, monkeypatch):
    train_npz = tmp_path / "c_activations.npz"
    fake_activations(train_npz)
    sweep = run_contrastive(tmp_path, monkeypatch, train_npz, "--cells", "0")
    other = tmp_path / "other_activations.npz"
    fake_judged(other, model="org/other-model")
    with pytest.raises(ValueError, match="probes are for"):
        evaluate(monkeypatch, sweep, other)
    fewer = tmp_path / "fewer_activations.npz"
    fake_judged(fewer)
    z = dict(np.load(fewer))
    z["response_L02"] = z.pop("response_L01")
    np.savez(fewer, **z)
    write_json(meta_path(fewer), {"model": "org/fake-model", "layers": [2]})
    with pytest.raises(ValueError, match="lacks"):
        evaluate(monkeypatch, sweep, fewer)
