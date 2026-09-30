import sys

import numpy as np

from probe_training import train_probes
from utils import probes
from utils.io import meta_path, read_json, read_jsonl, write_json


def fake_activations(path, n_prompts=20, n_pairs=2, hidden=8, drop_id=None):
    """Contrastive npz where label 1 shifts the first dimension, plus its meta."""
    rng = np.random.default_rng(0)
    fields = {"id": [], "label": [], "polarity": [], "pair_index": [], "cell": [], "pair_type": [], "prompt_id": []}
    for k in range(n_pairs):
        for label, tag in ((1, "pos"), (0, "neg")):
            for p in range(n_prompts):
                row_id = f"pair{k:02d}__{tag}__p{p:02d}"
                if row_id == drop_id:
                    continue
                fields["id"].append(row_id)
                fields["label"].append(label)
                fields["polarity"].append(tag)
                fields["pair_index"].append(k)
                fields["cell"].append(f"cell {k}")
                fields["pair_type"].append("taxonomy")
                fields["prompt_id"].append(f"p{p:02d}")
    X = rng.normal(size=(len(fields["id"]), hidden)).astype(np.float32)
    X[:, 0] += 3.0 * np.array(fields["label"])
    np.savez(path, response_L01=X, **{k: np.array(v) for k, v in fields.items()})
    write_json(meta_path(path), {"model": "org/fake-model", "layers": [1]})


def run(tmp_path, monkeypatch, npz, *extra):
    monkeypatch.setattr(train_probes, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_probes, "write_meta", lambda *a, **k: None)  # needs a git commit
    argv = ["train_probes", "--activations", str(npz), "--sweep", "s", "--methods", "logistic", "dim",
            "--positions", "response", "--layers", "1", "--C", "0.1", "1", "--max-iter", "1000",
            "--test-frac", "0.25", "--seed", "0", *extra]
    if "--val-frac" not in extra:
        argv += ["--val-frac", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    train_probes.main()
    return tmp_path / "probes" / "fake-model" / "s"


def test_pairs_never_split_and_manifest_matches_probes(tmp_path, monkeypatch):
    npz = tmp_path / "x_activations.npz"
    fake_activations(npz)
    out = run(tmp_path, monkeypatch, npz, "--cells", "all")

    split = read_json(out / "split.json")
    assert not set(split["train"]) & set(split["test"])
    assert len(split["train"]) + len(split["test"]) == 20

    manifest = read_jsonl(out / "manifest.jsonl")
    assert len(manifest) == 2 * (2 + 1)  # 2 cells x (logistic at 2 Cs + dim)
    assert all(m["n_test"] == 2 * len(split["test"]) for m in manifest)
    for m in manifest:
        with np.load(out / m["file"]) as z:
            assert f"{m['probe_id']}__direction" in z


def test_val_split_is_disjoint_and_reported(tmp_path, monkeypatch):
    npz = tmp_path / "x_activations.npz"
    fake_activations(npz)
    out = run(tmp_path, monkeypatch, npz, "--cells", "all", "--val-frac", "0.25")

    split = read_json(out / "split.json")
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert len(val) == len(test) == 5 and len(train) == 10
    assert not (train & val or train & test or val & test)
    assert len(split["train_ids"]) == 2 * 2 * len(train) and len(split["val_ids"]) == 2 * 2 * len(val)
    for m in read_jsonl(out / "manifest.jsonl"):
        assert m["n_train"] == 2 * len(train) and m["n_val"] == 2 * len(val) and m["n_test"] == 2 * len(test)
        assert m["val_auroc"] is not None and m["n_val_pairs"] == len(val)


def test_prompt_missing_a_partner_is_dropped(tmp_path, monkeypatch):
    npz = tmp_path / "x_activations.npz"
    fake_activations(npz, drop_id="pair00__neg__p03")
    out = run(tmp_path, monkeypatch, npz, "--cells", "0")
    manifest = read_jsonl(out / "manifest.jsonl")
    assert {m["n_train"] + m["n_test"] for m in manifest} == {2 * 19}
    assert all(m["n_val"] == 0 and m["val_auroc"] is None for m in manifest)


def test_complete_pairs():
    pids = np.array(["a", "a", "b", "c", "c", "d", "d"])
    labels = np.array([1, 0, 1, 1, 1, 0, 1])
    assert train_probes.complete_pairs(pids, labels).tolist() == [True, True, False, False, False, True, True]


def test_paired_win_rate():
    win, n = probes.paired_win_rate(np.array([2.0, 1.0, 0.0, 0.5]), np.array([1, 0, 1, 0]), ["a", "a", "b", "b"])
    assert (win, n) == (0.5, 2)
