import numpy as np
import pytest

from utils import probes
from utils.io import write_jsonl
from utils.splits import group_split, split_masks


def test_group_split_matches_old_algorithm():
    ids = [f"p{i:03d}" for i in range(50)]
    split = group_split(list(reversed(ids)), test_frac=0.2, seed=0)
    uniq = sorted(ids)
    order = np.random.default_rng(0).permutation(len(uniq))
    assert split["test"] == sorted(uniq[i] for i in order[:10])
    assert split["train"] == sorted(uniq[i] for i in order[10:])


def test_both_polarities_of_a_prompt_land_on_one_side():
    prompt_ids = [f"p{i}" for i in range(20) for _ in range(2)]
    tr, te = split_masks(prompt_ids, group_split(prompt_ids, 0.25, seed=3))
    for i in range(0, 40, 2):
        assert tr[i] == tr[i + 1]
    assert not {p for p, m in zip(prompt_ids, tr) if m} & {p for p, m in zip(prompt_ids, te) if m}


def test_split_masks_rejects_overlap_and_unassigned_groups():
    with pytest.raises(AssertionError, match="both"):
        split_masks(["a", "b"], {"train": ["a", "b"], "test": ["b"]})
    with pytest.raises(AssertionError, match="neither"):
        split_masks(["a", "c"], {"train": ["a"], "test": ["b"]})


def test_score_logistic_equals_sklearn_decision_function():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 16))
    y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
    X[:, 3] = 1.0  # zero-variance column
    probe = probes.fit_logistic(X, y, C=1.0, max_iter=2000)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X), y)
    np.testing.assert_allclose(probes.score_logistic(probe, X), clf.decision_function(scaler.transform(X)), atol=1e-9)


def test_dim_direction_uses_only_training_rows():
    X = np.array([[2.0, 0.0], [0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
    y = np.array([1, 0, 1, 0])
    probe = probes.fit_dim(X, y)
    np.testing.assert_allclose(probe["direction"], [1.0, 0.0])
    assert probe["threshold"] == pytest.approx(1.5)
    X_test = np.array([[2.0, 100.0], [0.0, -100.0]])
    s = probes.score_dim(probe, X_test)
    assert probes.auroc(np.array([1, 0]), s) == 1.0


def test_fit_requires_both_classes():
    X = np.ones((4, 2))
    with pytest.raises(ValueError, match="both classes"):
        probes.fit_dim(X, np.array([1, 1, 1, 1]))
    with pytest.raises(ValueError, match="both classes"):
        probes.fit_logistic(X, np.array([0, 0, 0, 0]), C=1.0, max_iter=100)


def test_auroc_undefined_for_one_class():
    assert probes.auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_contrastive_drop_takes_the_pair_partner(tmp_path):
    from train_contrastive import load_pair_rows

    gen = []
    for p, flags in {"p1": (None, None), "p2": ("repetitive", None), "p3": (None, "refusal"), "p4": (None, None)}.items():
        for pol, label, flag in (("pos", 1, flags[0]), ("neg", 0, flags[1])):
            gen.append({"id": f"s__{pol}__{p}", "prompt_id": p, "label": label, "degenerate": flag})
    write_jsonl(tmp_path / "s.jsonl", gen)
    kept_ids = [r["id"] for r in gen if r["id"] != "s__neg__p4"]
    write_jsonl(tmp_path / "act" / "s.index.jsonl", [{"row": i, "id": rid} for i, rid in enumerate(kept_ids)])
    write_jsonl(tmp_path / "act" / "s.skips.jsonl", [{"id": "s__neg__p4", "reason": "too_long"}])

    kept, rows, counts = load_pair_rows("s", tmp_path / "s.jsonl", tmp_path / "act")
    assert sorted(rows[i["id"]]["prompt_id"] for i in kept) == ["p1", "p1", "p3", "p3"]  # refusal is kept
    assert counts == {
        "n_in": 8,
        "excluded": {"extraction_too_long": 1, "degenerate_repetitive": 1, "pair_partner_dropped": 2},
        "n_out": 4,
    }


def test_skyline_excludes_and_counts_unlabeled_rows():
    from train_skyline import select_labeled

    rows = {
        "a": {"labels": {"v": 1}},
        "b": {"labels": {"v": None}},
        "c": {"labels": {"v": 0}},
        "d": {"labels": {"v": 1}},
    }
    index = [{"row": 0, "id": "a"}, {"row": 1, "id": "b"}, {"row": 2, "id": "c"}]
    kept, counts = select_labeled(rows, index, [{"id": "d", "reason": "too_long"}], "v")
    assert [i["id"] for i in kept] == ["a", "c"]
    assert counts == {"n_in": 4, "excluded": {"extraction_too_long": 1, "unlabeled": 1}, "n_out": 2}
    rows["a"]["labels"]["v"] = "1"
    with pytest.raises(ValueError):
        select_labeled(rows, index, [{"id": "d", "reason": "too_long"}], "v")
