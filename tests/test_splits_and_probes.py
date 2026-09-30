import numpy as np
import pytest

from utils import probes
from utils.splits import group_split, split_masks


def test_group_split_matches_old_algorithm():
    ids = [f"p{i:03d}" for i in range(50)]
    split = group_split(list(reversed(ids)), test_frac=0.2, val_frac=0, seed=0)
    uniq = sorted(ids)
    order = np.random.default_rng(0).permutation(len(uniq))
    assert split["test"] == sorted(uniq[i] for i in order[:10])
    assert split["train"] == sorted(uniq[i] for i in order[10:])
    assert split["val"] == []


def test_val_split_keeps_test_groups_and_takes_val_from_train():
    ids = [f"p{i:03d}" for i in range(50)]
    old = group_split(ids, test_frac=0.2, val_frac=0, seed=0)
    new = group_split(ids, test_frac=0.2, val_frac=0.1, seed=0)
    assert new["test"] == old["test"] and len(new["val"]) == 5
    assert sorted(new["train"] + new["val"]) == old["train"]
    with pytest.raises(ValueError, match="none for train"):
        group_split(ids[:3], test_frac=0.5, val_frac=0.5, seed=0)


def test_both_polarities_of_a_prompt_land_on_one_side():
    prompt_ids = [f"p{i}" for i in range(20) for _ in range(2)]
    tr, va, te = split_masks(prompt_ids, group_split(prompt_ids, 0.25, 0.25, seed=3))
    for i in range(0, 40, 2):
        assert tr[i] == tr[i + 1] and va[i] == va[i + 1]
    sides = [{p for p, m in zip(prompt_ids, mask) if m} for mask in (tr, va, te)]
    assert not (sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
    assert (tr.astype(int) + va + te == 1).all()


def test_split_masks_rejects_overlap_and_unassigned_groups():
    with pytest.raises(AssertionError, match="both"):
        split_masks(["a", "b"], {"train": ["a", "b"], "val": [], "test": ["b"]})
    with pytest.raises(AssertionError, match="both"):
        split_masks(["a", "b"], {"train": ["a"], "val": ["b"], "test": ["b"]})
    with pytest.raises(AssertionError, match="no side"):
        split_masks(["a", "c"], {"train": ["a"], "val": [], "test": ["b"]})


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
