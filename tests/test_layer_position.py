import numpy as np
import pytest

from probe_analysis.layer_position import auroc_se, rank_and_wins, variance_components


def test_variance_components_additive_array_has_no_interaction():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 1, 1)) + rng.normal(size=(1, 4, 1)) + rng.normal(size=(1, 1, 5))
    comps = {r["term"]: r for r in variance_components(a, ("x", "y", "z"))}
    assert sum(comps[t]["share"] for t in ("x", "y", "z")) == pytest.approx(1.0)
    for term in ("x x y", "x x z", "y x z", "higher-order interactions"):
        assert comps[term]["ss"] == pytest.approx(0.0, abs=1e-10)


def test_variance_components_sum_to_total_and_df():
    a = np.random.default_rng(1).normal(size=(2, 3, 4, 5))
    comps = variance_components(a, ("a", "b", "c", "d"))
    assert sum(r["ss"] for r in comps) == pytest.approx(((a - a.mean()) ** 2).sum())
    assert sum(r["df"] for r in comps) == a.size - 1
    two_way = next(r for r in comps if r["term"] == "a x b")
    m = a.mean(axis=(2, 3))
    effect = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()
    assert two_way["ss"] == pytest.approx((effect ** 2).sum() * 20)


def test_auroc_se_at_chance_matches_mann_whitney_null():
    assert auroc_se(0.5, 30, 50) == pytest.approx(np.sqrt((30 + 50 + 1) / (12 * 30 * 50)))


def test_rank_and_wins_split_ties():
    ranks, wins = rank_and_wins(np.array([[0.7, 0.5, 0.7]]))
    assert ranks.tolist() == [[1.5, 3.0, 1.5]]
    assert wins.tolist() == [[0.5, 0.0, 0.5]]
