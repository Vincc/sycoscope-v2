import numpy as np
import pytest

from interp.persona_judge_classifier import length_bins, score_metrics, top_hit_rate, within_bin_percentile


def test_length_bins_are_equal_count_and_ordered_by_length():
    n_tokens = np.random.default_rng(0).integers(1, 50, 1000)  # many ties
    bins = length_bins(n_tokens, 10)
    assert np.bincount(bins).tolist() == [100] * 10
    for b in range(9):
        assert n_tokens[bins == b].max() <= n_tokens[bins == b + 1].min()


def test_within_bin_percentile_ignores_a_per_bin_offset():
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 5, 400)
    s = rng.normal(size=400)
    shifted = s + 10.0 * bins  # a length effect that moves whole bins
    assert np.array_equal(within_bin_percentile(s, bins), within_bin_percentile(shifted, bins))
    p = within_bin_percentile(s, bins)
    for b in range(5):
        assert np.sort(p[bins == b]) == pytest.approx((np.arange((bins == b).sum()) + 0.5) / (bins == b).sum())


def test_top_hit_rate_shares_tied_places():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([5.0, 4.0, 3.0, 3.0, 3.0, 1.0])
    assert top_hit_rate(y, s, 2) == pytest.approx(0.5)  # 5 and 4
    assert top_hit_rate(y, s, 3) == pytest.approx((1 + 0 + 1 / 3) / 3)  # one place among three rows tied at 3, one of them positive
    assert top_hit_rate(y, s, 5) == pytest.approx(2 / 5)
    with pytest.raises(ValueError):
        top_hit_rate(y, s, 0)


def test_score_metrics_leave_auroc_undefined_with_one_class():
    rng = np.random.default_rng(0)
    s, judge, tokens = rng.normal(size=50), rng.uniform(0, 100, 50), rng.integers(1, 500, 50)
    groups = np.array([f"g{i}" for i in range(50)])
    out = score_metrics(np.zeros(50, dtype=int), s, groups, judge, tokens, 100, 0)
    assert out["auroc"] is None and out["ci_low"] is None and out["top_r_hit_rate"] is None and out["n_pos"] == 0
    y = (s > 0).astype(int)
    out = score_metrics(y, s, groups, judge, tokens, 100, 0)
    assert out["auroc"] == 1.0 and out["top_r_hit_rate"] == 1.0
