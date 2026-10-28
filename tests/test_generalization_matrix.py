import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from interp.generalization_matrix import assign_folds, unit_rows, weighted_auroc


def test_weighted_auroc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert weighted_auroc(y, s, np.ones((1, 300)))[0] == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 3, 300)
    expanded = np.repeat(np.arange(300), w)  # integer weights = repeating rows
    assert weighted_auroc(y, s, w[None, :].astype(float))[0] == pytest.approx(roc_auc_score(y[expanded], s[expanded]))


def test_folds_keep_each_group_whole_and_are_balanced():
    groups = [f"g{i % 37}" for i in range(500)]
    folds = assign_folds(groups, 5, 0)
    assert set(folds) == set(groups) and sorted(set(folds.values())) == [0, 1, 2, 3, 4]
    sizes = [list(folds.values()).count(k) for k in range(5)]  # groups per fold
    assert max(sizes) - min(sizes) <= 1
    assert assign_folds(groups, 5, 0) == folds


def test_unit_rows_drop_none_labels_and_score_flip_pairs_once():
    rows = [
        {"id": "a", "benchmark": "elephant", "source": "AITA-NTA-FLIP", "prompt_col": "original_post", "truncated": False,
         "labels": {"both_nta__j": 1}},
        {"id": "b", "benchmark": "elephant", "source": "AITA-NTA-FLIP", "prompt_col": "flipped_story", "truncated": False,
         "labels": {"both_nta__j": 1}},
        {"id": "c", "benchmark": "elephant", "source": "OEQ", "truncated": True,
         "labels": {"validation__j": None, "indirectness__j": None, "framing__j": None}},
        {"id": "d", "benchmark": "elephant", "source": "OEQ", "truncated": False,
         "labels": {"validation__j": 0, "indirectness__j": 1, "framing__j": None}},
    ]
    units, excluded = unit_rows(rows, "j")
    assert units == {"AITA-NTA-FLIP/both_nta": [("b", 1)], "OEQ/validation": [("d", 0)], "OEQ/indirectness": [("d", 1)]}
    assert excluded["OEQ/framing"] == {"truncated": 1, "label_none": 1} and excluded["OEQ/validation"] == {"truncated": 1}
