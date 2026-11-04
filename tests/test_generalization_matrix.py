import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from interp.cache_activations import cache_spans
from interp.generalization_matrix import PROBE_POSITION, assign_folds, contrastive_probes, unit_rows, weighted_auroc
from utils import probes
from utils.activations import position_spans
from utils.io import write_jsonl


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
    assert excluded["AITA-NTA-FLIP/both_nta"] == {"original_telling": 1}


def test_probe_positions_name_the_same_spans_as_cache_positions():
    for prompt_len, answer_start, resp_end in [(10, 10, 40), (10, 10, 13), (5, 5, 6)]:
        cache = cache_spans(prompt_len, answer_start, resp_end)
        trained = position_spans(prompt_len, resp_end, answer_start)
        assert all(cache[c] == trained[p] for c, p in PROBE_POSITION.items())


def test_contrastive_probes_keep_dim_and_logistic_at_C(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 8)), np.repeat([0, 1], 20)
    X[y == 1] += 1.0
    fitted, manifest = {}, []
    for method, position, C, kind in [("logistic", "last_prompt", 0.01, "contrastive"), ("logistic", "last_prompt", 1.0, "contrastive"),
                                      ("dim", "first5", None, "contrastive"), ("dim", "response", None, "skyline")]:
        pid = f"pair00__{probes.probe_key(method, position, 3)}" + ("" if C is None else f"__C{C:g}")
        fitted[pid] = probes.fit(method, X, y, C, 1000)
        manifest.append({"probe_id": pid, "file": "pair00.probes.npz", "probe_type": kind, "method": method,
                         "position": position, "layer": 3, "C": C, "model": "m"})
    probes.save_probes(tmp_path / "pair00.probes.npz", fitted)
    write_jsonl(tmp_path / "manifest.jsonl", manifest)
    out, counts = contrastive_probes(tmp_path, 0.01, "m")
    assert sorted(out) == [("prompt_last", 3), ("response_first5", 3)]
    (source, method, probe), = out["prompt_last", 3]
    assert (source, method) == (f"{tmp_path.name}:pair00", "logistic")
    assert np.allclose(probes.score(method, probe, X), probes.score("logistic", fitted[manifest[0]["probe_id"]], X))
    assert counts == {"n_in": 4, "excluded": {"other_C": 1, "not_contrastive": 1}, "n_out": 2}
    with pytest.raises(ValueError, match="not among"):
        contrastive_probes(tmp_path, 0.1, "m")
    with pytest.raises(ValueError, match="probes are for"):
        contrastive_probes(tmp_path, 0.01, "other")
