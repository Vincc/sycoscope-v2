import numpy as np
import pytest

from probe_analysis.nosys.compute_nosys import selected_probes, within_prompt_spread
from utils import probes
from utils.io import write_jsonl


def test_selected_probes_takes_best_val_auroc_and_smaller_c_on_ties(tmp_path):
    rows, fitted = [], {}
    for C, val in ((0.01, 0.8), (0.1, 0.9), (1.0, 0.9)):
        pid = f"pair00__logistic__response_mean_L03__C{C:g}"
        rows.append({"probe_id": pid, "file": "pair00.probes.npz", "pair_index": 0, "method": "logistic",
                     "position": "response_mean", "layer": 3, "C": C, "val_auroc": val})
        fitted[pid] = {"mean": np.zeros(2), "scale": np.ones(2), "coef": np.ones(2), "intercept": 0.0,
                       "direction": np.ones(2) / np.sqrt(2)}
    rows.append({"probe_id": "pair00__dim__response_mean_L03", "file": "pair00.probes.npz", "pair_index": 0,
                 "method": "dim", "position": "response_mean", "layer": 3, "C": None, "val_auroc": 0.7})
    fitted["pair00__dim__response_mean_L03"] = {"direction": np.array([1.0, 0.0]), "threshold": 0.0}
    probes.save_probes(tmp_path / "pair00.probes.npz", fitted)
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    sel = selected_probes(tmp_path)
    assert sel[(0, "logistic", "response_mean", 3)]["C"] == 0.1
    assert sel[(0, "dim", "response_mean", 3)]["probe"]["threshold"] == 0.0
    write_jsonl(tmp_path / "manifest.jsonl", [{**rows[0], "val_auroc": None}])
    with pytest.raises(ValueError, match="no validation AUROC"):
        selected_probes(tmp_path)


def test_within_prompt_spread():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    pids = np.array(["a", "a", "b", "b"])
    assert within_prompt_spread(x, pids) == 0.0
    x[1] = [1.0, 0.2]  # mean of a is (1, 0.1): each row is 0.1 away, |mean| = sqrt(1.01)
    assert within_prompt_spread(x, pids) == pytest.approx(0.1 / np.sqrt(1.01))
