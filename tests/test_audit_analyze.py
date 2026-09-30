import numpy as np

from audit import analyze
from utils.io import REPO_ROOT


def test_every_target_has_a_cell_and_matched_pair():
    targets = analyze.read_csv(REPO_ROOT / "reports" / "llama31_tefo_probe_transfer" / "targets.csv")
    assert len(targets) == 15
    for t in targets:
        cell, source = analyze.benchmark_cell(t["eval_dataset"])
        assert cell in analyze.CELLS and source in ("spec", "inferred") and cell in analyze.CELL_PAIR
    assert analyze.benchmark_cell("ss_framing_minority_balanced_seed0")[0] == "Position-Subjective / Implicit"
    assert analyze.benchmark_cell("aita_yta_validation_minority_balanced_seed0")[0] == "Person-Traits / Explicit"
    assert analyze.benchmark_cell("aita_nta_og_verdict_nta_minority_balanced_seed0") == ("Position-Subjective / Explicit", "inferred")


def test_terciles_partition_rows_and_bootstrap_brackets_point():
    rng = np.random.default_rng(0)
    y = np.arange(90) % 2
    s = y + rng.normal(size=90)
    length = rng.integers(1, 500, size=90)
    terc = analyze.tercile_aurocs(y, s, length)
    assert sum(t["n"] for t in terc) == 90
    lo, hi = analyze.bootstrap_ci(y, s, n_boot=200, seed=0)
    from utils import probes

    assert lo <= probes.auroc(y, s) <= hi
    assert analyze.bootstrap_ci(y, s, 200, 0) == (lo, hi)
