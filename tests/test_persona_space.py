import numpy as np
import pytest

from interp.persona_space import class_differences, pair_rows, prompt_centred_pc1


def test_class_differences_match_class_means_for_every_labeling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    Y = np.stack([rng.permutation(np.r_[np.ones(15), np.zeros(25)]) for _ in range(3)])
    D = class_differences(X, Y)
    for y, d in zip(Y, D):
        assert d == pytest.approx(X[y == 1].mean(0) - X[y == 0].mean(0))


def contrastive_table(n_prompts=4):
    """Rows of two pairs x n_prompts prompts x two sides, in shuffled order."""
    rows = [(p, f"q{q}", side) for p in (0, 1) for q in range(n_prompts) for side in ("sycophantic", "non_sycophantic")]
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    pair_index = np.array([r[0] for r in rows])
    return (np.array([r[2] for r in rows]), pair_index, np.where(pair_index == 0, "Base", "Cell A"),
            np.where(pair_index == 0, "baseline", "taxonomy"), np.array([r[1] for r in rows]))


def test_pair_rows_aligns_the_two_sides_by_prompt():
    polarity, pair_index, cell, pair_type, prompt_id = contrastive_table()
    pairs = pair_rows(polarity, pair_index, cell, pair_type, prompt_id)
    assert list(pairs) == ["00 Base", "01 Cell A"]
    for name, (t, s, n) in pairs.items():
        assert t == ("baseline" if name == "00 Base" else "taxonomy")
        assert (polarity[s] == "sycophantic").all() and (polarity[n] == "non_sycophantic").all()
        assert list(prompt_id[s]) == list(prompt_id[n]) == ["q0", "q1", "q2", "q3"]


def test_pair_rows_needs_both_sides_of_every_prompt():
    polarity, pair_index, cell, pair_type, prompt_id = contrastive_table()
    drop = np.flatnonzero((pair_index == 1) & (prompt_id == "q2") & (polarity == "sycophantic"))[0]
    keep = np.arange(len(polarity)) != drop
    with pytest.raises(ValueError, match="pair 1"):
        pair_rows(polarity[keep], pair_index[keep], cell[keep], pair_type[keep], prompt_id[keep])


def test_prompt_centred_pc1_ignores_prompt_offsets():
    rng = np.random.default_rng(0)
    direction = np.zeros(8)
    direction[3] = 1.0
    prompt_id = np.repeat(np.arange(30), 2)
    side = np.tile([1.0, -1.0], 30)
    X = 10 * rng.normal(size=(30, 8))[prompt_id] + side[:, None] * direction + 0.01 * rng.normal(size=(60, 8))
    pc1, share = prompt_centred_pc1(X, prompt_id)
    assert abs(pc1 @ direction) == pytest.approx(1.0, abs=1e-3)
    assert share > 0.99
