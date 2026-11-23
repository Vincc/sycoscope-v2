import numpy as np
import pytest

from interp.direction_cosine import dim_directions
from utils import probes


def test_dim_directions_match_fit_dim_for_each_label_vector():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    Y = np.stack([rng.permutation(np.r_[np.ones(15, int), np.zeros(45, int)]) for _ in range(4)])
    D = dim_directions(X, Y)
    for y, d in zip(Y, D):
        assert d == pytest.approx(probes.fit_dim(X, y)["direction"], abs=1e-5)  # fit_dim averages in float32
