"""Information-theoretic measure tests (pure functions, deterministic)."""

import numpy as np

from genesis.measure import (
    entropy_bits, mutual_information_bits, windowed_mi, brain_world_series,
)


def test_entropy_balanced_is_one_bit():
    assert entropy_bits([1, 1]) == 1.0
    assert entropy_bits([1, 0]) == 0.0
    assert abs(entropy_bits([3, 1]) - 0.8112781) < 1e-6


def test_mi_identical_sequences_equals_entropy():
    x = np.array([0, 1] * 200)
    assert abs(mutual_information_bits(x, x) - 1.0) < 1e-9   # perfectly informative


def test_mi_constant_target_is_zero():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 500)
    y = np.zeros(500, dtype=int)
    assert mutual_information_bits(x, y) == 0.0


def test_mi_independent_is_near_zero():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 4000)
    y = rng.integers(0, 2, 4000)
    assert mutual_information_bits(x, y) < 0.02            # only finite-sample noise


def test_mi_is_nonnegative():
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.integers(0, 3, 300); y = rng.integers(0, 3, 300)
        assert mutual_information_bits(x, y) >= 0.0


def test_windowed_mi_and_series_shapes():
    x = np.array([0, 1] * 300); y = np.roll(x, 1)
    ts, mi = windowed_mi(x, y, win=100, stride=10)
    assert len(ts) == len(mi)
    assert (mi >= 0).all()
    hist = [(t, t % 2, 0.0, 0.0, 1.0, -1.0, 0.0) for t in range(20)]
    b, w = brain_world_series(hist)
    assert set(np.unique(b)) <= {0, 1} and len(b) == 20
