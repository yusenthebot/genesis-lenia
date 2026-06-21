"""Foraging-world invariants: stability, eating, and the agency (taxis) property."""

import numpy as np

from genesis.world import LeniaParams
from genesis.foraging import ForagingWorld
from genesis.metrics import circular_centroid


def _glider_rule():
    # a viable 2D rule (from round 2); exact morphology not needed for these checks
    return LeniaParams(mu_g=0.147, sigma_g=0.0173, mu_k=0.665, sigma_k=0.191,
                       R=13.0, dt=0.1)


def _seed(w, size):
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    w.A = np.clip(np.exp(-0.5 * ((yy - c) ** 2 + (xx - c) ** 2) / 9.0 ** 2), 0, 1)


def test_state_and_food_stay_bounded():
    size = 80
    w = ForagingWorld((size, size), _glider_rule(), sense_sigma=14.0, gamma=6.0)
    _seed(w, size)
    w.add_food_blob((20, 20), radius=8.0)
    for _ in range(120):
        w.step()
    assert 0.0 <= w.A.min() and w.A.max() <= 1.0
    assert np.isfinite(w.A).all()
    assert w.F.min() >= -1e-9            # food never goes negative
    assert w.eaten >= 0.0


def test_eating_reduces_food():
    size = 60
    w = ForagingWorld((size, size), _glider_rule(), sense_sigma=10.0, gamma=0.0,
                      eta=0.3)
    _seed(w, size)
    w.add_food_blob((30, 30), radius=10.0)   # food under the body
    f0 = w.F.sum()
    for _ in range(20):
        w.step()
    assert w.F.sum() < f0
    assert w.eaten > 0.0


def test_roll_drift_conserves_mass():
    """The sensorimotor reflex uses np.roll, so a drift step must not change mass."""
    A = np.random.default_rng(0).random((64, 64))
    shifted = np.roll(A, 3, axis=0)
    # np.roll is an exact permutation; sums match within float reordering error
    assert abs(shifted.sum() - A.sum()) < 1e-9


def test_sensing_steers_toward_food():
    """Agency: with sensing on, the body drifts toward food; with it off, it does not.

    Lenia dynamics are frozen (dt=0) so this isolates the sensorimotor reflex: only
    the sense->drift coupling can move the (otherwise rigid) body.
    """
    size = 90
    food = (18, 45)                      # straight "north" of centre
    frozen = LeniaParams(mu_g=0.15, sigma_g=0.015, mu_k=0.5, sigma_k=0.15,
                         R=13.0, dt=0.0)

    def end_dist(gamma):
        w = ForagingWorld((size, size), frozen, sense_sigma=12.0, gamma=gamma)
        _seed(w, size)
        w.add_food_blob(food, radius=8.0)
        for _ in range(160):
            w.step()
        c = circular_centroid(w.A)
        d = (np.array(food) - c + size / 2) % size - size / 2
        return float(np.hypot(*d))

    assert end_dist(8.0) < end_dist(0.0) - 5.0   # sensing gets it closer to food


def test_metabolism_drains_and_kills_without_food():
    """With a metabolic cost and no food, the energy reserve drains and the body dies."""
    size = 70
    w = ForagingWorld((size, size), _glider_rule(), sense_sigma=12.0, gamma=0.0,
                      decay=0.02, feed=0.05, energy0=0.5)
    _seed(w, size)
    e0 = w.energy
    for _ in range(200):
        w.step()
    assert w.energy < e0                 # metabolism drained the reserve
    assert w.A.sum() < 1e-2 or not w.alive   # starved -> body dissipated


def test_eating_refills_energy():
    """Eating banks energy: with food, the reserve ends higher than starving alone."""
    size = 70

    def final_energy(with_food):
        w = ForagingWorld((size, size), _glider_rule(), sense_sigma=12.0,
                          gamma=0.0, eta=0.4, decay=0.01, feed=0.2, energy0=1.0)
        _seed(w, size)
        if with_food:
            w.add_food_blob((35, 35), radius=12.0, amp=1.0)  # food under the body
        for _ in range(15):
            w.step()
        return w.energy

    assert final_energy(True) > final_energy(False)
