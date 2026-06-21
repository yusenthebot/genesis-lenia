"""Harder-ToM invariants: actors detour; the observer beats the position oracle mid-detour."""

import numpy as np

from genesis.tom_obstacle import (
    K, T, NPARAMS, GOALS, RO, actor_trajectory, observer_beliefs,
    accuracy_by_step, naive_accuracy_by_step, evolve_es,
)


def test_actor_detours_and_reaches():
    rng = np.random.default_rng(0)
    reached = 0
    for g in range(K):
        for _ in range(20):
            _, path = actor_trajectory(g, rng)
            assert np.linalg.norm(path, axis=1).min() > RO - 1.5   # never deep inside the obstacle
            reached += np.linalg.norm(path[-1] - GOALS[g]) < 10
    assert reached > 0.7 * (K * 20)                                # most runs reach the goal via detour


def test_observer_beats_position_oracle_mid_detour():
    """The genuine ToM: at mid-detour the observer reads intent while the oracle is fooled."""
    theta, _ = evolve_es(gens=130, seed=0)
    acc = accuracy_by_step(theta, seed=777, n=250)
    naive = naive_accuracy_by_step(seed=777, n=250)
    abl = accuracy_by_step(theta, ablate=True, seed=777, n=250)[-1]
    mid = 3 * T // 4
    assert acc[mid - 1] > naive[mid - 1] + 0.15     # beats the position oracle mid-detour
    assert acc.mean() > naive.mean()                # and on average over the trajectory
    assert abl < 0.45                               # ablate the observation -> ~chance (0.33)
