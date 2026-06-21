"""Open-endedness invariants: simulate, binning, and illumination beats convergence."""

import numpy as np

from genesis.openended import (
    simulate, behaviour_cell, map_elites, GRID, MOVE_MAX,
)
from genesis.evolve import random_individual


def test_simulate_returns_metrics():
    ind = random_individual(np.random.default_rng(0), near_orbium=True)
    m, snap = simulate(ind, shape=(48, 48), T=60)
    assert "drift_speed" in m and "mean_mass" in m
    assert snap.shape == (48, 48)


def test_behaviour_cell_in_grid_or_none():
    ind = random_individual(np.random.default_rng(1), near_orbium=True)
    m, _ = simulate(ind, shape=(48, 48), T=60)
    cell = behaviour_cell(m)
    assert cell is None or (0 <= cell[0] < GRID and 0 <= cell[1] < GRID)


def test_illumination_finds_more_diversity_than_convergence():
    """A short MAP-Elites run retains many distinct behaviours, not just one."""
    archive, cov = map_elites(n_eval=140, seed=0, shape=(56, 56), T=80)
    assert len(archive) > 8                 # a real zoo, not a single converged type
    assert cov[-1][1] == len(archive)       # coverage curve tracks the archive
