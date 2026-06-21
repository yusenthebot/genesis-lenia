"""Coordination invariants: a team evolves a division of labour (distinct-site coverage)."""

import numpy as np

from genesis.coordinate import N, NPARAMS, _unpack, team_sites, coverage, evolve_es


def test_unpack_and_coverage():
    W1, b1, W2, b2 = _unpack(np.zeros(NPARAMS))
    assert W2.shape == (N, 12)
    assert 0 < coverage(np.zeros(NPARAMS)) <= 1.0


def test_division_of_labour_emerges():
    """Evolved: each role covers a distinct site (full coverage). Ablated (identical) -> 1/N."""
    theta, _ = evolve_es(gens=110, seed=0)
    assert coverage(theta) > 0.9                       # team covers (almost) all sites
    assert len(set(team_sites(theta))) == N            # a permutation: every site taken once
    assert coverage(theta, ablate=True) <= 1.0 / N + 1e-9   # no role distinction -> all collide
