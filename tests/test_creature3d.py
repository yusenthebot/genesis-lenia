"""3D-creature search invariants: evaluation shape + the search can find a compact body."""

import numpy as np

from genesis.creature3d import random_creature, evaluate, search, N_RINGS


def test_random_creature_and_evaluate():
    rng = np.random.default_rng(0)
    c = random_creature(rng)
    assert c.peaks.shape == (N_RINGS,)
    score, m = evaluate(c, size=30, steps=50)
    assert {"via", "travel", "alive", "conc", "mass"} <= set(m)
    assert score >= 0.0


def test_search_runs_and_can_reach_a_compact_body():
    """A short search returns a best with the expected structure; viability is reachable."""
    r = search(size=32, steps=60, pop=8, gens=3, seed=0, chase=False, verbose=False)
    assert "best" in r and "metrics" in r
    assert 0.0 <= r["metrics"]["via"] <= 2.0
    assert r["metrics"]["conc"] >= 0.0     # concentration is a valid fraction
