"""Flow-Lenia creature-search invariants: evaluation shape + the motion negative is robust."""

import numpy as np

from genesis.creature_flow import random_creature, evaluate, search


def test_random_creature_and_evaluate():
    rng = np.random.default_rng(0)
    c = random_creature(rng)
    score, m = evaluate(c, shape=(60, 60), steps=40)
    assert {"travel", "conc", "alive"} <= set(m)
    assert score >= 0.0 and m["travel"] >= 0.0


def test_short_search_runs():
    r = search(pop=8, gens=3, seed=0, shape=(60, 60), steps=40, verbose=False)
    assert "best" in r and "metrics" in r
    assert r["metrics"]["travel"] >= 0.0


def test_single_channel_flow_lenia_does_not_locomote():
    """The documented negative: a quick search stays far below locomotion (travel << 1 width).

    Single-channel Flow-Lenia uses a gradient flow (F=grad(G)) which relaxes to equilibrium, so
    the best evolved creature drifts only a fraction of its own radius — not locomotion. If this
    ever fails (best travel jumps high), the motion frontier has genuinely changed: re-examine.
    """
    r = search(pop=10, gens=5, seed=0, shape=(80, 80), steps=120, verbose=False)
    assert r["metrics"]["travel"] < 1.0          # < 1 R of net drift -> not a glider
