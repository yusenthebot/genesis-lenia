"""Open-ended-ratchet invariants: unbounded complexity accumulates, no target, only with both
transmission and innovation."""

import numpy as np

from genesis.open_ended import stable, add_blocks, W, run_lineage, BUDGET, GENS


def test_stability_and_add():
    assert stable([0.0, 0.0, 0.0])                 # a perfectly aligned stack is stable
    assert not stable([0.0, 5.0])                  # a wildly offset block topples
    grown = add_blocks(np.array([0.0]), BUDGET, np.random.default_rng(0))
    assert len(grown) <= 1 + BUDGET                # a lifetime adds at most BUDGET blocks
    assert stable(grown)                           # and whatever it builds stays stable


def test_open_ended_ratchet():
    cum = run_lineage("cumulative", seed=0)[0][-1]
    ind = run_lineage("individual", seed=0)[0][-1]
    tra = run_lineage("transmit", seed=0)[0][-1]
    assert cum > 3 * (1 + BUDGET)        # cumulative accumulates FAR past one lifetime's budget
    assert cum > 5 * ind                 # ... far beyond individual (restart each gen)
    assert ind <= 1 + BUDGET + 1         # individual capped near one lifetime
    assert tra <= 1                      # copy-only never grows (no innovation)
