"""Evolution-of-cooperation invariants: spatial structure lets cooperation survive; well-mixed kills it."""

import numpy as np

from genesis.cooperation import L, payoff, step, run, final_cooperation


def test_payoff_and_step():
    grid = np.ones((L, L), dtype=np.int8)              # all cooperators
    assert np.allclose(payoff(grid, 1.62), 8.0)        # 8 cooperating neighbours, 1 each
    nxt = step(grid, 1.62)
    assert nxt.shape == grid.shape and set(np.unique(nxt)).issubset({0, 1})


def test_spatial_enables_cooperation():
    """The major transition: cooperation persists spatially (via clustering) but collapses well-mixed."""
    sp = final_cooperation(1.62, well_mixed=False, seeds=(0, 1, 2))
    wm = final_cooperation(1.62, well_mixed=True, seeds=(0, 1, 2))
    assert sp > 0.15                 # cooperation persists with spatial structure
    assert wm < 0.05                 # ... and collapses to ~0 when well-mixed
    assert sp > 5 * (wm + 0.01)      # space is decisive
    # cooperators actually cluster (adjacency well above the random expectation)
    g = run(1.62, well_mixed=False, seed=0, capture=True)[1][-1]
    f = g.mean()
    adj = (np.mean(g * np.roll(g, 1, 0)) + np.mean(g * np.roll(g, 1, 1))) / 2
    assert adj / (f * f + 1e-9) > 1.4
