"""The evolution of cooperation — a spatial major transition (round 46).

The arc asked to "derive the world's evolution"; the one major evolutionary phenomenon not yet shown
is how COOPERATION survives. In a well-mixed world, defectors always win: they free-ride on the
benefit cooperators provide and out-reproduce them, so cooperation collapses to zero. But add SPATIAL
STRUCTURE (each agent interacts only with its grid neighbours) and cooperators can form CLUSTERS that
preferentially help each other -- "network reciprocity" (Nowak & May 1992) -- and cooperation
survives and persists. The transition is driven purely by space: same game, same update, but a 2D
neighbourhood vs a shuffled (well-mixed) one.

Spatial prisoner's dilemma on an LxL grid: a cooperator earns 1 per cooperating neighbour; a defector
earns b>1 per cooperating neighbour; defector neighbours pay nothing. Each cell imitates the
highest-scoring strategy in its Moore neighbourhood. Pure numpy.
"""

from __future__ import annotations

import numpy as np

L = 60
B = 1.62           # temptation to defect (Nowak-May regime where spatial cooperation persists)
STEPS = 80
_OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
_NEIGH = [o for o in _OFFS if o != (0, 0)]


def _coop_neighbours(grid):
    """Count cooperator neighbours (Moore, wrap) for every cell."""
    return sum(np.roll(np.roll(grid, dy, 0), dx, 1) for dy, dx in _NEIGH)


def payoff(grid, b):
    nC = _coop_neighbours(grid)
    return np.where(grid == 1, nC * 1.0, nC * b)       # C: 1/coop-neighbour; D: b/coop-neighbour


def _imitate_best(grid, score):
    """Each cell adopts the strategy of the highest-scoring cell in its 3x3 neighbourhood (incl self)."""
    best_s = score.copy()
    best_g = grid.copy()
    for dy, dx in _NEIGH:
        ss = np.roll(np.roll(score, dy, 0), dx, 1)
        gg = np.roll(np.roll(grid, dy, 0), dx, 1)
        take = ss > best_s
        best_s = np.where(take, ss, best_s)
        best_g = np.where(take, gg, best_g)
    return best_g


def step(grid, b, well_mixed=False, rng=None):
    g = grid
    if well_mixed:                                     # destroy spatial structure: shuffle positions
        flat = g.ravel().copy(); rng.shuffle(flat); g = flat.reshape(g.shape)
    sc = payoff(g, b)
    return _imitate_best(g, sc)


def run(b=B, steps=STEPS, well_mixed=False, seed=0, init_coop=0.6, capture=False):
    rng = np.random.default_rng(seed)
    grid = (rng.random((L, L)) < init_coop).astype(np.int8)
    frac = [grid.mean()]
    hist = [grid.copy()]
    for _ in range(steps):
        grid = step(grid, b, well_mixed, rng)
        frac.append(float(grid.mean()))
        if capture:
            hist.append(grid.copy())
    return np.array(frac), hist


def final_cooperation(b=B, well_mixed=False, seeds=(0, 1, 2), steps=STEPS):
    return float(np.mean([run(b, steps, well_mixed, s)[0][-1] for s in seeds]))
