"""Open-ended ratchet — cumulative culture that INVENTS its own complexity (round 43).

Round 41's ratchet climbed toward a GIVEN target (a star). The deeper, open-ended phenomenon has NO
target: complexity that grows unboundedly, that the culture discovers for itself. Here the artifact
is a TOWER of stacked blocks; the only rule is physics — the running centre of mass above every joint
must stay over the block below, or the tower topples. Complexity = how tall (and how far it reaches)
the stable tower is — unbounded; taller/farther is always "better", with no target shape.

Each generation INHERITS the tower and innovates a BOUNDED amount (adds a few stably-placed blocks,
biased to lean farther out). With both transmission and innovation the culture accumulates a tall,
leaning spire no single lifetime could build; individual learning (restart each generation) is capped
at one lifetime's few blocks; copy-only never grows. Open-ended: the tower keeps reaching with no
ceiling but stability itself. Pure numpy.
"""

from __future__ import annotations

import numpy as np

W = 1.0            # block width
BUDGET = 4         # blocks a single lifetime can add (bounded innovation)
TRIES = 24         # placement attempts per block within a lifetime
GENS = 40
DRIFT = 0.16       # rightward lean bias per block (the culture reaches outward)


def stable(centers):
    """True if the running centre of mass above every joint stays over the block below."""
    x = np.asarray(centers)
    n = len(x)
    for i in range(1, n):
        com_above = x[i:].mean()                 # CoM of blocks resting on block i-1
        if abs(com_above - x[i - 1]) > W / 2:
            return False
    return True


def add_blocks(centers, budget, rng):
    """One lifetime's bounded innovation: try to add `budget` stably-placed blocks, leaning out."""
    tower = list(centers)
    for _ in range(budget):
        placed = False
        best = None
        for _ in range(TRIES):
            cand = tower[-1] + DRIFT + rng.normal(0, 0.18)
            if stable(tower + [cand]):
                # prefer the candidate that reaches farthest (open-ended: extend the overhang)
                if best is None or cand > best:
                    best = cand
                    placed = True
        if not placed:
            break                                # couldn't extend further this lifetime
        tower.append(best)
    return np.array(tower)


def reach(centers):
    """Horizontal overhang: how far the tower extends beyond its base (in block-widths)."""
    return float(np.max(centers) - centers[0])


def height(centers):
    return len(centers)


def run_lineage(mode, gens=GENS, budget=BUDGET, seed=0):
    """mode: 'cumulative' (transmit+innovate), 'individual' (innovate, restart), 'transmit' (copy)."""
    rng = np.random.default_rng(seed)
    tower = np.array([0.0])                       # the base block
    h_curve = [height(tower)]
    r_curve = [reach(tower)]
    hist = [tower.copy()]
    for _ in range(gens):
        if mode == "cumulative":
            tower = add_blocks(tower, budget, rng)
        elif mode == "individual":
            tower = add_blocks(np.array([0.0]), budget, rng)
        elif mode == "transmit":
            tower = tower                         # faithful copy, no innovation
        else:
            raise ValueError(mode)
        h_curve.append(height(tower))
        r_curve.append(reach(tower))
        hist.append(tower.copy())
    return np.array(h_curve), np.array(r_curve), hist
