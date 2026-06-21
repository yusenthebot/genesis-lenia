"""Cumulative culture — the ratchet (round 41).

Round 34 showed a FIXED language survives cultural transmission. The deeper phenomenon (Tomasello's
ratchet) is cumulative IMPROVEMENT: each generation inherits the previous artifact, innovates a
BOUNDED amount within one lifetime, and passes on a slightly better one — so skill RATCHETS UP past
what any single lifetime could reach. The ratchet needs BOTH faithful transmission AND innovation:
- CUMULATIVE (transmit + innovate): inherits + improves each generation -> quality ratchets toward 1.
- INDIVIDUAL (innovate, no transmit): each generation restarts from scratch -> stuck at the
  single-lifetime ceiling (no accumulation).
- TRANSMIT-ONLY (transmit, no innovate): copies faithfully but never improves -> flat at generation 0.
Only when both are present does the artifact accumulate complexity no individual could invent alone.

The artifact is a cloud of points refined toward a target shape; the lifetime is a bounded budget of
stochastic hill-climbing. Pure numpy.
"""

from __future__ import annotations

import numpy as np

N = 60             # points in the artifact
LIFETIME = 70      # hill-climb steps per generation (bounded individual innovation)
GENS = 55
SIGMA = 1.6        # per-step perturbation


def target_shape():
    """A sharp 5-pointed star outline — a recognisably 'designed' artifact no single lifetime reaches.
    Built from 10 alternating outer/inner vertices, with N points interpolated along the polygon."""
    verts = []
    for k in range(10):
        r = 26.0 if k % 2 == 0 else 11.0
        a = k / 10 * 2 * np.pi - np.pi / 2       # start at the top
        verts.append([r * np.cos(a), r * np.sin(a)])
    verts = np.array(verts)
    pts = []
    seg = N // 10
    for k in range(10):
        v0, v1 = verts[k], verts[(k + 1) % 10]
        for j in range(seg):
            f = j / seg
            pts.append(v0 * (1 - f) + v1 * f)
    return np.array(pts[:N])


TARGET = target_shape()


def _dist(design):
    return float(np.mean(np.linalg.norm(design - TARGET, axis=1)))


# normalisation: mean distance of a random artifact (so quality ~0 random, ~1 perfect)
_D0 = float(np.mean([_dist(np.random.default_rng(s).normal(0, 18, (N, 2))) for s in range(40)]))


def quality(design):
    return max(0.0, 1.0 - _dist(design) / _D0)


def random_design(rng):
    return rng.normal(0, 18, (N, 2))


def lifetime(design, steps, rng):
    """Bounded individual innovation: `steps` of stochastic hill-climbing on the artifact."""
    d = design.copy()
    cur = _dist(d)
    for _ in range(steps):
        i = rng.integers(N)
        prop = d[i] + rng.normal(0, SIGMA, 2)
        before = np.linalg.norm(d[i] - TARGET[i])
        after = np.linalg.norm(prop - TARGET[i])
        if after < before:                       # keep improving moves (hill-climb)
            d[i] = prop
            cur += (after - before) / N
    return d


def run_lineage(mode, gens=GENS, steps=LIFETIME, seed=0):
    """mode: 'cumulative' (transmit+innovate), 'individual' (innovate, restart), 'transmit' (copy only)."""
    rng = np.random.default_rng(seed)
    design = random_design(rng)
    curve = [quality(design)]
    hist = [design.copy()]
    for _ in range(gens):
        if mode == "cumulative":
            design = lifetime(design, steps, rng)
        elif mode == "individual":
            design = lifetime(random_design(rng), steps, rng)
        elif mode == "transmit":
            design = design                       # faithful copy, no innovation
        else:
            raise ValueError(mode)
        curve.append(quality(design))
        hist.append(design.copy())
    return np.array(curve), np.array(hist)
