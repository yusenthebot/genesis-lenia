"""Measuring the mind — information the brain carries about the world.

A task score ("0.89 nutritious") says a creature does well; it does not say its mind
*knows* anything. Here we quantify knowing: the mutual information, in bits, between the
creature's internal state (which food its brain currently prefers) and the world's hidden
variable (which food is actually nutritious). A learner's brain tracks the hidden rule, so
its state carries ~1 bit about the world; a non-learner's fixed state carries ~0. Sweeping
how fast the world changes traces the mind's operating envelope — the rate of change it can
still keep up with.
"""

from __future__ import annotations

import numpy as np


def entropy_bits(counts) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(x, y) -> float:
    """MI(X;Y) in bits for two discrete integer sequences."""
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    n = len(x)
    if n == 0:
        return 0.0
    xs, ys = np.unique(x), np.unique(y)
    px = {v: np.mean(x == v) for v in xs}
    py = {v: np.mean(y == v) for v in ys}
    mi = 0.0
    for vx in xs:
        xmask = x == vx
        for vy in ys:
            pxy = np.mean(xmask & (y == vy))
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px[vx] * py[vy]))
    return float(max(mi, 0.0))


def windowed_mi(x, y, win=120, stride=4):
    """Sliding-window MI(X;Y) -> (times, mi) to watch knowledge rise and fall."""
    x = np.asarray(x); y = np.asarray(y)
    ts, vals = [], []
    for i in range(win, len(x) + 1, stride):
        vals.append(mutual_information_bits(x[i - win:i], y[i - win:i]))
        ts.append(i)
    return np.array(ts), np.array(vals)


def brain_world_series(hist):
    """From an embodied-learner hist -> (brain_preference, nutritious_type) as 0/1 ints."""
    h = np.array(hist)
    brain_pref = (h[:, 4] > h[:, 5]).astype(int)   # prefers food A if w_A > w_B
    world = h[:, 1].astype(int)                     # which food is nutritious
    return brain_pref, world
