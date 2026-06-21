"""Within-lifetime learning — a brain that adapts from its own experience.

Everything until now was a FIXED reflex whose gain evolution set across generations.
Here a creature carries a PLASTIC controller and changes its behaviour during a single
life. The task is reversal learning: two food sources, only one nutritious at a time,
and which one FLIPS partway through life. A fixed or purely-evolved policy cannot cope
with a rule that changes within one lifetime; a learner tracks it.

The controller is a model-free value learner (a one-layer plastic 'brain'): it keeps a
value V_s per source, updates it by the reward delta when it tastes a source
(Rescorla-Wagner), and chooses where to go epsilon-greedily on those values. Learning is
the lr; set lr=0 to ablate it (the same creature, plasticity off).
"""

from __future__ import annotations

import numpy as np


def run_episode(L=120, sources=((60, 32), (60, 88)), flip=320, T=2400,
                lr=0.25, eps=0.12, speed=2.2, eat_r=7.0, seed=0, record=False):
    """One life. Returns per-taste (t, source, hit_active) and optional frames."""
    rng = np.random.default_rng(seed)
    src = np.array(sources, dtype=np.float64)
    n = len(src)
    V = np.zeros(n)
    p = np.array([L / 2.0, L / 2.0])
    active = 0
    target = 0
    tastes = []         # (t, chosen_source, hit_active 0/1)
    frames = []
    for t in range(T):
        if t > 0 and t % flip == 0:
            active = (active + 1) % n               # the world's rule reverses
        d = src[target] - p
        dist = np.hypot(*d)
        p = p + speed * d / (dist + 1e-9) + rng.normal(0, 0.6, 2)
        p = np.clip(p, 0, L - 1)
        if np.hypot(*(p - src[target])) < eat_r:    # reached the source -> taste it
            reward = 1.0 if target == active else -0.3
            V[target] += lr * (reward - V[target])  # plastic update (lr=0 -> frozen)
            tastes.append((t, target, int(target == active)))
            target = (int(rng.integers(n)) if rng.random() < eps
                      else int(np.argmax(V)))        # epsilon-greedy on learned values
        if record and t % 6 == 0:
            frames.append((p.copy(), active, V.copy(), t))
    return tastes, frames


def windowed_accuracy(tastes, T, win=12):
    """Rolling fraction of tastes that hit the active source, over taste index."""
    if not tastes:
        return np.array([]), np.array([])
    ts = np.array([x[0] for x in tastes])
    hit = np.array([x[2] for x in tastes], dtype=float)
    acc = np.convolve(hit, np.ones(win) / win, mode="valid")
    return ts[win - 1:], acc


def summary(tastes):
    if not tastes:
        return dict(n=0, accuracy=0.0)
    hit = np.array([x[2] for x in tastes], dtype=float)
    return dict(n=len(tastes), accuracy=float(hit.mean()))
