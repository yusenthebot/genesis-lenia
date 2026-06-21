"""Within-lifetime learning invariants: it learns, it beats ablation, it re-adapts."""

import numpy as np

from genesis.learning import run_episode, summary, windowed_accuracy


def test_learner_finds_a_static_reward():
    """With no reversal, a plastic brain should reach the rewarding source reliably."""
    tastes, _ = run_episode(flip=10_000, T=1500, lr=0.25, seed=0)  # effectively no flip
    assert summary(tastes)["accuracy"] > 0.8


def test_plasticity_off_is_chance_on_reversal():
    tastes, _ = run_episode(flip=320, T=2400, lr=0.0, seed=0)
    acc = summary(tastes)["accuracy"]
    assert 0.35 < acc < 0.65            # a fixed choice tracks the active source ~half the time


def test_learner_beats_ablated_on_reversal():
    lr_acc, ab_acc = [], []
    for s in range(4):
        lt, _ = run_episode(flip=320, T=2400, lr=0.25, seed=s)
        at, _ = run_episode(flip=320, T=2400, lr=0.0, seed=s)
        lr_acc.append(summary(lt)["accuracy"]); ab_acc.append(summary(at)["accuracy"])
    assert np.mean(lr_acc) > np.mean(ab_acc) + 0.2   # learning clearly helps


def test_windowed_accuracy_shapes():
    tastes, _ = run_episode(flip=320, T=1200, lr=0.25, seed=1)
    x, a = windowed_accuracy(tastes, 1200, win=10)
    assert len(x) == len(a)
    assert a.min() >= 0.0 and a.max() <= 1.0
