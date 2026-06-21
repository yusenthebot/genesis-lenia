"""Memory-brain invariants: param packing, trial reward, and memory beats no-memory."""

import numpy as np

from genesis.memory_brain import (
    H, OBS, NPARAMS, unpack, run_trial, fitness, evolve_es,
)


def test_param_pack_unpack_shapes():
    theta = np.arange(NPARAMS, dtype=float)
    Wh, Wi, bh, wo = unpack(theta)
    assert Wh.shape == (H, H) and Wi.shape == (H, OBS)
    assert bh.shape == (H,) and wo.shape == (H,)
    assert NPARAMS == H * H + H * OBS + H + H


def test_run_trial_returns_binary_reward():
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.5, NPARAMS)
    r = run_trial(theta, cue=1.0, delay=5, recurrent=True)
    assert r in (0.0, 1.0)


def test_recurrent_solves_memory_task_feedforward_cannot():
    theta_r, _ = evolve_es(recurrent=True, gens=35, seed=0)
    theta_f, _ = evolve_es(recurrent=False, gens=35, seed=0)
    acc_r = fitness(theta_r, np.random.default_rng(99), recurrent=True, K=300)
    acc_f = fitness(theta_f, np.random.default_rng(99), recurrent=False, K=300)
    assert acc_r > 0.85                       # memory solves cue-recall
    assert acc_f < 0.65                       # a memoryless reflex is near chance
    assert acc_r > acc_f + 0.25


def test_feedforward_cannot_recall_after_delay():
    """Even a well-trained recurrent net, run WITHOUT recurrence, loses the cue."""
    theta_r, _ = evolve_es(recurrent=True, gens=35, seed=1)
    rng = np.random.default_rng(3)
    # same weights, recurrence ablated -> at the go step the cue is unreachable
    acc_ablated = fitness(theta_r, rng, recurrent=False, K=300)
    assert acc_ablated < 0.65
