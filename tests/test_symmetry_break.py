"""Symmetry-breaking invariants: identical agents differentiate only with interaction AND a trigger."""

import numpy as np

from genesis.symmetry_break import NPARAMS, _unpack, settle, split_rate, evolve_es, T


def test_unpack_and_settle():
    W1, b1, W2, b2 = _unpack(np.zeros(NPARAMS))
    assert W1.shape == (8, 2)
    s, tr = settle(np.zeros(NPARAMS), np.random.default_rng(0), return_traj=True)
    assert tr.shape == (T + 1, 2) and s.shape == (2,)


def test_roles_emerge_from_scratch():
    """Two IDENTICAL agents (one shared policy, no id) end in complementary roles -- but only with
    BOTH the interaction and a symmetry-breaking trigger."""
    theta, _ = evolve_es(gens=120, seed=0)
    full = split_rate(theta, seed=321)
    no_int = split_rate(theta, ablate_interaction=True, seed=321)
    no_trig = split_rate(theta, ablate_noise=True, seed=321)   # perfect symmetry, no trigger
    assert full > 0.85                 # identical agents reliably differentiate
    assert no_int < 0.6                # interaction is load-bearing
    assert no_trig < 0.05              # with perfect symmetry + no trigger, symmetry never breaks
