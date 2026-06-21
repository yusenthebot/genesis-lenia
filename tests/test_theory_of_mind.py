"""Theory-of-mind invariants: the observer infers a hidden goal from behaviour."""

import numpy as np

from genesis.theory_of_mind import (
    K, NPARAMS, _unpack, actor_trajectory, observer_beliefs, accuracy_by_step, evolve_es, T,
)


def test_unpack_and_trajectory():
    Wh, Wi, bh, Wo, bo = _unpack(np.zeros(NPARAMS))
    assert Wo.shape == (K, 12)
    sv, pos = actor_trajectory(0, np.random.default_rng(0))
    assert sv.shape == (T, 2)
    bel = observer_beliefs(np.zeros(NPARAMS), sv, rng=np.random.default_rng(0))
    assert bel.shape == (T, K) and np.allclose(bel.sum(1), 1.0)


def test_observer_reads_intent_from_behaviour():
    """After evolution: infers the goal well above chance from behaviour; ablated -> chance."""
    theta, _ = evolve_es(gens=110, seed=0)
    acc = accuracy_by_step(theta, seed=777, n=150)[-1]
    abl = accuracy_by_step(theta, ablate=True, seed=777, n=150)[-1]
    assert acc > 0.6                 # well above chance (1/K = 0.25)
    assert abl < 0.4                 # without the observation -> ~chance
    assert acc > 2 * abl             # the inference genuinely depends on observing behaviour
