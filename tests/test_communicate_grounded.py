"""Grounded-communication invariants: the channel carries actionable info for foraging."""

import numpy as np

from genesis.communicate_grounded import (
    NPARAMS, _unpack, episode, catch_rate, evolve_es,
)


def test_unpack_and_episode():
    sc, fo = _unpack(np.zeros(NPARAMS))
    assert len(sc) == 4 and len(fo) == 4
    caught, mind = episode(np.zeros(NPARAMS), 0.5, rng=np.random.default_rng(0))
    assert isinstance(bool(caught), bool) and mind >= 0.0


def test_comm_beats_ablated():
    """After joint evolution, the blind forager reaches food WITH the channel, not without it."""
    theta, _ = evolve_es(gens=90, seed=0)
    cr = catch_rate(theta, ablate=False, seed=777)
    cra = catch_rate(theta, ablate=True, seed=777)
    assert cr > 0.35                 # the pair forages
    assert cr > 4 * cra              # and the channel is what makes it work (ablated ~chance)
