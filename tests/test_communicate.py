"""Emergent-communication invariants: a code emerges, and the channel is genuinely used."""

import numpy as np

from genesis.communicate import (
    K, NPARAMS, _unpack, evolve_es, accuracy, mutual_information_bits,
)


def test_param_unpack_shapes():
    sp, li = _unpack(np.zeros(NPARAMS))
    assert len(sp) == 4 and len(li) == 4


def test_communication_emerges_and_is_used():
    """After evolution: listener beats chance and the channel carries real information;
    ablating the signal (random) collapses both to ~chance / ~0 bits."""
    theta, curve = evolve_es(gens=90, seed=0)
    acc = accuracy(theta)
    mi = mutual_information_bits(theta)
    acc_abl = accuracy(theta, ablate=True, rng=np.random.default_rng(9))
    mi_abl = mutual_information_bits(theta, ablate=True, rng=np.random.default_rng(9))

    assert acc > 0.6                       # well above chance (1/K = 0.25)
    assert mi > 1.0                        # real information (ceiling is log2(K)=2 bits)
    assert acc_abl < 0.45                  # ablated channel -> near chance
    assert mi_abl < 0.2                    # ablated channel -> almost no information
