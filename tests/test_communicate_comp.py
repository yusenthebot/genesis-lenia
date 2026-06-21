"""Compositional-communication invariants: naive is holistic; pressure adds structure."""

import numpy as np

from genesis.communicate_comp import (
    NPARAMS, _unpack, evolve_es, both_correct, topographic_similarity, TRAIN, HELDOUT,
)


def test_unpack_shapes():
    sp, li = _unpack(np.zeros(NPARAMS))
    assert len(sp) == 4 and len(li) == 6


def test_naive_communicates_but_is_holistic():
    """Naive: learns the training combos but does NOT generalise to held-out (holistic)."""
    th, _, _ = evolve_es(gens=120, seed=0, topo_weight=0.0)
    assert both_correct(th, TRAIN) > 0.7        # communicates the training meanings
    assert both_correct(th, HELDOUT) < 0.5      # but fails zero-shot (holistic, not compositional)


def test_structural_pressure_raises_compositionality():
    """Adding the topographic pressure makes the code more compositional (higher topo sim)."""
    th_n, _, _ = evolve_es(gens=140, seed=0, topo_weight=0.0)
    th_p, _, _ = evolve_es(gens=140, seed=0, topo_weight=0.8)
    assert topographic_similarity(th_p) > topographic_similarity(th_n) + 0.2
    assert topographic_similarity(th_p) > 0.55
