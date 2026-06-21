"""Iterated-learning invariants: a transmission bottleneck induces compositional structure."""

import numpy as np

from genesis.communicate_iterate import iterate, topo_similarity, _train_mlp, _features, ALL


def test_mlp_fits_and_topo_valid():
    X = _features(ALL); Y = np.random.default_rng(0).normal(0, 1, (len(ALL), 3))
    pred = _train_mlp(X, Y, epochs=200)
    assert pred(ALL).shape == (len(ALL), 3)
    assert -1.0 <= topo_similarity(Y) <= 1.0


def test_bottleneck_raises_compositionality_vs_full():
    """A transmission bottleneck raises topographic similarity; full transmission does not."""
    bn = np.mean([iterate(5, gens=40, seed=s)[1][-1] for s in (0, 1, 2)])
    full = np.mean([iterate(len(ALL), gens=40, seed=s)[1][-1] for s in (0, 1, 2)])
    assert bn > 0.15                 # compositionality emerges under the bottleneck
    assert bn > full + 0.1           # clearly more than full transmission (the control ~0)
