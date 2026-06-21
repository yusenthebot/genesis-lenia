"""Smoke tests for the 3D morphology GA (the tool from round 5's 3D exploration).

evolve3d isn't wired into a round driver (round 5 shipped a direct blob search), but it
is the tool for the open 'stable mobile 3D creature' frontier, so it must at least run
and return valid structures rather than rot.
"""

import numpy as np

from genesis.evolve3d import (
    PATCH, Individual3D, random_individual3, place_patch3, evolve3,
)


def test_random_individual3_valid():
    ind = random_individual3(np.random.default_rng(0))
    assert ind.patch.shape == (PATCH, PATCH, PATCH)
    assert 0.0 <= ind.patch.min() and ind.patch.max() <= 1.0


def test_place_patch3_centred_and_bounded():
    patch = random_individual3(np.random.default_rng(1)).patch
    field = place_patch3((40, 40, 40), patch)
    assert field.shape == (40, 40, 40)
    assert 0.0 <= field.min() and field.max() <= 1.0
    assert field[0, 0, 0] == 0.0           # empty at the corner


def test_evolve3_runs_tiny():
    res = evolve3(size=28, steps=30, seed=1, pop=4, gens=2, verbose=False)
    assert isinstance(res["individual"], Individual3D)
    assert len(res["trail"]) == 2
