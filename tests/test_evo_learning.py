"""Learning-under-selection invariants: bounded, plastic vs frozen, inheritance."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.evo_learning import EvoLearning


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_runs_bounded_and_composite_valid():
    rule, patch = _glider()
    eco = EvoLearning((110, 110), rule, patch, n0=8, flip=300, spawn=8,
                      food_decay=0.02, max_pop=16, seed=0)
    for _ in range(80):
        eco.step()
    assert len(eco.pop) <= 16
    assert all(np.isfinite(c.A).all() for c in eco.pop)
    rgb = eco.composite()
    assert rgb.shape == (110, 110, 3) and 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_learner_is_plastic_fixed_is_frozen():
    rule, patch = _glider()
    # one learner (w starts [0,0]) and we watch its weights move
    eco = EvoLearning((110, 110), rule, patch, n0=4, learner_frac=1.0, flip=10_000,
                      spawn=6, food_decay=0.02, lr=0.4, seed=1)
    w_before = [c.w.copy() for c in eco.pop]
    for _ in range(120):
        eco.step()
    moved = any(not np.allclose(c.w, wb) for c, wb in zip(eco.pop[:len(w_before)], w_before)
                if c.alive)
    assert moved                                    # plastic weights change with experience

    # a fixed creature's weights never change within its life
    eco2 = EvoLearning((110, 110), rule, patch, n0=4, learner_frac=0.0, flip=10_000,
                       spawn=6, food_decay=0.02, type_mut=0.0, seed=2)
    c = eco2.pop[0]; w0 = c.w.copy()
    for _ in range(80):
        eco2.step()
        if not c.alive:
            break
    assert np.allclose(c.w, w0)                      # fixed reflex: frozen within life


def test_learner_trait_is_heritable():
    """An all-learner population (no type mutation) stays all-learner as it reproduces."""
    rule, patch = _glider()
    eco = EvoLearning((120, 120), rule, patch, n0=4, learner_frac=1.0, flip=300,
                      spawn=5, food_decay=0.02, type_mut=0.0, max_pop=16, seed=3)
    for _ in range(150):
        eco.step()
    assert all(c.learner for c in eco.pop)
