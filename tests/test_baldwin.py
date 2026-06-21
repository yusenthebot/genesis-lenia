"""Baldwin-effect invariants: bounded, learning-rate is a heritable bounded gene."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.baldwin import BaldwinEcology, LR_MAX


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
    eco = BaldwinEcology((110, 110), rule, patch, n0=8, flip=200, spawn=8,
                         food_decay=0.02, max_pop=16, seed=0)
    for _ in range(80):
        eco.step()
    assert len(eco.pop) <= 16
    assert all(np.isfinite(c.A).all() for c in eco.pop)
    rgb = eco.composite()
    assert rgb.shape == (110, 110, 3) and 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_learning_rate_gene_stays_in_bounds():
    rule, patch = _glider()
    eco = BaldwinEcology((110, 110), rule, patch, n0=10, flip=200, spawn=6,
                         food_decay=0.02, lr_mut=0.3, max_pop=24, seed=1)
    for _ in range(150):
        eco.step()
        for c in eco.pop:
            assert 0.0 <= c.lr <= LR_MAX           # gene always within bounds


def test_higher_lr_converges_weights_faster():
    """A creature with a higher learning rate moves its plastic weights faster."""
    rule, patch = _glider()
    eco = BaldwinEcology((110, 110), rule, patch, n0=2, flip=10_000, spawn=6,
                         food_decay=0.02, max_pop=8, seed=2)
    eco.pop[0].lr = 0.6
    eco.pop[1].lr = 0.02
    for _ in range(120):
        eco.step()
    if len(eco.pop) >= 2:
        fast = max(eco.pop, key=lambda c: c.lr)
        slow = min(eco.pop, key=lambda c: c.lr)
        # the fast learner should have a more decided (larger-magnitude) preference
        assert np.abs(fast.w).max() >= np.abs(slow.w).max()
