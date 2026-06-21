"""Predator-prey invariants: bounded populations, predation transfers, starvation."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.predprey import PredPrey


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_populations_bounded_and_composite_valid():
    rule, patch = _glider()
    pp = PredPrey((110, 110), rule, patch, n_prey=6, n_pred=3, food_spawn=8,
                  max_prey=14, max_pred=8, seed=0)
    for _ in range(80):
        pp.step()
    assert len(pp.prey) <= 14 and len(pp.pred) <= 8
    rgb = pp.composite()
    assert rgb.shape == (110, 110, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert all(np.isfinite(c.A).all() for c in pp.prey + pp.pred)


def test_predation_hurts_prey():
    rule, patch = _glider()
    pp = PredPrey((110, 110), rule, patch, n_prey=1, n_pred=1, food_spawn=10_000,
                  seed=0)
    pp.pred[0].A = pp.prey[0].A.copy()        # predator right on the prey
    m0 = pp.prey[0].A.sum()
    for _ in range(8):
        pp.step()
    eaten = (len(pp.prey) == 0) or (pp.prey[0].A.sum() < m0)
    assert eaten                               # the prey lost mass / was eaten


def test_predators_starve_without_prey():
    rule, patch = _glider()
    pp = PredPrey((110, 110), rule, patch, n_prey=0, n_pred=4, food_spawn=10_000,
                  pred_decay=0.04, pred_e0=0.5, seed=1)
    for _ in range(250):
        pp.step()
    assert len(pp.pred) < 4                     # no prey -> predators die out
