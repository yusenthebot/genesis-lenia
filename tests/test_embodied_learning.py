"""Embodied-learning invariants: bounded, the brain learns, the learner out-eats ablation."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.embodied_learning import EmbodiedLearner, run_life


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _nutfrac(eco):
    h = np.array(eco.hist)
    active = h[:, 1].astype(int)
    nut = np.where(active == 0, h[:, 2], h[:, 3]); tot = h[:, 2] + h[:, 3]
    return float(nut.sum() / (tot.sum() + 1e-9))


def test_runs_bounded_and_composite_valid():
    rule, patch = _glider()
    eco = EmbodiedLearner((100, 100), rule, patch, lr=0.3, flip=10_000,
                          food_decay=0.015, seed=0)
    for _ in range(120):
        eco.step()
    assert 0.0 <= eco.A.min() and eco.A.max() <= 1.0
    assert eco.FA.min() >= -1e-9 and eco.FB.min() >= -1e-9
    rgb = eco.composite()
    assert rgb.shape == (100, 100, 3) and 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_brain_learns_which_food_is_nutritious():
    """With no reversal, the plastic weight for the nutritious type ends higher."""
    rule, patch = _glider()
    eco = EmbodiedLearner((110, 110), rule, patch, lr=0.35, flip=10_000,
                          food_decay=0.015, seed=1)   # active=0 (food A nutritious)
    for _ in range(450):
        eco.step()
    assert eco.w[0] > eco.w[1]                 # learned: prefer A


def test_learner_out_eats_ablation():
    rule, patch = _glider()
    el, _ = run_life(rule, patch, T=1200, lr=0.3, w0=(0, 0), seed=0, flip=400,
                     gain=0.6, food_decay=0.015, shape=(120, 120))
    ea, _ = run_life(rule, patch, T=1200, lr=0.0, w0=(1, 1), seed=0, flip=400,
                     gain=0.6, food_decay=0.015, shape=(120, 120))
    assert _nutfrac(el) > _nutfrac(ea) + 0.1   # learning clearly helps foraging
