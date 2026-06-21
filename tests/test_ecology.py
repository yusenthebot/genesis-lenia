"""Ecology invariants: it runs, food is shared/consumed, creatures can starve."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.ecology import Ecology


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_ecology_runs_and_composite_is_valid():
    rule, patch = _glider()
    specs = [(6.0, (40, 40)), (0.0, (90, 90))]
    eco = Ecology((120, 120), rule, patch, specs, spawn=40, frad=8.0, seed=0)
    for _ in range(60):
        eco.step()
    rgb = eco.composite()
    assert rgb.shape == (120, 120, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert all(np.isfinite(c.A).all() for c in eco.creatures)
    assert eco.F.min() >= -1e-9            # shared food never goes negative


def test_creature_starves_without_reachable_food():
    rule, patch = _glider()
    # high metabolism, tiny reserve, no foraging -> the creature must die
    eco = Ecology((120, 120), rule, patch, [(0.0, (60, 60))],
                  spawn=10_000, decay=0.06, energy0=0.3, seed=1)
    for _ in range(300):
        eco.step()
    assert not eco.creatures[0].alive


def test_food_is_shared_one_pool():
    """Eating reduces the single shared food field for everyone."""
    rule, patch = _glider()
    eco = Ecology((120, 120), rule, patch, [(0.0, (60, 60))], spawn=10_000, seed=2)
    eco.step()                              # absorb the t=0 auto-spawn
    eco.F[:] = 0.0
    # place food directly under the creature so it certainly eats
    c = eco.creatures[0]
    yc, xc = np.unravel_index(int(np.argmax(c.A)), c.A.shape)
    yy, xx = np.indices((120, 120))
    eco.F = np.exp(-0.5 * ((yy - yc) ** 2 + (xx - xc) ** 2) / 8.0 ** 2)
    before = eco.F.sum()
    for _ in range(10):
        eco.step()
    assert eco.F.sum() < before             # the body consumed shared food
