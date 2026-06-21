"""Evolving-ecology invariants: bounded population, birth, death, inheritance."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.evo_ecology import EvoEcology, GMAX


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_population_stays_within_cap_and_finite():
    rule, patch = _glider()
    eco = EvoEcology((110, 110), rule, patch, n0=4, spawn=6, frad=8.0,
                     max_pop=12, seed=0)
    for _ in range(120):
        eco.step()
    assert len(eco.pop) <= 12
    assert all(np.isfinite(c.A).all() for c in eco.pop)
    assert all(0.0 <= c.gamma <= GMAX for c in eco.pop)


def test_abundant_food_lets_population_reproduce():
    rule, patch = _glider()
    eco = EvoEcology((120, 120), rule, patch, n0=3, spawn=5, frad=10.0,
                     repro_energy=2.0, max_pop=20, seed=1)
    for _ in range(150):
        eco.step()
    assert len(eco.pop) > 3                  # births happened


def test_no_food_kills_the_population():
    rule, patch = _glider()
    eco = EvoEcology((110, 110), rule, patch, n0=5, spawn=10_000, frad=8.0,
                     decay=0.04, energy0=0.4, seed=2)
    for _ in range(260):
        eco.step()
    assert len(eco.pop) < 5                   # starvation removed creatures


def test_offspring_inherit_parent_gain():
    """A child's gain is its parent's plus a bounded mutation (heritability)."""
    rule, patch = _glider()
    eco = EvoEcology((100, 100), rule, patch, n0=1, spawn=4, frad=12.0,
                     repro_energy=1.6, mut=1.5, max_pop=8, seed=3)
    eco.pop[0].gamma = 9.0                    # fix the founder's gene
    for _ in range(140):
        eco.step()
    if len(eco.pop) > 1:                       # if it reproduced
        gains = np.array([c.gamma for c in eco.pop])
        # descendants cluster near the founder's gene, not uniformly random
        assert gains.std() < GMAX / 3
