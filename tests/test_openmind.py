"""Open-ended-minds invariants: descriptors, binning, illumination > convergence."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.openmind import run, behaviour_cell, map_elites, random_policy, GRID


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_run_returns_foraging_descriptors():
    rule, patch = _glider()
    m = run(rule, patch, random_policy(np.random.default_rng(0)), T=80, shape=(72, 72))
    assert {"eaten", "path", "spread", "traj"} <= set(m)
    assert m["path"] >= 0 and m["eaten"] >= 0


def test_behaviour_cell_in_grid_or_none():
    rule, patch = _glider()
    m = run(rule, patch, random_policy(np.random.default_rng(2)), T=80, shape=(72, 72))
    c = behaviour_cell(m)
    assert c is None or (0 <= c[0] < GRID and 0 <= c[1] < GRID)


def test_illumination_finds_many_strategies():
    """A short MAP-Elites run over policies retains several distinct foraging strategies."""
    rule, patch = _glider()
    archive, cov = map_elites(rule, patch, n_eval=120, seed=0, eval_seeds=(0,))
    assert len(archive) > 6                 # a zoo of strategies, not one
    assert cov[-1][1] == len(archive)
