"""Unification invariants: a bounded life, and every faculty is load-bearing."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.unified import run, survival


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_run_returns_bounded_life():
    rule, patch = _glider()
    surv, on_food, _ = run(rule, patch, "full", T=200, seed=0)
    assert 1 <= surv <= 200 and 0 <= on_food <= surv


def test_every_faculty_is_load_bearing():
    """full > memory_only > no_memory: removing a faculty costs survival (averaged)."""
    rule, patch = _glider()
    seeds = range(0, 12)
    full = survival(rule, patch, "full", seeds)
    mem = survival(rule, patch, "memory_only", seeds)
    nomem = survival(rule, patch, "no_memory", seeds)
    assert full > mem > nomem                 # the ablation ladder
    assert full > 1.4 * nomem                 # memory+prediction buy a clear survival margin


def test_full_creature_can_survive_a_whole_episode():
    """On a favourable seed the integrated creature survives the full episode."""
    rule, patch = _glider()
    surv, _, _ = run(rule, patch, "full", T=320, seed=0)
    assert surv == 320
