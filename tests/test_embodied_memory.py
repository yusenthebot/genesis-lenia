"""Embodied-memory invariants: packing, a bounded episode, and memory helps under flashing."""

import json
from pathlib import Path

import numpy as np
import pytest

from genesis.evolve import Genome
from genesis.embodied_memory import (
    NPARAMS, H, IN, OUT, unpack, episode, evolve_es,
)


def _glider():
    p = Path("outputs/round2_genome.json")
    if not p.exists():
        pytest.skip("round2 genome (evolved glider) not present")
    d = json.loads(p.read_text())
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def test_param_pack_unpack_shapes():
    theta = np.arange(NPARAMS, dtype=float)
    Wh, Wi, bh, Wo, bo = unpack(theta)
    assert Wh.shape == (H, H) and Wi.shape == (H, IN)
    assert Wo.shape == (OUT, H) and bo.shape == (OUT,)
    assert NPARAMS == H * H + H * IN + H + OUT * H + OUT


def test_episode_runs_bounded():
    rule, patch = _glider()
    theta = np.random.default_rng(0).normal(0, 0.4, NPARAMS)
    collected, eaten, t, _ = episode(rule, patch, theta, recurrent=True, T=120, seed=0)
    assert collected >= 0 and eaten >= 0.0 and t <= 120


def test_memory_helps_when_signal_flashes():
    """After a short evolution, the recurrent forager collects at least as much
    food as the memoryless one in the flashing world (usually strictly more)."""
    rule, patch = _glider()
    th_r, _ = evolve_es(rule, patch, recurrent=True, gens=12, pop=10, flash_off=16, seed=0)
    th_f, _ = evolve_es(rule, patch, recurrent=False, gens=12, pop=10, flash_off=16, seed=0)
    from genesis.embodied_memory import fitness
    cr = fitness(rule, patch, th_r, True, seeds=(30, 31, 32), flash_off=16)
    cf = fitness(rule, patch, th_f, False, seeds=(30, 31, 32), flash_off=16)
    assert cr >= cf                              # memory never hurts here
    assert cr > 0.0                              # and the memory forager actually forages
