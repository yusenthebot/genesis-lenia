"""Evolution-module invariants: bounds, seeding, fitness gates, a tiny EA run."""

import numpy as np
import pytest

from genesis.evolve import (
    GENE_BOUNDS, PATCH, Individual,
    random_individual, mutate, place_patch, fitness, evaluate, evolve,
)


def _rng():
    return np.random.default_rng(0)


def test_random_rule_within_bounds():
    ind = random_individual(_rng())
    for k, (lo, hi) in GENE_BOUNDS.items():
        assert lo <= getattr(ind.rule, k) <= hi


def test_mutate_keeps_rule_in_bounds():
    rng = _rng()
    ind = random_individual(rng)
    for _ in range(50):
        ind = mutate(ind, rng)
        for k, (lo, hi) in GENE_BOUNDS.items():
            assert lo <= getattr(ind.rule, k) <= hi
        assert 0.0 <= ind.patch.min() and ind.patch.max() <= 1.0


def test_place_patch_is_centred_and_bounded():
    patch = np.ones((PATCH, PATCH))
    field = place_patch((80, 80), patch)
    assert field.shape == (80, 80)
    assert 0.0 <= field.min() and field.max() <= 1.0
    # mass must sit near the centre, not at the edges
    assert field[40, 40] > 0.5
    assert field[0, 0] == 0.0


def test_fitness_rejects_dead_soup_and_rewards_clean_glider():
    dead = {"alive": False}
    assert fitness(dead) == 0.0
    soup = {"alive": True, "support_fraction": 0.2, "concentration": 0.2,
            "persistent": 1.0, "localized": 1.0, "travel_widths": 1.0,
            "straightness": 1.0}
    assert fitness(soup) == 0.0          # spread-out turbulence is not a creature
    glider = {"alive": True, "support_fraction": 0.05, "concentration": 1.0,
              "persistent": 1.0, "localized": 1.0, "travel_widths": 0.6,
              "straightness": 0.9}
    stationary = {**glider, "travel_widths": 0.0, "straightness": 0.0}
    assert fitness(glider) > fitness(stationary) > 0.0  # movement is rewarded


def test_evaluate_returns_fitness_and_metrics():
    ind = random_individual(_rng(), near_orbium=True)
    f, m = evaluate(ind, size=48, steps=60)
    assert isinstance(f, float) and f >= 0.0
    assert "concentration" in m and "travel_widths" in m


def test_tiny_evolution_runs_and_returns_individual():
    res = evolve(size=48, steps=60, seed=1, pop=6, gens=2, verbose=False)
    assert isinstance(res["individual"], Individual)
    assert res["individual"].patch.shape == (PATCH, PATCH)
    assert len(res["trail"]) == 2
