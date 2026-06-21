"""Open-endedness — does the world keep GENERATING creatures, or converge to one?

Every earlier round CONVERGES: fitness search finds the glider (round 2), the optimum gain
(round 7), the intercept (round 19). Convergence is the opposite of open-endedness. Here we
ask whether the Lenia substrate can keep producing QUALITATIVELY DIFFERENT creatures.

The tool is MAP-Elites (Mouret & Clune): instead of optimizing one score, we ILLUMINATE a
2-D behaviour space — (how much a creature MOVES) x (how BIG its body is) — keeping the most
VIABLE creature found in each behavioural niche. A generative substrate fills the map with a
zoo of distinct creatures; pure fitness search collapses into a single cell. We measure
coverage (filled niches) over evaluations and compare the two.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World
from genesis.metrics import analyze_history
from genesis.evolve import Individual, mutate, random_individual, place_patch, fitness

GRID = 8                       # behaviour map is GRID x GRID cells
MOVE_MAX = 0.85                # drift_speed axis range (cells/step) — varies 0..~0.83
MASS_MIN, MASS_MAX = 60.0, 1900.0   # body-mass (size) axis — varies ~80..~1900


def simulate(ind: Individual, shape=(72, 72), T=120):
    world = World(shape, ind.rule.to_params())
    A = place_patch(shape, ind.patch)
    hist = np.empty((T, *shape))
    for t in range(T):
        U = world.potential(A)
        A = np.clip(A + ind.rule.dt * world.growth(U), 0.0, 1.0)
        hist[t] = A
    return analyze_history(hist, window=0.2 * shape[0]), A


def behaviour_cell(m):
    """Map metrics -> (i, j) behaviour-grid cell, or None if not a viable body.

    Axis i = how much it MOVES (drift_speed); axis j = how BIG its body is (mean_mass).
    """
    if not m["alive"] or m["localized"] < 0.25:
        return None
    i = int(np.clip(m["drift_speed"] / MOVE_MAX, 0, 0.999) * GRID)
    size = (m["mean_mass"] - MASS_MIN) / (MASS_MAX - MASS_MIN)
    j = int(np.clip(size, 0, 0.999) * GRID)
    return i, j


def quality(m):
    return 0.6 * m["persistent"] + 0.4 * m["localized"]


def map_elites(n_eval=700, seed=0, shape=(72, 72), T=120, track_every=20):
    """Return (archive, coverage_curve). archive[cell] = dict(ind, q, m, snap)."""
    rng = np.random.default_rng(seed)
    archive = {}
    coverage = []
    init = 40
    for k in range(n_eval):
        if k < init or not archive:
            ind = random_individual(rng, near_orbium=(k % 3 == 0))
        else:
            keys = list(archive.keys())
            parent = archive[keys[rng.integers(len(keys))]]["ind"]
            ind = mutate(parent, rng, rate=0.5, scale=0.18)
        m, snap = simulate(ind, shape, T)
        cell = behaviour_cell(m)
        if cell is not None:
            q = quality(m)
            if cell not in archive or q > archive[cell]["q"]:
                archive[cell] = dict(ind=ind, q=q, m=m, snap=snap.copy())
        if k % track_every == 0:
            coverage.append((k, len(archive)))
    coverage.append((n_eval, len(archive)))
    return archive, coverage


def fitness_search(n_eval=700, seed=0, shape=(72, 72), T=120, pop=24, track_every=20):
    """A converging baseline: a population GA that optimizes the round-2 fitness.

    We track the diversity the population RETAINS — how many distinct behaviour cells the
    current population occupies. As selection drives it toward the single best body type,
    that diversity COLLAPSES (the opposite of MAP-Elites, which accumulates it).
    """
    rng = np.random.default_rng(seed)
    population = [random_individual(rng, near_orbium=(i % 2 == 0)) for i in range(pop)]
    scored = [(ind, *simulate(ind, shape, T)) for ind in population]   # (ind, m, snap)
    evals = pop
    coverage = []

    def pop_diversity(items):
        return len({behaviour_cell(m) for _, m, _ in items} - {None})

    coverage.append((evals, pop_diversity(scored)))
    best_f = max(fitness(m) for _, m, _ in scored)
    while evals < n_eval:
        scored.sort(key=lambda x: fitness(x[1]), reverse=True)
        survivors = scored[: pop // 2]                  # elitist selection -> convergence
        children = []
        for _ in range(pop - len(survivors)):
            parent = survivors[rng.integers(len(survivors))][0]
            child = mutate(parent, rng, rate=0.5, scale=0.12)
            m, snap = simulate(child, shape, T)
            children.append((child, m, snap))
            evals += 1
            best_f = max(best_f, fitness(m))
            if evals % track_every == 0:
                coverage.append((evals, pop_diversity(survivors + children)))
        scored = survivors + children
    coverage.append((evals, pop_diversity(scored)))
    return scored[0][0], best_f, coverage
