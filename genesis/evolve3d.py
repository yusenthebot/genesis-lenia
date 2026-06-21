"""3D evolution — the same idea as round 2, one dimension up.

The engine is already dimension-agnostic; the only 2D-specific pieces were the seed
morphology helpers. Here they are in 3D: an evolvable PATCH^3 seed, upsampled and
smoothed, co-evolved with the rule under the same concentration-gated fitness. This is
what turns a diffuse 3D cloud into a single compact creature.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from genesis.world import World
from genesis.metrics import analyze_history
from genesis.evolve import Genome, fitness, GENE_BOUNDS, ORBIUM_CENTER, _clip

PATCH = 6          # evolvable seed is PATCH^3
UPSCALE = 3        # upsampled to PATCH*UPSCALE cells, then smoothed
R3D = 8.0          # kernel radius for 3D (smaller grid than 2D)


@dataclass
class Individual3D:
    rule: Genome
    patch: np.ndarray         # (PATCH, PATCH, PATCH) in [0, 1]


def _blur3(a: np.ndarray, k: int = 2) -> np.ndarray:
    for _ in range(k):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)
             + np.roll(a, 1, 2) + np.roll(a, -1, 2)) / 7.0
    return a


def random_rule3(rng, near_orbium=True) -> Genome:
    g = {}
    for k, (lo, hi) in GENE_BOUNDS.items():
        if near_orbium:
            g[k] = _clip(k, ORBIUM_CENTER[k] + rng.normal(0, (hi - lo) * 0.18))
        else:
            g[k] = _clip(k, rng.uniform(lo, hi))
    return Genome(R=R3D, dt=0.1, **g)


def random_patch3(rng) -> np.ndarray:
    p = _blur3(rng.random((PATCH, PATCH, PATCH)), 2)
    zz, yy, xx = np.mgrid[0:PATCH, 0:PATCH, 0:PATCH]
    c = PATCH / 2 + rng.normal(0, 1.0, 3)
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (PATCH * 0.6)
    p = p * np.clip(1.0 - r, 0.0, 1.0)
    return p / (p.max() + 1e-9)


def mutate_patch3(patch, rng, rate=0.3, scale=0.25) -> np.ndarray:
    mask = rng.random(patch.shape) < rate
    out = patch + mask * rng.normal(0, scale, patch.shape)
    out = _blur3(np.clip(out, 0.0, 1.0), 1)
    return out / (out.max() + 1e-9)


def place_patch3(shape, patch) -> np.ndarray:
    up = _blur3(np.kron(patch, np.ones((UPSCALE, UPSCALE, UPSCALE))), 2)
    field = np.zeros(shape, dtype=np.float64)
    s = up.shape
    o = [shape[i] // 2 - s[i] // 2 for i in range(3)]
    field[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = np.clip(up, 0, 1)
    return field


def random_individual3(rng, near_orbium=True) -> Individual3D:
    return Individual3D(random_rule3(rng, near_orbium), random_patch3(rng))


def mutate3(ind, rng, rate=0.5, scale=0.10) -> Individual3D:
    g = {}
    for k, (lo, hi) in GENE_BOUNDS.items():
        v = getattr(ind.rule, k)
        if rng.random() < rate:
            v = _clip(k, v + rng.normal(0, (hi - lo) * scale))
        g[k] = v
    return Individual3D(replace(ind.rule, **g), mutate_patch3(ind.patch, rng))


def evaluate3(ind, size, steps, rec_stride=3) -> tuple[float, dict]:
    w = World((size, size, size), ind.rule.to_params())
    w.A = place_patch3((size, size, size), ind.patch)
    frames = []
    for i in range(steps):
        w.step()
        if i % rec_stride == 0:
            frames.append(w.A.copy())
    hist = np.stack(frames)
    m = analyze_history(hist, window=2.2 * ind.rule.R)
    return fitness(m), m


def evolve3(size=48, steps=150, seed=5, pop=14, gens=10, elite=4, verbose=True):
    rng = np.random.default_rng(seed)
    population = [random_individual3(rng, near_orbium=(i < pop * 3 // 4))
                 for i in range(pop)]
    best = None
    trail = []
    for gen in range(gens):
        scored = []
        for ind in population:
            f, m = evaluate3(ind, size, steps)
            scored.append((f, ind, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        elites = [ind for _, ind, _ in scored[:elite]]
        n_alive = sum(1 for _, _, m in scored if m["alive"])
        gb = scored[0]
        trail.append({"gen": gen, "best_fit": gb[0],
                      "conc": gb[2]["concentration"], "n_alive": n_alive})
        if verbose:
            print(f"  gen {gen:2d}: best_fit={gb[0]:.3f} "
                  f"conc={gb[2]['concentration']:.2f} support={gb[2]['support_fraction']:.3f} "
                  f"travel={gb[2]['travel_widths']:.2f} alive={n_alive}/{pop}")
        children = list(elites)
        while len(children) < pop - 2:
            children.append(mutate3(elites[rng.integers(len(elites))], rng))
        children.append(random_individual3(rng, near_orbium=True))
        children.append(random_individual3(rng, near_orbium=False))
        population = children
    return {"individual": best[1], "fitness": best[0], "metrics": best[2], "trail": trail}
