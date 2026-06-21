"""Round 25 — a serious search for a stable, MOBILE 3D creature.

R5 found robust 3D self-ORGANISATION but no compact mobile creature (a 3D glider), and an
honest negative was recorded. This is one well-resourced push at the negative. Three upgrades
over R5's search:

  1. MULTI-RING kernels (kernel_peaks) — what Bert Chan's 3D Lenia creatures actually use;
     R5 searched mostly single-ring.
  2. A SHAPED viability fitness — the round-2 fitness hard-gates on `alive`, so dead/exploded
     candidates score ~0 and the search has no gradient. Here partial credit (healthy mass,
     compactness, persistence) lets the GA CLIMB toward viability from a poor start.
  3. A MOBILITY bonus once viable — to pull viable blobs toward a moving creature.

Co-evolves rule + ring-heights + 3D seed. Reports honestly whether a mobile 3D creature is
found or the negative stands, with the search documented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis.world import World, LeniaParams
from genesis.metrics import analyze_history
from genesis.evolve3d import random_patch3, mutate_patch3, place_patch3, R3D

# rule gene bounds tuned to the fertile (narrow-growth) region for 3D
GB = {"mu_g": (0.08, 0.22), "sigma_g": (0.010, 0.045),
      "mu_k": (0.25, 0.62), "sigma_k": (0.07, 0.22)}
N_RINGS = 3


@dataclass
class Creature3D:
    rule: dict           # mu_g, sigma_g, mu_k, sigma_k
    peaks: np.ndarray    # (N_RINGS,) relative ring heights in [0,1]
    patch: np.ndarray    # (PATCH^3) seed


def _params(c: Creature3D) -> LeniaParams:
    return LeniaParams(R=R3D, dt=0.1, mu_g=c.rule["mu_g"], sigma_g=c.rule["sigma_g"],
                       mu_k=c.rule["mu_k"], sigma_k=c.rule["sigma_k"],
                       kernel_peaks=tuple(float(x) for x in c.peaks))


def random_creature(rng) -> Creature3D:
    rule = {k: rng.uniform(lo, hi) for k, (lo, hi) in GB.items()}
    peaks = rng.uniform(0.3, 1.0, N_RINGS)
    return Creature3D(rule, peaks, random_patch3(rng))


def mutate(c: Creature3D, rng, scale=0.12) -> Creature3D:
    rule = {}
    for k, (lo, hi) in GB.items():
        v = c.rule[k] + (rng.normal(0, (hi - lo) * scale) if rng.random() < 0.6 else 0)
        rule[k] = float(np.clip(v, lo, hi))
    peaks = np.clip(c.peaks + rng.normal(0, 0.12, N_RINGS), 0.1, 1.0)
    return Creature3D(rule, peaks, mutate_patch3(c.patch, rng))


def _mass_health(mean_mass, ncells):
    """1.0 when the body occupies a healthy ~1-10% of the grid, ->0 if dead/space-filling."""
    frac = mean_mass / ncells
    if frac < 1e-4:
        return 0.0
    target = 0.04
    return float(np.exp(-((np.log(frac + 1e-9) - np.log(target)) ** 2) / (2 * 0.9 ** 2)))


def evaluate(c: Creature3D, size=44, steps=110, rec_stride=3, chase=False, snap=False):
    w = World((size,) * 3, _params(c))
    w.A = place_patch3((size,) * 3, c.patch)
    frames = []
    for i in range(steps):
        w.step()
        if i % rec_stride == 0:
            frames.append(w.A.copy())
    hist = np.stack(frames)
    m = analyze_history(hist, window=2.2 * R3D)
    ncells = size ** 3
    # SHAPED viability: mass health * compactness, plus persistence; partial credit always
    health = _mass_health(m["mean_mass"], ncells)
    compact = float(m["concentration"]) if m["alive"] else 0.0
    persist = m["persistent"] if m["alive"] else 0.0
    via = 0.5 * health + 0.3 * compact + 0.2 * persist
    travel = float(m["travel_widths"])
    mobile = bool(m["alive"] and via > 0.5 and travel > 0.25)
    if chase:
        # AGGRESSIVELY reward directed motion among viable bodies (hunt the glider)
        score = via + (2.6 * min(travel, 1.6) if (m["alive"] and via > 0.5) else 0.0)
    else:
        # viability-focused: a stable COMPACT body, with a gentle motion nudge
        score = via + (0.3 * min(travel, 1.0) if (m["alive"] and via > 0.5) else 0.0)
    m2 = dict(via=via, travel=travel, alive=bool(m["alive"]), conc=float(m["concentration"]),
              mass=m["mean_mass"], mass_cv=m["mass_cv"], support=m["support_fraction"],
              mobile=mobile, health=health)
    if snap:
        m2["field"] = hist[-1]
    return score, m2


def search(size=44, steps=110, pop=18, gens=18, seed=0, chase=False, verbose=True):
    rng = np.random.default_rng(seed)
    population = [random_creature(rng) for _ in range(pop)]
    best = None
    trail = []
    for gen in range(gens):
        scored = [(s, c, m) for c in population
                  for s, m in [evaluate(c, size, steps, chase=chase)]]
        scored.sort(key=lambda t: t[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        gb = scored[0]
        n_via = sum(1 for s, _, m in scored if m["via"] > 0.55)
        n_mob = sum(1 for s, _, m in scored if m["mobile"] and m["travel"] > 0.3)
        trail.append(dict(gen=gen, best=gb[0], via=gb[2]["via"], travel=gb[2]["travel"],
                          n_via=n_via, n_mob=n_mob))
        if verbose:
            print(f"  gen {gen:2d}: best={gb[0]:.2f} via={gb[2]['via']:.2f} "
                  f"travel={gb[2]['travel']:.2f} conc={gb[2]['conc']:.2f} "
                  f"viable={n_via}/{pop} mobile={n_mob}")
        elites = [c for _, c, _ in scored[:max(3, pop // 4)]]
        children = list(elites)
        while len(children) < pop - 2:
            children.append(mutate(elites[rng.integers(len(elites))], rng))
        children += [random_creature(rng), random_creature(rng)]
        population = children
    return dict(best=best[1], score=best[0], metrics=best[2], trail=trail)
