"""Evolution over world-genomes — discover creatures, don't hand-place them.

An individual is a world *rule* (``LeniaParams``) **plus an evolvable seed morphology**
(a small smooth patch). We learned the hard way that a glider is a narrow attractor:
no Gaussian blob lands in it under any rule, so the seed shape must be evolved too.
We mutate + select on a behavioural objective. Round 2's objective is *locomotion*: a
single, compact, persistent body that travels straight. The famous Orbium glider is not
coded in — selection is free to find it (or any other mover) on its own. We only bias
part of the initial rule population toward the region where 2D life is known to exist
(prior knowledge of *where* to look, not a hand-built creature).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from genesis.world import World, LeniaParams
from genesis.metrics import analyze_history


GENE_BOUNDS = {
    "mu_g": (0.05, 0.30),
    "sigma_g": (0.004, 0.060),
    "mu_k": (0.20, 0.80),
    "sigma_k": (0.05, 0.30),
}
ORBIUM_CENTER = {"mu_g": 0.15, "sigma_g": 0.015, "mu_k": 0.5, "sigma_k": 0.15}

PATCH = 10          # evolvable seed is PATCH x PATCH, in [0, 1]
UPSCALE = 3         # upsampled to PATCH*UPSCALE cells, then smoothed
R_DEFAULT = 13.0


@dataclass(frozen=True)
class Genome:
    mu_g: float
    sigma_g: float
    mu_k: float
    sigma_k: float
    R: float = R_DEFAULT
    dt: float = 0.1

    def to_params(self) -> LeniaParams:
        return LeniaParams(R=self.R, mu_k=self.mu_k, sigma_k=self.sigma_k,
                           mu_g=self.mu_g, sigma_g=self.sigma_g, dt=self.dt)


@dataclass
class Individual:
    rule: Genome
    patch: np.ndarray         # (PATCH, PATCH) in [0, 1]


def _clip(name: str, value: float) -> float:
    lo, hi = GENE_BOUNDS[name]
    return float(min(hi, max(lo, value)))


def _blur2d(a: np.ndarray, k: int = 2) -> np.ndarray:
    for _ in range(k):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def random_patch(rng: np.random.Generator) -> np.ndarray:
    """A smooth, asymmetric, blob-like seed (low-frequency, off-centre)."""
    p = _blur2d(rng.random((PATCH, PATCH)), 2)
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    cy = PATCH / 2 + rng.normal(0, 1.3)
    cx = PATCH / 2 + rng.normal(0, 1.3)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (PATCH * 0.55)
    p = p * np.clip(1.0 - r, 0.0, 1.0)
    return p / (p.max() + 1e-9)


def mutate_patch(patch: np.ndarray, rng: np.random.Generator,
                 rate: float = 0.3, scale: float = 0.25) -> np.ndarray:
    mask = rng.random(patch.shape) < rate
    out = patch + mask * rng.normal(0, scale, patch.shape)
    out = _blur2d(np.clip(out, 0.0, 1.0), 1)
    return out / (out.max() + 1e-9)


def place_patch(shape, patch: np.ndarray) -> np.ndarray:
    """Upsample + smooth the patch and drop it in the centre of an empty field."""
    up = _blur2d(np.kron(patch, np.ones((UPSCALE, UPSCALE))), 2)
    field = np.zeros(shape, dtype=np.float64)
    h, w = up.shape
    r0, c0 = shape[0] // 2 - h // 2, shape[1] // 2 - w // 2
    field[r0:r0 + h, c0:c0 + w] = np.clip(up, 0.0, 1.0)
    return field


def random_rule(rng: np.random.Generator, near_orbium: bool) -> Genome:
    g = {}
    for k, (lo, hi) in GENE_BOUNDS.items():
        if near_orbium:
            g[k] = _clip(k, ORBIUM_CENTER[k] + rng.normal(0, (hi - lo) * 0.15))
        else:
            g[k] = _clip(k, rng.uniform(lo, hi))
    return Genome(**g)


def random_individual(rng, near_orbium=False) -> Individual:
    return Individual(random_rule(rng, near_orbium), random_patch(rng))


def mutate(ind: Individual, rng: np.random.Generator, rate=0.5, scale=0.10) -> Individual:
    g = {}
    for k, (lo, hi) in GENE_BOUNDS.items():
        v = getattr(ind.rule, k)
        if rng.random() < rate:
            v = _clip(k, v + rng.normal(0, (hi - lo) * scale))
        g[k] = v
    return Individual(replace(ind.rule, **g), mutate_patch(ind.patch, rng))


def fitness(m: dict) -> float:
    """Reward a single compact persistent body that travels straight.

    Hard gates: alive, single compact body (concentration), not space-filling
    (support). base requires survival as one coherent body; move requires directed
    locomotion. A stable non-mover floors at ~0.2*base; a clean glider approaches base.
    """
    if not m["alive"]:
        return 0.0
    if not (0.003 < m["support_fraction"] < 0.35):
        return 0.0
    if m["concentration"] < 0.6:        # one compact creature, not turbulence
        return 0.0
    base = m["persistent"] * (0.3 + 0.7 * m["localized"]) * m["concentration"]
    move = min(1.0, m["travel_widths"] / 0.5) * min(1.0, m["straightness"] / 0.6)
    return float(base * (0.2 + 0.8 * move))


def evaluate(ind: Individual, size: int, steps: int,
             rec_stride: int = 2) -> tuple[float, dict]:
    w = World((size, size), ind.rule.to_params())
    w.A = place_patch((size, size), ind.patch)
    frames = []
    for i in range(steps):
        w.step()
        if i % rec_stride == 0:
            frames.append(w.A.copy())
    hist = np.stack(frames)
    m = analyze_history(hist, window=2.2 * ind.rule.R)
    return fitness(m), m


def evolve(size=84, steps=340, seed=7, pop=40, gens=30, elite=8, verbose=True):
    """(mu + lambda)-style evolution toward a travelling creature."""
    rng = np.random.default_rng(seed)
    population = [random_individual(rng, near_orbium=(i < pop // 2))
                 for i in range(pop)]
    best = None
    trail = []
    for gen in range(gens):
        scored = []
        for ind in population:
            f, m = evaluate(ind, size, steps)
            scored.append((f, ind, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        elites = [ind for _, ind, _ in scored[:elite]]
        n_alive = sum(1 for _, _, m in scored if m["alive"])
        n_move = sum(1 for _, _, m in scored if m["travel_widths"] > 0.1
                     and m["concentration"] > 0.6)
        gb = scored[0]
        trail.append({"gen": gen, "best_fit": gb[0],
                      "best_travel": gb[2]["travel_widths"], "n_alive": n_alive,
                      "n_move": n_move})
        if verbose:
            print(f"  gen {gen:2d}: best_fit={gb[0]:.3f} "
                  f"travel={gb[2]['travel_widths']:.2f}w straight={gb[2]['straightness']:.2f} "
                  f"conc={gb[2]['concentration']:.2f} movers={n_move}/{pop}")
        children = list(elites)
        while len(children) < pop - 3:
            parent = elites[rng.integers(len(elites))]
            children.append(mutate(parent, rng))
        children.append(random_individual(rng, near_orbium=True))
        children.append(random_individual(rng, near_orbium=True))
        children.append(random_individual(rng, near_orbium=False))
        population = children
    return {"individual": best[1], "fitness": best[0], "metrics": best[2],
            "trail": trail}
