"""Round 28 — evolve a MOBILE Flow-Lenia creature (the round-2 search, on the new substrate).

R27 showed Flow-Lenia forms robust compact creatures but they are stationary; random/asymmetric
seeds and rotated flows did not move (F = grad(G) is a gradient flow -> relaxes to equilibrium).
But a gradient flow CAN sustain a glider IF the creature's shape keeps the affinity persistently
shifted forward (as Orbium does in plain Lenia) — a special, EVOLVED configuration. Round 2's
lesson: the glider seed must be EVOLVED, not random. So this is the proper attempt: a GA over the
Flow-Lenia rule + kernel asymmetry + the SEED SHAPE, rewarding directed motion of a compact body.

Mass is conserved, so a found mover cannot dissipate — the search gets a clean signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis.world import LeniaParams
from genesis.flowlenia import FlowWorld
from genesis.evolve import random_patch, mutate_patch, PATCH, UPSCALE, _blur2d

GB = {"mu_g": (0.08, 0.22), "sigma_g": (0.012, 0.045),
      "mu_k": (0.30, 0.60), "sigma_k": (0.08, 0.20), "R": (10.0, 15.0)}


@dataclass
class FlowCreature:
    rule: dict
    asym: float
    phi: float
    patch: np.ndarray


def random_creature(rng):
    rule = {k: rng.uniform(lo, hi) for k, (lo, hi) in GB.items()}
    return FlowCreature(rule, rng.uniform(0.0, 0.8), rng.uniform(0, 6.28), random_patch(rng))


def mutate(c, rng, scale=0.12):
    rule = {}
    for k, (lo, hi) in GB.items():
        v = c.rule[k] + (rng.normal(0, (hi - lo) * scale) if rng.random() < 0.6 else 0)
        rule[k] = float(np.clip(v, lo, hi))
    asym = float(np.clip(c.asym + rng.normal(0, 0.12), 0.0, 0.95))
    phi = float(c.phi + rng.normal(0, 0.4))
    return FlowCreature(rule, asym, phi, mutate_patch(c.patch, rng))


def _place(shape, patch):
    up = _blur2d(np.kron(patch, np.ones((UPSCALE, UPSCALE))), 2)
    field = np.zeros(shape)
    h, w = up.shape
    r0, c0 = shape[0] // 2 - h // 2, shape[1] // 2 - w // 2
    field[r0:r0 + h, c0:c0 + w] = np.clip(up, 0, 1)
    return field * 0.6


def evaluate(c, shape=(100, 100), steps=150):
    p = LeniaParams(R=c.rule["R"], mu_k=c.rule["mu_k"], sigma_k=c.rule["sigma_k"],
                    mu_g=c.rule["mu_g"], sigma_g=c.rule["sigma_g"], dt=0.2)
    w = FlowWorld(shape, p, flow_clip=0.9, asym=c.asym, phi=c.phi)
    w.A = _place(shape, c.patch)
    yy, xx = np.indices(shape)
    tot0 = w.A.sum() + 1e-9
    cy0, cx0 = (w.A * yy).sum() / tot0, (w.A * xx).sum() / tot0
    for _ in range(steps):
        w.step()
        if (w.A > 0.05).mean() > 0.4 or w.A.max() < 0.08:
            return 0.0, dict(travel=0.0, conc=0.0, alive=False)
    tot = w.A.sum() + 1e-9
    cy, cx = (w.A * yy).sum() / tot, (w.A * xx).sum() / tot
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    conc = float(w.A[d2 < (2.5 * c.rule["R"]) ** 2].sum() / tot)
    # wrap-aware net displacement
    dy = min(abs(cy - cy0), shape[0] - abs(cy - cy0))
    dx = min(abs(cx - cx0), shape[1] - abs(cx - cx0))
    travel = float(np.hypot(dy, dx) / c.rule["R"])
    score = (travel if conc > 0.6 else 0.3 * conc)     # reward motion of a compact body
    return score, dict(travel=travel, conc=conc, alive=True)


def search(pop=22, gens=30, seed=0, shape=(100, 100), steps=150, verbose=True):
    rng = np.random.default_rng(seed)
    population = [random_creature(rng) for _ in range(pop)]
    best = None
    trail = []
    for gen in range(gens):
        scored = [(s, c, m) for c in population for s, m in [evaluate(c, shape, steps)]]
        scored.sort(key=lambda t: t[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        gb = scored[0]
        trail.append(dict(gen=gen, best=gb[0], travel=gb[2]["travel"], conc=gb[2]["conc"]))
        if verbose:
            print(f"  gen {gen:2d}: best_travel={gb[2]['travel']:.2f}R conc={gb[2]['conc']:.2f} "
                  f"score={gb[0]:.2f}")
        elites = [c for _, c, _ in scored[:max(3, pop // 4)]]
        children = list(elites)
        while len(children) < pop - 2:
            children.append(mutate(elites[rng.integers(len(elites))], rng))
        children += [random_creature(rng), random_creature(rng)]
        population = children
    return dict(best=best[1], score=best[0], metrics=best[2], trail=trail)
