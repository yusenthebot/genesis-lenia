"""Open-ended MINDS — a zoo of foraging STRATEGIES, not body shapes.

Round 22 illuminated a zoo of body morphologies. This goes deeper: keep the body FIXED (the
evolved glider) and vary the MIND — a small foraging policy — then illuminate the space of
foraging BEHAVIOURS. The creature senses scattered food and drifts under a 4-parameter policy
(sensing range, gain, exploration, momentum). Different policies forage in qualitatively
different ways: beeliners go straight to food, wanderers cover ground, sitters stay local.

MAP-Elites illuminates a behaviour space (how much it ROAMS x how DIRECTLY it moves), keeping
the best-eating forager in each niche -> a zoo of strategies; fitness search converges to one.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch

GRID = 8
ROAM_MAX = 680.0       # total path length axis (cells/episode) — paths vary ~140..670
SPREAD_MAX = 0.30      # trajectory spread axis (gyration / arena) — local vs roving
N_FOOD = 16            # scattered food; different strategies harvest different amounts
# policy genome = [gain, sense_sigma, explore, momentum] in these bounds:
BOUNDS = np.array([[0.3, 3.2], [6.0, 20.0], [0.0, 1.6], [0.0, 0.9]])


def random_policy(rng):
    return BOUNDS[:, 0] + rng.random(4) * (BOUNDS[:, 1] - BOUNDS[:, 0])


def mutate_policy(p, rng, scale=0.18):
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    out = p + rng.normal(0, 1, 4) * span * scale
    return np.clip(out, BOUNDS[:, 0], BOUNDS[:, 1])


class ForagerMind:
    def __init__(self, shape, rule: Genome, patch, policy, n_food=7, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.gain, self.sigma, self.explore, self.momentum = policy
        self.rng = np.random.default_rng(seed)
        self._build_sense(self.sigma)
        self.A = place_patch(shape, patch)
        for _ in range(20):
            self._lenia()
        yy, xx = np.indices(shape)
        self.foods = [self.rng.uniform(0.12, 0.88, 2) * np.array(shape)
                      for _ in range(n_food)]
        self.frad = 7.0
        self._render_food()
        self.drift = np.zeros(2)
        self.last = np.zeros(2)
        self.eaten = 0
        self.path = 0.0
        self.start = self.body_center()
        self.traj = [self.start.copy()]

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2); Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _lenia(self):
        U = self.world.potential(self.A)
        self.A = np.clip(self.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _render_food(self):
        yy, xx = np.indices(self.shape)
        F = np.zeros(self.shape)
        for p in self.foods:
            F = np.maximum(F, np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2)
                                     / self.frad ** 2))
        self.F = F

    def body_center(self):
        tot = self.A.sum() + 1e-9
        yy, xx = np.indices(self.shape)
        return np.array([float((self.A * yy).sum() / tot),
                         float((self.A * xx).sum() / tot)])

    def _grad(self):
        ax = (0, 1)
        S = np.fft.irfftn(np.fft.rfftn(self.F, axes=ax) * self.Ks_fft, s=self.shape, axes=ax)
        g = np.gradient(S)
        tot = self.A.sum() + 1e-9
        d = np.array([float((self.A * g[0]).sum() / tot), float((self.A * g[1]).sum() / tot)])
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(2)

    def step(self):
        self._lenia()
        grad = self._grad()
        noise = self.rng.normal(0, 1, 2)
        cmd = self.gain * grad + self.explore * noise + self.momentum * self.last
        self.last = cmd
        moved = np.zeros(2)
        for axn in range(2):
            self.drift[axn] += float(cmd[axn])
            s = int(round(self.drift[axn]))
            if s:
                self.A = np.roll(self.A, s, axis=axn); self.drift[axn] -= s; moved[axn] = s
        bc = self.body_center()
        self.path += np.linalg.norm(moved)
        self.traj.append(bc.copy())
        # eat nearby food (remove blob, fixed budget — no respawn)
        kept = []
        for p in self.foods:
            if np.linalg.norm(bc - p) < self.frad + 4.0:
                self.eaten += 1
            else:
                kept.append(p)
        if len(kept) != len(self.foods):
            self.foods = kept; self._render_food()


def run(rule, patch, policy, T=170, shape=(110, 110), seed=0, n_food=N_FOOD):
    f = ForagerMind(shape, rule, patch, policy, n_food=n_food, seed=seed)
    for _ in range(T):
        f.step()
        if not f.foods:
            break
    traj = np.array(f.traj)
    spread = float(np.linalg.norm(traj - traj.mean(0), axis=1).mean() / max(shape))
    return dict(eaten=f.eaten, path=f.path, spread=spread, traj=traj, n_food=n_food)


def behaviour_cell(m):
    """(roaming x trajectory-spread) -> grid cell. None if it barely moved."""
    if m["path"] < 6.0:
        return None
    i = int(np.clip(m["path"] / ROAM_MAX, 0, 0.999) * GRID)
    j = int(np.clip(m["spread"] / SPREAD_MAX, 0, 0.999) * GRID)
    return i, j


def map_elites(rule, patch, n_eval=560, seed=0, eval_seeds=(0, 1, 2), track_every=20):
    rng = np.random.default_rng(seed)
    archive, coverage = {}, []
    for k in range(n_eval):
        if k < 40 or not archive:
            pol = random_policy(rng)
        else:
            keys = list(archive.keys())
            pol = mutate_policy(archive[keys[rng.integers(len(keys))]]["pol"], rng)
        ms = [run(rule, patch, pol, seed=s) for s in eval_seeds]
        eaten = float(np.mean([m["eaten"] for m in ms]))
        m = ms[0]
        cell = behaviour_cell(m)
        if cell is not None and (cell not in archive or eaten > archive[cell]["q"]):
            archive[cell] = dict(pol=pol, q=eaten, m=m)
        if k % track_every == 0:
            coverage.append((k, len(archive)))
    coverage.append((n_eval, len(archive)))
    return archive, coverage


def fitness_search(rule, patch, n_eval=560, seed=0, pop=20, eval_seeds=(0, 1, 2),
                   track_every=20):
    """Population GA maximizing food eaten; track the population's behavioural diversity."""
    rng = np.random.default_rng(seed)

    def score(pol):
        return float(np.mean([run(rule, patch, pol, seed=s)["eaten"] for s in eval_seeds]))

    pop_list = [random_policy(rng) for _ in range(pop)]
    scored = [(p, score(p), run(rule, patch, p, seed=0)) for p in pop_list]
    evals = pop
    coverage = []

    def diversity(items):
        return len({behaviour_cell(m) for _, _, m in items} - {None})

    coverage.append((evals, diversity(scored)))
    while evals < n_eval:
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = scored[: pop // 2]
        children = []
        for _ in range(pop - len(survivors)):
            par = survivors[rng.integers(len(survivors))][0]
            child = mutate_policy(par, rng, scale=0.12)
            children.append((child, score(child), run(rule, patch, child, seed=0)))
            evals += 1
            if evals % track_every == 0:
                coverage.append((evals, diversity(survivors + children)))
        scored = survivors + children
    coverage.append((evals, diversity(scored)))
    return scored[0][0], coverage
