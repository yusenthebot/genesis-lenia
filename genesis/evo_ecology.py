"""Evolution running INSIDE the ecology — selection that acts, not a sweep.

Round 6 measured a fitness landscape by hand (sweep foraging gain, read lifetimes).
Here selection RUNS: foraging gain is a HERITABLE gene. A creature that forages well
banks energy and REPRODUCES (offspring inherit its gain plus a small mutation); a poor
forager starves and its gene dies with it. Starting from a population of random gains,
the population should DISCOVER the optimal foraging strategy on its own — mean gain
converging to the ecology's optimum with nobody specifying it.

Each creature is a Lenia body in its own channel sharing one food field (as in ecology.py);
the additions here are birth, death, inheritance and mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch

GMAX = 18.0          # gene bound on foraging gain


@dataclass
class Indiv:
    A: np.ndarray
    gamma: float            # heritable foraging gain
    energy: float
    alive: bool = True
    age: int = 0
    born_mass: float = 0.0
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class EvoEcology:
    def __init__(self, shape, rule: Genome, patch, *, n0=14, sense_sigma=14.0,
                 eta=0.3, decay=0.012, feed=0.06, energy0=1.3, spawn=22, frad=8.0,
                 repro_energy=3.2, repro_cost=2.0, mut=1.6, max_pop=34, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.patch = patch
        self.eta, self.decay, self.feed, self.energy0 = eta, decay, feed, energy0
        self.spawn, self.frad = spawn, frad
        self.repro_energy, self.repro_cost, self.mut = repro_energy, repro_cost, mut
        self.max_pop = max_pop
        self.F = np.zeros(shape)
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._build_sense(sense_sigma)
        self.pop = []
        H, W = shape
        for _ in range(n0):
            g = float(self.rng.uniform(0, GMAX))           # random starting gains
            pos = self.rng.integers([0, 0], [H, W])
            self.pop.append(self._make(g, pos))
        for c in self.pop:                                  # settle into gliders
            for _ in range(25):
                self._lenia(c)
            c.born_mass = float(c.A.sum())
        self.history = []          # (t, pop, mean_gamma, std_gamma)

    # -- construction helpers ---------------------------------------------
    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _make(self, gamma, pos):
        H, W = self.shape
        A = place_patch(self.shape, self.patch)
        A = np.roll(A, (int(pos[0]) - H // 2, int(pos[1]) - W // 2), axis=(0, 1))
        return Indiv(A=A, gamma=float(np.clip(gamma, 0, GMAX)), energy=self.energy0)

    # -- dynamics ----------------------------------------------------------
    def _lenia(self, c):
        U = self.world.potential(c.A)
        c.A = np.clip(c.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _sense(self):
        ax = (0, 1)
        return np.fft.irfftn(np.fft.rfftn(self.F, axes=ax) * self.Ks_fft,
                             s=self.shape, axes=ax)

    def add_food(self, n=1):
        H, W = self.shape
        yy, xx = np.indices(self.shape)
        for _ in range(n):
            p = self.rng.integers([0, 0], [H, W])
            self.F = self.F + np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2)
                                     / self.frad ** 2)

    def _centroid(self, A):
        m = A.sum() + 1e-9
        yy, xx = np.indices(self.shape)
        return int((A * yy).sum() / m), int((A * xx).sum() / m)

    def step(self):
        if self.t % self.spawn == 0:
            self.add_food(1)
        S = self._sense()
        gS = np.gradient(S)
        births = []
        for k in self.rng.permutation(len(self.pop)):
            c = self.pop[k]
            if not c.alive:
                continue
            c.age += 1
            U = self.world.potential(c.A)
            c.A = c.A + self.dt * self.world.growth(U)
            bite = np.clip(self.eta * c.A, 0.0, 1.0) * self.F
            self.F = self.F - bite
            c.energy += self.feed * float(bite.sum()) - self.decay
            if c.energy <= 0.0:                              # starve -> die
                c.A = c.A * 0.9
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False
            c.A = np.clip(c.A, 0.0, 1.0)
            if c.gamma:                                      # forage
                tot = c.A.sum() + 1e-9
                for ax in range(2):
                    g = float((c.A * gS[ax]).sum() / tot)
                    c.drift[ax] += c.gamma * g
                    sft = int(round(c.drift[ax]))
                    if sft:
                        c.A = np.roll(c.A, sft, axis=ax)
                        c.drift[ax] -= sft
            # reproduce when well-fed and there is room
            n_alive = sum(x.alive for x in self.pop) + len(births)
            if (c.alive and c.energy >= self.repro_energy and c.age > 30
                    and n_alive < self.max_pop):
                c.energy -= self.repro_cost
                cy, cx = self._centroid(c.A)
                off = self.rng.integers(-25, 26, 2)
                child_g = c.gamma + self.rng.normal(0, self.mut)
                child = self._make(child_g, (cy + off[0], cx + off[1]))
                child.born_mass = child.A.sum()
                births.append(child)
        self.pop.extend(births)
        self.pop = [c for c in self.pop if c.alive]          # bury the dead
        self.t += 1
        gammas = np.array([c.gamma for c in self.pop]) if self.pop else np.array([0.0])
        self.history.append((self.t, len(self.pop), float(gammas.mean()),
                             float(gammas.std())))

    # -- reporting ---------------------------------------------------------
    def mean_gamma(self):
        return float(np.mean([c.gamma for c in self.pop])) if self.pop else 0.0

    def composite(self):
        H, W = self.shape
        rgb = np.zeros((H, W, 3))
        for c in self.pop:
            hue = min(c.gamma / GMAX, 1.0)
            col = np.array([1.0 - 0.7 * hue, 0.15, 0.3 + 0.7 * hue])
            rgb += c.A[..., None] * col[None, None, :]
        rgb[..., 1] += np.clip(self.F, 0, 1) * 0.55
        return np.clip(rgb, 0, 1)
