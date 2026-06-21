"""Ecology — many creatures in one world, competing for shared, finite food.

Each creature is a Lenia body in its own channel; all channels share ONE food field.
Every creature senses the food gradient and drifts up it (the round-3 reflex), eats
food under its body into a private energy reserve (the round-4 metabolism), and dies
when that reserve empties. Food is finite per spawn, so a meal taken by one creature is
denied to the others — exploitation competition. No interaction is coded beyond that
shared resource; who lives and who starves is decided by foraging skill under contest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.world import World, LeniaParams
from genesis.evolve import Genome, place_patch


@dataclass
class Creature:
    A: np.ndarray
    gamma: float                 # sensorimotor gain (its foraging skill)
    energy: float
    label: int
    alive: bool = True
    born_mass: float = 0.0
    death_t: int = -1
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class Ecology:
    def __init__(self, shape, rule: Genome, patch, specs, *, sense_sigma=14.0,
                 eta=0.3, decay=0.012, feed=0.06, energy0=1.3, spawn=70, frad=8.0,
                 seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())   # shared kernel + growth
        self.dt = rule.dt
        self.eta, self.decay, self.feed = eta, decay, feed
        self.spawn, self.frad = spawn, frad
        self.F = np.zeros(shape)
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._build_sense(sense_sigma)
        self.creatures = []
        H, W = shape
        for i, (gamma, center) in enumerate(specs):
            A = place_patch(shape, patch)
            # move the seeded body to its start location
            A = np.roll(A, (center[0] - H // 2, center[1] - W // 2), axis=(0, 1))
            c = Creature(A=A, gamma=float(gamma), energy=energy0, label=i)
            self.creatures.append(c)
        for c in self.creatures:
            for _ in range(25):                       # settle bodies into gliders
                self._lenia_only(c)
            c.born_mass = float(c.A.sum())

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _lenia_only(self, c):
        U = self.world.potential(c.A)
        c.A = np.clip(c.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _sense(self):
        ax = (0, 1)
        return np.fft.irfftn(np.fft.rfftn(self.F, axes=ax) * self.Ks_fft,
                             s=self.shape, axes=ax)

    def add_food(self, n=1):
        H, W = self.shape
        for _ in range(n):
            pos = self.rng.integers([0, 0], [H, W])
            yy, xx = np.indices(self.shape)
            d2 = ((yy - pos[0]) ** 2 + (xx - pos[1]) ** 2)
            self.F = self.F + np.exp(-0.5 * d2 / self.frad ** 2)

    def step(self):
        if self.t % self.spawn == 0:
            self.add_food(1)
        S = self._sense()
        gS = np.gradient(S)
        order = self.rng.permutation(len(self.creatures))   # fair contest order
        for k in order:
            c = self.creatures[k]
            if not c.alive:
                continue
            U = self.world.potential(c.A)
            c.A = c.A + self.dt * self.world.growth(U)
            bite = np.clip(self.eta * c.A, 0.0, 1.0) * self.F
            self.F = self.F - bite                          # food removed for everyone
            c.energy += self.feed * float(bite.sum()) - self.decay
            if c.energy <= 0.0:
                c.energy = 0.0
                c.A = c.A * 0.9
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False
                    c.death_t = self.t
            c.A = np.clip(c.A, 0.0, 1.0)
            if c.gamma:                                     # forage: drift up gradient
                tot = c.A.sum() + 1e-9
                for ax in range(2):
                    g = float((c.A * gS[ax]).sum() / tot)
                    c.drift[ax] += c.gamma * g
                    sft = int(round(c.drift[ax]))
                    if sft:
                        c.A = np.roll(c.A, sft, axis=ax)
                        c.drift[ax] -= sft
        self.t += 1

    def n_alive(self):
        return sum(c.alive for c in self.creatures)

    def composite(self):
        """RGB frame: each creature a colour by its gamma, food in green."""
        H, W = self.shape
        rgb = np.zeros((H, W, 3))
        gmax = max((c.gamma for c in self.creatures), default=1.0) or 1.0
        for c in self.creatures:
            if not c.alive:
                continue
            hue = c.gamma / gmax
            col = np.array([1.0 - 0.7 * hue, 0.2, 0.3 + 0.7 * hue])  # red=poor, blue=skilled
            rgb += c.A[..., None] * col[None, None, :]
        rgb[..., 1] += np.clip(self.F, 0, 1) * 0.6                   # food green
        return np.clip(rgb, 0, 1)
