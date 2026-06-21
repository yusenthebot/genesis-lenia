"""Learning under selection — does knowing more make a creature win?

Round 12 measured that a learner's brain carries more information about the world. Here
we ask whether that pays: learners (plastic brains that adapt within life) and fixed-reflex
creatures (a non-plastic strategy, inherited and tuned only across generations) compete in
ONE world for the same food, with the 'is_learner' trait heritable. Two food types, only
one nutritious, the rule reversing at a tunable rate.

The expectation (Baldwin / evolution of learning): in a STABLE world the fixed strategy
wins (it pays no learning cost); in a CHANGING world learners win (they adapt within life
while the fixed strategy, tuned only across generations, cannot keep up). We watch the
fraction of learners in the population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch


@dataclass
class Being:
    A: np.ndarray
    w: np.ndarray            # drift weights [A, B]
    learner: bool
    energy: float
    alive: bool = True
    age: int = 0
    born_mass: float = 0.0
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class EvoLearning:
    def __init__(self, shape, rule: Genome, patch, *, n0=20, learner_frac=0.5,
                 sense_sigma=14.0, lr=0.3, gain=0.6, eta=0.3, feed=0.06, pen=0.04,
                 decay=0.012, energy0=1.4, flip=300, frad=8.0, spawn=10,
                 food_decay=0.02, repro_energy=2.6, repro_cost=1.6, type_mut=0.04,
                 w_mut=0.25, max_pop=40, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.patch = patch
        self.lr, self.gain, self.eta = lr, gain, eta
        self.feed, self.pen, self.decay, self.energy0 = feed, pen, decay, energy0
        self.flip, self.frad, self.spawn, self.food_decay = flip, frad, spawn, food_decay
        self.repro_energy, self.repro_cost = repro_energy, repro_cost
        self.type_mut, self.w_mut, self.max_pop = type_mut, w_mut, max_pop
        self.rng = np.random.default_rng(seed)
        self.FA = np.zeros(shape); self.FB = np.zeros(shape)
        self.active = 0
        self.t = 0
        self._build_sense(sense_sigma)
        self.pop = []
        for i in range(n0):
            learner = i < int(n0 * learner_frac)
            w = np.zeros(2) if learner else self.rng.choice([[1.0, -1.0], [-1.0, 1.0]])
            self.pop.append(self._make(np.array(w, float), learner))
        for c in self.pop:
            for _ in range(20):
                self._lenia(c)
            c.born_mass = float(c.A.sum())
        self.history = []      # (t, n_learner, n_fixed)

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _make(self, w, learner):
        H, W = self.shape
        pos = self.rng.integers([0, 0], [H, W])
        A = np.roll(place_patch(self.shape, self.patch),
                    (int(pos[0]) - H // 2, int(pos[1]) - W // 2), axis=(0, 1))
        return Being(A=A, w=np.array(w, float), learner=bool(learner),
                     energy=self.energy0)

    def _lenia(self, c):
        U = self.world.potential(c.A)
        c.A = np.clip(c.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _sense(self, F):
        ax = (0, 1)
        return np.fft.irfftn(np.fft.rfftn(F, axes=ax) * self.Ks_fft, s=self.shape, axes=ax)

    def add_food(self):
        H, W = self.shape
        yy, xx = np.indices(self.shape)
        for F in (self.FA, self.FB):
            p = self.rng.integers([0, 0], [H, W])
            F += np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / self.frad ** 2)

    def step(self):
        if self.t > 0 and self.t % self.flip == 0:
            self.active = 1 - self.active
        self.FA *= (1 - self.food_decay); self.FB *= (1 - self.food_decay)
        if self.t % self.spawn == 0:
            self.add_food()
        gA = np.gradient(self._sense(self.FA))         # shared sensed gradients
        gB = np.gradient(self._sense(self.FB))
        births = []
        for k in self.rng.permutation(len(self.pop)):
            c = self.pop[k]
            if not c.alive:
                continue
            c.age += 1
            self._lenia(c)
            tot = c.A.sum() + 1e-9
            ga = [float((c.A * gA[ax]).sum() / tot) for ax in range(2)]
            gb = [float((c.A * gB[ax]).sum() / tot) for ax in range(2)]
            for ax in range(2):
                v = self.gain * (c.w[0] * ga[ax] + c.w[1] * gb[ax]) + self.rng.normal(0, 0.18)
                c.drift[ax] += v
                s = int(round(c.drift[ax]))
                if s:
                    c.A = np.roll(c.A, s, axis=ax); c.drift[ax] -= s
            biteA = np.clip(self.eta * c.A, 0, 1) * self.FA
            biteB = np.clip(self.eta * c.A, 0, 1) * self.FB
            self.FA -= biteA; self.FB -= biteB
            ateA, ateB = float(biteA.sum()), float(biteB.sum())
            if c.learner:                              # plastic: learn each type's value
                for X, ate in ((0, ateA), (1, ateB)):
                    if ate > 0.5:
                        val = 1.0 if self.active == X else -1.0
                        c.w[X] += self.lr * (val - c.w[X])
            nut = ateA if self.active == 0 else ateB
            non = ateB if self.active == 0 else ateA
            c.energy += self.feed * nut - self.pen * non - self.decay
            if c.energy <= 0:
                c.A *= 0.9
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False
            c.A = np.clip(c.A, 0, 1)
            n_alive = sum(x.alive for x in self.pop) + len(births)
            if (c.alive and c.energy >= self.repro_energy and c.age > 30
                    and n_alive < self.max_pop):
                c.energy -= self.repro_cost
                child_learner = c.learner
                if self.rng.random() < self.type_mut:
                    child_learner = not child_learner          # rare strategy mutation
                if child_learner:
                    w = np.zeros(2)                             # learner re-learns from scratch
                else:
                    base = c.w if not c.learner else np.array([1.0, -1.0])
                    w = base + self.rng.normal(0, self.w_mut, 2)  # fixed: inherit + mutate
                child = self._make(w, child_learner)
                child.born_mass = child.A.sum()
                births.append(child)
        self.pop.extend(births)
        self.pop = [c for c in self.pop if c.alive]
        self.t += 1
        nl = sum(c.learner for c in self.pop)
        self.history.append((self.t, nl, len(self.pop) - nl))

    def learner_fraction(self):
        return (sum(c.learner for c in self.pop) / len(self.pop)) if self.pop else 0.0

    def composite(self):
        H, W = self.shape
        rgb = np.zeros((H, W, 3))
        for c in self.pop:
            if c.learner:
                rgb[..., 2] += c.A                      # learners = blue
            else:
                rgb[..., 0] += c.A                      # fixed = red
        rgb[..., 1] += np.clip(self.FA + self.FB, 0, 1) * 0.5
        return np.clip(rgb, 0, 1)
