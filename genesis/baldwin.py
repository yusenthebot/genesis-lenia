"""The Baldwin effect — evolving HOW to learn, not just whether.

Round 13 made learning on/off heritable with a fixed learning rate. Here the LEARNING
RATE itself is a heritable gene: each creature is born naive (brain weights reset) and
learns the current rule at its own inherited rate, which mutates and is selected across
generations. The tradeoff that makes this interesting: a high rate adapts fast but
overreacts to noise; a low rate is stable but slow. So the optimal rate should depend on
how fast the world changes — and evolution should discover it. We watch the population's
mean learning rate self-tune to its world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch

LR_MAX = 0.8


@dataclass
class Being:
    A: np.ndarray
    lr: float                # heritable LEARNING RATE (0 = never learns)
    w: np.ndarray            # plastic drift weights [A, B], reset each life
    energy: float
    alive: bool = True
    age: int = 0
    born_mass: float = 0.0
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class BaldwinEcology:
    def __init__(self, shape, rule: Genome, patch, *, n0=22, sense_sigma=14.0,
                 gain=0.6, eta=0.3, feed=0.06, pen=0.04, decay=0.012, energy0=1.4,
                 flip=300, frad=8.0, spawn=10, food_decay=0.02, repro_energy=2.6,
                 repro_cost=1.6, lr_mut=0.10, max_pop=38, reward_noise=0.0, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.patch = patch
        self.gain, self.eta = gain, eta
        self.feed, self.pen, self.decay, self.energy0 = feed, pen, decay, energy0
        self.flip, self.frad, self.spawn, self.food_decay = flip, frad, spawn, food_decay
        self.repro_energy, self.repro_cost = repro_energy, repro_cost
        self.lr_mut, self.max_pop, self.reward_noise = lr_mut, max_pop, reward_noise
        self.rng = np.random.default_rng(seed)
        self.FA = np.zeros(shape); self.FB = np.zeros(shape)
        self.active = 0
        self.t = 0
        self._build_sense(sense_sigma)
        self.pop = [self._make(float(self.rng.uniform(0, LR_MAX))) for _ in range(n0)]
        for c in self.pop:
            for _ in range(20):
                self._lenia(c)
            c.born_mass = float(c.A.sum())
        self.history = []     # (t, pop, mean_lr, std_lr)

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _make(self, lr):
        H, W = self.shape
        pos = self.rng.integers([0, 0], [H, W])
        A = np.roll(place_patch(self.shape, self.patch),
                    (int(pos[0]) - H // 2, int(pos[1]) - W // 2), axis=(0, 1))
        return Being(A=A, lr=float(np.clip(lr, 0, LR_MAX)), w=np.zeros(2),
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
        gA = np.gradient(self._sense(self.FA))
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
            for X, ate in ((0, ateA), (1, ateB)):
                if ate > 0.5:
                    val = 1.0 if self.active == X else -1.0
                    if self.reward_noise and self.rng.random() < self.reward_noise:
                        val = -val                      # noisy reward: a high lr overreacts
                    c.w[X] += c.lr * (val - c.w[X])     # learn at the creature's own rate
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
                child = self._make(c.lr + self.rng.normal(0, self.lr_mut))  # inherit + mutate lr
                child.born_mass = child.A.sum()
                births.append(child)
        self.pop.extend(births)
        self.pop = [c for c in self.pop if c.alive]
        self.t += 1
        lrs = np.array([c.lr for c in self.pop]) if self.pop else np.array([0.0])
        self.history.append((self.t, len(self.pop), float(lrs.mean()), float(lrs.std())))

    def mean_lr(self):
        return float(np.mean([c.lr for c in self.pop])) if self.pop else 0.0

    def composite(self):
        H, W = self.shape
        rgb = np.zeros((H, W, 3))
        for c in self.pop:
            hue = min(c.lr / LR_MAX, 1.0)               # low lr = red, high lr = blue
            col = np.array([1.0 - hue, 0.1, hue])
            rgb += c.A[..., None] * col[None, None, :]
        rgb[..., 1] += np.clip(self.FA + self.FB, 0, 1) * 0.5
        return np.clip(rgb, 0, 1)
