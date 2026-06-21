"""Embodied learning — the plastic brain inside a Lenia body.

Until now the learner (round 9) was a detached point and the body (rounds 2-8) was a
fixed reflex. Here they are ONE creature: a Lenia glider whose foraging drift is a
PLASTIC policy over two food types,

    drift = w_A * grad(sense F_A) + w_B * grad(sense F_B),

with w learned from experience. Only one food type is nutritious at a time, and which
one FLIPS mid-life. The creature must learn, with its real body, which food to chase —
and re-learn after each reversal. Set lr=0 (and fix w) to ablate the learning.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch


class EmbodiedLearner:
    def __init__(self, shape, rule: Genome, patch, *, sense_sigma=14.0, lr=0.2,
                 eta=0.3, feed=0.06, pen=0.04, decay=0.0, flip=350, frad=8.0,
                 spawn=12, w0=(0.0, 0.0), seed=0, explore=0.18, gain=6.0,
                 food_decay=0.0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.patch = patch
        self.lr, self.eta, self.feed, self.pen, self.decay = lr, eta, feed, pen, decay
        self.flip, self.frad, self.spawn, self.explore = flip, frad, spawn, explore
        self.gain = gain                          # how decisively learned weights steer
        self.food_decay = food_decay              # food evaporates -> stays bounded
        self.rng = np.random.default_rng(seed)
        self.FA = np.zeros(shape)
        self.FB = np.zeros(shape)
        self.w = np.array(w0, dtype=np.float64)   # plastic drift weights [A, B]
        self.active = 0
        self.t = 0
        self.energy = 2.0
        self._drift = np.zeros(2)
        self._build_sense(sense_sigma)
        self.A = place_patch(shape, patch)
        for _ in range(25):
            self._lenia()
        self.hist = []        # (t, active, ateA, ateB, wA, wB, energy)

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _lenia(self):
        U = self.world.potential(self.A)
        self.A = np.clip(self.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _sense(self, F):
        ax = (0, 1)
        return np.fft.irfftn(np.fft.rfftn(F, axes=ax) * self.Ks_fft, s=self.shape, axes=ax)

    def add_food(self):
        H, W = self.shape
        yy, xx = np.indices(self.shape)
        for F in (self.FA, self.FB):
            p = self.rng.integers([0, 0], [H, W])
            F += np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / self.frad ** 2)

    def _body_grad(self, g):
        tot = self.A.sum() + 1e-9
        return [float((self.A * g[ax]).sum() / tot) for ax in range(2)]

    def step(self):
        if self.t > 0 and self.t % self.flip == 0:
            self.active = 1 - self.active            # the rule reverses
        if self.food_decay:
            self.FA *= (1.0 - self.food_decay)       # food evaporates if uneaten
            self.FB *= (1.0 - self.food_decay)
        if self.t % self.spawn == 0:
            self.add_food()
        self._lenia()
        gA = self._body_grad(np.gradient(self._sense(self.FA)))
        gB = self._body_grad(np.gradient(self._sense(self.FB)))
        for ax in range(2):
            v = (self.gain * (self.w[0] * gA[ax] + self.w[1] * gB[ax])
                 + self.rng.normal(0, self.explore))   # exploration keeps sampling both
            self._drift[ax] += v
            s = int(round(self._drift[ax]))
            if s:
                self.A = np.roll(self.A, s, axis=ax)
                self._drift[ax] -= s
        biteA = np.clip(self.eta * self.A, 0.0, 1.0) * self.FA
        biteB = np.clip(self.eta * self.A, 0.0, 1.0) * self.FB
        self.FA -= biteA; self.FB -= biteB
        ateA, ateB = float(biteA.sum()), float(biteB.sum())
        for X, ate in ((0, ateA), (1, ateB)):
            if ate > 0.5:                            # tasted type X -> learn its value
                val = 1.0 if self.active == X else -1.0
                self.w[X] += self.lr * (val - self.w[X])   # lr=0 -> frozen
        nut = ateA if self.active == 0 else ateB
        non = ateB if self.active == 0 else ateA
        self.energy += self.feed * nut - self.pen * non - self.decay
        self.t += 1
        self.hist.append((self.t, self.active, ateA, ateB, self.w[0], self.w[1],
                          self.energy))

    def composite(self):
        H, W = self.shape
        a_on = 1.0 if self.active == 0 else 0.35
        b_on = 1.0 if self.active == 1 else 0.35
        rgb = np.zeros((H, W, 3))
        rgb[..., 0] += self.A                         # body -> red/magenta
        rgb[..., 2] += self.A
        rgb[..., 1] += np.clip(self.FA, 0, 1) * 0.7 * a_on   # food A -> green-ish
        rgb[..., 2] += np.clip(self.FA, 0, 1) * 0.7 * a_on
        rgb[..., 0] += np.clip(self.FB, 0, 1) * 0.8 * b_on   # food B -> orange
        rgb[..., 1] += np.clip(self.FB, 0, 1) * 0.4 * b_on
        return np.clip(rgb, 0, 1)


def nutritious_fraction(hist, win=40):
    """Rolling fraction of eaten mass that was the nutritious type."""
    h = np.array(hist)
    if len(h) == 0:
        return np.array([]), np.array([])
    active = h[:, 1].astype(int)
    ateA, ateB = h[:, 2], h[:, 3]
    nut = np.where(active == 0, ateA, ateB)
    tot = ateA + ateB
    k = np.ones(win)
    nut_w = np.convolve(nut, k, mode="valid")
    tot_w = np.convolve(tot, k, mode="valid")
    frac = nut_w / (tot_w + 1e-9)
    return h[win - 1:, 0], frac


def run_life(rule, patch, T=2400, lr=0.2, w0=(0.0, 0.0), seed=0, shape=(140, 140),
             record=False, **kw):
    eco = EmbodiedLearner(shape, rule, patch, lr=lr, w0=w0, seed=seed, **kw)
    frames = []
    for _ in range(T):
        eco.step()
        if record and eco.t % 26 == 0:        # sparse frames -> compact GIF
            frames.append((eco.composite(), eco.active, eco.w.copy(), eco.t))
    return eco, frames
