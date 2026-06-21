"""Embodied memory — a recurrent brain inside a Lenia creature, where memory PAYS.

This reunites the two tracks: the embodied Lenia forager (rounds 3-13) and the recurrent
mind (rounds 15-16). The creature's foraging drift is produced by a small recurrent net.
The world FLASHES: the food gradient is visible only briefly, then goes dark while the food
stays put. A feedforward (memoryless) brain only drifts while it can see the gradient and
stalls in the dark; a recurrent brain holds the last-seen direction in its hidden state and
keeps coasting to the food. We evolve the net (ES) and ask whether memory lets the embodied
creature EAT MORE — payoff, not just capability.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch

H = 8                 # hidden units
IN = 3                # input [grad_x, grad_y, bias]
OUT = 2               # output drift [dx, dy]
NPARAMS = H * H + H * IN + H + OUT * H + OUT


def unpack(theta):
    i = 0
    Wh = theta[i:i + H * H].reshape(H, H); i += H * H
    Wi = theta[i:i + H * IN].reshape(H, IN); i += H * IN
    bh = theta[i:i + H]; i += H
    Wo = theta[i:i + OUT * H].reshape(OUT, H); i += OUT * H
    bo = theta[i:i + OUT]
    return Wh, Wi, bh, Wo, bo


class EmbodiedMemoryForager:
    def __init__(self, shape, rule: Genome, patch, theta, recurrent=True, *,
                 sense_sigma=12.0, gain=3.0, eta=0.3, feed=0.06, decay=0.006,
                 energy0=1.5, flash_on=6, flash_off=16, frad=8.0, gscale=12.0, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.recurrent = recurrent
        self.params = unpack(theta)
        self.gain, self.eta, self.feed, self.decay = gain, eta, feed, decay
        self.flash_on, self.flash_off, self.gscale = flash_on, flash_off, gscale
        self.frad = frad
        self.rng = np.random.default_rng(seed)
        self._build_sense(sense_sigma)
        self.A = place_patch(shape, patch)
        for _ in range(20):
            self._lenia()
        self.F = np.zeros(shape)
        self.spawn_food()
        self.h = np.zeros(H)
        self.drift = np.zeros(2)
        self.energy = energy0
        self.eaten = 0.0
        self.collected = 0
        self.t = 0
        self.alive = True

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

    def spawn_food(self):
        H_, W_ = self.shape
        yy, xx = np.indices(self.shape)
        p = self.rng.integers([0, 0], [H_, W_])
        self.F = np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / self.frad ** 2)

    def _grad(self):
        ax = (0, 1)
        S = np.fft.irfftn(np.fft.rfftn(self.F, axes=ax) * self.Ks_fft, s=self.shape, axes=ax)
        g = np.gradient(S)
        tot = self.A.sum() + 1e-9
        return (float((self.A * g[0]).sum() / tot), float((self.A * g[1]).sum() / tot))

    def step(self):
        self._lenia()
        dark = (self.t % (self.flash_on + self.flash_off)) >= self.flash_on
        if dark:
            gy, gx = 0.0, 0.0                 # the gradient signal is invisible
        else:
            gy, gx = self._grad()
        o = np.array([self.gscale * gy, self.gscale * gx, 1.0])
        Wh, Wi, bh, Wo, bo = self.params
        if not self.recurrent:
            Wh = np.zeros_like(Wh)            # memory ablated
        self.h = np.tanh(Wh @ self.h + Wi @ o + bh)
        out = Wo @ self.h + bo               # drift command
        for axn in range(2):
            self.drift[axn] += self.gain * float(out[axn])
            s = int(round(self.drift[axn]))
            if s:
                self.A = np.roll(self.A, s, axis=axn)
                self.drift[axn] -= s
        bite = np.clip(self.eta * self.A, 0, 1) * self.F
        self.F = self.F - bite
        meal = float(bite.sum())
        self.eaten += meal
        self.energy += self.feed * meal - self.decay
        if self.F.sum() < 0.15 * (2 * np.pi * self.frad ** 2):   # food collected -> respawn
            self.collected += 1
            self.spawn_food()
        if self.energy <= 0:
            self.alive = False
        self.energy = min(self.energy, 6.0)
        self.t += 1


def episode(rule, patch, theta, recurrent, T=240, shape=(110, 110), seed=0,
            flash_off=16, record=False):
    """flash_off=0 -> a STEADY world (gradient always visible); >0 -> flashing."""
    f = EmbodiedMemoryForager(shape, rule, patch, theta, recurrent,
                              flash_off=flash_off, seed=seed)
    frames = []
    for _ in range(T):
        f.step()
        if record and f.t % 6 == 0:
            dark = (f.t % (f.flash_on + f.flash_off)) >= f.flash_on if f.flash_off else False
            frames.append((np.clip(f.A, 0, 1).copy(), np.clip(f.F, 0, 1).copy(),
                           f.collected, f.t, dark))
        if not f.alive:
            break
    return f.collected, f.eaten, f.t, frames


def fitness(rule, patch, theta, recurrent, seeds=(0, 1, 2), flash_off=16):
    return float(np.mean([episode(rule, patch, theta, recurrent, seed=s,
                                  flash_off=flash_off)[0] for s in seeds]))


def evolve_es(rule, patch, recurrent=True, gens=30, pop=20, sigma=0.3, lr=0.2,
              flash_off=16, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.4, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(rule, patch, theta + sigma * eps[i], recurrent,
                                 seeds=(0, 1), flash_off=flash_off) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(fitness(rule, patch, theta, recurrent, seeds=(0, 1, 2),
                             flash_off=flash_off))
    return theta, np.array(curve)
