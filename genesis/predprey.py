"""Predator and prey — a world that pushes back.

Two species share one world. PREY are the foragers of earlier rounds: they eat a food
field, carry energy, reproduce, and now also FLEE predators (heritable evasion gain).
PREDATORS eat PREY (not food): they sense the prey-density field, chase it, consume prey
mass into their own energy, reproduce, and starve without prey (heritable pursuit gain).

Each creature is a Lenia body in its own channel. Predation is a field interaction:
predators impose a removal field on the shared prey-density field, gaining energy and
taking mass from whichever prey they overlap. Two coupled heritable strategies -> a
moving fitness landscape: population cycles and a co-evolutionary arms race.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch

EVADE_MAX = 12.0
PURSUE_MAX = 16.0
FORAGE_GAIN = 4.0     # prey foraging gain fixed at the round-7 evolved optimum


@dataclass
class Agent:
    A: np.ndarray
    gene: float                 # prey: evade gain; predator: pursuit gain
    energy: float
    alive: bool = True
    age: int = 0
    born_mass: float = 0.0
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class PredPrey:
    def __init__(self, shape, rule: Genome, patch, *, n_prey=16, n_pred=5,
                 sense_sigma=14.0, seed=0,
                 # prey economy
                 eta=0.3, prey_decay=0.012, prey_feed=0.06, prey_e0=1.4,
                 food_spawn=20, frad=8.0, prey_repro=2.6, prey_cost=1.6,
                 # predator economy
                 pred_decay=0.02, pred_feed=2.2, pred_eat=0.35, pred_e0=2.5,
                 pred_repro=3.2, pred_cost=2.0,
                 mut=1.4, max_prey=40, max_pred=16):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.patch = patch
        self.rng = np.random.default_rng(seed)
        self.eta, self.frad, self.food_spawn = eta, frad, food_spawn
        self.prey_decay, self.prey_feed, self.prey_e0 = prey_decay, prey_feed, prey_e0
        self.prey_repro, self.prey_cost = prey_repro, prey_cost
        self.pred_decay, self.pred_feed = pred_decay, pred_feed
        self.pred_eat, self.pred_e0 = pred_eat, pred_e0
        self.pred_repro, self.pred_cost = pred_repro, pred_cost
        self.mut, self.max_prey, self.max_pred = mut, max_prey, max_pred
        self.F = np.zeros(shape)
        self.t = 0
        self._build_sense(sense_sigma)
        self.prey = [self._make(self.rng.uniform(0, EVADE_MAX), self.prey_e0)
                     for _ in range(n_prey)]
        self.pred = [self._make(self.rng.uniform(3, PURSUE_MAX), self.pred_e0)
                     for _ in range(n_pred)]
        for c in self.prey + self.pred:
            for _ in range(25):
                self._lenia(c)
            c.born_mass = float(c.A.sum())
        self.history = []          # (t, n_prey, n_pred, mean_evade, mean_pursue)

    def _build_sense(self, sigma):
        idx = np.indices(self.shape, dtype=np.float64)
        cc = np.array([s // 2 for s in self.shape]).reshape((2, 1, 1))
        d2 = ((idx - cc) ** 2).sum(0)
        Ks = np.exp(-0.5 * d2 / sigma ** 2)
        Ks /= Ks.max()
        self.Ks_fft = np.fft.rfftn(np.fft.ifftshift(Ks))

    def _make(self, gene, e0):
        H, W = self.shape
        pos = self.rng.integers([0, 0], [H, W])
        A = np.roll(place_patch(self.shape, self.patch),
                    (int(pos[0]) - H // 2, int(pos[1]) - W // 2), axis=(0, 1))
        return Agent(A=A, gene=float(gene), energy=float(e0))

    def _lenia(self, c):
        U = self.world.potential(c.A)
        c.A = np.clip(c.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _sense(self, fieldarr):
        ax = (0, 1)
        return np.fft.irfftn(np.fft.rfftn(fieldarr, axes=ax) * self.Ks_fft,
                             s=self.shape, axes=ax)

    def add_food(self, n=1):
        H, W = self.shape
        yy, xx = np.indices(self.shape)
        for _ in range(n):
            p = self.rng.integers([0, 0], [H, W])
            self.F = self.F + np.exp(-0.5 * ((yy - p[0]) ** 2 + (xx - p[1]) ** 2)
                                     / self.frad ** 2)

    def _drift(self, c, vec):
        for ax in range(2):
            c.drift[ax] += vec[ax]
            s = int(round(c.drift[ax]))
            if s:
                c.A = np.roll(c.A, s, axis=ax)
                c.drift[ax] -= s

    def _grad_at_body(self, c, gfield):
        tot = c.A.sum() + 1e-9
        return [float((c.A * gfield[ax]).sum() / tot) for ax in range(2)]

    def step(self):
        if self.t % self.food_spawn == 0:
            self.add_food(1)
        P = sum(c.A for c in self.prey) if self.prey else np.zeros(self.shape)
        Q = sum(c.A for c in self.pred) if self.pred else np.zeros(self.shape)
        gF = np.gradient(self._sense(self.F))
        gP = np.gradient(self._sense(P))
        gQ = np.gradient(self._sense(Q))

        # --- prey: forage food, flee predators, eat, metabolise, reproduce ---
        prey_births = []
        for k in self.rng.permutation(len(self.prey)):
            c = self.prey[k]
            if not c.alive:
                continue
            c.age += 1
            self._lenia(c)
            bite = np.clip(self.eta * c.A, 0.0, 1.0) * self.F
            self.F = self.F - bite
            c.energy += self.prey_feed * float(bite.sum()) - self.prey_decay
            if c.energy <= 0:
                c.A *= 0.9
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False
            c.A = np.clip(c.A, 0.0, 1.0)
            gf = self._grad_at_body(c, gF)
            gq = self._grad_at_body(c, gQ)
            self._drift(c, [FORAGE_GAIN * gf[a] - c.gene * gq[a] for a in range(2)])
            n_alive = sum(x.alive for x in self.prey) + len(prey_births)
            if c.alive and c.energy >= self.prey_repro and c.age > 30 and n_alive < self.max_prey:
                c.energy -= self.prey_cost
                ch = self._make(c.gene + self.rng.normal(0, self.mut), self.prey_e0)
                ch.gene = float(np.clip(ch.gene, 0, EVADE_MAX))
                ch.born_mass = ch.A.sum()
                prey_births.append(ch)
        self.prey.extend(prey_births)

        # --- predation: predators remove mass from the prey field, gain energy ---
        P = sum(c.A for c in self.prey if c.alive) if self.prey else np.zeros(self.shape)
        removal = np.zeros(self.shape)
        for c in self.pred:
            if not c.alive:
                continue
            eaten = np.clip(self.pred_eat * c.A, 0.0, 1.0) * P
            removal += eaten
            c.energy += self.pred_feed * float(eaten.sum())
        removal = np.minimum(removal, P)
        frac = removal / (P + 1e-9)              # share of prey mass removed per cell
        for c in self.prey:
            if c.alive:
                c.A = c.A * (1.0 - frac)
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False              # eaten

        # --- predators: chase prey, metabolise, reproduce, starve ---
        pred_births = []
        for k in self.rng.permutation(len(self.pred)):
            c = self.pred[k]
            if not c.alive:
                continue
            c.age += 1
            self._lenia(c)
            c.energy -= self.pred_decay
            if c.energy <= 0:
                c.A *= 0.9
                if c.A.sum() < 0.2 * c.born_mass:
                    c.alive = False
            c.A = np.clip(c.A, 0.0, 1.0)
            gp = self._grad_at_body(c, gP)
            self._drift(c, [c.gene * gp[a] for a in range(2)])
            n_alive = sum(x.alive for x in self.pred) + len(pred_births)
            if c.alive and c.energy >= self.pred_repro and c.age > 30 and n_alive < self.max_pred:
                c.energy -= self.pred_cost
                ch = self._make(c.gene + self.rng.normal(0, self.mut), self.pred_e0)
                ch.gene = float(np.clip(ch.gene, 0, PURSUE_MAX))
                ch.born_mass = ch.A.sum()
                pred_births.append(ch)
        self.pred.extend(pred_births)

        self.prey = [c for c in self.prey if c.alive]
        self.pred = [c for c in self.pred if c.alive]
        self.t += 1
        me = np.mean([c.gene for c in self.prey]) if self.prey else 0.0
        mp = np.mean([c.gene for c in self.pred]) if self.pred else 0.0
        self.history.append((self.t, len(self.prey), len(self.pred), float(me), float(mp)))

    def composite(self):
        H, W = self.shape
        rgb = np.zeros((H, W, 3))
        for c in self.prey:
            rgb[..., 2] += c.A                  # prey = blue
        for c in self.pred:
            rgb[..., 0] += c.A                  # predators = red
        rgb[..., 1] += np.clip(self.F, 0, 1) * 0.5   # food = green
        return np.clip(rgb, 0, 1)
