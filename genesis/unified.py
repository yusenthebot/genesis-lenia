"""Unification — one creature, one mind, all four faculties at once.

The arc demonstrated the faculties in isolation and showed each is EVOLVABLE: a body that
forages (rounds 3-13), a brain that remembers (15) and predicts (16), embodied memory (18),
and planning (19). This round integrates them into ONE organism on ONE survival task that
needs all of them, and proves each is load-bearing by ABLATION.

  A Lenia creature must stay on a piece of food that MOVES (constant-velocity drift, bouncing
  off the walls) and that FLASHES — visible only briefly, then dark — under METABOLISM, so
  tracking the food is literally survival. One controller integrates:
    PERCEIVE  - sense the food's relative position when it is visible
    REMEMBER  - hold an estimate of where the food is across the dark
    PREDICT   - propagate that estimate by the remembered velocity (dead-reckoning)
    PLAN/ACT  - drive the Lenia body's drift toward the predicted intercept

Ablations (each removes one faculty):
    full        - all of the above -> tracks the food, survives
    memory_only - remembers the last seen position but does NOT predict its motion ->
                  heads to a stale spot, lags the moving food
    no_memory   - retains nothing across the dark -> stalls whenever the food is invisible
Each ablation starves earlier: every faculty earns its place.
"""

from __future__ import annotations

import numpy as np

from genesis.world import World
from genesis.evolve import Genome, place_patch


class UnifiedCreature:
    def __init__(self, shape, rule: Genome, patch, *, mode="full",
                 gain=2.4, decay=0.01, feed=0.045, energy0=1.2, lead=6.0,
                 food_speed=1.5, flash_on=5, flash_off=24, frad=8.0,
                 feed_r=9.0, margin=16.0, seed=0):
        self.shape = shape
        self.world = World(shape, rule.to_params())
        self.dt = rule.dt
        self.mode = mode
        self.gain, self.decay, self.feed, self.lead = gain, decay, feed, lead
        self.flash_on, self.flash_off = flash_on, flash_off
        self.feed_r, self.frad, self.margin = feed_r, frad, margin
        self.rng = np.random.default_rng(seed)
        self.A = place_patch(shape, patch)
        for _ in range(20):
            self._lenia()
        L0, L1 = shape
        # food: random start + constant velocity (bounces off the inner box)
        self.pf = self.rng.uniform([margin, margin], [L0 - margin, L1 - margin])
        ang = self.rng.uniform(0, 2 * np.pi)
        self.vf = food_speed * np.array([np.cos(ang), np.sin(ang)])
        self.F = np.zeros(shape)
        self._render_food()
        # controller internal state (memory)
        self.pf_est = None
        self.vf_est = np.zeros(2)
        self.prev_obs = None
        self.energy = energy0
        self.t = 0
        self.alive = True
        self.on_food = 0

    def _lenia(self):
        U = self.world.potential(self.A)
        self.A = np.clip(self.A + self.dt * self.world.growth(U), 0.0, 1.0)

    def _advance_food(self):
        self.pf = self.pf + self.vf
        for i in range(2):
            if self.pf[i] < self.margin:
                self.pf[i] = 2 * self.margin - self.pf[i]; self.vf[i] *= -1
            elif self.pf[i] > self.shape[i] - self.margin:
                self.pf[i] = 2 * (self.shape[i] - self.margin) - self.pf[i]; self.vf[i] *= -1

    def _render_food(self):
        yy, xx = np.indices(self.shape)
        self.F = np.exp(-0.5 * ((yy - self.pf[0]) ** 2 + (xx - self.pf[1]) ** 2)
                        / self.frad ** 2)

    def body_center(self):
        tot = self.A.sum() + 1e-9
        yy, xx = np.indices(self.shape)
        return np.array([float((self.A * yy).sum() / tot),
                         float((self.A * xx).sum() / tot)])

    def _controller(self, visible):
        """Return the target point the body should drive toward (or None to hold)."""
        if visible:
            obs = self.pf.copy()                       # PERCEIVE: see the food's position
            if self.prev_obs is not None:              # estimate its velocity
                self.vf_est = 0.5 * self.vf_est + 0.5 * (obs - self.prev_obs)
            self.prev_obs = obs
            self.pf_est = obs
        else:
            if self.mode == "no_memory":
                self.pf_est = None                     # retain nothing across the dark
            elif self.mode == "memory_only":
                pass                                   # remember last position, do NOT predict
            else:                                      # full: PREDICT (dead-reckon) the motion
                if self.pf_est is not None:
                    self.pf_est = self.pf_est + self.vf_est
        if self.pf_est is None:
            return None
        if self.mode == "full":                        # PLAN: lead the intercept
            return self.pf_est + self.vf_est * self.lead
        return self.pf_est                             # memory_only / re-acquired: aim at estimate

    def step(self):
        self._lenia()
        self._advance_food()
        self._render_food()
        visible = (self.t % (self.flash_on + self.flash_off)) < self.flash_on
        target = self._controller(visible)
        if target is not None:
            d = target - self.body_center()
            n = np.linalg.norm(d)
            if n > 1e-6:
                step_vec = self.gain * d / n
                for axn in range(2):
                    s = int(round(step_vec[axn]))
                    if s:
                        self.A = np.roll(self.A, s, axis=axn)
        on = np.linalg.norm(self.body_center() - self.pf) < self.feed_r
        self.energy += (self.feed if on else 0.0) - self.decay
        self.energy = min(self.energy, 3.0)
        self.on_food += int(on)
        if self.energy <= 0:
            self.alive = False
        self.t += 1


def run(rule, patch, mode, T=320, shape=(120, 120), seed=0, record=False, **kw):
    c = UnifiedCreature(shape, rule, patch, mode=mode, seed=seed, **kw)
    frames = []
    for _ in range(T):
        c.step()
        if record and c.t % 4 == 0:
            visible = (c.t % (c.flash_on + c.flash_off)) < c.flash_on
            frames.append((np.clip(c.A, 0, 1).copy(), c.pf.copy(), c.energy, c.t, visible))
        if not c.alive:
            break
    return c.t, c.on_food, frames


def survival(rule, patch, mode, seeds, T=320, **kw):
    return float(np.mean([run(rule, patch, mode, T=T, seed=s, **kw)[0] for s in seeds]))
