"""Emergent roles from scratch — symmetry-breaking with no pre-given id (round 44).

Rounds 37/40 had agents DIVIDE labour, but the role distinction was GIVEN (each agent carried a
distinct id). The harder, genuine phenomenon is two IDENTICAL agents -- one shared policy, no id, the
same initial state -- nonetheless ending up in COMPLEMENTARY roles. That needs spontaneous
SYMMETRY-BREAKING: a learned mutual-inhibition dynamic where tiny noise tips the symmetric tie and
the two agents' 'leanings' DIVERGE to opposite roles (a pitchfork bifurcation), so two food patches
both get covered. Both ingredients are necessary: INTERACTION (each agent reacts to the other) and
NOISE (to break the exact tie). Remove either and the identical agents collide on one role.

The shared policy is a tiny net evolved by ES; the dynamics run in pure numpy.
"""

from __future__ import annotations

import numpy as np

H = 8
T = 18             # settling steps
NOISE = 0.12
# shared policy: [s_self, s_other] -> H -> d(lean)   (the SAME params drive BOTH agents)
NPARAMS = H * 2 + H + H + 1


def _unpack(theta):
    i = 0
    W1 = theta[i:i + H * 2].reshape(H, 2); i += H * 2
    b1 = theta[i:i + H]; i += H
    W2 = theta[i:i + H]; i += H
    b2 = theta[i:i + 1]
    return W1, b1, W2, b2


def _step(pol, s_self, s_other):
    W1, b1, W2, b2 = pol
    h = np.tanh(W1 @ np.array([s_self, s_other]) + b1)
    return float(W2 @ h + b2[0])


def settle(theta, rng, ablate_interaction=False, ablate_noise=False, return_traj=False):
    """Run the two IDENTICAL agents from a near-symmetric start; return their final leanings."""
    pol = _unpack(theta)
    # ablate_noise = NO symmetry-breaking trigger at all: a PERFECTLY symmetric start and no noise.
    s = np.zeros(2) if ablate_noise else rng.normal(0, 0.02, 2)
    traj = [s.copy()]
    for _ in range(T):
        o0 = 0.0 if ablate_interaction else s[1]
        o1 = 0.0 if ablate_interaction else s[0]
        d0 = _step(pol, s[0], o0)
        d1 = _step(pol, s[1], o1)
        nz = 0.0 if ablate_noise else NOISE
        s = np.tanh(s + np.array([d0, d1]) + rng.normal(0, nz, 2))
        traj.append(s.copy())
    if return_traj:
        return s, np.array(traj)
    return s


def split_rate(theta, ablate_interaction=False, ablate_noise=False, seed=0, n=400):
    """Fraction of runs where the two identical agents end in DIFFERENT roles (sign of leaning)."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        s = settle(theta, rng, ablate_interaction, ablate_noise)
        ok += (np.sign(s[0]) != np.sign(s[1])) and (abs(s[0]) > 0.3) and (abs(s[1]) > 0.3)
    return ok / n


def fitness(theta, rng):
    ok = 0.0
    for _ in range(20):
        s = settle(theta, rng)
        # reward: opposite signs AND both committed (|s| large) -> two distinct, decisive roles
        ok += (np.sign(s[0]) != np.sign(s[1])) * min(abs(s[0]), abs(s[1]))
    return ok / 20


def evolve_es(gens=140, pop=44, sigma=0.3, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], np.random.default_rng(seed + 1 + i))
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(split_rate(theta, seed=999, n=80))
    return theta, np.array(curve)
