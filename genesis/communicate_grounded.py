"""Grounded communication — a signal that drives action in the world (round 33).

Rounds 30-31 communicated abstract labels. Here the signal is GROUNDED: a SCOUT sees where food
is and emits a signal; a BLIND FORAGER sees only the signal and must navigate to the food. The
pair is evolved jointly. With the channel, the blind forager reaches the food (coordination);
ablate the channel (random signal) and it is lost. So the emitted signal carries ACTIONABLE
SPATIAL information that a body uses to forage — communication that does something in the world,
fusing the embodied track (navigation) with the social track (signalling).
"""

from __future__ import annotations

import numpy as np

M = 3          # signal dimensionality
H = 10
D = 32.0       # food distance from the forager's start
V = 2.2        # forager speed
T = 24         # steps the forager gets
CATCH = 5.0    # catch radius
# scout: [cos th, sin th] -> H -> M ; forager: M -> H -> 2 (heading)
NP_S = H * 2 + H + M * H + M
NP_F = H * M + H + 2 * H + 2
NPARAMS = NP_S + NP_F


def _unpack(theta):
    i = 0
    Ws1 = theta[i:i + H * 2].reshape(H, 2); i += H * 2
    bs1 = theta[i:i + H]; i += H
    Ws2 = theta[i:i + M * H].reshape(M, H); i += M * H
    bs2 = theta[i:i + M]; i += M
    Wf1 = theta[i:i + H * M].reshape(H, M); i += H * M
    bf1 = theta[i:i + H]; i += H
    Wf2 = theta[i:i + 2 * H].reshape(2, H); i += 2 * H
    bf2 = theta[i:i + 2]; i += 2
    return (Ws1, bs1, Ws2, bs2), (Wf1, bf1, Wf2, bf2)


def scout_signal(sc, theta_food):
    Ws1, bs1, Ws2, bs2 = sc
    h = np.tanh(Ws1 @ np.array([np.cos(theta_food), np.sin(theta_food)]) + bs1)
    return Ws2 @ h + bs2


def forager_heading(fo, signal):
    Wf1, bf1, Wf2, bf2 = fo
    h = np.tanh(Wf1 @ signal + bf1)
    d = Wf2 @ h + bf2
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(2)


def episode(theta, theta_food, ablate=False, rng=None, return_traj=False):
    sc, fo = _unpack(theta)
    sig = (rng.normal(0, 1, M) if ablate else scout_signal(sc, theta_food))
    head = forager_heading(fo, sig)
    food = D * np.array([np.cos(theta_food), np.sin(theta_food)])
    pos = np.zeros(2)
    mind = np.linalg.norm(pos - food)
    traj = [pos.copy()]
    for _ in range(T):
        pos = pos + V * head
        mind = min(mind, np.linalg.norm(pos - food))
        traj.append(pos.copy())
    caught = mind < CATCH
    if return_traj:
        return caught, mind, np.array(traj), food
    return caught, mind


def catch_rate(theta, ablate=False, seed=0, n=60):
    rng = np.random.default_rng(seed)
    return float(np.mean([episode(theta, rng.uniform(0, 2 * np.pi), ablate, rng)[0]
                          for _ in range(n)]))


def fitness(theta, rng):
    out = []
    for _ in range(12):
        th = rng.uniform(0, 2 * np.pi)
        caught, mind = episode(theta, th, rng=rng)
        out.append(1.0 if caught else max(0.0, 1.0 - mind / D))   # dense closeness reward
    return float(np.mean(out))


def evolve_es(gens=120, pop=40, sigma=0.3, lr=0.22, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], np.random.default_rng(seed + 1 + i))
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(catch_rate(theta, seed=999))
    return theta, np.array(curve)
