"""Harder theory of mind — inferring intent when behaviour MISLEADS (round 38).

Round 35's observer read intent from motion, but a position oracle did just as well because the actor
walked straight at its goal (behaviour was honest). Here the actor must DETOUR around a central
obstacle to reach goals behind it, so its early motion is TANGENTIAL — it moves sideways, away from
the goal. A position/heading-based naive observer is FOOLED (the actor is on the wrong side, heading
the wrong way); only an observer that has learned the obstacle + the actor's go-around policy infers
the true goal. This is the genuine theory-of-mind test: modelling goal-directed behaviour under a
constraint, where surface behaviour points the wrong way. The observer BEATS the position oracle.
"""

from __future__ import annotations

import numpy as np

K = 3              # candidate goals (behind / around the obstacle)
H = 16
T = 34             # steps observed (long enough to complete the go-around detour)
V = 2.3            # actor speed
RO = 12.0          # obstacle (disk at origin) radius
INFLU = 21.0       # repulsion influence radius
REPEL = 70.0       # repulsion strength
NOISE = 0.32
START = np.array([0.0, -30.0])
GOALS = np.array([[-21.0, 24.0], [0.0, 30.0], [21.0, 24.0]])   # all behind the central obstacle
NPARAMS = H * H + H * 2 + H + K * H + K


def _unpack(theta):
    i = 0
    Wh = theta[i:i + H * H].reshape(H, H); i += H * H
    Wi = theta[i:i + H * 2].reshape(H, 2); i += H * 2
    bh = theta[i:i + H]; i += H
    Wo = theta[i:i + K * H].reshape(K, H); i += K * H
    bo = theta[i:i + K]
    return Wh, Wi, bh, Wo, bo


def _softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def actor_trajectory(goal, rng):
    """Goal-directed navigation that goes AROUND the obstacle: when the straight line to the goal is
    blocked by the disk, steer tangentially around it (consistent side), else head straight. Returns
    per-step displacements + the path. The detour makes early motion point AWAY from the goal."""
    pos = START.copy()
    steps = []; path = [pos.copy()]
    gp = GOALS[goal]
    for _ in range(T):
        to_goal = gp - pos
        dist_goal = np.linalg.norm(to_goal) + 1e-9
        dir_goal = to_goal / dist_goal
        # closest approach of the ray pos->goal to the obstacle centre
        tclear = np.clip(-pos @ dir_goal, 0.0, dist_goal)
        closest = pos + tclear * dir_goal
        blocked = (np.linalg.norm(closest) < RO + 4.0) and (0.0 < tclear < dist_goal)
        od = np.linalg.norm(pos) + 1e-9
        rad = pos / od
        if blocked:                                      # steer tangentially around the disk
            tang = np.array([-rad[1], rad[0]])
            proj = tang @ dir_goal
            if abs(proj) < 0.15:                          # ~directly opposite: pick a consistent side
                if (tang @ np.array([np.sign(gp[0]) or 1.0, 0.0])) < 0:
                    tang = -tang
            elif proj < 0:
                tang = -tang
            d = tang + 0.35 * dir_goal
        else:
            d = dir_goal
        if od < RO + 3.0:                                # close repulsion (don't graze the disk)
            d = d + 1.5 * rad
        d = d / (np.linalg.norm(d) + 1e-9) + NOISE * rng.normal(0, 1, 2)
        step = V * d / (np.linalg.norm(d) + 1e-9)
        nxt = pos + step
        if np.linalg.norm(nxt) < RO:                     # hard wall: don't enter the disk
            nxt = pos - 0.3 * step
        pos = nxt; steps.append(pos - path[-1]); path.append(pos.copy())
    return np.array(steps), np.array(path)


def observer_beliefs(theta, steps, ablate=False, rng=None):
    Wh, Wi, bh, Wo, bo = _unpack(theta)
    h = np.zeros(H); out = []
    for d in steps:
        inp = (rng.normal(0, V, 2) if ablate else d)
        h = np.tanh(Wh @ h + Wi @ inp + bh)
        out.append(_softmax(Wo @ h + bo))
    return np.array(out)


def accuracy_by_step(theta, ablate=False, seed=0, n=200):
    rng = np.random.default_rng(seed)
    correct = np.zeros(T)
    for _ in range(n):
        g = rng.integers(K)
        steps, _ = actor_trajectory(g, rng)
        correct += (observer_beliefs(theta, steps, ablate, rng).argmax(1) == g)
    return correct / n


def naive_accuracy_by_step(seed=0, n=200):
    """Position oracle: guess the goal NEAREST the actor's current position (fooled by detours)."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(T)
    for _ in range(n):
        g = rng.integers(K)
        _, path = actor_trajectory(g, rng)
        for t in range(T):
            guess = np.argmin(np.linalg.norm(GOALS - path[t + 1], axis=1))
            correct[t] += (guess == g)
    return correct / n


def fitness(theta, rng):
    correct = 0.0
    for _ in range(14):
        g = rng.integers(K)
        steps, _ = actor_trajectory(g, rng)
        correct += np.mean(observer_beliefs(theta, steps, rng=rng).argmax(1) == g)
    return correct / 14


def evolve_es(gens=150, pop=44, sigma=0.3, lr=0.22, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], np.random.default_rng(seed + 1 + i))
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(accuracy_by_step(theta, seed=999, n=60)[-1])
    return theta, np.array(curve)
