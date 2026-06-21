"""Theory of mind — inferring another agent's hidden GOAL from its behaviour (round 35).

A distinct social-intelligence axis (not communication). An ACTOR picks a hidden goal (one of K
targets) and moves toward it with NOISE; an OBSERVER sees only the actor's step-by-step motion and
must infer WHICH target it intends. The observer is a RECURRENT net (evolved) that integrates the
noisy trajectory into a belief over goals — mentalising intent from behaviour. It reads the goal
EARLY (from a few steps) and its belief SHARPENS as it watches; ablate the observation (random
motion) and it collapses to chance. Honest scope: inferring "which target an agent walks toward"
is partly geometric (a position oracle also solves it well, since the actor cooperatively reveals
its goal) — the real result here is the LEARNED belief-updating from motion alone + the ablation
dependence; behaviour that MISLEADS (detours/obstacles) is a future frontier.
"""

from __future__ import annotations

import numpy as np

K = 4          # number of candidate goals
H = 12
D = 30.0       # target distance from the actor's start
V = 2.6        # actor speed
T = 18         # steps the observer watches
NOISE = 1.5    # heavy heading noise: single steps are AMBIGUOUS, so the observer must
               # INTEGRATE the trajectory (de-noise the intent) to beat instantaneous position
TARGETS = D * np.array([[np.cos(a), np.sin(a)]
                        for a in np.linspace(0, 2 * np.pi, K, endpoint=False)])
# observer (recurrent): displacement(2) -> H (recurrent) -> K belief
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
    e = np.exp(x - x.max())
    return e / e.sum()


def actor_trajectory(goal, rng):
    """Noisy goal-directed walk; returns the per-step displacement vectors."""
    pos = np.zeros(2)
    steps = []
    for _ in range(T):
        d = TARGETS[goal] - pos
        d = d / (np.linalg.norm(d) + 1e-9)
        d = d + NOISE * rng.normal(0, 1, 2)
        d = V * d / (np.linalg.norm(d) + 1e-9)
        pos = pos + d
        steps.append(d.copy())
    return np.array(steps), pos


def observer_beliefs(theta, steps, ablate=False, rng=None):
    """Return the observer's belief over goals after each observed step (T, K)."""
    Wh, Wi, bh, Wo, bo = _unpack(theta)
    h = np.zeros(H)
    out = []
    for d in steps:
        inp = (rng.normal(0, V, 2) if ablate else d)
        h = np.tanh(Wh @ h + Wi @ inp + bh)
        out.append(_softmax(Wo @ h + bo))
    return np.array(out)


def accuracy_by_step(theta, ablate=False, seed=0, n=200):
    """Inference accuracy after each number of observed steps (length T)."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(T)
    for _ in range(n):
        g = rng.integers(K)
        steps, _ = actor_trajectory(g, rng)
        bel = observer_beliefs(theta, steps, ablate, rng)
        correct += (bel.argmax(1) == g)
    return correct / n


def naive_accuracy_by_step(seed=0, n=200):
    """Baseline: guess the target NEAREST the actor's current position after each step."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(T)
    for _ in range(n):
        g = rng.integers(K)
        pos = np.zeros(2)
        for t in range(T):
            dd = TARGETS[g] - pos; dd = dd / (np.linalg.norm(dd) + 1e-9)
            dd = dd + NOISE * rng.normal(0, 1, 2); dd = V * dd / (np.linalg.norm(dd) + 1e-9)
            pos = pos + dd
            guess = np.argmin(np.linalg.norm(TARGETS - pos, axis=1))
            correct[t] += (guess == g)
    return correct / n


def fitness(theta, rng):
    """Reward being right as EARLY as possible (mean accuracy across observed steps)."""
    correct = 0.0
    for _ in range(16):
        g = rng.integers(K)
        steps, _ = actor_trajectory(g, rng)
        bel = observer_beliefs(theta, steps, rng=rng)
        correct += np.mean(bel.argmax(1) == g)
    return correct / 16


def evolve_es(gens=130, pop=44, sigma=0.3, lr=0.22, seed=0):
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
