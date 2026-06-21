"""A deeper controller — a brain with MEMORY.

Every mind so far was a memoryless reflex: action = f(current sensation). Here the
controller is a tiny recurrent network with a hidden state that carries information
across time. The task requires it: a CUE is shown, then hidden, and after a random DELAY
the creature must act on the cue it can no longer see. A feedforward (memoryless)
controller is at chance once the cue is gone; a recurrent one holds the cue in its hidden
state. We evolve the network weights (an evolution strategy, pure numpy) and can read the
memory straight out of the hidden state.

Observation o_t = [cue (+/-1 at the cue step, else 0), go (1 at the decision step, else 0),
bias 1]. The creature outputs a scalar; at the go step its sign must match the cue.
"""

from __future__ import annotations

import numpy as np

H = 6        # hidden units
OBS = 3      # observation dims [cue, go, bias]
NPARAMS = H * H + H * OBS + H + H


def unpack(theta):
    i = 0
    Wh = theta[i:i + H * H].reshape(H, H); i += H * H
    Wi = theta[i:i + H * OBS].reshape(H, OBS); i += H * OBS
    bh = theta[i:i + H]; i += H
    wo = theta[i:i + H]
    return Wh, Wi, bh, wo


def run_trial(theta, cue, delay, recurrent=True, return_trace=False):
    Wh, Wi, bh, wo = unpack(theta)
    if not recurrent:
        Wh = np.zeros_like(Wh)             # ablate memory: no recurrence
    h = np.zeros(H)
    obs_seq = [[cue, 0, 1]] + [[0, 0, 1]] * delay + [[0, 1, 1]]
    a_go = 0.0
    hs = []
    for o in obs_seq:
        h = np.tanh(Wh @ h + Wi @ np.array(o, float) + bh)
        hs.append(h.copy())
        if o[1] == 1:                      # go step -> read the action
            a_go = float(wo @ h)
    reward = 1.0 if np.sign(a_go) == np.sign(cue) else 0.0
    if return_trace:
        return reward, np.array(hs), obs_seq
    return reward


def fitness(theta, rng, recurrent=True, K=24, dmin=2, dmax=9):
    r = 0.0
    for _ in range(K):
        cue = rng.choice([-1.0, 1.0])
        delay = int(rng.integers(dmin, dmax + 1))
        r += run_trial(theta, cue, delay, recurrent)
    return r / K


def evolve_es(recurrent=True, gens=60, pop=40, sigma=0.25, lr=0.2, seed=0):
    """OpenAI-style evolution strategy on the network weights."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], rng, recurrent)
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(fitness(theta, rng, recurrent, K=80))
    return theta, np.array(curve)
