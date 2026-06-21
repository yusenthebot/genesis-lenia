"""A brain that PREDICTS — anticipating the next world-state, not just reacting.

Round 15 gave the brain memory (it held the past). Here it must foresee the future. The
world emits a periodic pattern [0,0,1,1] in which each symbol is AMBIGUOUS: a 0 is followed
by 0 at one phase and by 1 at another, and likewise for 1. So you cannot predict the next
state from the current one alone — you must track WHERE in the cycle you are and anticipate
the flip BEFORE it lands. A recurrent model learns the pattern and predicts every step,
including the transitions; a feedforward (reactive) controller is stuck near chance on the
ambiguous steps. We evolve the weights (ES, pure numpy) and read the predictive information
out of the hidden state.
"""

from __future__ import annotations

import numpy as np

PATTERN = [0, 0, 1, 1]        # the world's hidden temporal structure (period 4)
H = 8                          # hidden units
OBS = 2                        # observation [current state (+/-1), bias]
NPARAMS = H * H + H * OBS + H + H


def unpack(theta):
    i = 0
    Wh = theta[i:i + H * H].reshape(H, H); i += H * H
    Wi = theta[i:i + H * OBS].reshape(H, OBS); i += H * OBS
    bh = theta[i:i + H]; i += H
    wo = theta[i:i + H]
    return Wh, Wi, bh, wo


def run_seq(theta, recurrent=True, n_periods=40, warmup=8, return_trace=False):
    """Run the agent through the world sequence; predict next-state each step."""
    Wh, Wi, bh, wo = unpack(theta)
    if not recurrent:
        Wh = np.zeros_like(Wh)             # reactive: no hidden dynamics across time
    seq = PATTERN * n_periods
    h = np.zeros(H)
    preds, truths, cur, hs = [], [], [], []
    for t in range(len(seq) - 1):
        o = np.array([1.0 if seq[t] == 1 else -1.0, 1.0])
        h = np.tanh(Wh @ h + Wi @ o + bh)
        pred = 1 if (wo @ h) > 0 else 0
        if t >= warmup:                    # let the recurrent net lock onto the phase
            preds.append(pred); truths.append(seq[t + 1]); cur.append(seq[t])
        hs.append(h.copy())
    preds = np.array(preds); truths = np.array(truths); cur = np.array(cur)
    acc = float((preds == truths).mean())
    trans = cur != truths                  # steps where the world FLIPS (the hard ones)
    acc_trans = float((preds[trans] == truths[trans]).mean()) if trans.any() else 0.0
    if return_trace:
        return acc, acc_trans, preds, truths, cur
    return acc, acc_trans


def fitness(theta, recurrent=True):
    acc, _ = run_seq(theta, recurrent, n_periods=40)
    return acc


def evolve_es(recurrent=True, gens=60, pop=40, sigma=0.25, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], recurrent) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(fitness(theta, recurrent))
    return theta, np.array(curve)
