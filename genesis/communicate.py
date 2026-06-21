"""Emergent communication — two agents evolve a shared code from scratch (round 30).

The whole mind arc (memory, prediction, planning) was SINGLE-agent. Communication is a hallmark
of intelligence that needs two: a SPEAKER sees a hidden referent and emits a continuous SIGNAL;
a LISTENER sees only the signal and must recover the referent. Neither is told a code — we evolve
both jointly (OpenAI-ES, pure numpy) to maximise the listener getting it right. A shared language
EMERGES: from random init the listener's accuracy rises from chance (1/K) to ~1.0, and the realised
information I(referent ; listener-action) climbs from 0 to the log2(K)-bit ceiling. Ablating the
channel (feeding the listener a random signal) collapses it back to chance -> the signal is USED.
"""

from __future__ import annotations

import numpy as np

K = 4          # number of referents (hidden states to communicate)
M = 2          # signal dimensionality (continuous)
H = 8          # hidden units per agent
# speaker: onehot(K) -> H -> M ; listener: M -> H -> K
NP_S = H * K + H + M * H + M
NP_L = H * M + H + K * H + K
NPARAMS = NP_S + NP_L


def _unpack(theta):
    i = 0
    Ws1 = theta[i:i + H * K].reshape(H, K); i += H * K
    bs1 = theta[i:i + H]; i += H
    Ws2 = theta[i:i + M * H].reshape(M, H); i += M * H
    bs2 = theta[i:i + M]; i += M
    Wl1 = theta[i:i + H * M].reshape(H, M); i += H * M
    bl1 = theta[i:i + H]; i += H
    Wl2 = theta[i:i + K * H].reshape(K, H); i += K * H
    bl2 = theta[i:i + K]; i += K
    return (Ws1, bs1, Ws2, bs2), (Wl1, bl1, Wl2, bl2)


def speak(sp, s):
    Ws1, bs1, Ws2, bs2 = sp
    onehot = np.zeros(K); onehot[s] = 1.0
    h = np.tanh(Ws1 @ onehot + bs1)
    return Ws2 @ h + bs2                       # continuous signal (M,)


def listen(li, signal):
    Wl1, bl1, Wl2, bl2 = li
    h = np.tanh(Wl1 @ signal + bl1)
    return int(np.argmax(Wl2 @ h + bl2))       # chosen referent


def confusion(theta, ablate=False, rng=None):
    """K x K matrix C[s, a] = P(listener says a | referent s).

    Ablation feeds the listener signals DECOUPLED from the referent (many random draws per
    referent), so the action distribution becomes referent-independent -> I -> 0 at chance.
    """
    sp, li = _unpack(theta)
    C = np.zeros((K, K))
    rng = rng or np.random.default_rng(0)
    reps = 300 if ablate else 1
    for s in range(K):
        for _ in range(reps):
            sig = rng.normal(0, 1, M) if ablate else speak(sp, s)
            C[s, listen(li, sig)] += 1.0
    return C


def accuracy(theta, ablate=False, rng=None):
    C = confusion(theta, ablate, rng)
    return float(np.trace(C) / C.sum())


def mutual_information_bits(theta, ablate=False, rng=None):
    """I(referent ; listener-action) in bits, from the confusion matrix (uniform referents)."""
    C = confusion(theta, ablate, rng)
    P = C / C.sum()
    ps = P.sum(1, keepdims=True); pa = P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (ps @ pa)[nz])).sum())


def fitness(theta):
    return accuracy(theta)


def evolve_es(gens=120, pop=40, sigma=0.3, lr=0.25, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i]) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(accuracy(theta))
    return theta, np.array(curve)
