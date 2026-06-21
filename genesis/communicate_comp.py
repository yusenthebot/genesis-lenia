"""Compositional communication — does the emerged language factorise? (round 31)

Round 30 communicated a single category. Here a referent has TWO attributes (A shapes x B
colours), and the speaker must convey BOTH. The deep question: is the emerged code COMPOSITIONAL
(part of the signal = shape, part = colour, so UNSEEN combinations decode zero-shot), or HOLISTIC
(each combo an arbitrary code that does not generalise)? We hold out some combinations during
evolution and test them; we also measure topographic similarity (do similar meanings get similar
signals?). A standard, rigorous probe of whether emergent language has compositional structure.
"""

from __future__ import annotations

import numpy as np

A = 3          # attribute 1 (e.g. shape) values
B = 3          # attribute 2 (e.g. colour) values
M = 3          # signal dimensionality (continuous)
H = 12
NP_S = H * (A + B) + H + M * H + M
NP_L = H * M + H + A * H + A + B * H + B
NPARAMS = NP_S + NP_L

ALL = [(a, b) for a in range(A) for b in range(B)]
HELDOUT = [(0, 0), (1, 2), (2, 1)]            # zero-shot test combos
TRAIN = [m for m in ALL if m not in HELDOUT]


def _unpack(theta):
    i = 0
    Ws1 = theta[i:i + H * (A + B)].reshape(H, A + B); i += H * (A + B)
    bs1 = theta[i:i + H]; i += H
    Ws2 = theta[i:i + M * H].reshape(M, H); i += M * H
    bs2 = theta[i:i + M]; i += M
    Wl1 = theta[i:i + H * M].reshape(H, M); i += H * M
    bl1 = theta[i:i + H]; i += H
    Wla = theta[i:i + A * H].reshape(A, H); i += A * H
    bla = theta[i:i + A]; i += A
    Wlb = theta[i:i + B * H].reshape(B, H); i += B * H
    blb = theta[i:i + B]; i += B
    return (Ws1, bs1, Ws2, bs2), (Wl1, bl1, Wla, bla, Wlb, blb)


def speak(sp, m):
    Ws1, bs1, Ws2, bs2 = sp
    x = np.zeros(A + B); x[m[0]] = 1.0; x[A + m[1]] = 1.0
    h = np.tanh(Ws1 @ x + bs1)
    return Ws2 @ h + bs2


def listen(li, signal):
    Wl1, bl1, Wla, bla, Wlb, blb = li
    h = np.tanh(Wl1 @ signal + bl1)
    return int(np.argmax(Wla @ h + bla)), int(np.argmax(Wlb @ h + blb))


def accuracy(theta, combos):
    sp, li = _unpack(theta)
    ok = 0.0
    for m in combos:
        a, b = listen(li, speak(sp, m))
        ok += 0.5 * (a == m[0]) + 0.5 * (b == m[1])
    return ok / len(combos)


def both_correct(theta, combos):
    sp, li = _unpack(theta)
    return float(np.mean([listen(li, speak(sp, m)) == m for m in combos]))


def topographic_similarity(theta):
    """Spearman corr between meaning-distance (Hamming) and signal-distance (Euclidean)."""
    sp, _ = _unpack(theta)
    sigs = {m: speak(sp, m) for m in ALL}
    md, sd = [], []
    for i, m1 in enumerate(ALL):
        for m2 in ALL[i + 1:]:
            md.append((m1[0] != m2[0]) + (m1[1] != m2[1]))
            sd.append(np.linalg.norm(sigs[m1] - sigs[m2]))
    md = np.array(md, float); sd = np.array(sd)
    rm = md.argsort().argsort(); rs = sd.argsort().argsort()
    rm = rm - rm.mean(); rs = rs - rs.mean()
    return float((rm @ rs) / (np.sqrt((rm @ rm) * (rs @ rs)) + 1e-9))


def signals(theta):
    """The K signals the evolved speaker emits, one per referent (for visualising the code)."""
    sp, _ = _unpack(theta)
    return np.array([speak(sp, m) for m in ALL])


def evolve_es(gens=200, pop=48, sigma=0.3, lr=0.22, seed=0, topo_weight=0.0,
              capture=False):
    """Evolve speaker+listener. topo_weight>0 adds a STRUCTURAL pressure (reward topographic
    similarity) -> the emerged language becomes COMPOSITIONAL rather than holistic."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve, sig_hist = [], []

    def fit(th):
        f = accuracy(th, TRAIN)
        return f + topo_weight * topographic_similarity(th) if topo_weight else f

    for g in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fit(theta + sigma * eps[i]) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(topographic_similarity(theta))
        if capture and g % 4 == 0:
            sig_hist.append(signals(theta))
    return theta, np.array(curve), sig_hist
