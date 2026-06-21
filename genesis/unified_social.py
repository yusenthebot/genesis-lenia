"""Unified social world — communication AND coordination together (round 40).

The social-arc capstone: the multi-agent analogue of round 21's unification of the single mind.
The rounds R30-38 each isolated ONE social faculty; here they must work TOGETHER in one foraging
world. Each round a few sites are RICH (hold food) but only a SCOUT sees which; it emits a signal.
A team of foragers (each with a role) hears the signal and must COVER the rich sites -- spread
across them, since piling two foragers on one site wastes an agent. The team's yield needs BOTH
faculties: COMMUNICATION (to know which sites are rich) and COORDINATION (to divide labour across
them). Ablate the channel and the team forages blind; ablate the role-split and it piles up. Each
social faculty is shown LOAD-BEARING, exactly as round 21 did for memory/prediction/planning.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

N = 2              # foragers (FEWER than sites, so they must CHOOSE which to cover -> comm needed)
K = 4              # sites
R = 2              # rich sites per round
M = 3              # signal dimensionality
H = 18
PATTERNS = list(combinations(range(K), R))     # all C(4,2)=6 rich-site patterns

# scout: rich-pattern (K bits) -> H -> signal (M)
NP_S = H * K + H + M * H + M
# forager (shared policy): [onehot(role,N), signal(M)] -> H -> site logits (K)
NP_F = H * (N + M) + H + K * H + K
NPARAMS = NP_S + NP_F


def _unpack(theta):
    i = 0
    Ws1 = theta[i:i + H * K].reshape(H, K); i += H * K
    bs1 = theta[i:i + H]; i += H
    Ws2 = theta[i:i + M * H].reshape(M, H); i += M * H
    bs2 = theta[i:i + M]; i += M
    Wf1 = theta[i:i + H * (N + M)].reshape(H, N + M); i += H * (N + M)
    bf1 = theta[i:i + H]; i += H
    Wf2 = theta[i:i + K * H].reshape(K, H); i += K * H
    bf2 = theta[i:i + K]; i += K
    return (Ws1, bs1, Ws2, bs2), (Wf1, bf1, Wf2, bf2)


def scout_signal(sc, pattern_bits):
    Ws1, bs1, Ws2, bs2 = sc
    h = np.tanh(Ws1 @ pattern_bits + bs1)
    return Ws2 @ h + bs2


def forager_site(fo, role, signal, ablate_coord=False):
    Wf1, bf1, Wf2, bf2 = fo
    rid = np.zeros(N)
    rid[0 if ablate_coord else role] = 1.0
    x = np.concatenate([rid, signal])
    h = np.tanh(Wf1 @ x + bf1)
    return int(np.argmax(Wf2 @ h + bf2))


def yield_on(theta, pattern, ablate_comm=False, ablate_coord=False, rng=None):
    sc, fo = _unpack(theta)
    bits = np.zeros(K); bits[list(pattern)] = 1.0
    sig = (rng.normal(0, 1, M) if ablate_comm else scout_signal(sc, bits))
    sites = [forager_site(fo, r, sig, ablate_coord) for r in range(N)]
    rich = set(pattern)
    return len(rich & set(sites)) / len(rich)        # fraction of rich sites covered


def team_yield(theta, ablate_comm=False, ablate_coord=False, seed=0, reps=4):
    rng = np.random.default_rng(seed)
    ys = [yield_on(theta, p, ablate_comm, ablate_coord, rng)
          for _ in range(reps) for p in PATTERNS]
    return float(np.mean(ys))


def fitness(theta, rng):
    return float(np.mean([yield_on(theta, p, rng=rng) for p in PATTERNS]))


def evolve_es(gens=160, pop=48, sigma=0.3, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i], np.random.default_rng(seed + 1 + i))
                         for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(team_yield(theta, seed=999))
    return theta, np.array(curve)
