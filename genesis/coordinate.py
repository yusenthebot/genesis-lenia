"""Multi-agent coordination — division of labour (round 37).

The social vein so far: a language emerges, factorises, grounds in action; agents read each other's
goals (theory of mind). The missing piece is ACTING TOGETHER. Here N agents must COVER N food sites:
the team's yield is the number of DISTINCT sites occupied (a doubled-up site wastes an agent; an
empty site yields nothing). With distinct ROLES the team evolves a DIVISION OF LABOUR — a permutation
assigning each agent its own site (full coverage). Ablate the role distinction (identical agents) and
they all pick the same site -> redundancy, coverage collapses to 1/N. Coordination needs broken
symmetry; the assignment is an emergent convention. A team capability, distinct from communication.
"""

from __future__ import annotations

import numpy as np

N = 4          # agents == sites
H = 12
SITES = np.array([[np.cos(a), np.sin(a)]
                  for a in np.linspace(0, 2 * np.pi, N, endpoint=False)]) * 26.0
# ONE shared policy read by every agent: onehot(role id, N) -> H -> site logits (N)
NPARAMS = H * N + H + N * H + N


def _unpack(theta):
    i = 0
    W1 = theta[i:i + H * N].reshape(H, N); i += H * N
    b1 = theta[i:i + H]; i += H
    W2 = theta[i:i + N * H].reshape(N, H); i += N * H
    b2 = theta[i:i + N]
    return W1, b1, W2, b2


def agent_site(theta, role, ablate=False):
    """Site chosen by the agent in a given role (ablate -> all agents share role 0 = no distinction)."""
    W1, b1, W2, b2 = _unpack(theta)
    x = np.zeros(N)
    x[0 if ablate else role] = 1.0
    h = np.tanh(W1 @ x + b1)
    return int(np.argmax(W2 @ h + b2))


def team_sites(theta, ablate=False):
    return [agent_site(theta, r, ablate) for r in range(N)]


def coverage(theta, ablate=False):
    """Fraction of sites covered by >=1 agent (distinct sites occupied / N)."""
    return len(set(team_sites(theta, ablate))) / N


def fitness(theta):
    # smooth team yield: reward distinct coverage; a small penalty per collision discourages piling up
    sites = team_sites(theta)
    distinct = len(set(sites))
    return distinct - 0.1 * (N - distinct)


def evolve_es(gens=120, pop=40, sigma=0.4, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i]) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(coverage(theta))
    return theta, np.array(curve)
