"""Round 37 — multi-agent coordination: division of labour.

N agents must COVER N food sites; the team's yield is the number of DISTINCT sites occupied. With
distinct ROLES the team evolves a DIVISION OF LABOUR — a permutation assigning each agent its own
site (full coverage 1.00), beating an independent-random team (~0.68, some collide by chance) and
crushing an ablated identical team (0.25, all pile on one site). Coordination needs broken symmetry;
the assignment is an emergent convention. A team capability distinct from communication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import genesis.coordinate as CO  # noqa: E402

AGENT_COLORS = ["#d04030", "#2a9d2a", "#2060d0", "#d0a020", "#8040c0", "#20a0a0"]


def _random_coverage(seed=0, trials=2000):
    rng = np.random.default_rng(seed)
    cov = [len(set(rng.integers(CO.N, size=CO.N))) / CO.N for _ in range(trials)]
    return float(np.mean(cov))


def _draw_team(ax, theta, ablate, title):
    sites = CO.team_sites(theta, ablate)
    for s in CO.SITES:
        ax.plot(*s, "s", color="#ccc", ms=20, mec="#999", zorder=1)
    # jitter agents that share a site so overlaps are visible
    from collections import defaultdict
    seen = defaultdict(int)
    for r, s in enumerate(sites):
        k = seen[s]; seen[s] += 1
        off = np.array([np.cos(k * 1.4), np.sin(k * 1.4)]) * 3.2 * (k > 0)
        p = CO.SITES[s] + off
        ax.plot([0, p[0]], [0, p[1]], color=AGENT_COLORS[r], lw=1.4, alpha=0.6, zorder=2)
        ax.plot(*p, "o", color=AGENT_COLORS[r], ms=11, mec="k", zorder=3, label=f"agent {r}")
    ax.plot(0, 0, "k+", ms=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-34, 34); ax.set_ylim(-34, 34); ax.set_title(title, fontsize=10)


def figure(theta, out: Path):
    cov = CO.coverage(theta)
    cova = CO.coverage(theta, ablate=True)
    rnd = _random_coverage()

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.5), facecolor="white")
    a0.bar(["evolved team\n(division of\nlabour)", "random\n(independent)", "ablated\n(identical\nagents)"],
           [cov, rnd, cova], color=["#2a9d2a", "#888", "#b04030"])
    for i, v in enumerate([cov, rnd, cova]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a0.set_ylim(0, 1.08); a0.set_ylabel("site coverage (distinct sites / N)")
    a0.set_title("A team self-organises to COVER all sites\n(no role distinction -> all collide)")

    _draw_team(a1, theta, False, "evolved: each agent its OWN site (coverage 1.0)")
    _draw_team(a2, theta, True, "ablated: identical agents pile up (coverage 0.25)")

    fig.suptitle("Round 37 — multi-agent coordination: a team evolves a DIVISION OF LABOUR "
                 "(emergent role assignment)", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(evolved=cov, random=rnd, ablated=cova, assignment=CO.team_sites(theta))


def make_gif(theta, out: Path):
    sites_e = CO.team_sites(theta, False)
    sites_a = CO.team_sites(theta, True)
    fig, (axe, axa) = plt.subplots(1, 2, figsize=(8.8, 4.6))
    frames = 16
    for ax, t in zip((axe, axa), ("division of labour", "ablated (identical)")):
        ax.set_xlim(-34, 34); ax.set_ylim(-34, 34); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=10)
        for s in CO.SITES:
            ax.plot(*s, "s", color="#ccc", ms=20, mec="#999")
    dots_e = [axe.plot([], [], "o", color=AGENT_COLORS[r], ms=11, mec="k")[0] for r in range(CO.N)]
    dots_a = [axa.plot([], [], "o", color=AGENT_COLORS[r], ms=11, mec="k")[0] for r in range(CO.N)]

    def targets(sites):
        from collections import defaultdict
        seen = defaultdict(int); out = []
        for s in sites:
            k = seen[s]; seen[s] += 1
            out.append(CO.SITES[s] + np.array([np.cos(k * 1.4), np.sin(k * 1.4)]) * 3.2 * (k > 0))
        return out

    te, ta = targets(sites_e), targets(sites_a)

    def upd(f):
        u = f / (frames - 1)
        for r in range(CO.N):
            pe = u * te[r]; pa = u * ta[r]
            dots_e[r].set_data([pe[0]], [pe[1]]); dots_a[r].set_data([pa[0]], [pa[1]])
        return dots_e + dots_a

    FuncAnimation(fig, upd, frames=frames, blit=False).save(out, writer=PillowWriter(fps=8))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 37: multi-agent coordination")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== multi-agent coordination: division of labour ==")
    theta, _ = CO.evolve_es(gens=130, seed=0)
    res = figure(theta, outdir / "round37_coordination.png")
    print(f"  coverage: evolved {res['evolved']:.2f} | random {res['random']:.2f} | "
          f"ablated {res['ablated']:.2f} (assignment {res['assignment']})")
    print(f"wrote {outdir/'round37_coordination.png'}")
    if args.gif:
        make_gif(theta, outdir / "round37_coordination.gif")
        print(f"wrote {outdir/'round37_coordination.gif'}")
    (outdir / "round37_coordination.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
