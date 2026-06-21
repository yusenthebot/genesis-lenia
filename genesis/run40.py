"""Round 40 — unified social world: communication AND coordination together.

The social-arc capstone (the multi-agent analogue of round 21's unification of the single mind).
Each round a few sites are RICH but only a SCOUT sees which; it signals a team of foragers who must
COVER the rich sites without piling up. Full yield needs BOTH faculties: communication (to know which
sites are rich) and coordination (to divide labour across them). Ablate the channel -> the team
forages blind (~0.5); ablate the role-split -> the foragers pile onto one site (~0.5). Each social
faculty is shown LOAD-BEARING, exactly as round 21 did for memory/prediction/planning.
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

import genesis.unified_social as US  # noqa: E402

SITES = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, US.K, endpoint=False)]) * 24.0
FCOL = ["#d04030", "#2060d0"]
SAMPLE = (1, 3)        # a fixed rich-site pattern for the vignettes


def _sites_for(theta, pattern, ablate_comm, ablate_coord, rng):
    sc, fo = US._unpack(theta)
    bits = np.zeros(US.K); bits[list(pattern)] = 1.0
    sig = (rng.normal(0, 1, US.M) if ablate_comm else US.scout_signal(sc, bits))
    return [US.forager_site(fo, r, sig, ablate_coord) for r in range(US.N)]


def _vignette(ax, theta, pattern, ablate_comm, ablate_coord, title, rng):
    for k in range(US.K):
        rich = k in pattern
        ax.plot(*SITES[k], "*" if rich else "o", color="#d0a020" if rich else "#bbb",
                ms=22 if rich else 14, mec="k", zorder=2)
    ax.plot(0, 0, "ks", ms=9, zorder=2)                # scout at centre
    chosen = _sites_for(theta, pattern, ablate_comm, ablate_coord, rng)
    from collections import defaultdict
    seen = defaultdict(int)
    for r, s in enumerate(chosen):
        kk = seen[s]; seen[s] += 1
        off = np.array([np.cos(kk * 1.8), np.sin(kk * 1.8)]) * 3.4 * (kk > 0)
        p = SITES[s] + off
        ax.plot([0, p[0]], [0, p[1]], color=FCOL[r], lw=1.6, alpha=0.6, zorder=1)
        ax.plot(*p, "o", color=FCOL[r], ms=11, mec="k", zorder=3)
    cov = len(set(pattern) & set(chosen)) / len(pattern)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)
    ax.set_title(f"{title}\ncovered {cov:.0%} of rich sites", fontsize=9.5)


def figure(theta, out: Path):
    full = US.team_yield(theta, seed=555)
    noc = US.team_yield(theta, ablate_comm=True, seed=555)
    ncd = US.team_yield(theta, ablate_coord=True, seed=555)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), facecolor="white")
    a0 = axes[0]
    a0.bar(["full\n(comm +\ncoord)", "no\ncomm", "no\ncoord"], [full, noc, ncd],
           color=["#2a9d2a", "#b04030", "#b07020"])
    for i, v in enumerate([full, noc, ncd]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a0.set_ylim(0, 1.08); a0.set_ylabel("rich-site coverage (team yield)")
    a0.set_title("BOTH faculties are load-bearing\n(ablate either -> yield halves)")

    _vignette(axes[1], theta, SAMPLE, False, False, "FULL: scout tells, team splits",
              np.random.default_rng(0))
    # pick a REPRESENTATIVE no-comm draw (one that misses a rich site), not a lucky one
    nc_rng = np.random.default_rng(0)
    for s in range(200):
        r = np.random.default_rng(s)
        if len(set(SAMPLE) & set(_sites_for(theta, SAMPLE, True, False, r))) < len(SAMPLE):
            nc_rng = np.random.default_rng(s); break
    _vignette(axes[2], theta, SAMPLE, True, False, "NO COMM: forage blind", nc_rng)
    _vignette(axes[3], theta, SAMPLE, False, True, "NO COORD: pile on one site",
              np.random.default_rng(0))

    fig.suptitle("Round 40 — unified social world: agents must COMMUNICATE and COORDINATE to forage "
                 "(the social-arc capstone)", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(full=full, no_comm=noc, no_coord=ncd)


def make_gif(theta, out: Path):
    rng = np.random.default_rng(1)
    conds = [(False, False, "FULL"), (False, True, "NO COORD")]
    chosen = [_sites_for(theta, SAMPLE, ac, acoord, rng) for ac, acoord, _ in conds]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.5))
    targs = []
    for ax, (ac, acoord, name), ch in zip(axs, conds, chosen):
        ax.set_xlim(-32, 32); ax.set_ylim(-32, 32); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=10)
        for k in range(US.K):
            rich = k in SAMPLE
            ax.plot(*SITES[k], "*" if rich else "o", color="#d0a020" if rich else "#bbb",
                    ms=20 if rich else 13, mec="k")
        ax.plot(0, 0, "ks", ms=9)
        from collections import defaultdict
        seen = defaultdict(int); t = []
        for s in ch:
            kk = seen[s]; seen[s] += 1
            t.append(SITES[s] + np.array([np.cos(kk * 1.8), np.sin(kk * 1.8)]) * 3.4 * (kk > 0))
        targs.append(t)
    dots = [[ax.plot([], [], "o", color=FCOL[r], ms=11, mec="k")[0] for r in range(US.N)] for ax in axs]
    frames = 16

    def upd(f):
        u = f / (frames - 1)
        for c in range(2):
            for r in range(US.N):
                p = u * targs[c][r]
                dots[c][r].set_data([p[0]], [p[1]])
        return [d for row in dots for d in row]

    FuncAnimation(fig, upd, frames=frames, blit=False).save(out, writer=PillowWriter(fps=8))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 40: unified social world")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== unified social world: communication AND coordination ==")
    theta, _ = US.evolve_es(gens=220, seed=0)
    res = figure(theta, outdir / "round40_unified_social.png")
    print(f"  team yield: full {res['full']:.2f} | no-comm {res['no_comm']:.2f} | "
          f"no-coord {res['no_coord']:.2f}")
    print(f"wrote {outdir/'round40_unified_social.png'}")
    if args.gif:
        make_gif(theta, outdir / "round40_unified_social.gif")
        print(f"wrote {outdir/'round40_unified_social.gif'}")
    (outdir / "round40_unified_social.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
