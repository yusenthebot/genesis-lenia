"""Round 44 — emergent roles from scratch: symmetry-breaking with no pre-given id.

Two IDENTICAL agents (one shared policy, no id, the same near-symmetric start) must end in
COMPLEMENTARY roles. A learned mutual-inhibition dynamic + a tiny symmetry-breaking trigger make
their 'leanings' DIVERGE to opposite roles (a pitchfork bifurcation), so two patches both get
covered. Both ingredients are necessary: split rate 1.00 full vs ~0.4 with no interaction vs 0.00
with perfect symmetry (no trigger). Roles emerge from scratch, not from a handed-out label.
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

import genesis.symmetry_break as SB  # noqa: E402


def figure(theta, out: Path):
    full = SB.split_rate(theta, seed=777)
    noint = SB.split_rate(theta, ablate_interaction=True, seed=777)
    notrig = SB.split_rate(theta, ablate_noise=True, seed=777)

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    a0.bar(["full\n(interaction\n+ trigger)", "no\ninteraction", "no trigger\n(perfect\nsymmetry)"],
           [full, noint, notrig], color=["#2a9d2a", "#b04030", "#b07020"])
    for i, v in enumerate([full, noint, notrig]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a0.set_ylim(0, 1.08); a0.set_ylabel("role-differentiation (split) rate")
    a0.set_title("Identical agents differentiate -- but only with\nBOTH interaction and a trigger")

    # the bifurcation: leanings diverge from ~0 to opposite roles
    rng = np.random.default_rng(1)
    t = np.arange(SB.T + 1)
    for _ in range(14):
        _, tr = SB.settle(theta, rng, return_traj=True)
        a1.plot(t, tr[:, 0], color="#2060d0", lw=1.1, alpha=0.7)
        a1.plot(t, tr[:, 1], color="#d04030", lw=1.1, alpha=0.7)
    a1.axhline(0, color="#aaa", lw=0.8, ls=":")
    a1.set_xlabel("settling step"); a1.set_ylabel("agent 'leaning' (role)")
    a1.set_ylim(-1.1, 1.1); a1.set_title("SYMMETRY BREAKS: identical agents diverge\nto opposite roles (a bifurcation)")

    # contrast: perfect symmetry -> no differentiation (stuck at the tie)
    rng = np.random.default_rng(2)
    for _ in range(8):
        _, tr = SB.settle(theta, rng, ablate_noise=True, return_traj=True)
        a2.plot(t, tr[:, 0], color="#2060d0", lw=1.4, alpha=0.8)
        a2.plot(t, tr[:, 1], color="#d04030", lw=1.4, alpha=0.8, ls="--")
    a2.axhline(0, color="#aaa", lw=0.8, ls=":")
    a2.set_xlabel("settling step"); a2.set_ylim(-1.1, 1.1)
    a2.set_title("perfect symmetry, no trigger:\nthey stay tied (no roles)")

    fig.suptitle("Round 44 — emergent roles from scratch: identical agents break symmetry into "
                 "complementary roles", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(full=full, no_interaction=noint, no_trigger=notrig)


def make_gif(theta, out: Path):
    rng = np.random.default_rng(7)
    runs = [SB.settle(theta, rng, return_traj=True)[1] for _ in range(3)]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 2.6); ax.set_yticks([])
    ax.set_xticks([-1, 1]); ax.set_xticklabels(["patch A", "patch B"])
    ax.axvline(0, color="#ddd", lw=1)
    ax.set_title("two identical agents settle into two roles", fontsize=10)
    dots0 = [ax.plot([], [], "o", color="#2060d0", ms=13, mec="k")[0] for _ in runs]
    dots1 = [ax.plot([], [], "o", color="#d04030", ms=13, mec="k")[0] for _ in runs]

    def upd(f):
        for r, tr in enumerate(runs):
            y = 0.4 + r
            dots0[r].set_data([tr[f, 0]], [y]); dots1[r].set_data([tr[f, 1]], [y])
        return dots0 + dots1

    FuncAnimation(fig, upd, frames=SB.T + 1, blit=False).save(out, writer=PillowWriter(fps=5))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 44: emergent roles from scratch")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== emergent roles from scratch: symmetry-breaking with no id ==")
    theta, _ = SB.evolve_es(gens=150, seed=0)
    res = figure(theta, outdir / "round44_symmetry_break.png")
    print(f"  split rate: full {res['full']:.2f} | no-interaction {res['no_interaction']:.2f} | "
          f"no-trigger {res['no_trigger']:.2f}")
    print(f"wrote {outdir/'round44_symmetry_break.png'}")
    if args.gif:
        make_gif(theta, outdir / "round44_symmetry_break.gif")
        print(f"wrote {outdir/'round44_symmetry_break.gif'}")
    (outdir / "round44_symmetry_break.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
