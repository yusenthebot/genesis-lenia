"""Round 25 — a serious push at the stable, MOBILE 3D creature (the R5 negative).

The search (genesis/creature3d.py) upgrades R5's: multi-ring kernels + a shaped viability
gradient + a motion reward. The honest result:

  POSITIVE — a stable, COMPACT 3D creature is found (a single localised body, concentration
  ~1.0), an upgrade on R5 which only delivered diffuse 3D self-organisation.
  NEGATIVE (sharpened) — a MOBILE 3D creature (a 3D glider) is still NOT found. The search
  locates exactly WHY: compact bodies are stationary attractors, while the structures that
  MOVE are diffuse (not creatures). The glider is the unmet INTERSECTION — compact AND moving.

This driver renders the best compact creature in 3D and plots the compactness x motion
trade-off that makes the 3D glider knife-edge.
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

from genesis.creature3d import (  # noqa: E402
    random_creature, mutate, evaluate, Creature3D,
)

THRESH = 0.12


def _scatter(ax, A, max_pts=6000):
    zc, yc, xc = np.where(A > THRESH)
    v = A[A > THRESH]
    if len(v) > max_pts:
        idx = np.random.default_rng(0).choice(len(v), max_pts, replace=False)
        xc, yc, zc, v = xc[idx], yc[idx], zc[idx], v[idx]
    ax.scatter(xc, yc, zc, c=v, cmap="magma", s=5, alpha=0.55, vmin=0, vmax=1)
    n = A.shape[0]
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _tradeoff_points(n=70, seed=3):
    """Screen creatures, recording (compactness, travel) to show the empty glider corner."""
    rng = np.random.default_rng(seed)
    pts = []
    pool = [random_creature(rng) for _ in range(8)]
    for k in range(n):
        c = random_creature(rng) if k % 3 == 0 else mutate(pool[rng.integers(len(pool))], rng)
        _, m = evaluate(c, size=36, steps=80)
        if m["alive"]:
            pts.append((m["conc"], min(m["travel"], 2.0)))
            if m["conc"] > 0.5:
                pool.append(c)
    return np.array(pts) if pts else np.zeros((0, 2))


def figure(field, meta, out: Path):
    pts = _tradeoff_points()
    fig = plt.figure(figsize=(13.5, 5.0), facecolor="white")
    for i, az in enumerate((30, 120, 210)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        _scatter(ax, field)
        ax.view_init(elev=20, azim=az)
        if i == 1:
            ax.set_title(f"a STABLE COMPACT 3D creature found\n(concentration {meta['conc']:.2f}, "
                         f"mass {int(meta['mass'])}) — but it does NOT move", fontsize=9)

    ax = fig.add_subplot(1, 4, 4)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=22, c="#3070c0", alpha=0.7)
    ax.axhspan(0.4, 2.0, xmin=0.62, color="#d04030", alpha=0.10)
    ax.text(0.66, 1.5, "the 3D GLIDER\nlives here —\nbut is EMPTY", fontsize=8.5,
            color="#b03020", ha="left", va="center")
    ax.set_xlabel("compactness (concentration) ->")
    ax.set_ylabel("motion (travel, widths) ->")
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 2.05)
    ax.set_title("Why the 3D glider is knife-edge:\ncompact OR moving, not both", fontsize=9)

    fig.suptitle("Round 25 — the 3D creature: compact body found, mobile glider still the open negative",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=118)
    plt.close(fig)


def rotating_gif(field, out: Path, frames=48):
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, field)
    ax.set_title("a stable compact 3D creature", fontsize=10)

    def upd(k):
        ax.view_init(elev=20, azim=k * 360 / frames)
        return []

    FuncAnimation(fig, upd, frames=frames, blit=False).save(
        out, writer=PillowWriter(fps=18))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 25: 3D creature")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    # reconstruct the creature from its saved genome (deterministic) — no big .npy needed
    g = json.load(open(outdir / "round25_creature.json"))
    c = Creature3D(g["rule"], np.array(g["peaks"]), np.array(g["patch"]))
    _, m = evaluate(c, size=48, steps=140, chase=False, snap=True)
    field = m["field"]
    meta = {k: m[k] for k in ("conc", "travel", "mass", "alive", "via")}
    print(f"== 3D creature: compact={meta['conc']:.2f} travel={meta['travel']:.2f} "
          f"mass={int(meta['mass'])} alive={meta['alive']} ==")
    figure(field, meta, outdir / "round25_creature3d.png")
    print(f"wrote {outdir/'round25_creature3d.png'}")
    if args.gif:
        rotating_gif(field, outdir / "round25_creature3d.gif")
        print(f"wrote {outdir/'round25_creature3d.gif'}")


if __name__ == "__main__":
    main()
