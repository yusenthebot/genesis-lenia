"""Round 43 — open-ended ratchet: cumulative culture that invents its own complexity.

No target shape (unlike round 41). The artifact is a TOWER; the only rule is physics (the running
centre of mass above every joint must stay supported). Complexity = HEIGHT, unbounded. With both
transmission and innovation the culture accumulates a tall spire (~150 blocks) no single lifetime
could build; individual learning (restart each generation) is capped at one lifetime's few blocks;
copy-only never grows. The tower keeps reaching upward with no ceiling but stability itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import genesis.open_ended as OE  # noqa: E402


def _draw_tower(ax, centers, title, hmax):
    cmap = plt.get_cmap("viridis")
    n = len(centers)
    for i, x in enumerate(centers):
        ax.add_patch(Rectangle((x - OE.W / 2, i), OE.W, 1.0,
                               facecolor=cmap(i / max(hmax, 1)), edgecolor="k", linewidth=0.3))
    ax.set_xlim(-2.2, 3.2); ax.set_ylim(-2, hmax + 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{title}\nheight {n}", fontsize=9.5)


def figure(out: Path):
    ch, cr, chist = OE.run_lineage("cumulative", seed=0)
    ih, ir, ihist = OE.run_lineage("individual", seed=0)
    th, tr, thist = OE.run_lineage("transmit", seed=0)
    g = np.arange(OE.GENS + 1)
    hmax = int(ch[-1])

    fig = plt.figure(figsize=(14, 5.0), facecolor="white")
    a0 = fig.add_axes([0.05, 0.12, 0.42, 0.76])
    a0.plot(g, ch, color="#2a9d2a", lw=2.8, label="cumulative (transmit + innovate)")
    a0.plot(g, ih, color="#b04030", lw=2.2, label="individual (restart each gen)")
    a0.plot(g, th, color="#888", lw=2.0, ls="--", label="transmit-only (no innovation)")
    a0.set_xlabel("generation"); a0.set_ylabel("tower height (open-ended complexity)")
    a0.set_title("OPEN-ENDED: no target -- height just keeps accumulating", fontsize=10)
    a0.legend(fontsize=8.5, loc="upper left")

    a1 = fig.add_axes([0.52, 0.14, 0.15, 0.74]); _draw_tower(a1, chist[-1], "cumulative", hmax)
    a2 = fig.add_axes([0.70, 0.14, 0.15, 0.74]); _draw_tower(a2, ihist[-1], "individual", hmax)
    a3 = fig.add_axes([0.86, 0.14, 0.13, 0.74]); _draw_tower(a3, thist[-1], "copy-only", hmax)

    fig.suptitle("Round 43 — open-ended ratchet: a culture builds a tower no single lifetime could",
                 fontsize=11, fontweight="bold", y=0.965)
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(cumulative_height=int(ch[-1]), individual_height=int(ih[-1]),
                transmit_height=int(th[-1]), cumulative_reach=float(cr[-1]))


def make_gif(out: Path):
    ch, _, hist = OE.run_lineage("cumulative", seed=0)
    hmax = int(ch[-1])
    # sample ~30 frames across generations
    idx = np.linspace(0, len(hist) - 1, 30).astype(int)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(3.6, 5.2))

    def upd(f):
        ax.clear()
        centers = hist[idx[f]]
        for i, x in enumerate(centers):
            ax.add_patch(Rectangle((x - OE.W / 2, i), OE.W, 1.0,
                                   facecolor=cmap(i / max(hmax, 1)), edgecolor="k", linewidth=0.3))
        ax.set_xlim(-3, 5); ax.set_ylim(-1, hmax + 3); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"generation {idx[f]} · height {len(centers)}", fontsize=10)
        return []

    FuncAnimation(fig, upd, frames=len(idx), blit=False).save(out, writer=PillowWriter(fps=7))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 43: open-ended ratchet")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== open-ended ratchet: a culture builds a tower no single lifetime could ==")
    res = figure(outdir / "round43_open_ended.png")
    print(f"  final height: cumulative {res['cumulative_height']} | individual "
          f"{res['individual_height']} | copy-only {res['transmit_height']}")
    print(f"wrote {outdir/'round43_open_ended.png'}")
    if args.gif:
        make_gif(outdir / "round43_open_ended.gif")
        print(f"wrote {outdir/'round43_open_ended.gif'}")
    (outdir / "round43_open_ended.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
