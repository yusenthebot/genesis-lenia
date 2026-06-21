"""Round 46 — the evolution of cooperation: a spatial major transition.

In a WELL-MIXED world defectors always win and cooperation collapses to zero. Add SPATIAL STRUCTURE
and cooperators form CLUSTERS that preferentially help each other (network reciprocity, Nowak & May
1992), so cooperation survives. Same prisoner's-dilemma game and imitate-best update -- the only
difference is a 2D neighbourhood vs a shuffled one. Cooperation persists at ~0.31 spatially (with a
clustering index ~2.2, i.e. cooperators clump far more than random) but goes to 0 well-mixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import genesis.cooperation as CO  # noqa: E402

CMAP = ListedColormap(["#202830", "#2ad07a"])     # defector dark, cooperator green


def _clustering(grid):
    f = grid.mean()
    adj = (np.mean(grid * np.roll(grid, 1, 0)) + np.mean(grid * np.roll(grid, 1, 1))) / 2
    return adj / (f * f + 1e-9)


def figure(out: Path):
    sp_frac, sp_hist = CO.run(CO.B, well_mixed=False, seed=0, capture=True)
    wm_frac, _ = CO.run(CO.B, well_mixed=True, seed=0)
    t = np.arange(len(sp_frac))

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.8, 4.5), facecolor="white")
    a0.plot(t, sp_frac, color="#2a9d2a", lw=2.7, label="spatial (neighbours)")
    a0.plot(t, wm_frac, color="#b04030", lw=2.4, label="well-mixed (shuffled)")
    a0.set_xlabel("generation"); a0.set_ylabel("fraction cooperating")
    a0.set_ylim(-0.02, 1.0); a0.set_title("Cooperation SURVIVES with space,\nCOLLAPSES when well-mixed")
    a0.legend(fontsize=9)

    g = sp_hist[-1]
    a1.imshow(g, cmap=CMAP, interpolation="nearest"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"spatial grid: cooperators CLUSTER\n(green=C, dark=D; clustering {_clustering(g):.1f}x random)")

    bs = np.linspace(1.35, 1.95, 13)
    sp = [CO.final_cooperation(b, well_mixed=False, seeds=(0, 1)) for b in bs]
    wm = [CO.final_cooperation(b, well_mixed=True, seeds=(0, 1)) for b in bs]
    a2.plot(bs, sp, "o-", color="#2a9d2a", lw=2.2, label="spatial")
    a2.plot(bs, wm, "s-", color="#b04030", lw=2.0, label="well-mixed")
    a2.axvline(CO.B, color="#888", ls=":", lw=1.2)
    a2.set_xlabel("temptation to defect  b"); a2.set_ylabel("final cooperation")
    a2.set_ylim(-0.02, 1.0); a2.set_title("space sustains cooperation up to a\ncritical temptation; well-mixed never")
    a2.legend(fontsize=9)

    fig.suptitle("Round 46 — the evolution of cooperation: spatial clustering lets altruism survive "
                 "(a major transition)", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(spatial_final=float(sp_frac[-1]), well_mixed_final=float(wm_frac[-1]),
                clustering=float(_clustering(g)), b=CO.B)


def make_gif(out: Path):
    sp = CO.run(CO.B, well_mixed=False, seed=0, capture=True)[1]
    wm = CO.run(CO.B, well_mixed=True, seed=0, capture=True)[1]
    fig, (axs, axw) = plt.subplots(1, 2, figsize=(8.2, 4.4))
    for ax, t in zip((axs, axw), ("spatial", "well-mixed")):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=10)
    im_s = axs.imshow(sp[0], cmap=CMAP, interpolation="nearest")
    im_w = axw.imshow(wm[0], cmap=CMAP, interpolation="nearest")
    n = min(len(sp), len(wm))

    def upd(k):
        im_s.set_data(sp[k]); im_w.set_data(wm[k])
        axs.set_xlabel(f"coop {sp[k].mean():.2f}"); axw.set_xlabel(f"coop {wm[k].mean():.2f}")
        return [im_s, im_w]

    FuncAnimation(fig, upd, frames=n, blit=False).save(out, writer=PillowWriter(fps=8))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 46: evolution of cooperation")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== the evolution of cooperation: a spatial major transition ==")
    res = figure(outdir / "round46_cooperation.png")
    print(f"  final cooperation: spatial {res['spatial_final']:.2f} | well-mixed "
          f"{res['well_mixed_final']:.2f} | clustering {res['clustering']:.1f}x (b={res['b']})")
    print(f"wrote {outdir/'round46_cooperation.png'}")
    if args.gif:
        make_gif(outdir / "round46_cooperation.gif")
        print(f"wrote {outdir/'round46_cooperation.gif'}")
    (outdir / "round46_cooperation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
