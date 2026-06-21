"""Round 34 — iterated learning: compositionality from cultural transmission.

Round 31 made language compositional with a HAND-ADDED structural reward. Here it emerges from
the principled mechanism (Kirby): cultural transmission through a LEARNABILITY BOTTLENECK. Each
"generation" learns the language from only a SUBSET of meanings, then must produce all of them.
Holistic codes can't be reconstructed from few examples and degrade; compositional ones survive.
Under the bottleneck topographic similarity RISES (~0 -> ~0.35); with FULL transmission it stays
flat. No structure term added — structure emerges because compositional languages are learnable.
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

from genesis.communicate_iterate import iterate, ALL, A  # noqa: E402

SHAPES = ["o", "s", "^"]
COLORS = ["#d04030", "#2a9d2a", "#2060d0"]
GENS = 45
SEEDS = (0, 1, 2, 3, 4)


def _pca2(X):
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def _curves(bottleneck):
    cs = [iterate(bottleneck, gens=GENS, seed=s)[1] for s in SEEDS]
    return np.array(cs)


def figure(out: Path):
    full = _curves(len(ALL))
    bn = _curves(5)
    g = np.arange(GENS + 1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")

    a0.plot(g, bn.mean(0), color="#2a9d2a", lw=2.6, label="bottleneck 5/9 (transmission)")
    a0.fill_between(g, bn.min(0), bn.max(0), color="#2a9d2a", alpha=0.15)
    a0.plot(g, full.mean(0), color="#b04030", lw=2.6, label="full transmission (control)")
    a0.fill_between(g, full.min(0), full.max(0), color="#b04030", alpha=0.12)
    a0.set_xlabel("generation"); a0.set_ylabel("topographic similarity (compositionality)")
    a0.set_title("Compositionality emerges from a TRANSMISSION\nbottleneck — no hand-added pressure")
    a0.legend(fontsize=9, loc="center right")

    # the bottleneck-evolved language, organised by meaning
    _, curve_b, hist = iterate(5, gens=GENS, seed=0, return_hist=True)
    s2 = _pca2(hist[-1])
    for i, (a, b) in enumerate(ALL):
        a1.scatter(s2[i, 0], s2[i, 1], marker=SHAPES[a], color=COLORS[b], s=190,
                   edgecolor="k", linewidth=0.6)
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"the transmitted language is STRUCTURED\n(shape=attr1, colour=attr2) "
                 f"topo {curve_b[-1]:.2f}")

    fig.suptitle("Round 34 — iterated learning: structure emerges from cultural transmission "
                 "(Kirby), not a hand-added term", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(full_final=float(full.mean(0)[-1]), bottleneck_final=float(bn.mean(0)[-1]),
                bottleneck_peak=float(bn.mean(0).max()))


def make_gif(out: Path):
    _, _, hist = iterate(5, gens=GENS, seed=0, return_hist=True)
    proj = _pca2(np.vstack(hist)).reshape(len(hist), len(ALL), 2)
    lo = proj.reshape(-1, 2).min(0) - 0.3; hi = proj.reshape(-1, 2).max(0) + 0.3
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_xticks([]); ax.set_yticks([])
    pts = [ax.scatter([], [], marker=SHAPES[a], color=COLORS[b], s=190, edgecolor="k",
                      linewidth=0.6) for a, b in ALL]
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        for i in range(len(ALL)):
            pts[i].set_offsets(proj[k, i][None, :])
        ttl.set_text(f"generation {k}: language transmitted through a bottleneck")
        return pts

    FuncAnimation(fig, upd, frames=len(hist), blit=False).save(
        out, writer=PillowWriter(fps=8))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 34: iterated learning")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== iterated learning: compositionality from cultural transmission ==")
    res = figure(outdir / "round34_iterated.png")
    print(f"  topo final: full transmission {res['full_final']:.2f} vs bottleneck "
          f"{res['bottleneck_final']:.2f} (peak {res['bottleneck_peak']:.2f})")
    print(f"wrote {outdir/'round34_iterated.png'}")
    if args.gif:
        make_gif(outdir / "round34_iterated.gif")
        print(f"wrote {outdir/'round34_iterated.gif'}")
    (outdir / "round34_iterated.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
