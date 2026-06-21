"""Round 31 — does the emerged language become COMPOSITIONAL?

Two agents must communicate a referent with TWO attributes (3 shapes x 3 colours). Round 30's
trick (evolve speaker+listener) gives a working but HOLISTIC code: it memorises each meaning,
fails on held-out combinations, and has low topographic similarity. Add a STRUCTURAL pressure
(reward topographic similarity) and the language becomes COMPOSITIONAL: the signals organise
into a MEANING GRID (one direction per attribute), topographic similarity jumps 0.2 -> 0.8, and
partial zero-shot generalisation appears. Compositionality is not free — it emerges under pressure.
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

import genesis.communicate_comp as C  # noqa: E402

SHAPES = ["o", "s", "^"]               # attribute 1 -> marker
COLORS = ["#d04030", "#2a9d2a", "#2060d0"]  # attribute 2 -> colour


def _pca2(X):
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def _scatter(ax, sig2):
    for i, (a, b) in enumerate(C.ALL):
        ax.scatter(sig2[i, 0], sig2[i, 1], marker=SHAPES[a], color=COLORS[b],
                   s=180, edgecolor="k", linewidth=0.6)
    ax.set_xticks([]); ax.set_yticks([])


def figure(out: Path):
    th_n, _, _ = C.evolve_es(gens=220, seed=0, topo_weight=0.0)
    th_p, _, _ = C.evolve_es(gens=220, seed=0, topo_weight=0.8)
    topo_n, topo_p = C.topographic_similarity(th_n), C.topographic_similarity(th_p)
    ho_n, ho_p = C.both_correct(th_n, C.HELDOUT), C.both_correct(th_p, C.HELDOUT)
    s_n = _pca2(C.signals(th_n)); s_p = _pca2(C.signals(th_p))

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.5), facecolor="white")
    a0.bar(["naive\n(communicate)", "structural\npressure", "ideal\ncompositional"],
           [topo_n, topo_p, 1.0], color=["#b04030", "#2a9d2a", "#888"])
    for i, v in enumerate([topo_n, topo_p, 1.0]):
        a0.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    a0.set_ylim(0, 1.1); a0.set_ylabel("topographic similarity (compositionality)")
    a0.set_title(f"Compositionality emerges under pressure\n(held-out zero-shot {ho_n:.2f} -> {ho_p:.2f})")

    _scatter(a1, s_n)
    a1.set_title(f"naive code: HOLISTIC (scrambled)\ntopo {topo_n:.2f}")
    _scatter(a2, s_p)
    a2.set_title(f"pressured code: STRUCTURED by attribute\n(shape=marker, colour=hue) topo {topo_p:.2f}")

    fig.suptitle("Round 31 — compositional communication: a structured language emerges under "
                 "pressure (topographic similarity 0.2 -> 0.8)", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(topo_naive=topo_n, topo_pressured=topo_p, heldout_naive=ho_n,
                heldout_pressured=ho_p)


def make_gif(out: Path):
    _, _, hist = C.evolve_es(gens=220, seed=0, topo_weight=0.8, capture=True)
    proj = _pca2(np.vstack(hist))            # consistent projection across frames
    hist2 = proj.reshape(len(hist), len(C.ALL), 2)
    lo = hist2.reshape(-1, 2).min(0) - 0.3; hi = hist2.reshape(-1, 2).max(0) + 0.3
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([]); ax.set_yticks([])
    scat = [ax.scatter([], [], marker=SHAPES[a], color=COLORS[b], s=180,
                       edgecolor="k", linewidth=0.6) for a, b in C.ALL]
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        for i in range(len(C.ALL)):
            scat[i].set_offsets(hist2[k, i][None, :])
        ttl.set_text(f"gen {k*4}: signals organise into a meaning grid")
        return scat

    FuncAnimation(fig, upd, frames=len(hist2), blit=False).save(
        out, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 31: compositional communication")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== compositional communication: does the emerged language factorise? ==")
    res = figure(outdir / "round31_compositional.png")
    print(f"  topographic similarity: naive {res['topo_naive']:.2f} -> pressured {res['topo_pressured']:.2f}; "
          f"held-out zero-shot {res['heldout_naive']:.2f} -> {res['heldout_pressured']:.2f}")
    print(f"wrote {outdir/'round31_compositional.png'}")
    if args.gif:
        make_gif(outdir / "round31_compositional.gif")
        print(f"wrote {outdir/'round31_compositional.gif'}")
    (outdir / "round31_compositional.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
