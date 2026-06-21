"""Round 22 — open-endedness: does the world keep GENERATING creatures, or converge?

Every earlier round converges. Here we ILLUMINATE a 2-D behaviour space (how much a creature
MOVES x how BIG it is) with MAP-Elites, keeping the most viable creature in each niche, and
compare against pure fitness search. MAP-Elites fills the map with a ZOO of distinct creatures
while fitness collapses to one -> the Lenia substrate is generative across this behaviour space.
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

from genesis.world import World  # noqa: E402
from genesis.openended import (  # noqa: E402
    map_elites, fitness_search, simulate, place_patch, GRID, MOVE_MAX,
    MASS_MIN, MASS_MAX,
)

N_EVAL = 760


def _sim_frames(ind, shape=(72, 72), T=120, stride=3):
    world = World(shape, ind.rule.to_params())
    A = place_patch(shape, ind.patch)
    frames = []
    for t in range(T):
        U = world.potential(A)
        A = np.clip(A + ind.rule.dt * world.growth(U), 0.0, 1.0)
        if t % stride == 0:
            frames.append(A.copy())
    return frames


def _gallery_cells(archive):
    """Pick a spread of cells across the behaviour map for the zoo gallery."""
    targets = [(0, 0), (7, 0), (0, 7), (7, 7), (4, 2), (2, 5), (6, 4), (3, 7)]
    keys = list(archive.keys())
    chosen, used = [], set()
    for tgt in targets:
        best = min((k for k in keys if k not in used),
                   key=lambda k: (k[0] - tgt[0]) ** 2 + (k[1] - tgt[1]) ** 2,
                   default=None)
        if best is not None:
            chosen.append(best); used.add(best)
    return chosen


def figure(archive, cov_me, cov_fit, out: Path):
    fig = plt.figure(figsize=(13.5, 8.0), facecolor="white")

    # behaviour map (top-left, spans 4 of 8 columns)
    qmap = np.full((GRID, GRID), np.nan)
    for (i, j), e in archive.items():
        qmap[j, i] = e["q"]
    ax0 = plt.subplot2grid((2, 8), (0, 0), colspan=4)
    im = ax0.imshow(qmap, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax0.set_xlabel("how much it MOVES  (drift speed) ->")
    ax0.set_ylabel("how BIG its body is  (mass) ->")
    ax0.set_title(f"MAP-Elites illuminates {len(archive)} distinct creatures "
                  f"({100*len(archive)//(GRID*GRID)}% of the map)")
    fig.colorbar(im, ax=ax0, fraction=0.046, label="viability")

    # coverage curve (top-right)
    ax1 = plt.subplot2grid((2, 8), (0, 4), colspan=4)
    ax1.plot([c[0] for c in cov_me], [c[1] for c in cov_me], color="#2a9d2a", lw=2.4,
             label="MAP-Elites (illuminate diversity)")
    ax1.plot([c[0] for c in cov_fit], [c[1] for c in cov_fit], color="#b04030", lw=2.4,
             label="fitness search (converges)")
    ax1.set_xlabel("creatures evaluated"); ax1.set_ylabel("distinct behaviours retained")
    ax1.set_title("Diversity keeps growing vs collapses to one")
    ax1.legend(fontsize=9, loc="center right")

    # zoo gallery (bottom row of 8 snapshots)
    cells = _gallery_cells(archive)
    for n, cell in enumerate(cells[:8]):
        ax = plt.subplot2grid((2, 8), (1, n))
        ax.imshow(archive[cell]["snap"], cmap="magma", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        m = archive[cell]["m"]
        ax.set_title(f"v={m['drift_speed']:.2f}\nmass={int(m['mean_mass'])}", fontsize=7)

    fig.suptitle("A ZOO of evolved creatures — one substrate, many bodies & behaviours",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=118)
    plt.close(fig)


def make_gif(archive, out: Path):
    corners = _gallery_cells(archive)[:4]
    seqs = [_sim_frames(archive[c]["ind"]) for c in corners]
    labels = [f"v={archive[c]['m']['drift_speed']:.2f}  m={int(archive[c]['m']['mean_mass'])}"
              for c in corners]
    n = min(len(s) for s in seqs)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    ims = []
    for ax, s, lab in zip(axes, seqs, labels):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=9)
        ims.append(ax.imshow(s[0], cmap="magma", vmin=0, vmax=1))
    fig.suptitle("four creatures from the zoo — one substrate, distinct behaviours", fontsize=11)

    def upd(k):
        for im, s in zip(ims, seqs):
            im.set_data(s[min(k, len(s) - 1)])
        return ims

    FuncAnimation(fig, upd, frames=n, blit=False).save(out, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 22: open-endedness")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--evals", type=int, default=N_EVAL)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== open-endedness: MAP-Elites illuminates a zoo vs fitness converges ==")
    archive, cov_me = map_elites(n_eval=args.evals, seed=0)
    _, fit_best, cov_fit = fitness_search(n_eval=max(400, args.evals // 2), seed=0)
    print(f"  MAP-Elites filled {len(archive)}/{GRID*GRID} niches; "
          f"fitness search retains 1 (converged, fitness {fit_best:.2f})")
    figure(archive, cov_me, cov_fit, outdir / "round22_openended.png")
    print(f"wrote {outdir/'round22_openended.png'}")
    if args.gif:
        make_gif(archive, outdir / "round22_openended.gif")
        print(f"wrote {outdir/'round22_openended.gif'}")

    res = dict(niches_filled=len(archive), grid=GRID * GRID, fitness_best=fit_best,
               coverage_curve=cov_me)
    (outdir / "round22_openended.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir/'round22_openended.json'}")


if __name__ == "__main__":
    main()
