"""Round 24 — open-ended MINDS: a zoo of foraging STRATEGIES (not body shapes).

Same body (the evolved glider), but the MIND — a 4-parameter foraging policy — varies. We
illuminate the space of foraging BEHAVIOURS (how much it ROAMS x how widely its path SPREADS)
with MAP-Elites, keeping the best-eating forager in each niche, vs a fitness GA. MAP-Elites
fills the map with distinct strategies (sitters, pacers, rovers) whose trajectories look
visibly different; fitness converges to one. Open-endedness of minds, not just morphologies.
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

from genesis.evolve import Genome  # noqa: E402
from genesis.openmind import (  # noqa: E402
    map_elites, fitness_search, ForagerMind, GRID, ROAM_MAX, SPREAD_MAX, N_FOOD,
)

N_EVAL = 480


def load_glider(path="outputs/round2_genome.json"):
    d = json.load(open(path))
    rule = Genome(**{k: d["rule"][k] for k in
                     ["mu_g", "sigma_g", "mu_k", "sigma_k", "R", "dt"]})
    return rule, np.array(d["patch"])


def _gallery_cells(archive):
    targets = [(1, 7), (4, 6), (7, 7), (2, 3), (5, 4), (7, 3)]
    keys, chosen, used = list(archive.keys()), [], set()
    for t in targets:
        b = min((k for k in keys if k not in used),
                key=lambda k: (k[0] - t[0]) ** 2 + (k[1] - t[1]) ** 2, default=None)
        if b is not None:
            chosen.append(b); used.add(b)
    return chosen


def _sim_body_frames(rule, patch, policy, shape=(110, 110), T=170, stride=4):
    f = ForagerMind(shape, rule, patch, policy, n_food=N_FOOD, seed=0)
    frames = []
    for t in range(T):
        f.step()
        if t % stride == 0:
            frames.append((np.clip(f.A, 0, 1).copy(),
                           np.array(f.foods).copy() if f.foods else np.zeros((0, 2))))
        if not f.foods:
            break
    return frames


def figure(rule, patch, archive, cov_me, cov_fit, out: Path):
    fig = plt.figure(figsize=(13.5, 8.0), facecolor="white")

    qmap = np.full((GRID, GRID), np.nan)
    for (i, j), e in archive.items():
        qmap[j, i] = e["q"]
    ax0 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    im = ax0.imshow(qmap, origin="lower", cmap="viridis", aspect="auto")
    ax0.set_xlabel("how much it ROAMS  (path length) ->")
    ax0.set_ylabel("how WIDE its path spreads ->")
    ax0.set_title(f"MAP-Elites: {len(archive)} foraging strategies")
    fig.colorbar(im, ax=ax0, fraction=0.046, label="food eaten")

    ax1 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax1.plot([c[0] for c in cov_me], [c[1] for c in cov_me], color="#2a9d2a", lw=2.4,
             label="MAP-Elites (illuminate)")
    ax1.plot([c[0] for c in cov_fit], [c[1] for c in cov_fit], color="#b04030", lw=2.4,
             label="fitness GA (converges)")
    ax1.set_xlabel("policies evaluated"); ax1.set_ylabel("distinct strategies retained")
    ax1.set_title("Strategy diversity: grow vs collapse")
    ax1.legend(fontsize=8)

    # one example trajectory (top-right)
    cells = _gallery_cells(archive)
    axd = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    m = archive[cells[0]]["m"]
    tr = m["traj"]
    axd.plot(tr[:, 1], tr[:, 0], color="#2060d0", lw=1.2)
    axd.plot(tr[0, 1], tr[0, 0], "ko", ms=5)
    axd.set_aspect("equal"); axd.set_xticks([]); axd.set_yticks([])
    axd.set_title("a single forager's path", fontsize=10)

    # trajectory gallery (bottom row of 6 strategies)
    for n, cell in enumerate(cells[:6]):
        ax = plt.subplot2grid((2, 6), (1, n))
        m = archive[cell]["m"]
        tr = m["traj"]
        ax.plot(tr[:, 1], tr[:, 0], color="#1a6", lw=1.0)
        ax.plot(tr[0, 1], tr[0, 0], "k.", ms=6)
        ax.set_xlim(0, 110); ax.set_ylim(0, 110); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"path={int(m['path'])}\nate={m['eaten']}", fontsize=7)

    fig.suptitle("A ZOO of foraging MINDS — same body, different strategies",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=118)
    plt.close(fig)


def make_gif(rule, patch, archive, out: Path):
    cells = _gallery_cells(archive)[:4]
    seqs = [_sim_body_frames(rule, patch, archive[c]["pol"]) for c in cells]
    labels = [f"path={int(archive[c]['m']['path'])} ate={archive[c]['m']['eaten']}"
              for c in cells]
    n = max(len(s) for s in seqs)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    ims, dots = [], []
    for ax, s, lab in zip(axes, seqs, labels):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=9)
        A0, food0 = s[0]
        ims.append(ax.imshow(np.dstack([A0, np.zeros_like(A0), np.zeros_like(A0)])))
        dots.append(ax.scatter(food0[:, 1], food0[:, 0], c="#20d020", s=18) if len(food0)
                    else ax.scatter([], [], c="#20d020", s=18))
    fig.suptitle("four foraging minds — same body, distinct strategies", fontsize=11)

    def upd(k):
        for im, dot, s in zip(ims, dots, seqs):
            A, food = s[min(k, len(s) - 1)]
            im.set_data(np.dstack([A, np.zeros_like(A), np.zeros_like(A)]))
            dot.set_offsets(food[:, ::-1] if len(food) else np.zeros((0, 2)))
        return ims

    FuncAnimation(fig, upd, frames=n, blit=False).save(out, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 24: open-ended minds")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--evals", type=int, default=N_EVAL)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rule, patch = load_glider()
    print("== open-ended minds: a zoo of foraging strategies (same body) ==")
    archive, cov_me = map_elites(rule, patch, n_eval=args.evals, seed=0, eval_seeds=(0, 1))
    _, cov_fit = fitness_search(rule, patch, n_eval=max(300, args.evals // 2), seed=0,
                                eval_seeds=(0, 1))
    print(f"  MAP-Elites: {len(archive)}/{GRID*GRID} strategies; fitness GA converges to "
          f"{cov_fit[-1][1]} distinct")
    figure(rule, patch, archive, cov_me, cov_fit, outdir / "round24_openmind.png")
    print(f"wrote {outdir/'round24_openmind.png'}")
    if args.gif:
        make_gif(rule, patch, archive, outdir / "round24_openmind.gif")
        print(f"wrote {outdir/'round24_openmind.gif'}")

    res = dict(strategies=len(archive), grid=GRID * GRID, fitness_diversity=cov_fit[-1][1],
               coverage=cov_me)
    (outdir / "round24_openmind.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir/'round24_openmind.json'}")


if __name__ == "__main__":
    main()
