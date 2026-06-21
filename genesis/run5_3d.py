"""Round 5 — the third dimension: the SAME engine self-organises in 3D.

The whole architecture was built so dimensionality is just ``len(shape)``. Here we run
the identical Lenia engine on a 3D field and it spontaneously self-organises a single
seed into a robust, persistent 3D structure (homeostatic mass, reproducible across
seeds) — the 3D analogue of round 1's 1D lattice. This validates the dimensional-
generality bet end to end (1D -> 2D -> 3D, one codebase).

Honest scope: a *compact, mobile* 3D creature (the 3D analogue of the round-2 glider)
is NOT delivered here. Across many searches (single/multi-ring kernels, growth-width
sweeps, evolved 3D morphology) the 3D dynamics in reach are knife-edge — they die,
diffuse into foam, or proliferate — and stable localised creatures need the heavy
specialised search the 3D-Lenia literature uses. That is the open 3D frontier; what is
solid and shown here is robust 3D *self-organisation*.
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

from genesis.world import World, LeniaParams  # noqa: E402

# the robust persistent-3D-structure regime found by search (stable across seeds)
PARAMS = dict(R=8.0, mu_k=0.5, sigma_k=0.15, mu_g=0.12, sigma_g=0.024, dt=0.1)
N = 50
THRESH = 0.15


def run3d(seed=1, steps=260, record_times=None):
    p = LeniaParams(**PARAMS)
    w = World((N, N, N), p)
    rng = np.random.default_rng(seed)
    w.seed_blob(center=(N // 2,) * 3, radius=7, amp=1.0, asymmetry=4.0, rng=rng)
    snaps, mass = {}, []
    record_times = set(record_times or [])
    for t in range(steps + 1):
        if t in record_times:
            snaps[t] = w.A.copy()
        mass.append(float(w.A.sum()))
        w.step()
    return snaps, np.array(mass), w


def _scatter(ax, A, max_pts=6000, rng=None):
    zc, yc, xc = np.where(A > THRESH)
    v = A[A > THRESH]
    if len(v) > max_pts:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(v), max_pts, replace=False)
        xc, yc, zc, v = xc[idx], yc[idx], zc[idx], v[idx]
    ax.scatter(xc, yc, zc, c=v, cmap="magma", s=4, alpha=0.5, vmin=0, vmax=1)
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_zlim(0, N)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def figure(out: Path):
    times = [0, 30, 70, 130, 200, 260]
    snaps, mass, _ = run3d(seed=1, steps=260, record_times=times)
    fig = plt.figure(figsize=(15, 5.2), facecolor="white")
    for i, t in enumerate(times):
        ax = fig.add_subplot(2, 4, i + 1 if i < 3 else i + 2, projection="3d")
        _scatter(ax, snaps[t])
        ax.set_title(f"t={t}   mass={mass[t]:.0f}", fontsize=9)
    axm = fig.add_subplot(2, 4, 4)
    axm.plot(mass, color="#a00"); axm.set_title("mass(t) — homeostasis", fontsize=9)
    axm.set_xlabel("time"); axm.set_ylim(0, mass.max() * 1.1)
    axm.text(0.5, 0.1, f"tail mass_cv = {mass[-60:].std()/mass[-60:].mean():.4f}",
             transform=axm.transAxes, ha="center", fontsize=8, family="monospace")
    fig.suptitle("Round 5 — the SAME engine self-organises in 3D: a seed becomes a "
                 "robust, persistent 3D structure (1D -> 2D -> 3D, one codebase)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return mass


def rotating_gif(out: Path, frames=60):
    _, _, w = run3d(seed=1, steps=240)
    A = w.A
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, A)
    ax.set_title("emergent 3D structure", fontsize=10)

    def upd(k):
        ax.view_init(elev=22, azim=k * 360 / frames)
        return ()

    FuncAnimation(fig, upd, frames=frames, blit=False).save(
        out, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 5: 3D self-organisation")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== 3D world: same engine, len(shape)=3 ==")
    mass = figure(outdir / "round5_3d.png")
    cv = mass[-60:].std() / mass[-60:].mean()
    # robustness check across seeds
    finals = []
    for s in (1, 2, 3):
        _, m, _ = run3d(seed=s, steps=240)
        finals.append(m[-1])
    print(f"  persistent 3D structure: tail mass_cv={cv:.4f}, "
          f"final mass across seeds {[int(x) for x in finals]}")
    print(f"wrote {outdir/'round5_3d.png'}")
    if args.gif:
        rotating_gif(outdir / "round5_3d.gif")
        print(f"wrote {outdir/'round5_3d.gif'}")

    (outdir / "round5_3d.json").write_text(json.dumps(
        {"params": PARAMS, "N": N, "tail_mass_cv": float(cv),
         "final_mass_seeds": [float(x) for x in finals]}, indent=2))
    print(f"wrote {outdir/'round5_3d.json'}")


if __name__ == "__main__":
    main()
