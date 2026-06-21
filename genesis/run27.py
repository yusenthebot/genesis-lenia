"""Round 27 — Flow-Lenia: a mass-conserving substrate (pure numpy).

Plain Lenia grows/clips mass in place, so it is not conserved — a moving body dissipates, and
3D creatures are knife-edge (R5/R25: die / foam / proliferate). Flow-Lenia conserves mass by
MOVING it along the affinity gradient with mass-conserving advection. This round builds it in
numpy and shows what the conservation buys:

  - mass is conserved EXACTLY (a flat line), in 2D and 3D;
  - a seed self-organises into a COMPACT creature in 2D AND robustly in 3D (where plain Lenia 3D
    was knife-edge) — mass conservation removes the dissipation/explosion failure modes;
  - MULTIPLE creatures coexist in one world, sharing the conserved mass.

Honest: a MOBILE creature is not delivered here either — with a symmetric radial kernel the
clumps are stationary attractors; motion needs a search (as in round 2) or multi-channel kernels.
But it is now on a substrate where, once found, a moving body cannot dissipate.
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

from genesis.world import LeniaParams  # noqa: E402
from genesis.flowlenia import FlowWorld  # noqa: E402

P2 = dict(R=12.0, mu_k=0.5, sigma_k=0.15, mu_g=0.15, sigma_g=0.018, dt=0.2)
P3 = dict(R=9.0, mu_k=0.5, sigma_k=0.15, mu_g=0.15, sigma_g=0.02, dt=0.2)


def _blob(shape, centre, sigma, amp=0.5):
    idx = np.indices(shape, dtype=float)
    d2 = sum((idx[i] - centre[i]) ** 2 for i in range(len(shape)))
    return amp * np.exp(-0.5 * d2 / sigma ** 2)


def _blur(a, k=10):
    for _ in range(k):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def _count_creatures(A):
    """Count well-separated creatures: blur away internal texture, then count blobs."""
    b = _blur(A, 12)
    return _label2d(b > 0.35 * b.max())


def _label2d(mask):
    """Count 4-connected components in a 2D boolean mask (tiny numpy flood fill)."""
    lab = np.zeros(mask.shape, int)
    nxt = 0
    for s in zip(*np.where(mask)):
        if lab[s]:
            continue
        nxt += 1
        stack = [s]
        while stack:
            y, x = stack.pop()
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] and not lab[y, x]:
                lab[y, x] = nxt
                stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return nxt


def _run_single(shape, params, seed_sigma, steps):
    w = FlowWorld(shape, LeniaParams(**params), flow_clip=0.85)
    w.A = _blob(shape, [s // 2 for s in shape], seed_sigma)
    m0 = w.mass()
    ratios = [1.0]
    for i in range(steps):
        w.step()
        if i % 8 == 0:
            ratios.append(w.mass() / m0)
    return w, np.array(ratios)


def figure(out: Path):
    # 2D single creature
    w2, r2 = _run_single((110, 110), P2, 8.0, 180)
    # 3D creature
    w3, r3 = _run_single((52, 52, 52), P3, 7.0, 130)
    # multi-creature 2D
    wm = FlowWorld((130, 130), LeniaParams(**P2), flow_clip=0.85)
    A = np.zeros((130, 130))
    for cyx in [(38, 40), (40, 92), (92, 45), (88, 95)]:
        A += _blob((130, 130), cyx, 7.0)
    wm.A = A
    m0 = wm.mass()
    for _ in range(170):
        wm.step()
    # count whole creatures: each has fine internal texture + a dotted ring halo that naive
    # thresholding fragments, so blur away the texture first, then count well-separated blobs
    nclump = _count_creatures(wm.A)

    fig = plt.figure(figsize=(13.5, 4.6), facecolor="white")
    a0 = fig.add_subplot(1, 4, 1)
    a0.imshow(w2.A, cmap="magma"); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title("2D: a compact creature\n(mass conserved)", fontsize=9)

    a1 = fig.add_subplot(1, 4, 2)
    a1.plot(np.linspace(0, 180, len(r2)), r2, color="#2a9d2a", lw=2.2, label="2D")
    a1.plot(np.linspace(0, 130, len(r3)), r3, color="#2060d0", lw=2.2, label="3D")
    a1.set_ylim(0.9, 1.1); a1.axhline(1.0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("step"); a1.set_ylabel("mass / initial mass")
    a1.set_title("Mass is conserved EXACTLY", fontsize=9); a1.legend(fontsize=8)

    a2 = fig.add_subplot(1, 4, 3, projection="3d")
    zc, yc, xc = np.where(w3.A > 0.1)
    v = w3.A[w3.A > 0.1]
    if len(v) > 5000:
        idx = np.random.default_rng(0).choice(len(v), 5000, replace=False)
        zc, yc, xc, v = zc[idx], yc[idx], xc[idx], v[idx]
    a2.scatter(xc, yc, zc, c=v, cmap="magma", s=5, alpha=0.5)
    a2.set_xticks([]); a2.set_yticks([]); a2.set_zticks([])
    a2.set_title("3D: a ROBUST compact creature\n(plain Lenia 3D was knife-edge)", fontsize=9)

    a3 = fig.add_subplot(1, 4, 4)
    a3.imshow(wm.A, cmap="magma"); a3.set_xticks([]); a3.set_yticks([])
    a3.set_title(f"multi-creature: {nclump} creatures\ncoexist, sharing conserved mass", fontsize=9)

    fig.suptitle("Round 27 — Flow-Lenia: a mass-conserving substrate; robust 2D & 3D creatures, "
                 "multi-creature worlds", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(mass_ratio_2d=float(r2[-1]), mass_ratio_3d=float(r3[-1]), clumps=int(nclump))


def make_gif(out: Path):
    wm = FlowWorld((130, 130), LeniaParams(**P2), flow_clip=0.85)
    A = np.zeros((130, 130))
    for cyx in [(38, 40), (40, 92), (92, 45), (88, 95)]:
        A += _blob((130, 130), cyx, 7.0)
    wm.A = A
    frames = []
    for i in range(180):
        wm.step()
        if i % 4 == 0:
            frames.append((wm.A.copy(), wm.mass()))
    m0 = frames[0][1]
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0][0], cmap="magma", vmin=0, vmax=2.0)
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        A, m = frames[k]
        im.set_data(A)
        ttl.set_text(f"Flow-Lenia — 4 creatures, mass {m/m0:.3f}x initial (conserved)")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 27: Flow-Lenia")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== Flow-Lenia: mass-conserving substrate (2D + 3D + multi-creature) ==")
    res = figure(outdir / "round27_flowlenia.png")
    print(f"  mass ratio 2D={res['mass_ratio_2d']:.4f} 3D={res['mass_ratio_3d']:.4f}; "
          f"multi-creature clumps={res['clumps']}")
    print(f"wrote {outdir/'round27_flowlenia.png'}")
    if args.gif:
        make_gif(outdir / "round27_flowlenia.gif")
        print(f"wrote {outdir/'round27_flowlenia.gif'}")
    (outdir / "round27_flowlenia.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir/'round27_flowlenia.json'}")


if __name__ == "__main__":
    main()
