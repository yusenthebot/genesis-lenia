"""Round 29 — multi-channel Flow-Lenia: the substrate built; motion still walled off.

The user chose to build multi-channel Flow-Lenia (the paper's glider mechanism). It is built
(genesis/flowlenia_mc.py): C channels, each advected by its own affinity gradient, COUPLED via
cross-channel kernels; each channel conserves its own mass. Coupled, structured 2-channel
creatures form. But a serious GA over the coupling + a less-diffusive (reintegration-tracking)
advection BOTH leave motion at ~0.1R — the same wall as single-channel.

This sharpens round 28's diagnosis into its final form: motion is walled off by the GRADIENT
FLOW itself (F = grad(G) relaxes to a stationary equilibrium), NOT by the channel count or the
advection scheme. Translation needs a self-sustaining asymmetric (uniform-drift) attractor that
these numpy formulations do not reach. The mobile creature is now an exhaustively-tested,
well-explained negative; the remaining path is the gated differentiable-Lenia apparatus.
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

from genesis.flowlenia_mc import FlowWorldMC, two_channel_conn  # noqa: E402

N = 110
LOCO_R = 7.5  # round-2 plain-Lenia glider, ~3.78 widths ~ 7.5 R (the bar)


def _seed(w):
    yy, xx = np.indices((N, N)); c = N // 2
    w.A[0] = 0.5 * np.exp(-0.5 * ((yy - c) ** 2 + (xx - c) ** 2) / 8.0 ** 2)
    w.A[1] = 0.5 * np.exp(-0.5 * ((yy - c - 3) ** 2 + (xx - c + 2) ** 2) / 7.0 ** 2)


def _run(steps=180, record=False):
    w = FlowWorldMC((N, N), 2, two_channel_conn(R=12.0, cross_off=4), flow_clip=0.9)
    _seed(w)
    masses = [w.total().copy()]
    frames = []
    for i in range(steps):
        w.step()
        masses.append(w.total().copy())
        if record and i % 4 == 0:
            frames.append((w.A[0].copy(), w.A[1].copy()))
    return w, np.array(masses), frames


def _rgb(a0, a1):
    return np.clip(np.dstack([a0 / 2.0, a1 / 2.0, np.zeros_like(a0)]), 0, 1)


def figure(out: Path):
    w, masses, _ = _run(180)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")

    a0.imshow(_rgb(w.A[0], w.A[1])); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title("a 2-CHANNEL coupled creature\n(red = ch0, green = ch1, structured)")

    t = np.arange(len(masses))
    a1.plot(t, masses[:, 0] / masses[0, 0], color="#c03030", lw=2.2, label="channel 0")
    a1.plot(t, masses[:, 1] / masses[0, 1], color="#2a9d2a", lw=2.2, label="channel 1")
    a1.set_ylim(0.9, 1.1); a1.axhline(1.0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("step"); a1.set_ylabel("mass / initial")
    a1.set_title("each channel conserves its mass"); a1.legend(fontsize=8)

    labels = ["single-channel\n(R28 GA)", "multi-channel\n(R29 GA)", "locomotion\n(round-2 glider)"]
    vals = [0.20, 0.11, LOCO_R]
    cols = ["#d09030", "#d09030", "#2a9d2a"]
    a2.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.15, f"{v:.2f}R" if i < 2 else f"{v:.1f}R", ha="center", fontsize=10)
    a2.set_ylabel("net travel (creature radii)")
    a2.set_title("motion: same wall with or without\nmulti-channel -> it's the GRADIENT FLOW")

    fig.suptitle("Round 29 — multi-channel Flow-Lenia built; coupled creatures form + conserve mass, "
                 "but motion stays walled off (gradient flow)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)


def make_gif(out: Path):
    _, _, frames = _run(180, record=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.7))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(_rgb(*frames[0]))
    ax.set_title("2-channel coupled creature (stays put)", fontsize=10)

    def upd(k):
        im.set_data(_rgb(*frames[k])); return [im]

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 29: multi-channel Flow-Lenia")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== multi-channel Flow-Lenia: substrate built; motion still walled off (gradient flow) ==")
    figure(outdir / "round29_multichannel.png")
    print(f"wrote {outdir/'round29_multichannel.png'}")
    if args.gif:
        make_gif(outdir / "round29_multichannel.gif")
        print(f"wrote {outdir/'round29_multichannel.gif'}")
    (outdir / "round29_multichannel.json").write_text(json.dumps(
        dict(single_channel_travel_R=0.20, multi_channel_travel_R=0.11, locomotion_R=LOCO_R),
        indent=2))


if __name__ == "__main__":
    main()
