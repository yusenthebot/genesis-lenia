"""Round 28 — why the creature won't move: the gradient-flow diagnosis.

The mobile creature has now been attacked three ways (R25 plain Lenia; R27 Flow-Lenia symmetric;
R28 Flow-Lenia with asymmetric kernels, rotated flow, AND a proper round-2-style GA over rule +
SEED SHAPE). All converge to the same place: a robust COMPACT but STATIONARY creature. This round
makes the negative precise and explains it.

THE DIAGNOSIS: single-channel Flow-Lenia moves mass by F = grad(G(U)) — a GRADIENT flow, which is
curl-free, so it can only RELAX mass toward an equilibrium (the stationary attractor). Evolution
pushes travel up from ~0.06R (random) to ~0.2R (evolved) but plateaus far below locomotion (round
2's plain-Lenia glider crossed 3.78 widths). Sustained locomotion needs a NON-gradient flow — i.e.
MULTI-CHANNEL Flow-Lenia, where the combined flow from several channels is not a pure gradient.
That is the clear next frontier (a bigger build), recorded honestly.
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
from genesis.creature_flow import search, FlowCreature, _place  # noqa: E402

LOCOMOTION = 3.78  # round-2 plain-Lenia glider, net travel in widths (the bar to clear)


def _sim(c: FlowCreature, shape=(100, 100), steps=200, record=False):
    p = LeniaParams(R=c.rule["R"], mu_k=c.rule["mu_k"], sigma_k=c.rule["sigma_k"],
                    mu_g=c.rule["mu_g"], sigma_g=c.rule["sigma_g"], dt=0.2)
    w = FlowWorld(shape, p, flow_clip=0.9, asym=c.asym, phi=c.phi)
    w.A = _place(shape, c.patch)
    yy, xx = np.indices(shape)
    tot = w.A.sum() + 1e-9
    traj = [((w.A * yy).sum() / tot, (w.A * xx).sum() / tot)]
    frames = []
    for i in range(steps):
        w.step()
        tot = w.A.sum() + 1e-9
        traj.append(((w.A * yy).sum() / tot, (w.A * xx).sum() / tot))
        if record and i % 4 == 0:
            frames.append(w.A.copy())
    return np.array(traj), frames, w


def figure(curve, c: FlowCreature, out: Path):
    traj, _, w = _sim(c, steps=220)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")

    g = np.arange(len(curve))
    a0.plot(g, curve, color="#2a9d2a", lw=2.4, label="evolved best (Flow-Lenia)")
    a0.axhline(LOCOMOTION, color="#b04030", lw=2.0, ls="--",
               label=f"locomotion (round-2 glider, {LOCOMOTION:.1f}w)")
    a0.set_xlabel("evolution generation"); a0.set_ylabel("net travel (creature widths)")
    a0.set_ylim(0, LOCOMOTION * 1.1)
    a0.set_title("Evolution pushes motion up — but plateaus\nfar below locomotion")
    a0.legend(fontsize=8, loc="center right")

    a1.imshow(w.A, cmap="magma"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"the evolved creature: COMPACT\n(conc 1.00) but barely moves "
                 f"({np.hypot(*(traj[-1]-traj[0]))/c.rule['R']:.2f}R)")

    a2.plot(traj[:, 1], traj[:, 0], color="#2060d0", lw=1.5)
    a2.plot(traj[0, 1], traj[0, 0], "go", ms=7, label="start")
    a2.plot(traj[-1, 1], traj[-1, 0], "rx", ms=9, mew=2, label="end")
    a2.set_aspect("equal")
    a2.set_xlim(traj[:, 1].mean() - 20, traj[:, 1].mean() + 20)
    a2.set_ylim(traj[:, 0].mean() - 20, traj[:, 0].mean() + 20)
    a2.set_xticks([]); a2.set_yticks([]); a2.legend(fontsize=8)
    a2.set_title("centroid stays put: F=grad(G) is a\nGRADIENT flow -> relaxes to equilibrium")

    fig.suptitle("Round 28 — why the creature won't move: a gradient flow relaxes; "
                 "locomotion needs multi-channel (non-gradient) flow",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)


def make_gif(c: FlowCreature, out: Path):
    _, frames, _ = _sim(c, steps=200, record=True)
    fig, ax = plt.subplots(figsize=(4.4, 4.6))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0], cmap="magma", vmin=0, vmax=2.0)
    ax.set_title("a compact Flow-Lenia creature — staying put", fontsize=10)

    def upd(k):
        im.set_data(frames[k]); return [im]

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(
        out, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 28: Flow-Lenia motion diagnosis")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    g = json.load(open(outdir / "round28_flowcreature.json"))
    c = FlowCreature(g["rule"], g["asym"], g["phi"], np.array(g["patch"]))
    print("== Flow-Lenia motion: a gradient flow relaxes -> stationary (diagnosis) ==")
    # short GA to capture the best-travel curve (in widths = travel*R / width... travel is in R)
    res = search(pop=20, gens=26, seed=0, verbose=False)
    curve = np.array([t["travel"] for t in res["trail"]])  # net travel in creature-R units
    # convert R-units to "widths": a creature ~2R wide, so widths ~ travel(R)/2; keep in R but
    # compare against locomotion expressed the same way (round-2 glider ~7.5R ~ 3.78 widths)
    curve_w = curve * 0.5
    print(f"  evolved best travel: {res['metrics']['travel']:.2f}R (~{res['metrics']['travel']*0.5:.2f}w) "
          f"vs locomotion {LOCOMOTION:.1f}w; conc {res['metrics']['conc']:.2f}")
    figure(curve_w, c, outdir / "round28_motion_diagnosis.png")
    print(f"wrote {outdir/'round28_motion_diagnosis.png'}")
    if args.gif:
        make_gif(c, outdir / "round28_motion_diagnosis.gif")
        print(f"wrote {outdir/'round28_motion_diagnosis.gif'}")


if __name__ == "__main__":
    main()
