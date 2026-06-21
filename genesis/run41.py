"""Round 41 — cumulative culture: the ratchet.

Each generation inherits the previous artifact, innovates a BOUNDED amount within one lifetime, and
passes on a slightly better one -> skill RATCHETS UP past any single lifetime. The ratchet needs BOTH
faithful transmission AND innovation: cumulative (both) climbs to a sharp star (quality ~0.9), while
individual (innovate, restart each generation) is stuck at the single-lifetime ceiling (~0.06) and
transmit-only (copy, no innovation) never improves (~0.03). The accumulated artifact -- a star -- is
something no individual lifetime here could build.
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

import genesis.cumulative_culture as CC  # noqa: E402


def _draw_artifact(ax, design, title, q):
    tgt = np.vstack([CC.TARGET, CC.TARGET[0]])
    ax.plot(tgt[:, 0], tgt[:, 1], color="#ccc", lw=1.2, zorder=1)
    ax.scatter(design[:, 0], design[:, 1], s=14, color="#2a9d2a", zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)
    ax.set_title(f"{title}\nquality {q:.2f}", fontsize=9.5)


def figure(out: Path):
    cum_c, cum_h = CC.run_lineage("cumulative", seed=0)
    ind_c, ind_h = CC.run_lineage("individual", seed=0)
    tra_c, _ = CC.run_lineage("transmit", seed=0)
    g = np.arange(CC.GENS + 1)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor="white")
    a0 = axes[0]
    a0.plot(g, cum_c, color="#2a9d2a", lw=2.7, label="cumulative (transmit + innovate)")
    a0.plot(g, ind_c, color="#b04030", lw=2.2, label="individual (restart each gen)")
    a0.plot(g, tra_c, color="#888", lw=2.0, ls="--", label="transmit-only (no innovation)")
    a0.set_xlabel("generation"); a0.set_ylabel("artifact quality")
    a0.set_ylim(0, 1.0); a0.set_title("The RATCHET: skill accumulates only with\nBOTH transmission and innovation")
    a0.legend(fontsize=8, loc="center right")

    mid = CC.GENS // 2
    _draw_artifact(axes[1], cum_h[0], "cumulative · generation 0", cum_c[0])
    _draw_artifact(axes[2], cum_h[mid], f"cumulative · generation {mid}", cum_c[mid])
    _draw_artifact(axes[3], cum_h[-1], f"cumulative · generation {CC.GENS}", cum_c[-1])

    fig.suptitle("Round 41 — cumulative culture: an artifact RATCHETS toward a star no single "
                 "lifetime could build", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=116)
    plt.close(fig)
    return dict(cumulative_final=float(cum_c[-1]), individual_final=float(ind_c[-1]),
                transmit_final=float(tra_c[-1]), individual_artifact_quality=float(ind_c[-1]))


def make_gif(out: Path):
    cum_c, cum_h = CC.run_lineage("cumulative", seed=0)
    tgt = np.vstack([CC.TARGET, CC.TARGET[0]])
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot(tgt[:, 0], tgt[:, 1], color="#ccc", lw=1.2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)
    scat = ax.scatter(cum_h[0][:, 0], cum_h[0][:, 1], s=16, color="#2a9d2a")
    ttl = ax.set_title("")

    def upd(k):
        scat.set_offsets(cum_h[k])
        ttl.set_text(f"generation {k} · quality {cum_c[k]:.2f} (each builds on the last)")
        return [scat, ttl]

    FuncAnimation(fig, upd, frames=len(cum_h), blit=False).save(out, writer=PillowWriter(fps=7))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 41: cumulative culture")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== cumulative culture: the ratchet ==")
    res = figure(outdir / "round41_cumulative.png")
    print(f"  final quality: cumulative {res['cumulative_final']:.2f} | individual "
          f"{res['individual_final']:.2f} | transmit-only {res['transmit_final']:.2f}")
    print(f"wrote {outdir/'round41_cumulative.png'}")
    if args.gif:
        make_gif(outdir / "round41_cumulative.gif")
        print(f"wrote {outdir/'round41_cumulative.gif'}")
    (outdir / "round41_cumulative.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
