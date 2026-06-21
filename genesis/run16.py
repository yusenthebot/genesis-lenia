"""Round 16 — a brain that predicts: foreseeing the next world-state.

The world emits a periodic, ambiguous pattern; predicting the next state requires an
internal MODEL and phase-tracking, not just reaction. A recurrent brain (evolved by an ES)
predicts every step including the FLIPS, with 1 full bit of predictive information about the
future; a feedforward (reactive) brain is at chance. We show the evolution curves, the
prediction trace anticipating each flip, and a GIF where the brain calls the next symbol
before it arrives.
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

from genesis.predict_brain import evolve_es, run_seq, PATTERN  # noqa: E402
from genesis.measure import mutual_information_bits  # noqa: E402

GENS = 60
SEEDS = 3


def train(recurrent):
    curves, thetas, accs = [], [], []
    for s in range(SEEDS):
        theta, curve = evolve_es(recurrent=recurrent, gens=GENS, seed=s)
        a, _ = run_seq(theta, recurrent, n_periods=80)
        curves.append(curve); thetas.append(theta); accs.append(a)
    best = thetas[int(np.argmax(accs))]
    return np.array(curves), best, float(np.mean(accs))


def pred_info(theta, recurrent):
    _, _, preds, truths, _ = run_seq(theta, recurrent, n_periods=80, return_trace=True)
    return mutual_information_bits(preds, truths)


def figure(rec_curves, ff_curves, theta_rec, out: Path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    g = np.arange(1, GENS + 1)
    for c in rec_curves:
        a0.plot(g, c, color="#2a9d2a", lw=1, alpha=0.4)
    for c in ff_curves:
        a0.plot(g, c, color="#b04030", lw=1, alpha=0.4)
    a0.plot(g, rec_curves.mean(0), color="#2a9d2a", lw=2.5, label="recurrent (models the world)")
    a0.plot(g, ff_curves.mean(0), color="#b04030", lw=2.5, label="feedforward (reactive)")
    a0.axhline(0.5, color="#bbb", ls=":", lw=1)
    a0.set_ylim(0.3, 1.02); a0.set_xlabel("evolution generation")
    a0.set_ylabel("next-state prediction accuracy")
    a0.set_title("Predicting an ambiguous world needs a model")
    a0.legend(fontsize=9, loc="center right")

    # prediction trace: the brain calls the next state, including the flips
    _, _, preds, truths, cur = run_seq(theta_rec, True, n_periods=80, return_trace=True)
    k = 16
    x = np.arange(k)
    a1.step(x, cur[:k], where="post", color="#888", lw=2, label="world state now")
    a1.plot(x, preds[:k] + 0.04, "o", color="#2a9d2a", ms=7,
            label="brain's prediction of NEXT")
    a1.plot(x, truths[:k] - 0.04, "x", color="#2060d0", ms=7, label="actual NEXT state")
    flips = np.where(cur[:k] != truths[:k])[0]
    for f in flips:
        a1.axvspan(f - 0.4, f + 0.4, color="#ffe6a6", alpha=0.7, lw=0)
    a1.set_yticks([0, 1]); a1.set_ylim(-0.3, 1.3)
    a1.set_xlabel("time step (shaded = a FLIP — the world reverses; must be anticipated)")
    a1.set_title(f"The brain foresees every flip (predictive info {pred_info(theta_rec, True):.2f} bits)")
    a1.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_gif(theta, out: Path):
    _, _, preds, truths, cur = run_seq(theta, True, n_periods=10, return_trace=True)
    n = min(28, len(preds))
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1]); ax.set_xticks([])
    ttl = ax.set_title("", fontsize=10)
    (world_line,) = ax.step([], [], where="post", color="#888", lw=2)
    pred_pt = ax.scatter([], [], color="#2a9d2a", s=80, zorder=5)

    def upd(k):
        world_line.set_data(range(k + 1), cur[:k + 1])
        pred_pt.set_offsets([[k + 1, preds[k]]])      # prediction of the NEXT step
        flip = cur[k] != truths[k]
        ok = preds[k] == truths[k]
        msg = "FLIP — anticipated!" if (flip and ok) else ("predicting..." if ok else "miss")
        ttl.set_text(f"world now={cur[k]}  ->  brain predicts next={preds[k]}  ({msg})")
        return world_line, pred_pt, ttl

    FuncAnimation(fig, upd, frames=n, blit=False).save(
        out, writer=PillowWriter(fps=2))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 16: a brain that predicts")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"== a brain that predicts: anticipate the world pattern {PATTERN} ==")
    rec_curves, theta_rec, acc_rec = train(recurrent=True)
    ff_curves, theta_ff, acc_ff = train(recurrent=False)
    pir, pif = pred_info(theta_rec, True), pred_info(theta_ff, False)
    print(f"  prediction accuracy: recurrent {acc_rec:.2f} vs feedforward {acc_ff:.2f}")
    print(f"  predictive info: {pir:.2f} bits (recurrent) vs {pif:.2f} bits (feedforward)")
    figure(rec_curves, ff_curves, theta_rec, outdir / "round16_predict.png")
    print(f"wrote {outdir/'round16_predict.png'}")
    if args.gif:
        make_gif(theta_rec, outdir / "round16_predict.gif")
        print(f"wrote {outdir/'round16_predict.gif'}")

    (outdir / "round16_predict.json").write_text(json.dumps(
        {"acc_recurrent": acc_rec, "acc_feedforward": acc_ff,
         "predictive_bits_recurrent": pir, "predictive_bits_feedforward": pif,
         "pattern": PATTERN}, indent=2))
    print(f"wrote {outdir/'round16_predict.json'}")


if __name__ == "__main__":
    main()
