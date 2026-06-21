"""Round 15 — a brain with memory: holding a cue across a delay.

Every controller until now was a memoryless reflex. Here a tiny recurrent network is
evolved on a cue-recall task: a cue appears, vanishes, and after a random delay the
creature must act on it. A feedforward (memoryless) controller is stuck at chance; the
recurrent one holds the cue in its hidden state and acts correctly. We show the evolution
curves (recurrent solves it, feedforward cannot), the MEMORY read straight out of the
hidden state (in bits), and a GIF where you watch the cue set the memory, the memory
persist through the delay, and the correct action fire.
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

from genesis.memory_brain import evolve_es, fitness, run_trial, unpack, H  # noqa: E402
from genesis.measure import mutual_information_bits  # noqa: E402

GENS = 60
SEEDS = 3


def train(recurrent):
    curves, thetas = [], []
    for s in range(SEEDS):
        theta, curve = evolve_es(recurrent=recurrent, gens=GENS, seed=s)
        curves.append(curve); thetas.append(theta)
    accs = [fitness(t, np.random.default_rng(900 + i), recurrent, K=400)
            for i, t in enumerate(thetas)]
    best = thetas[int(np.argmax(accs))]
    return np.array(curves), best, float(np.mean(accs))


def memory_bits(theta, recurrent, n=400):
    """MI(decoded-cue-during-delay ; true cue) — how many bits of memory the brain holds."""
    rng = np.random.default_rng(7)
    Wh, Wi, bh, wo = unpack(theta)
    if not recurrent:
        Wh = np.zeros_like(Wh)
    cues, decoded = [], []
    for _ in range(n):
        cue = rng.choice([-1.0, 1.0]); delay = int(rng.integers(4, 9))
        h = np.zeros(H)
        seq = [[cue, 0, 1]] + [[0, 0, 1]] * delay + [[0, 1, 1]]
        mid = 1 + delay // 2                        # sample mid-delay (cue long gone)
        for t, o in enumerate(seq):
            h = np.tanh(Wh @ h + Wi @ np.array(o, float) + bh)
            if t == mid:
                decoded.append(1 if (wo @ h) > 0 else 0)
                cues.append(1 if cue > 0 else 0)
    return mutual_information_bits(decoded, cues)


def memory_trace(theta):
    """Readout (wo.h) over time for cue=+1 and cue=-1, long delay -> the held memory."""
    traces = {}
    for cue in (+1.0, -1.0):
        _, hs, seq = run_trial(theta, cue, delay=10, recurrent=True, return_trace=True)
        Wh, Wi, bh, wo = unpack(theta)
        traces[cue] = (np.array([wo @ h for h in hs]), seq)
    return traces


def figure(rec_curves, ff_curves, theta_rec, theta_ff, out: Path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    g = np.arange(1, GENS + 1)
    for c in rec_curves:
        a0.plot(g, c, color="#2a9d2a", lw=1, alpha=0.4)
    for c in ff_curves:
        a0.plot(g, c, color="#b04030", lw=1, alpha=0.4)
    a0.plot(g, rec_curves.mean(0), color="#2a9d2a", lw=2.5, label="recurrent (has memory)")
    a0.plot(g, ff_curves.mean(0), color="#b04030", lw=2.5, label="feedforward (no memory)")
    a0.axhline(0.5, color="#bbb", ls=":", lw=1)
    a0.set_ylim(0.3, 1.02); a0.set_xlabel("evolution generation")
    a0.set_ylabel("cue-recall accuracy")
    a0.set_title("Memory solves a task a reflex cannot")
    a0.legend(fontsize=9, loc="center right")

    traces = memory_trace(theta_rec)
    for cue, color in [(+1.0, "#2060d0"), (-1.0, "#d06010")]:
        tr, seq = traces[cue]
        a1.plot(range(len(tr)), tr, "o-", color=color, lw=1.8, ms=4,
                label=f"cue = {int(cue):+d}")
    go_t = len(traces[1.0][0]) - 1
    a1.axvline(0, color="#888", ls="--", lw=1); a1.text(0, a1.get_ylim()[1], " cue", fontsize=8)
    a1.axvline(go_t, color="#888", ls="--", lw=1); a1.text(go_t, a1.get_ylim()[1], " go", fontsize=8, ha="right")
    a1.axhline(0, color="#ccc", lw=0.8)
    mb_r = memory_bits(theta_rec, True); mb_f = memory_bits(theta_ff, False)
    a1.set_xlabel("time step (cue at 0, go at end; flat middle = the delay)")
    a1.set_ylabel("brain output (sign = decision)")
    a1.set_title(f"The hidden state HOLDS the cue  (memory: {mb_r:.2f} bits vs {mb_f:.2f} ablated)")
    a1.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return mb_r, mb_f


def make_gif(theta, out: Path):
    cue, delay = 1.0, 9
    Wh, Wi, bh, wo = unpack(theta)
    seq = [[cue, 0, 1]] + [[0, 0, 1]] * delay + [[0, 1, 1]]
    h = np.zeros(H); mems = []
    for o in seq:
        h = np.tanh(Wh @ h + Wi @ np.array(o, float) + bh)
        mems.append(float(wo @ h))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.set_xlim(-0.5, len(seq) - 0.5); ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color="#ccc", lw=0.8); ax.set_xticks([]); ax.set_yticks([])
    (line,) = ax.plot([], [], "o-", color="#2a9d2a", lw=2)
    ttl = ax.set_title("", fontsize=10)

    def upd(k):
        line.set_data(range(k + 1), mems[:k + 1])
        o = seq[k]
        if o[0] != 0:
            s = f"CUE = {int(cue):+d} shown  ->  memory set"
        elif o[1] == 1:
            act = "+1" if mems[k] > 0 else "-1"
            s = f"GO!  act {act}  ({'correct' if (mems[k]>0)==(cue>0) else 'wrong'})"
        else:
            s = f"delay (cue gone) — holding memory = {'+' if mems[k]>0 else '-'}"
        ttl.set_text(f"t={k}   {s}")
        return line, ttl

    FuncAnimation(fig, upd, frames=len(seq), blit=False).save(
        out, writer=PillowWriter(fps=2))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 15: a brain with memory")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== a brain with memory: cue-recall across a delay ==")
    rec_curves, theta_rec, acc_rec = train(recurrent=True)
    ff_curves, theta_ff, acc_ff = train(recurrent=False)
    print(f"  cue-recall accuracy: recurrent {acc_rec:.2f} vs feedforward {acc_ff:.2f}")
    mb_r, mb_f = figure(rec_curves, ff_curves, theta_rec, theta_ff,
                        outdir / "round15_memory.png")
    print(f"  memory held: {mb_r:.2f} bits (recurrent) vs {mb_f:.2f} bits (feedforward)")
    print(f"wrote {outdir/'round15_memory.png'}")
    if args.gif:
        make_gif(theta_rec, outdir / "round15_memory.gif")
        print(f"wrote {outdir/'round15_memory.gif'}")

    (outdir / "round15_memory.json").write_text(json.dumps(
        {"acc_recurrent": acc_rec, "acc_feedforward": acc_ff,
         "memory_bits_recurrent": mb_r, "memory_bits_feedforward": mb_f}, indent=2))
    print(f"wrote {outdir/'round15_memory.json'}")


if __name__ == "__main__":
    main()
