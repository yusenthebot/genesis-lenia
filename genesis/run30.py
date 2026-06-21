"""Round 30 — emergent communication: two agents evolve a shared code from scratch.

A SPEAKER sees a hidden referent and emits a continuous 2-D signal; a LISTENER sees only the
signal and must name the referent. Neither is given a code; we evolve both jointly (ES). A
language EMERGES: accuracy climbs from chance to 1.0, the signals separate into a discrete code,
and the realised information I(referent; listener-action) reaches the log2(K)-bit ceiling.
Ablating the channel (random signal) collapses it to chance -> the signal is genuinely used.
First SOCIAL-intelligence round (the mind arc so far was single-agent).
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

from genesis.communicate import (  # noqa: E402
    K, M, NPARAMS, _unpack, speak, accuracy, mutual_information_bits, fitness,
)

COLORS = ["#d04030", "#2a9d2a", "#2060d0", "#d0a020", "#a040c0", "#20b0a0"]


def evolve_capture(gens=110, pop=40, sigma=0.3, lr=0.25, seed=0):
    """Evolve and capture, per generation, the accuracy and the speaker's K signals."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, NPARAMS)
    curve, sig_hist = [], []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, NPARAMS))
        fits = np.array([fitness(theta + sigma * eps[i]) for i in range(pop)])
        adv = (fits - fits.mean()) / (fits.std() + 1e-9)
        theta = theta + lr / (pop * sigma) * (eps.T @ adv)
        curve.append(accuracy(theta))
        sp, _ = _unpack(theta)
        sig_hist.append(np.array([speak(sp, s) for s in range(K)]))
    return theta, np.array(curve), np.array(sig_hist)


def figure(out: Path):
    theta, curve, sig = evolve_capture(seed=0)
    acc = accuracy(theta); mi = mutual_information_bits(theta)
    accA = accuracy(theta, ablate=True, rng=np.random.default_rng(9))
    miA = mutual_information_bits(theta, ablate=True, rng=np.random.default_rng(9))

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    a0.plot(np.arange(len(curve)), curve, color="#2a9d2a", lw=2.4)
    a0.axhline(1.0 / K, color="#b04030", ls="--", lw=1.6, label=f"chance ({1.0/K:.2f})")
    a0.set_xlabel("evolution generation"); a0.set_ylabel("listener accuracy")
    a0.set_ylim(0, 1.05); a0.set_title("A shared language emerges"); a0.legend(fontsize=8)

    fin = sig[-1]
    for s in range(K):
        a1.scatter(*fin[s], s=180, color=COLORS[s], label=f"referent {s}")
        a1.text(fin[s, 0], fin[s, 1], f" {s}", fontsize=9, va="center")
    a1.set_xlabel("signal dim 1"); a1.set_ylabel("signal dim 2")
    a1.set_title("the evolved CODE: signals\nseparate into distinct words"); a1.legend(fontsize=7)

    x = ["evolved", "ablated\n(random signal)"]
    a2.bar(x, [mi, miA], color=["#2a9d2a", "#b04030"], width=0.5)
    a2.axhline(np.log2(K), color="k", ls=":", lw=1.2, label=f"ceiling ({np.log2(K):.0f} bits)")
    for i, v in enumerate([mi, miA]):
        a2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11)
    a2.set_ylabel("I(referent ; listener-action), bits")
    a2.set_ylim(0, np.log2(K) * 1.15)
    a2.set_title(f"information transmitted\n(accuracy {acc:.2f} vs ablated {accA:.2f})")
    a2.legend(fontsize=8)

    fig.suptitle("Round 30 — emergent communication: two agents evolve a shared code, "
                 "measured in bits", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118)
    plt.close(fig)
    return dict(accuracy=acc, accuracy_ablated=accA, bits=mi, bits_ablated=miA,
                ceiling=float(np.log2(K)))


def make_gif(out: Path):
    _, _, sig = evolve_capture(seed=0)
    lo = sig.reshape(-1, M).min(0) - 0.5; hi = sig.reshape(-1, M).max(0) + 0.5
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("signal dim 1"); ax.set_ylabel("signal dim 2")
    pts = [ax.scatter([], [], s=180, color=COLORS[s]) for s in range(K)]
    ttl = ax.set_title("", fontsize=10)

    def upd(g):
        for s in range(K):
            pts[s].set_offsets(sig[g, s][None, :])
        ttl.set_text(f"gen {g}: the 4 signals separate into a code")
        return pts

    FuncAnimation(fig, upd, frames=len(sig), blit=False).save(
        out, writer=PillowWriter(fps=14))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Round 30: emergent communication")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("== emergent communication: two agents evolve a shared code ==")
    res = figure(outdir / "round30_communication.png")
    print(f"  accuracy {res['accuracy']:.2f} (ablated {res['accuracy_ablated']:.2f}); "
          f"I={res['bits']:.2f} bits (ablated {res['bits_ablated']:.2f}, ceiling {res['ceiling']:.0f})")
    print(f"wrote {outdir/'round30_communication.png'}")
    if args.gif:
        make_gif(outdir / "round30_communication.gif")
        print(f"wrote {outdir/'round30_communication.gif'}")
    (outdir / "round30_communication.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
