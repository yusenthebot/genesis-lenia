# genesis

**A sandbox world where intelligence is grown, not coded — and the same engine carries
that world from 1D to 2D to 3D.**

The substrate is a continuous cellular automaton (Lenia family): the world is an
N-dimensional real-valued field, and one update is a radial-kernel convolution followed
by a growth map.

```
U = K * A                # neighbourhood sum (circular FFT convolution)
A = clip(A + dt * G(U))  # growth toward a preferred neighbourhood
```

Because the kernel and convolution are built from `shape` alone, the *identical* code
runs in any dimension — `len(shape)` **is** the world's dimensionality. Nothing above
the engine is hand-placed: persistent structure, locomotion, evolution and agency all
*emerge* from local rules and selection.

## The arc so far

| # | Frontier | Status |
|---|----------|--------|
| 1 | **Emergence** — self-maintaining structure from local rules | ✅ round 1 (1D) |
| 2 | **Locomotion** — a creature that travels | ✅ round 2 (evolved 2D glider) |
| 3 | **Agency** — sense the world and act to pursue a goal | ✅ round 3 (evolved forager) |
| 4 | **Survival / ecology / learning / 3D** | → next |

---

## Round 1 — emergence (1D)

From a single seed, the world spontaneously self-organises into a **persistent,
homeostatic lattice** of localised structures (mass coefficient-of-variation ≈ 0).
Directed motion appears but is *transient* — a measured finding that persistent gliders
are a 2D phenomenon.

![1D self-organisation](outputs/round1_1d_selforg.png)

## Round 2 — locomotion (2D)

We **co-evolve the rule and the seed morphology** under a locomotion objective, and
selection discovers a genuine **glider** de novo: a single compact body that crosses the
field in a near-straight line (net travel ≈ 3.8 widths, straightness 0.99) while holding
constant mass. Orbium is *not* hardcoded; the creature is found.

![2D evolved glider](outputs/round2_2d_creature.png)

![glider animation](outputs/round2_2d_creature.gif)

> Lesson, caught twice by *looking* rather than trusting a metric: a glider is a narrow
> attractor (no Gaussian blob reaches it under any rule → the seed shape must be evolved),
> and a space-filling "soup" games naive travel/support metrics → a scale-aware
> **mass-concentration** gate is required to mean "one creature."

## Round 3 — agency (2D)

The glider gets a **sensorimotor reflex**: it senses the food gradient over its body and
drifts up it (a mass-exact `np.roll` translation — the body stays an emergent Lenia
creature). We evolve the reflex and benchmark it on **unseen** random food layouts:

![foraging benchmark](outputs/round3_benchmark.png)

The evolved forager eats **85%** of the food vs **18%** for the sensing-ablated body and
**18%** for a creature that drifts the same amount in a *random* direction. The random
control is the clincher — equal motion, no food-sensing, no better than ablated — so the
advantage is specifically *food-directed* sensing, not movement. The creature turns
toward food **in every direction**, then stays and consumes it:

![foraging trajectories](outputs/round3_agency.png)

![foraging animation](outputs/round3_forage.gif)

*(red = creature, green = food)*

---

## How it works

- **Engine** (`genesis/world.py`) — `World(shape, LeniaParams)`: an N-D field, an FFT
  radial-kernel convolution and a Gaussian growth map. Same code in 1D/2D/3D.
- **Metrics** (`genesis/metrics.py`) — emergence measured, not assumed: alive /
  localized / persistent / locomotion, plus a scale-aware **mass-concentration** that
  tells one compact creature from space-filling turbulence. All dimension-agnostic.
- **Evolution** (`genesis/evolve.py`) — a `(μ+λ)` GA over an `Individual` = rule
  (`LeniaParams`) + an evolvable **seed morphology** (a small smooth patch).
- **Agency** (`genesis/foraging.py`) — `ForagingWorld` adds a food field, a sensing
  kernel, a sensorimotor drift reflex, and eating; `genesis/run3.py` evolves the forager
  and runs the ablation/random-drift benchmark.

## Run

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

.venv/bin/python -m genesis.run1d         # 1D self-organisation
.venv/bin/python -m genesis.run2d --gif   # evolve a 2D glider, render strip + gif
.venv/bin/python -m genesis.run3  --gif   # evolve a forager, agency benchmark + gif
.venv/bin/python -m pytest -q             # engine + evolution + foraging invariants
```

Outputs (figures, GIFs, evolved genomes) are written to `outputs/`.

## Layout

```
genesis/
  world.py      dimension-agnostic Lenia engine
  metrics.py    emergence + concentration metrics (any dimension)
  evolve.py     co-evolution of rule + seed morphology
  foraging.py   food field + sensorimotor reflex (agency)
  run1d.py run2d.py run3.py   round drivers + visualisation
tests/          engine (1D/2D/3D) + evolution + foraging invariants
docs/           STATUS.md + progress.md (autonomous-loop resume state)
outputs/        figures, GIFs, evolved genomes
```

## Method note

Built autonomously, round by round, under a simple rule: **never trust a metric — run
the world and look at it.** Every round here records its honest dead ends (a soup that
gamed the locomotion metric; three failed taxis couplings) right next to what worked, in
`docs/progress.md`.
