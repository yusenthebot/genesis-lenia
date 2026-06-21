# genesis

**A sandbox world where intelligence is grown, not coded — and the same engine carries
that world from 1D to 2D to 3D.**

![progress overview: emergence → locomotion → agency](outputs/progress_overview.png)

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
| 4 | **Survival** — forage to stay alive, or die | ✅ round 4 (metabolism) |
| 5 | **3D** — the same engine, one more axis | ✅ round 5 (3D self-organisation) |
| 6 | **Ecology** — many creatures compete for food | ✅ round 6 (stabilizing selection) |
| 7 | **Evolution running** — the population self-tunes | ✅ round 7 (discovers the optimum) |
| 8 | **Predator–prey** — a world that pushes back | ✅ round 8 (top-down regulation) |
| 9 | **Within-lifetime learning** | → next |

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

## Round 4 — survival (2D)

Now metabolism is on: the creature carries an **energy reserve** that drains every step
and is refilled only by eating; when it hits zero the body dissipates and the creature
**dies**. Food appears at random places over time, so staying alive demands *continual,
aimed* foraging. We evolve the forager to maximise lifetime and benchmark on **unseen**
food schedules:

![survival: energy curves + lifetimes](outputs/round4_survival.png)

The evolved forager banks an energy surplus and lives indefinitely (≈ **832** of 900
steps), while the sensing-ablated body starves at **158** and a random-drifter at **299**.
Intelligence now has teeth — it is the difference between living and dying.

![survival animation](outputs/round4_survival.gif)

## Round 5 — the third dimension (3D)

The whole engine was built so dimensionality is just `len(shape)`. Here the **identical
code** runs a 3D world, and a single seed spontaneously self-organises into a **robust,
persistent 3D structure** — homeostatic mass (`mass_cv ≈ 0.0008`), reproducible across
seeds. This completes the **1D → 2D → 3D** arc on one codebase.

![3D self-organisation](outputs/round5_3d.png)

![rotating 3D structure](outputs/round5_3d.gif)

> Honest scope: a *compact, mobile* 3D creature (the 3D analogue of the 2D glider) is **not**
> delivered. Across five search strategies (single/multi-ring kernels, growth-width sweeps,
> evolved 3D morphology) the reachable 3D dynamics are knife-edge — they die, foam, or
> proliferate, and flip on grid size and seed. Stable localised 3D creatures need the heavy
> specialised search the 3D-Lenia literature uses. What is solid here is robust 3D
> *self-organisation*; the 3D *creature* is an open frontier.

## Round 6 — ecology (2D)

Many creatures now share **one world and one finite food field** — each is a Lenia body
in its own channel, with its own energy and foraging reflex, and food eaten by one is
denied to the others (exploitation competition). Sweeping foraging skill across the
population reveals **stabilizing selection**: survival peaks at an *intermediate* skill —
under-foragers can't find food and starve, over-foragers overshoot and forage
inefficiently. The optimum sits at gain ≈ 4 and is robust across food-scarcity regimes.

![ecology: stabilizing selection + population dynamics](outputs/round6_ecology.png)

![ecology animation](outputs/round6_ecology.gif)

*(creatures coloured by foraging skill — red = low, blue = high; green = food)*

## Round 7 — evolution running (2D)

In round 6 we *measured* the fitness landscape by hand. Here selection **runs**: foraging
gain is a **heritable gene**, well-fed creatures **reproduce** (offspring inherit the gain
plus a mutation), and starving ones die. Under scarce food the population is food-limited,
so foraging skill decides who reproduces — and starting from **random** gains, the
population **discovers the optimum on its own**: mean gain converges to ≈ 3.5 across
independent runs, matching round 6's independently-swept optimum (~4). Two methods, one answer.

![evolution converging to the optimum](outputs/round7_evolution.png)

![evolving population animation](outputs/round7_evolution.gif)

*(population colour = foraging gain; over time it shifts toward the optimal strategy)*

## Round 8 — predator & prey (2D)

A **second species** now hunts the foragers. Prey forage food *and* flee predators;
predators sense the prey-density field, chase it, and eat prey into their own energy.
Both carry energy, reproduce, and have a heritable strategy gene. The world starts to
push back:

![predator–prey dynamics + co-evolution](outputs/round8_predprey.png)

- **Top-down regulation:** predators hold the prey population at ~14, four times below
  its predator-free ceiling of ~55 — the prey's very numbers are now set by something
  hunting it, and the prey shows vivid boom-bust fluctuations.
- **Co-evolution (the honest surprise):** selection acts, but prey evolve *lower* evasion
  (7 → 3), not higher — fleeing costs more foraging than it saves, so "forage hard and
  out-breed predation" wins. The evolved answer to predators is fecundity, not flight.
  *(A clean escalating arms race did not emerge — documented honestly; see `docs/progress.md`.)*

![predator–prey animation](outputs/round8_predprey.gif)

*(prey = blue, predators = red, food = green)*

---

## How it works

- **Engine** (`genesis/world.py`) — `World(shape, LeniaParams)`: an N-D field, an FFT
  radial-kernel convolution and a Gaussian growth map. Same code in 1D/2D/3D.
- **Metrics** (`genesis/metrics.py`) — emergence measured, not assumed: alive /
  localized / persistent / locomotion, plus a scale-aware **mass-concentration** that
  tells one compact creature from space-filling turbulence. All dimension-agnostic.
- **Evolution** (`genesis/evolve.py`) — a `(μ+λ)` GA over an `Individual` = rule
  (`LeniaParams`) + an evolvable **seed morphology** (a small smooth patch).
- **Agency + survival** (`genesis/foraging.py`) — `ForagingWorld` adds a food field, a
  sensing kernel, a sensorimotor drift reflex, eating, and a scalar **energy reserve**
  (metabolism drains it, food refills it, zero = death). `genesis/run3.py` evolves the
  forager (agency benchmark); `genesis/run4.py` turns on metabolism (survival benchmark).

## Run

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

.venv/bin/python -m genesis.run1d         # 1D self-organisation
.venv/bin/python -m genesis.run2d --gif   # evolve a 2D glider, render strip + gif
.venv/bin/python -m genesis.run3  --gif   # evolve a forager, agency benchmark + gif
.venv/bin/python -m genesis.run4  --gif   # turn on metabolism, survival benchmark + gif
.venv/bin/python -m genesis.run5_3d --gif # 3D self-organisation + rotating gif
.venv/bin/python -m genesis.run6  --gif   # ecology / competition + gif
.venv/bin/python -m genesis.run7  --gif   # evolution running inside the ecology + gif
.venv/bin/python -m genesis.run8  --gif   # predator-prey + gif
.venv/bin/python -m genesis.overview      # rebuild the progress montage
.venv/bin/python -m pytest -q             # engine + evolution + foraging invariants
```

Outputs (figures, GIFs, evolved genomes) are written to `outputs/`.

## Layout

```
genesis/
  world.py      dimension-agnostic Lenia engine
  metrics.py    emergence + concentration metrics (any dimension)
  evolve.py     co-evolution of rule + seed morphology (2D)
  evolve3d.py   3D morphology GA (PATCH^3 seed + rule)
  foraging.py   food field + sensorimotor reflex + metabolism (agency, survival)
  ecology.py    many creatures sharing one food field (competition)
  evo_ecology.py  heritable gain + birth/death/mutation (evolution running)
  predprey.py   two species: prey + predators (co-evolution, regulation)
  run1d.py … run8.py   round drivers + visualisation
  overview.py   stitches the per-round figures into one progress montage
tests/          engine (1D/2D/3D) + evolution + foraging invariants
docs/           STATUS.md + progress.md (autonomous-loop resume state)
outputs/        figures, GIFs, evolved genomes
```

## Method note

Built autonomously, round by round, under a simple rule: **never trust a metric — run
the world and look at it.** Every round here records its honest dead ends (a soup that
gamed the locomotion metric; three failed taxis couplings) right next to what worked, in
`docs/progress.md`.
