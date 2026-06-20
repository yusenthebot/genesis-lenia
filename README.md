# genesis

A sandbox world where intelligence is **grown, not coded** — and where the same
engine carries that world from **1D to 2D to 3D**.

The substrate is a continuous cellular automaton in the Lenia family: the world is
an N-dimensional real-valued field, and one update is a radial-kernel convolution
followed by a growth map,

```
U = K * A                # neighbourhood sum (circular FFT convolution)
A = clip(A + dt * G(U))  # growth toward a preferred neighbourhood
```

Because the kernel and convolution are built from `shape` alone, the *identical*
code runs in any dimension — `len(shape)` is the world's dimensionality. Nothing
above the engine is hand-placed: persistent structures, locomotion, and (later)
evolution and agency are meant to *emerge* from local rules.

## North star

1. **Emergence** — self-maintaining structures arise from local rules. ✅ (round 1, 1D)
2. **Locomotion** — creatures that travel (proto-agency). ✅ (round 2, evolved 2D glider)
3. **Evolution** — rule + morphology under selection → open-ended complexity. ✅ (round 2)
4. **Agency / intelligence** — sensing + acting + selection over a world the creature
   must cope with (food/resource); intelligence *measured*, not asserted. → round 3
5. **3D** — the same engine, one more axis.

## Round 1 result (1D)

From a single seed, the world spontaneously self-organises into a **persistent,
homeostatic lattice** of localised structures (mass CV ≈ 0, support ≈ 12%).
Directed motion appears but is **transient** (net drift ~0.4 widths early → ~0
late) — a measured finding that persistent gliders are a 2D phenomenon.
Evidence: `outputs/round1_1d_selforg.png`, `outputs/round1_1d_locomotion.png`.

## Round 2 result (2D) — an evolved creature that travels

We **co-evolve the rule and the seed morphology** (a small smooth patch) under a
locomotion objective, and selection discovers a genuine **glider** de novo — a
single compact body that crosses the field in a near-straight line (net travel
≈ 3.8 widths, straightness 0.99, concentration 1.0) while holding constant mass
(homeostasis, mass CV ≈ 0.002). Orbium is *not* hardcoded; the creature is found.

Key lesson (twice caught by *looking*, not trusting metrics): a glider is a narrow
attractor — no Gaussian blob reaches it under any rule, so the **seed shape must be
evolved too**; and a space-filling "soup" games naive travel/support metrics, so a
scale-aware **mass-concentration** gate is required to mean "one creature."

Evidence: `outputs/round2_2d_creature.png` (snapshot strip + trajectory), `.gif`.

## Run

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m genesis.run2d --gif   # evolve a 2D glider, render strip + gif
.venv/bin/python -m genesis.run1d         # round-1 1D self-organisation
.venv/bin/python -m pytest -q             # engine + evolution invariants
```

## Layout

- `genesis/world.py` — dimension-agnostic Lenia engine (`World`, `LeniaParams`).
- `genesis/metrics.py` — emergence metrics (alive / localized / persistent /
  locomotion / concentration / gyration), all dimension-agnostic.
- `genesis/evolve.py` — co-evolution of rule + seed morphology (`Individual`, GA).
- `genesis/run1d.py`, `genesis/run2d.py` — round drivers + visualisation.
- `tests/` — engine invariants (1D/2D/3D) + evolution invariants.
- `docs/STATUS.md`, `docs/progress.md` — autonomous-loop resume state.
