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
2. **Locomotion** — creatures that travel (proto-agency). → 2D Orbium gliders (round 2)
3. **Evolution** — genomes (`LeniaParams`) under selection → open-ended complexity.
4. **Agency / intelligence** — sensing + neural control + selection; intelligence
   measured, not asserted.
5. **3D** — the same engine, one more axis.

## Round 1 result (1D)

From a single seed, the world spontaneously self-organises into a **persistent,
homeostatic lattice** of localised structures (mass coefficient-of-variation
≈ 0, support ≈ 12% of space). Directed motion appears but is **transient**: net
centroid drift decays from ~0.4 world-widths early to ~0 late — a clean,
*measured* finding that persistent gliders are a 2D phenomenon, motivating round 2.

Evidence: `outputs/round1_1d_selforg.png`, `outputs/round1_1d_locomotion.png`.

## Run

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m genesis.run1d        # search + render to outputs/
.venv/bin/python -m pytest -q            # engine invariants (1D/2D/3D)
```

## Layout

- `genesis/world.py` — dimension-agnostic Lenia engine (`World`, `LeniaParams`).
- `genesis/metrics.py` — emergence metrics (alive / localized / persistent / locomotion).
- `genesis/run1d.py` — round-1 search, simulation, and space-time visualisation.
- `tests/` — engine invariants across 1D/2D/3D.
- `docs/STATUS.md`, `docs/progress.md` — autonomous-loop resume state.
