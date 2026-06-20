# progress — genesis

## Round 1 — dimension-agnostic engine + 1D emergence (DONE, committed)

### What worked
- Lenia substrate as the engine: continuous CA defined identically in any dimension
  via radial-kernel FFT convolution + Gaussian growth map. `len(shape)` = ndim, so
  1D/2D/3D are one codebase. Confirmed by parametrized tests on (64,), (24,24),
  (12,12,12).
- Searching, not assuming, emergence: sweeping the growth map + seed asymmetry and
  scoring with emergence metrics reliably finds alive configs (139/360 alive).
- 1D spontaneous self-organisation into a persistent homeostatic lattice
  (mass_cv ~ 0, support ~ 0.12) — verified by running the real sim and reading the
  space-time diagram, not by trusting a number.

### What did NOT work / honest negatives
- No PERSISTENT single 1D glider in the Orbium-neighbourhood search. Directed motion
  is transient: net drift ~0.40 widths early decays to ~0.00 late as structures lock
  into a stationary lattice. (Consistent with the literature: clean gliders are a 2D
  phenomenon. This is the round-2 hook, not a failure to paper over.)
- First locomotion metric counted breathing/oscillation as "motion" (path length) and
  picked a symmetric, stationary breather as a "mover". Fixed: net displacement vs
  oscillation + straightness; movers must break symmetry (asymmetry > 0).
- Re-running the "best" config didn't reproduce the search exactly until per-candidate
  seeding was made deterministic (rng = default_rng((seed, trial))).

### Next-round seed
Round 2 = 2D. Goal: an unambiguous travelling creature (Orbium-class glider) in the
SAME engine — persistent locomotion, the behaviour 1D could only show transiently.
Then bootstrap EVOLUTION: treat LeniaParams (+ IC) as a genome, mutate + select on an
objective (persistence, then locomotion, then foraging) -> open-ended complexity.
Add a 2D renderer (frames -> mp4/gif) and a field snapshot visual.

## Frontier (durable ambition horizon — what ORIENT is pulled up by)

- CURRENT CEILING: emergent persistent structures in 1D; engine proven N-D capable.
- NEXT FRONTIER(S), ranked by ambition x feasibility:
  1. 2D living creatures + locomotion (Orbium glider) — high feasibility, direct.
  2. Evolution over the genome: mutation + selection -> novelty/complexity growth,
     open-endedness metrics (e.g. structure diversity over time).
  3. Sensorimotor AGENCY: put a resource/food field in the world; select creatures
     that move toward and consume it. This is where "intelligence" starts being real
     (sensing -> action -> self-maintenance), per the Flow-Lenia / sensorimotor-Lenia
     line of work.
  4. Neural-controlled agents: a small learned controller modulating local growth;
     within-lifetime adaptation, not just evolution.
  5. Intelligence MEASURED: information integration, predictive ability, goal
     achievement under perturbation — not asserted.
  6. 3D worlds: same engine, one more axis; 3D creatures + volumetric rendering.
- FIDELITY / STACK ESCALATION LADDER:
  numpy CPU (now) -> vectorised batch search -> torch + MPS/GPU for large 2D/3D and
  for differentiable/neural controllers -> real-time interactive viewer.
- RADICAL IDEAS WEIGHED: Flow-Lenia (mass-conserving, lets multiple species share a
  world — strong for open-ended ecology); Particle-Lenia (particle substrate, cheap
  agency); differentiable Lenia (gradient-evolve creatures). Park for after 2D works.
- AMBITION CRITIC (what an expert would find unimpressive today): it's still a CA with
  no agent, no objective, no learning. "Intelligence" is not yet present — only
  autopoiesis. The ratchet to honor that: agency (frontier #3) is the real bar, and
  every round should move toward sensing+acting under selection, not prettier patterns.
