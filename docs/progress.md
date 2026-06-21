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

## Round 2 — evolve a 2D travelling creature (DONE, committed)

### What worked
- Co-evolving RULE + an evolvable SEED MORPHOLOGY (smooth patch) under a locomotion
  fitness FOUND a genuine glider de novo: net travel 3.78 widths, straightness 0.99,
  concentration 1.0, mass_cv 0.0021. Movers first appeared at gen 8 and spread through
  the population. Verified by eye (snapshot strip + gif) AND metrics.
- mass-concentration metric (fraction of mass within ~2.2R of the centroid) is the
  gate that separates ONE compact creature from space-filling turbulence. Essential.
- Unwrapped centroid trajectory gives an honest, wrap-immune travel + straightness;
  analyze_history's wrap-aware net UNDERCOUNTS once travel > half a width (only safe
  for short search runs).

### What did NOT work / honest negatives (Goodhart twice — eye caught both)
- v1 fitness rewarded a SPACE-FILLING SOUP: its centroid jitter read as "travel" and
  sparse support slipped the gate. Fix: concentration gate.
- A Gaussian/asymmetric blob CANNOT reach a glider basin under ANY rule (probed the
  canonical Orbium region: best blob travel 0.0014w). The glider is a narrow attractor
  -> the SEED SHAPE itself must be evolved. This reframed the whole round.
- With concentration gate but blob ICs, the EA collapsed to a stationary compact dot
  (no locomotion). Only evolving the morphology unlocked movement.

### Next-round seed
Round 3 = AGENCY. Add a resource/food scalar field to the world; couple it to growth
(creature consumes/needs it). Fitness: creatures that SENSE a gradient and MOVE toward
food to sustain themselves. That is sensing->action->self-maintenance = the first real
"intelligence", not asserted. Parallel track: open-endedness — evolve a ZOO of distinct
creatures, add a behavioural-diversity metric, watch complexity grow.

## Round 3 — AGENCY: a creature that senses food and forages (DONE, committed)

### What worked
- ForagingWorld = the round-2 Lenia glider + a food field + a SENSORIMOTOR REFLEX:
  sense the food gradient over the body, rigidly translate the body up-gradient with
  np.roll (an exact permutation -> mass-exact, no blow-up), and eat food under the body.
- Evolving the reflex (gain + sensing radius) gives genuine chemotaxis: on UNSEEN
  random food layouts the evolved forager eats 85% vs 18% ablated vs 18% random-drift.
- The random-drift control is the clincher: it moves exactly as much but toward a random
  direction, and scores like ablated -> the advantage is FOOD-DIRECTED sensing, not motion.
  Trajectories visibly bend toward food from every direction and then buzz on it, eating.

### What did NOT work / honest negatives (3 dead ends before the reflex)
- Isotropic food->growth bonus: no steering (gamma=0 already ate ~0.31 ballistically;
  higher gamma just destabilised the body).
- Non-conservative advection -gamma*(grad S . grad A): produced strong "taxis" BUT
  ballooned the creature ~10x (covers all food = soup-gaming again). Not real steering.
- Mass-conserving advection -gamma*div(A grad S): still blew up ~10x — explicit
  central-difference advection of a clipped field is numerically unstable; the clip
  rectifies oscillations and injects mass. Abandoned PDE advection.
- WHAT FIXED IT: peak-normalise the sensing kernel (sum-norm made grad S ~0), then steer
  by a RIGID np.roll drift (mass-exact). Robust, stable, strong taxis (gain~5-7).

### Next-round seed
Round 4 = deepen agency. Top pick: METABOLIC SURVIVAL — turn on decay+feed (already in
foraging.py) so the creature DIES without food; fitness = lifetime. Foraging stops being
free reward and becomes survival = real homeostatic intelligence. Then ecology (many
creatures + competition) and/or a learned within-lifetime controller (torch+MPS).

## Round 4 — SURVIVAL: forage to stay alive (DONE, committed)

### What worked
- A scalar ENERGY reserve as metabolism: while energy>0 the body is a normal glider
  (constant mass); metabolism drains energy each step, eating refills it, at 0 the body
  dissipates (death). This keeps the survival currency separate from body mass.
- Food that SPAWNS over time (a new small cluster every 85 steps at a random place) makes
  survival demand CONTINUAL, AIMED foraging.
- Regime decay=0.012 feed=0.06 energy0=1.3 frad=7: on UNSEEN food schedules the evolved
  forager lives ~832 steps (banks an energy surplus) vs 158 ablated vs 299 random (cap 900).
  Death is real; directed sensing is the difference between living and dying.

### What did NOT work / honest negatives
- First metabolism model added eaten food directly to BODY MASS (+feed*bite): the creature
  ballooned ~30x into a space-filler that trivially "survives" by growth. Replaced with
  the scalar-energy model (eating fills a reserve, not the body) -> mass stays glider-sized.
- Tuning the stakes is a balance: a big energy buffer lets EVERYONE (even non-foragers)
  coast and survive; too small and the forager starves before reaching food. The lever
  that worked was a SHALLOW buffer + a SKILLED forager (higher gain/sense reaches small
  food fast) -> only directed foraging survives, non-foragers and random-drift die.

### Next-round seed
Round 5 = ECOLOGY or OPEN-ENDEDNESS or LEARNING or 3D (see Frontier). Leading pick:
ECOLOGY — multiple creatures sharing one world with finite/contested food, so selection
acts in a social/competitive setting (then predator-prey). Needs per-creature identity +
energy bookkeeping; Flow-Lenia-style mass handling helps. Alternatively bank the
"1D->2D->3D" headline goal by doing 3D now that agency+survival work in 2D.

## Round 5 — the third dimension (DONE, committed)

### What worked
- The dimension-agnostic bet pays off end to end: the IDENTICAL engine runs a 3D world
  (len(shape)=3, no code changes) and self-organises a seed into a ROBUST, persistent 3D
  structure (tail mass_cv=0.0008; final mass ~22150 reproducible across 3 seeds) — the 3D
  analogue of round 1's 1D lattice. Regime: mu_g=0.12, mu_k=0.5, sigma_g=0.024, R=8 on 50^3.
- genesis/run5_3d.py renders it: 3D scatter over time + homeostasis curve + a rotating GIF.
- genesis/evolve3d.py: a 3D morphology GA (PATCH^3 seed + rule), reusing the ND fitness.

### What did NOT work / honest negative (the open 3D frontier)
- A compact MOBILE 3D creature (round-2-glider analogue) was NOT found. Five principled
  search strategies all failed: (1) single-ring growth sweep -> diffuse foam (conc~0.34);
  (2) kernel-shape sweep -> mu_k=0.3 gave a transiently compact creature (conc 0.56) that
  then proliferates; (3) multi-ring kernels -> 0 stable compact; (4) narrow growth -> 0;
  (5) evolved 3D morphology -> mostly dies. The dynamics are KNIFE-EDGE: the same params
  die on 56^3 but proliferate on 52^3 -> no robust localised attractor in reach. 3D Lenia
  creatures need the heavy specialised search (multi-ring kernels + CMA-ES/gradient) the
  literature uses. Honest result: 3D *self-organisation* is solid; 3D *creatures* are open.

### Next-round seed
Round 6 = ECOLOGY (2D). Put MULTIPLE creatures in one food world: per-creature identity +
energy, shared/finite food -> competition; then predator/prey. Implementation options: a
multi-channel field (one channel per creature) or tracked bodies in one field with
collision/competition for food. Social/competitive selection -> richer intelligence.

## Frontier (durable ambition horizon — what ORIENT is pulled up by)

- CURRENT CEILING: a 2D creature with AGENCY + SURVIVAL (senses food, forages to stay alive,
  proven vs controls) AND the engine validated in 3D (robust 3D self-organisation). The
  1D->2D->3D engine arc is COMPLETE. Missing: other creatures / ecology, within-lifetime
  learning, stable mobile 3D creatures (open), deeper intelligence measures.
- NEXT FRONTIER(S), ranked by ambition x feasibility:
  1. ECOLOGY: many creatures + contested food -> competition; then predator/prey. Social
     selection is where richer intelligence pressure comes from. (Round 6 pick.)
  2. Open-endedness: a ZOO of distinct survivors + behavioural-diversity / novelty metrics.
  3. Within-lifetime LEARNING: a neural controller modulating drift/growth; adaptation,
     not just evolution. (torch + MPS here.)
  4. Stable MOBILE 3D creatures: multi-ring kernels + CMA-ES/gradient search (hard, open).
  5. Intelligence MEASURED: prediction, info-integration, goal achievement under
     perturbation — not asserted.
- FIDELITY / STACK ESCALATION LADDER:
  numpy CPU (now) -> vectorised batch search -> torch + MPS/GPU for large 2D/3D and
  for differentiable/neural controllers -> real-time interactive viewer.
- RADICAL IDEAS WEIGHED: Flow-Lenia (mass-conserving, lets multiple species share a
  world — strong for open-ended ecology + foraging); Particle-Lenia (particle substrate,
  cheap agency); differentiable Lenia (gradient-evolve creatures / learned controllers).
  Flow-Lenia is now the leading candidate substrate for round 3+ (food + ecology).
- AMBITION CRITIC (what an expert would still find unimpressive): the engine now spans
  1D->2D->3D and the 2D creature forages to survive — but it still lives ALONE, with no
  other creatures to compete/cooperate with and no within-lifetime learning, and the 3D leg
  is only self-organisation (no 3D creature yet). The ratchet: round 6 MUST add OTHER
  CREATURES (ecology/competition — where richer intelligence pressure lives); then learning.
  A lone survivor is real but lonely intelligence; an ecology is where minds get interesting.
