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

## Round 6 — ECOLOGY: many creatures, contested food (DONE, committed)

### What worked
- genesis/ecology.py: K creatures, each a Lenia body in its OWN channel, all sharing ONE
  food field; each carries the round-3 foraging reflex + round-4 energy/metabolism. Food
  eaten by one is gone for the others (exploitation competition); per-step creature order
  is randomised for fairness. composite() renders all creatures (coloured by skill) + food.
- The headline finding (run6.py): STABILIZING SELECTION. Sweeping foraging gain across a
  population in one contested world, mean lifetime is an inverted-U — peaks at intermediate
  gain (~4), with gain=0 (under-foragers) starving first and gain=16 (over-foragers)
  overshooting and doing worse. ROBUST across food scarcity (spawn 38/50/65 -> optimum=4 in
  all 3; gain=0 always worst). This matches round 3's finding that gain ~5-7 forages best.
- Visuals: survival-vs-skill curve + population-of-one-contest dynamics + a competition GIF.

### What did NOT work / notes
- Single-regime stats are noisy (position luck: a creature near a food spawn survives
  regardless of skill) -> needed many seeds + randomised start-slot assignment to see the
  curve cleanly. Adversarially re-checked across 3 scarcity regimes before believing it.

### Next-round seed
Round 7 = EVOLVE the foragers INSIDE the ecology: let gain/sense (and morphology) mutate and
be selected by competitive survival, and watch the population converge to the optimal
strategy on its own (open-ended evolution in a social world). Or PREDATOR/PREY (a species
that eats creatures -> arms race). Or within-lifetime LEARNING (neural controller, torch+MPS).

## Round 7 — evolution RUNNING inside the ecology (DONE, committed)

### What worked
- genesis/evo_ecology.py: foraging gain is a HERITABLE gene. Well-fed creatures reproduce
  (offspring = body placed near parent + parent gain + Gaussian mutation), starving ones die,
  hard cap on population for compute. Selection now ACTS rather than being measured.
- The key result (run7.py): under scarce food the population is food-limited (not cap-limited),
  so foraging skill decides who reproduces. From RANDOM gains (mean ~9) the population self-
  tunes to ~3.5 across 3 independent runs — MATCHING round 6's swept optimum (~4). Two
  independent methods (sweep vs running evolution) agree on the optimum. Start gains uniform
  0-18 -> evolved gains concentrated in the good-forager band (peak ~3-4).

### What did NOT work / honest notes
- First attempt used abundant food: the population instantly hit the hard cap and EVERYONE
  reproduced -> no selection (cap-limited, mean gain frozen ~9-11). Selection only RUNS when
  the population is FOOD-limited (scarce food, spawn ~75) so poor foragers die before reproducing.
- The evolved distribution is a band, not a spike: round 6's landscape is a plateau (gain 2-6
  all forage well), so selection + mutation settle on a distribution around the optimum. Honest.

### Next-round seed
Round 8 = PREDATOR/PREY (leading): a second species that eats CREATURES (not food) -> two
coupled heritable strategies, a co-evolutionary arms race (prey evolve evasion, predators
pursuit). Or evolve MORPHOLOGY in the ecology (body+brain), or within-lifetime LEARNING.

## Round 8 — predator & prey: a world that pushes back (DONE, committed)

### What worked
- genesis/predprey.py: two species in one world. Prey forage food AND flee predators (heritable
  evasion gene); predators hunt+eat prey via a field interaction (impose a removal field on the
  prey-density field, gaining energy, taking prey mass), chasing the prey gradient (heritable
  pursuit gene). Both have energy/metabolism/reproduction.
- TOP-DOWN REGULATION (clean, robust): predators hold prey ~14 vs ~55 without predators (4x), with
  vivid prey boom-bust fluctuations. COEXISTENCE for the full run. The prey's numbers are now set
  by something hunting it — the world pushes back.
- Visuals: prey/predator population dynamics with a no-predator control + gene trajectories +
  a chase GIF (prey blue, predators red, food green).

### What did NOT work / honest negative
- No clean ESCALATING arms race. Across 5 tuned regimes the prey EVASION gene was ALWAYS selected
  DOWN (7 -> ~3): fleeing costs more foraging than it saves, so "forage hard, out-breed predation"
  beats "flee". Predator pursuit stays ~10 (predators at cap, weak selection). The honest finding
  is that the evolved answer to predation here is fecundity, not evasion.
- The balance is knife-edge: strong predation / high predator cap -> prey EXTINCTION; moderate ->
  predators pin at their cap (prey-limited cycles never fully form). Sustained Lotka-Volterra
  cycles (both populations oscillating) were not achieved — a known difficulty of agent-based PP.

### Next-round seed
Round 9 = WITHIN-LIFETIME LEARNING (leading): give a creature a small plastic/neural controller
that ADAPTS during its life (not only across generations). Learning is the missing ingredient of
"real intelligence". Or QUANTIFY intelligence (prediction / info-integration / goal under
perturbation). Or evolve morphology in-ecology / revisit stable mobile 3D creature.

## Round 9 — within-lifetime learning (DONE, committed)

### What worked
- genesis/learning.py: a creature with a PLASTIC value-learning brain (Rescorla-Wagner delta
  rule per source; epsilon-greedy choice) on a REVERSAL task: two sources, one nutritious, the
  rule flips every 320 steps. The learner re-learns after EVERY reversal -> accuracy 0.87 vs 0.49
  for the same creature with plasticity off (lr=0 = chance). Clean ablation, 8-life stats, and a
  curve that visibly dips at each flip then recovers. Pure numpy, fast (~8s), no torch (no gate).
- This closes the last big conceptual gap the ambition critic kept naming: adaptation WITHIN a
  life from a creature's OWN experience, not only across generations.

### What did NOT work / honest notes
- The brain is intentionally minimal (one-layer value learner), not a deep net. It demonstrates
  the PRINCIPLE (within-life plasticity + reversal) cleanly; richer controllers (multi-layer,
  recurrent, embodied in the Lenia drift policy) are deferred. The agent here is a point in an
  arena (rendered as a marker), a deliberate departure from the Lenia body to make learning crisp
  and measurable — re-uniting brain + Lenia body is a future round.

### Next-round seed
Round 10 is a natural MILESTONE. Options: (a) a REVIEW round — adversarially re-verify all 9
claims by re-running every driver and checking the picture matches the metric, prune/polish;
(b) MEASURE intelligence quantitatively (predictive info, goal under perturbation); (c) marry the
plastic brain to the embodied Lenia creature (learned drift policy in the ecology); (d) Baldwin
effect (evolve the learning rule). Leaning toward a REVIEW round at this milestone, then (c)/(b).

## Frontier (durable ambition horizon — what ORIENT is pulled up by)

- CURRENT CEILING (after 9 rounds): a continuous-CA world with ONE engine across 1D/2D/3D; an
  embodied creature that emerges, moves, senses+forages (agency), forages-to-survive (metabolism);
  a social ECOLOGY with stabilizing selection and EVOLUTION RUNNING (discovers the optimum); a
  two-species predator-prey world (top-down regulation); and a creature that LEARNS within its
  life and re-learns under reversal. The arc emergence->locomotion->agency->survival->3D->ecology
  ->evolution->predation->learning is complete in skeleton.
- NEXT FRONTIER(S), ranked by ambition x feasibility:
  1. REVIEW + consolidate (milestone): adversarially re-verify all claims; tighten the story.
  2. MEASURE intelligence: predictive info / goal achievement under perturbation -> mind, measured.
  3. Embodied LEARNING: plastic brain drives the Lenia creature's drift in the ecology.
  4. Baldwin effect: evolve the learning rule / priors (learning x evolution).
  5. Evolve MORPHOLOGY in-ecology; stable mobile 3D creature (multi-ring + CMA-ES, hard).
- FIDELITY / STACK ESCALATION LADDER:
  numpy CPU (now) -> vectorised batch search -> torch + MPS/GPU for large 2D/3D and
  for differentiable/neural controllers -> real-time interactive viewer.
- RADICAL IDEAS WEIGHED: Flow-Lenia (mass-conserving, lets multiple species share a
  world — strong for open-ended ecology + foraging); Particle-Lenia (particle substrate,
  cheap agency); differentiable Lenia (gradient-evolve creatures / learned controllers).
  Flow-Lenia is now the leading candidate substrate for round 3+ (food + ecology).
- AMBITION CRITIC (what an expert would still find unimpressive): a creature now LEARNS within its
  life and re-learns under reversal — but the brain is a one-layer value learner on a 2-choice task,
  and it lives DETACHED from the Lenia body/ecology (a point agent). The mind and the embodied
  world have not yet met, the learning is shallow (no prediction, no planning, no memory of
  sequences), and intelligence is still shown by a task score, not MEASURED information-theoretically.
  The ratchet: marry the plastic brain to the embodied creature in the ecology, deepen the
  controller, and measure the mind (predictive info / goal under perturbation). Also: consolidate —
  after 9 rounds, a REVIEW that adversarially re-verifies every claim is overdue.
