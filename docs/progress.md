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

## Round 10 — REVIEW / milestone (DONE, committed)

Adversarial re-verification of all 9 rounds (re-ran drivers, checked metric vs claim, tried
fresh seeds to catch cherry-picking):
- R2 glider artifact still glides: net 4.05 widths, straightness 0.99, concentration 1.00, mass_cv 0.0021.
- R3 agency: 85% / 18% / 18% (evolved / ablated / random) — exact.
- R4 survival: 832 / 158 / 299 — exact.
- R6 ecology: inverted-U, optimum gain ~4 — holds.
- R9 learning: 0.87 vs 0.49; and 0.87 vs 0.57 on FRESH unused seeds 200-207 -> NOT cherry-picked.
- R5/R7/R8 were verified across seeds/regimes within their own rounds (kept).
- 39 tests green; all 19 README images resolve; outputs/ clean (no temp files).
- Hygiene: only orphan module was evolve3d.py (round-5 3D-GA, used by no driver) -> added smoke
  tests, kept for the open 3D-creature frontier. ~2950 LOC, repo ~22M.
- No overclaiming found. The honest negatives (no stable mobile 3D creature; no clean predator-prey
  arms race; transient 1D locomotion) are all still documented and intact.
VERDICT: the project stands up. Nine capabilities, each verified by a real run + a picture matching
its metric. Resume building at round 11.

## Round 11 — embodied learning (DONE, committed)

### What worked
- genesis/embodied_learning.py: the round-9 plasticity now lives INSIDE a Lenia creature. The body
  forages two food types; its drift is a plastic policy w_A*grad(sense F_A)+w_B*grad(sense F_B);
  only one type is nutritious; the rule reverses mid-life; food EVAPORATES if uneaten so the creature
  must actively chase fresh food. Reward-modulated delta rule updates w on tasting each type.
- THE BRAIN LEARNS: the drift weights track the nutritious type and FLIP at every reversal
  (argmax(w)==active ~93%) — a clean anti-phase square-wave plot aligned with the active-food shading.
- BEHAVIOURAL PAYOFF: the embodied learner eats 0.89 nutritious vs 0.52 for the same body with
  plasticity off (lr=0, fixed w). Body (R2) + agency (R3) + within-life learning (R9) in ONE creature.

### What did NOT work / honest notes
- A large drift GAIN HURT (0.50, chance): high gain makes the body lock onto one region and stop
  sampling, so it can't track reversals. Gentle steering (gain~0.6) works.
- Without food evaporation the world saturated with food (muddy + the eating signal washed out to
  0.66). Adding evaporation (food_decay 0.015) made foraging matter -> payoff jumped to 0.89 and the
  visual cleaned up. The brain metric (~93%) is the clean signal; the body is an imperfect actuator.

### Next-round seed
Round 12 = MEASURE intelligence (leading): a NUMBER for the mind — predictive information the brain
carries about the world / goal achievement under perturbation — not a task score. Or put learners
INTO the ecology/predator-prey (do learners out-compete non-learners?). Or Baldwin / deeper controller.

## Round 12 — measuring the mind (DONE, committed)

### What worked
- genesis/measure.py: quantify intelligence information-theoretically. I(brain;world) in BITS =
  mutual information between the creature's internal state (which food its brain prefers, sign of
  w_A - w_B) and the world's hidden variable (which food is nutritious). Pure, tested info-theory.
- RESULT: the learner's brain carries 0.69 bits about the world; the same body with plasticity off
  carries 0.00. A literal number for "the mind knows about the world", not a task score.
- OPERATING ENVELOPE: sweep how fast the world reverses -> I(brain;world) rises monotonically from
  0.17 bits (flip 100, world changes faster than it can learn) to 0.88 bits (flip 1600, near the
  1-bit ceiling). This is the timescale of environmental change the mind can track.
- Visuals: MI bar (learner vs ablated) + the envelope curve + a GIF with a live "knowledge meter"
  (windowed I(brain;world)) rising and falling as the creature learns and the world flips.

### What did NOT work / honest notes
- MI within a single stable epoch is 0 (no variation in the hidden variable -> nothing to be
  informative about); the meaningful measure is over windows that SPAN reversals. So the live meter
  uses a window > one flip period. I(brain;world) measures KNOWING (epistemic state), one component
  of intelligence — not planning or prediction-of-the-future, which remain open.

### Next-round seed
Round 13 = LEARNING UNDER SELECTION (leading): drop learners AND fixed-reflex creatures into the
ecology under competition — does learning WIN (does plasticity pay its metabolic cost)? Measure
I(brain;world) of the survivors. Then Baldwin effect, deeper controllers.

## Round 13 — learning under selection (DONE, committed)

### What worked
- genesis/evo_learning.py: learners (plastic brain, w starts [0,0], re-learns each life) and
  fixed-reflex creatures (non-plastic w, inherited + mutated across generations) COMPETE in one
  world for two food types; 'is_learner' is heritable (rare strategy mutation); nutritious type
  reverses at a tunable rate. Shared sensed gradients -> only N+2 FFTs/step (efficient).
- THE EVOLUTION-OF-LEARNING (BALDWIN) RESULT, clean: CHANGING world -> learner fraction 0.5 -> 1.00
  (learning wins); STABLE world -> 0.5 -> 0.39 (fixed reflexes win). The change-rate sweep shows a
  sharp transition (~interval 1000-2000): learning pays ONLY in worlds that change within a lifetime.
  This closes the loop from round 12: a learner doesn't just KNOW more, it WINS — conditionally.
- Visuals: learner-fraction-over-time (stable vs changing) + when-does-learning-pay transition curve
  + a GIF of learners (blue) taking over fixed-reflex creatures (red) in a changing world.

### What did NOT work / honest notes
- Run is slow (~6 min, ~9 competitive sims of ~36 creatures x 2600 steps). Kept population/time
  modest; the signal is strong enough to be unambiguous at this scale.
- The fixed-reflex baseline is the fair one: it inherits + evolves its w across generations, so it is
  NOT static — it just can't track within-life reversals. That is exactly why it loses only when the
  world changes faster than generational evolution can follow.

### Next-round seed
Round 14 = BALDWIN EFFECT (leading): let the learning RULE / brain priors evolve, not just the
is_learner flag -> does evolution tune HOW to learn? Or deeper controller (memory/recurrence), or
predictive/planning intelligence, or a milestone review (13 rounds in).

## Round 14 — the Baldwin effect (DONE, committed; mixed result)

### What worked
- genesis/baldwin.py: the LEARNING RATE is a heritable gene. Each creature is born naive (weights
  reset) and learns at its own inherited rate, which mutates + is selected. From RANDOM rates the
  population self-tunes to ~0.57 (distribution shifts from uniform toward high rates) -> evolution
  discovers a good amount of plasticity (favouring fast learning in a changing world). The Baldwin
  effect operates on the rate itself. Visual: mean-lr convergence + random->concentrated histogram + GIF.

### What did NOT work / honest negative (the predicted result did NOT appear)
- I predicted the evolved rate would TRACK the world's change-rate (high for fast worlds, low for
  stable) — the textbook volatility->learning-rate law. It does NOT here: across flip 80..3000, and
  with added reward noise (0.25, 0.35), the evolved rate stayed ~0.5 (one noise level even ran the
  WRONG way). The clean Bayesian relationship is washed out by the embodied foraging dynamics
  (born-naive each life, food competition, body inertia, the cap at LR_MAX biting). Reframed the round
  honestly around the real positive (evolution tunes the rate to a good value) + this documented
  negative, rather than forcing the change-rate story.

### Next-round seed
Round 15 = DEEPER CONTROLLER (leading): the brain is a one-cue value reflex; give it MEMORY / a hidden
recurrent state so it can handle sequences and anticipate, not just react. Or predictive intelligence
(a creature that predicts the next world-state). Or a milestone review (14 rounds in).

## Round 15 — a brain with memory (DONE, committed)

### What worked
- genesis/memory_brain.py: a tiny RECURRENT controller (6 hidden units, ~66 weights) evolved by an
  OpenAI-style ES (pure numpy, no torch) on a CUE-RECALL task — a cue appears, vanishes, and after a
  random delay the creature must act on it. The recurrent net reaches 1.00 accuracy and holds exactly
  1.00 BIT of memory across the delay (measured by MI between the mid-delay hidden readout and the
  cue). A FEEDFORWARD net (recurrence ablated) is stuck at 0.53 / 0.00 bits — it cannot, the cue is
  gone when it must act. Memory unlocks a task class impossible for a reflex.
- The memory is both MEASURED (1 bit, reusing measure.py) and VISUALISED (the two cue trajectories
  separate at the cue and stay separated, flat, through the whole delay until the go step). Fast (~6s).
- This is the first controller in the project with a HIDDEN STATE — the depth-of-mind frontier the
  ambition critic kept naming. It also directly explains round 14's failure: a memoryless learner
  cannot be Bayes-optimal about volatility.

### What did NOT work / honest notes
- The task is abstract (a clean cue-recall benchmark, point-agent style like round 9), not embodied in
  a Lenia body yet — deliberate, to isolate the memory result. Re-embodying the recurrent brain in the
  ecology is a future round. The net is small (6 units); deeper/longer-memory tasks remain open.

### Next-round seed
Round 16 = PREDICTIVE intelligence (leading): with memory in hand, give the brain a task/objective that
rewards ANTICIPATING the next world-state, and measure predictive information I(state_t ; obs_{t+1}) —
a mind that predicts, not just remembers. Or embody the recurrent brain in the ecology.

## Round 16 — a brain that predicts (DONE, committed)

### What worked
- genesis/predict_brain.py: a recurrent net (8 hidden units, ES, pure numpy) must PREDICT the next
  state of a periodic AMBIGUOUS world (pattern [0,0,1,1]: a 0 is followed by 0 at one phase and 1 at
  another, so the current symbol does NOT determine the next — you must track phase and model the
  pattern). Recurrent: 1.00 accuracy + 1.00 BIT of predictive information I(brain;next-state),
  anticipating EVERY flip BEFORE it lands. Feedforward (reactive): 0.50 + 0.00 bits (cannot disambiguate).
- This is the deepest mind-property so far: not reacting (rounds 3-13), not remembering (round 15), but
  FORESEEING — the brain builds an internal model of the world's dynamics and predicts the future. The
  predictive information is measured in bits (reusing measure.py), continuing the "measure the mind" thread.
- Visuals: ES curves (recurrent solves / feedforward stuck at chance) + a prediction trace where the
  brain's predicted next-state matches the actual next-state at every step incl. the flips + a GIF.

### What did NOT work / honest notes
- Like round 15, the task is an abstract benchmark (clean, isolates prediction), not embodied in a Lenia
  creature yet. And predicting is not yet PLANNING — the brain foresees the world but does not use the
  model to CHOOSE actions that steer the future. That (model-based control) is the next depth.

### Next-round seed
Round 17 = MILESTONE REVIEW (leading; 16 rounds in): adversarially re-verify rounds 11-16, consolidate,
run the ambition critic. Then: embody the predictive brain (does foresight PAY under selection?), or
PLANNING (use the model to act). 

## Round 17 — REVIEW / milestone (DONE, committed)

Adversarial re-verification of rounds 11-16 (re-ran drivers, checked metric vs claim, fresh seeds):
- R11 embodied learning: brain 93% correct, nutritious 0.89 vs 0.52 ablated — exact.
- R12 measuring the mind: I(brain;world) 0.69 bits vs 0.00; envelope 0.17->0.88 — exact.
- R13 learning under selection: changing world learner-fraction ->1.00, stable ->0.44 — holds (spot-check).
- R14 Baldwin: mean learning rate 0.48 (random) -> 0.59 (evolved) — holds (spot-check); honest negative intact.
- R15 memory: recurrent 1.00 / 1.00 bit vs feedforward 0.53 / 0.00; on FRESH seeds 300-302: 1.00 vs 0.48.
- R16 prediction: recurrent 1.00 / 1.00 bit vs feedforward 0.50 / 0.00; on FRESH seeds: 1.00 vs 0.50.
- 62 tests green; no orphan modules; all README images resolve; honest negatives (R5 3D, R8 arms race,
  R14 change-rate) all intact and documented. Repo growing (.git ~50M, outputs ~46M from committed GIFs).
VERDICT: the newer half stands up. The two open structural gaps are INTEGRATION (deep brain 15-16 lives
in abstract benchmarks, not embodied) and PROOF-OF-PAYOFF (is memory/prediction adaptive under selection?).
Resume building at round 18 by reuniting the tracks.

## Round 18 — embodied memory: the two tracks reunited + payoff proven (DONE, committed)

### What worked
- genesis/embodied_memory.py: a recurrent net (ES, pure numpy) drives a LENIA forager's drift. The world
  FLASHES — the food gradient is visible briefly, then dark while the food persists — so reaching food
  requires holding the direction across the dark. This reunites the EMBODIED track (Lenia body, drift,
  metabolism; rounds 3-13) with the RECURRENT-MIND track (rounds 15-16) in ONE creature.
- PAYOFF (food collected/episode, averaged over 2 ES seeds, unseen eval seeds): recurrent-flashing 1.9 vs
  feedforward-flashing 1.0; feedforward-STEADY control 3.4. Reading: a steady signal lets the memoryless
  forager thrive (3.4); FLASHING crushes it (-> 1.0, a 70% collapse); MEMORY roughly doubles it (-> 1.9),
  recovering much of the loss. Memory PAYS exactly when the world hides information across time — and the
  steady control proves the feedforward forager is not broken, it just cannot cope with intermittency.

### What did NOT work / honest notes
- High variance: a single ES seed gave anywhere from feedforward=0.0 to 2.0 in flashing -> averaged over 2
  ES seeds + 10 eval seeds for a stable estimate, and HARDENED the regime (longer dark, flash_off=26) so the
  gap is robust. The magnitude is modest in absolute terms (a Lenia glider is a clumsy navigator), but the
  RELATIVE effect (memory ~2x feedforward under flashing; flashing -70% for the reflex) is clear and honest.
- Did not also embody PREDICTION (16) yet — only memory (15). That, and PLANNING, are next.

### Next-round seed
Round 19 = PLANNING / model-based control (leading): the brain can predict (16) — now make it USE the model
to choose actions that steer the future toward a goal (anticipatory interception), not just react. Or embody
prediction (a foresight-requiring task: moving/scheduled food).

## Round 19 — planning: acting on foresight (the mind's capstone) (DONE, committed)

### What worked
- genesis/planning.py: a target ORBITS on a circle; an agent (slightly faster) must intercept. A REACTIVE
  pursuer heads at where the target IS and tail-chases around the circle; a model-based PLANNER heads where
  it WILL be and cuts across the interior. Result: planner catches in 24 steps vs reactive pursuit 54
  (~2.2x faster) — the classic pursuit-curve-vs-interception picture. Acting on prediction beats reacting.
- LEARNED: an EVOLVED RECURRENT controller (sees only relative position each step, so it must INFER the
  target's motion) discovers anticipation -> catch rate 0.45 vs 0.05 for a feedforward (reaction-only)
  controller (9x). Recurrence is what lets a creature learn to lead a moving target.
- This completes the mind's core loop: perceive -> model -> predict -> ACT-TO-ACHIEVE. React (3-13),
  remember (15), predict (16), and now INTEND (19) are all present and measured.

### What did NOT work / honest notes
- Getting here took 4 task redesigns: a big speed margin made pursuit ~= planner (planning didn't matter);
  a bouncing target broke long-horizon prediction; an open-space straight target let the ES policy let the
  target escape (rate 0.10). A CIRCULAR (bounded, periodic, predictable) target with a small speed margin
  is the regime where pursuit clearly tail-chases AND ES can learn. Documented as the lesson.
- The evolved recurrent (0.45) does NOT reach the hand-coded planner (1.00) — interception is hard to learn
  from scratch via ES; the honest, real result is the 9x gap over feedforward, not parity with the planner.

### Next-round seed
Round 20 is a natural MILESTONE (the mind's core loop is complete). Options: a review + capstone README/story
pass; or UNIFY (one creature with body+memory+prediction+planning in the ecology); or open-endedness (a ZOO
of evolved minds + diversity metric). Leaning toward review+capstone, then unification.

## Round 20 — MILESTONE REVIEW + capstone (DONE, committed)

Adversarial re-verification of R18-19 + a capstone README pass at the natural milestone (mind's core loop complete).
- R19 (planning): reproduces EXACTLY — pursuit 54 vs planner 24; evolved recurrent 0.45 vs feedforward 0.05.
- R18 (embodied memory): the review CAUGHT AN OVERCLAIM. A quick under-trained probe FLIPPED the result
  (recurrent 0.2 < feedforward 1.0). A careful re-check (4 ES seeds, committed settings, 12 unseen eval seeds)
  gives recurrent mean 1.90 vs feedforward 1.21, recurrent wins 3/4 seeds -> the effect is REAL and directionally
  robust but MODERATE and seed-sensitive, not the clean "2x / memoryless collects nothing" the single figure
  implied. FIXED: montage caption + README now state ~1.9 vs ~1.2 (3/4 seeds) with an explicit seed caveat.
  This is exactly what a review is for: looking harder than the first run did.
- Capstone README: rewrote the intro to tell the whole arc (emergence -> ... -> planning) + an HONEST-SCOPE box
  (focused demonstrations not one organism; small-scale numpy-only; 3D creature still an open negative). Montage
  suptitle updated to the full arc. 69 tests green; all README images resolve; honest negatives intact.
VERDICT: the arc holds; one optimistic headline (R18) corrected to honest numbers. Build resumes at UNIFICATION.

## Round 21 — unification: one creature, all four faculties (DONE, committed)

### What worked
- genesis/unified.py: a Lenia creature must stay on food that MOVES (constant-velocity, bouncing) and FLASHES
  (visible briefly, then dark) under METABOLISM -> tracking it IS survival. ONE integrated controller:
  perceive -> REMEMBER (hold the position estimate across the dark) -> PREDICT (dead-reckon by remembered
  velocity) -> PLAN/ACT (drive the Lenia body's np.roll drift toward the lead/intercept).
- ABLATION LADDER (steps survived, max 320, 20 seeds): full 263 > memory_only 188 (keeps memory but does not
  predict the motion -> aims at a stale spot, lags) > no_memory 140 (retains nothing across the dark -> stalls).
  Each step removes one faculty and costs survival -> every faculty is LOAD-BEARING. The energy-over-life panel
  shows the mechanism: full keeps eating (sawtooth recoveries) and lives; no_memory declines monotonically.
- Body (3-13) + memory (15/18) + prediction (16) + planning (19) are now demonstrably ONE organism.

### What did NOT work / honest notes
- The ES route FAILED first (twice): training one recurrent net to learn all four jointly gave recurrent == feedforward
  (memory not load-bearing on a too-regular orbit) and, when hardened, both collapsed (too hard for ES). This is the
  same R18/R19 tension a THIRD time: where memory/prediction is necessary, a tiny ES net can't learn it. PIVOTED to a
  hand-INTEGRATED controller (the faculties were already shown EVOLVABLE in 15/16/19; unification's claim is that they
  COMPOSE and each is necessary, which ablation proves cleanly and deterministically).
- The prediction LEAD occasionally overshoots at a wall bounce (dead-reckons the old velocity through the bounce), so
  `full` is not best on every seed (best on average + 10/16 seeds survive the whole episode). Documented, not hidden.

### Next-round seed
Round 22 = OPEN-ENDEDNESS (leading): does the world keep GENERATING novelty or converge? a ZOO of evolved minds/bodies
+ a behavioural-diversity / novelty metric — the one big theme not yet shown. Or the long-parked stable 3D creature.

## Round 22 — open-endedness: does the world keep generating, or converge? (DONE, committed)

### What worked
- genesis/openended.py: MAP-Elites (Mouret & Clune) ILLUMINATES a 2-D behaviour space — (drift-speed) x (body-mass)
  — keeping the most VIABLE creature found in each of 8x8 niches, vs a fitness-GA baseline.
- RESULT: MAP-Elites fills 54/64 niches (84%) spanning the FULL move x size range; coverage ACCUMULATES (climbs
  24->54 over evaluations, still rising). The fitness GA's population diversity COLLAPSES / hovers ~8 as elitist
  selection drives it to one body type. So the Lenia substrate is GENERATIVE across this behaviour space — diversity
  accumulates under illumination where it converges under optimization. This is the first round about CREATIVITY /
  open-endedness rather than convergence — the one big theme the prior 21 rounds hadn't shown.
- Visuals: the behaviour map (filled niches coloured by viability), the coverage curve (grow vs collapse), and a
  zoo GALLERY of 8 distinct creatures + a GIF of 4 of them animating their different behaviours.

### What did NOT work / honest notes
- First axes were WRONG: drift-speed clipped (MOVE_MAX 0.32 vs real 0.83) and gyration SATURATED (~0.41 for all) ->
  only 9/64 cells, plateaued. A feature diagnostic (sweep random creatures) showed drift-speed and mean_mass spread
  well -> fixed axes (move 0..0.85 x mass 60..1900) -> 54/64. Lesson: pick behaviour axes that actually vary.
- First fitness baseline was DISHONEST: it counted behaviour cells TOUCHED during hill-climbing (30), not RETAINED ->
  made convergence look diverse. Fixed to a population GA tracking the POPULATION's diversity, which honestly collapses.
- The "zoo" is HONESTLY MIXED: low-mass niches hold a clean single glider; high-mass niches hold multi-blob "foam"
  textures, not discrete organisms (the localized>0.25 filter admits sparse foam). So the substrate fills the BEHAVIOUR
  space with viable PATTERNS, not all single creatures; and the map is bounded -> no claim of UNBOUNDED open-endedness.

### Next-round seed
Round 23 is a natural MILESTONE (22 rounds, last review R20): re-verify R21-22 adversarially, prune, refresh the arc.
Or the long-parked stable 3D creature; or novelty-search in the MIND space (open-ended brains, not just bodies).

## Round 23 — MILESTONE REVIEW (DONE, committed)

Adversarial re-verification of R21-22 on FRESH unseen seeds (the last full review was R20).
- R21 (unification): the ablation ladder HOLDS on fresh seeds 30-49 -> full 288 > memory_only 192 > no_memory 157
  (committed 263/188/140 on seeds 0-19 reproduces directionally). Each faculty still load-bearing; not cherry-picked.
- R22 (open-endedness): on a FRESH seed (2), MAP-Elites fills 54/64 niches and the fitness-GA's population diversity
  collapses to 8 -> reproduces the committed numbers EXACTLY. The generativity result is robust, not seed-specific.
- 75 tests green; all README images resolve; no orphan modules; every honest negative/caveat intact (R18 seed-
  sensitivity, R21 hand-wired controller + bounce overshoot, R22 foam textures + bounded map, 3D creature negative).
- NOTE: repo growing (.git 62M, outputs 51M) from committed GIFs/PNGs — inherent to the visual README; acceptable.
VERDICT: the newer rounds stand up on unseen seeds. Unlike R18 (corrected at R20), R21-22 needed no correction.

## Round 24 — open-ended MINDS: a zoo of foraging strategies (DONE, committed)

### What worked
- genesis/openmind.py: same body (the evolved glider), but the MIND varies — a 4-parameter foraging policy
  (sensing range, gain, exploration, momentum). The creature forages scattered food; MAP-Elites illuminates the
  foraging-BEHAVIOUR space (roaming x trajectory-spread), keeping the best-eating forager per niche; vs a fitness GA.
- RESULT: MAP-Elites fills 28 niches (~the reachable max) with VISIBLY DISTINCT trajectories — compact efficient
  foragers (path 136, ate 16), sprawling rovers (path 360-600, ate 16), columnar pacers (path 212, ate 7) — while the
  fitness GA's population diversity collapses to ~5. Open-endedness holds for MINDS, not just bodies (R22). The
  trajectory gallery is the centrepiece: distinct minds forage in qualitatively different ways.

### What did NOT work / honest notes
- Same axis-tuning lesson as R22, twice: first axes clipped/saturated (ROAM_MAX too small; directness compressed to
  0..0.32 because scattered-food foragers wander) -> swapped to roaming x trajectory-SPREAD and widened ROAM_MAX -> a
  good spread (28 vs 6). Lesson restated: pick behaviour axes that actually vary, verify with a diagnostic sweep.
- The reachable behaviour region is a TRIANGLE (low-roam physically can't be high-spread), so 28/64 is ~the max, not a
  failure to fill. And in this small world a compact path eats as much as a long rover -> the diversity is in STYLE,
  not all in foraging success; "more roaming" is not "more food" here. Stated, not hidden.

### Next-round seed
Round 25 = the STABLE MOBILE 3D CREATURE (the biggest parked negative, since R5) — one serious push with a heavier
search (multi-ring/shell kernels + CMA-ES / large random screen). May stay negative; report honestly either way.

## Round 25 — the 3D creature: compact body found, mobile glider still negative (DONE, committed)

### What worked
- genesis/creature3d.py: a serious, well-resourced push at the R5 negative (the mobile 3D creature). Three upgrades
  over R5's search: (1) MULTI-RING kernels (kernel_peaks) — what Bert Chan's 3D Lenia creatures actually use, where
  R5 searched mostly single-ring; (2) a SHAPED viability gradient — the round-2 fitness hard-gates on `alive`, so
  dead/exploded candidates score ~0 and the GA has NO gradient; partial credit (healthy mass + compactness +
  persistence) lets it CLIMB; (3) a motion reward.
- POSITIVE RESULT: the search RELIABLY finds a STABLE, COMPACT 3D creature — a single localised body, concentration
  1.00, mass ~3162 — across multiple seeds. This is a genuine upgrade on R5, which only delivered diffuse 3D
  self-organisation (a lattice). So "a 3D creature" (stable, compact, single body) IS now delivered.

### What did NOT work / honest notes (the sharpened negative)
- A MOBILE 3D creature (a 3D glider) is STILL NOT found. The aggressive-motion search (chase=True) found structures
  that MOVE (travel ~0.5) but they are DIFFUSE (conc ~0.3), not compact creatures; the viability search finds compact
  bodies (conc 1.0) that are STATIONARY (travel 0). The compactness x motion scatter makes it precise: the two
  ingredients exist SEPARATELY, but their intersection — compact AND moving — is EMPTY. The 3D glider is the
  knife-edge between a stationary-compact attractor and a diffuse-moving wave. This is a far more precise, honest
  negative than R5's "knife-edge, dies/foams/proliferates."
- Why: a compact symmetric blob has no preferred direction (no net motion); a 3D glider needs self-sustaining
  ASYMMETRY that translates without dispersing — rare, and the GA converges to the easy stationary-compact basin.

### Next-round seed
Round 26 = CAPSTONE REVIEW + "what is this" essay (most frontiers now explored; the last big negative is sharply
characterised). Or one more 3D-glider attempt from the diffuse-moving side. A richer substrate (Flow-Lenia/torch) is GATED.

## Round 26 — CAPSTONE REVIEW + essay (DONE, committed)

Consolidation at the natural plateau (25 build rounds explored; last big negative sharply characterised).
- R25 re-verified on a FRESH seed (11): the NEGATIVE is ROBUST — the motion search's best stays DIFFUSE (conc 0.40),
  no compact-AND-moving creature. The POSITIVE (compact 3D creature) has SEARCH VARIANCE — committed seeds 0/7 reach
  conc 1.00 and the saved genome reproduces conc 1.00, but a shorter run on seed 11 reached only 0.39 -> findable, not
  on every seed/budget. Recorded honestly; the core claims (compact found; mobile not) hold with this nuance.
- Wrote docs/CAPSTONE.md — the definitive one-page "what is this": the one-paragraph version, the arc as one line, what
  it IS and is NOT, the honest negatives (kept on purpose), and the gated next-step decision. README links to it.
- 80 tests green; all images resolve; no orphan modules. Repo .git 67M / outputs 52M (growing; visual-README cost).
VERDICT: the project is COMPLETE IN KIND and honest about its limits. CORRECTION (made same round): Flow-Lenia is an
ALGORITHM buildable in PURE NUMPY -> NOT gated; it is the natural autonomous next frontier (mass conservation may unlock
the mobile 3D creature + multi-creature worlds). Only torch/GPU + differentiable Lenia stay gated. So round 27 = a
Flow-Lenia spike (ungated, spike-then-migrate, keep plain Lenia intact), not a stop. A real frontier leap, not a stall.

## Round 27 — Flow-Lenia: a mass-conserving substrate (DONE, committed)

### What worked
- genesis/flowlenia.py: Flow-Lenia (Plantec et al. 2022) in PURE NUMPY — the ungated substrate leap. Plain Lenia
  does A += dt*G(U) and clips (mass NOT conserved). Flow-Lenia instead treats growth as an AFFINITY, computes a flow
  F = clip(grad(growth(U))), and MOVES mass along it with a mass-conserving UPWIND finite-volume advection
  (every face flux that leaves a cell enters its neighbour -> total mass exactly constant). Dimension-agnostic;
  additive (world.py untouched).
- RESULTS: (1) mass conserved EXACTLY (ratio 1.0000) in 2D AND 3D — verified + unit-tested. (2) a seed self-organises
  into a COMPACT creature with an emergent orbium-like RING in 2D, and a ROBUST compact 3D creature (conc ~0.98) —
  this is the big one: plain Lenia 3D was knife-edge (die/foam/proliferate, R5/R25), but mass conservation removes
  the dissipation/explosion failure modes, so 3D creatures are now ROBUST and EASY. (3) 4 creatures COEXIST in one
  world sharing the conserved mass (multi-creature — what plain Lenia struggles with).
- This is a genuine substrate level-up that directly addresses R25's diagnosis (mobile candidates were diffuse
  because mass dissipates) and opens multi-creature ecology.

### What did NOT work / honest notes
- A MOBILE creature is STILL not delivered. With a symmetric radial kernel the Flow-Lenia clumps are STATIONARY
  attractors (travel ~0); an offset/asymmetric kernel did NOT unlock motion in a quick test. Like plain Lenia,
  a moving creature is a narrow attractor needing a SEARCH (round-2 style) or multi-channel kernels. The win is that
  the substrate now CONSERVES mass, so a found mover won't dissipate -> the mobile glider is more reachable, not yet reached.
- Counting "how many creatures" needed a blur-first step (the creatures have fine internal texture + a dotted ring
  halo that naive thresholding fragments into ~80-110 components; blur-then-label gives the true 4).

### Next-round seed
Round 28 = EVOLVE a MOBILE Flow-Lenia creature (search over rule + asymmetric/multi-channel kernel + seed; mass
conservation means a found mover persists) — the real shot at the mobile 2D/3D glider. Or an open-ended ecology on Flow-Lenia.

## Round 28 — why the creature won't move: the gradient-flow diagnosis (DONE, committed)

### What worked (a precise NEGATIVE + its mechanism)
- genesis/creature_flow.py: the proper round-2-equivalent search — a GA over Flow-Lenia rule + kernel asymmetry +
  EVOLVED SEED SHAPE (round 2's lesson: the glider seed must be evolved, not random), rewarding directed motion of
  a compact body. Evolution pushed net travel from ~0.06R (random screen) to ~0.2R (evolved) — but PLATEAUED far
  below locomotion (round-2 plain-Lenia glider crossed 3.78 widths ~ 7.5R). Also ruled out by probes: asymmetric
  kernels, rotated (chiral) flow, and multi-creature competition (creatures coexist inertly, no mass transfer).
- THE DIAGNOSIS (the real result): single-channel Flow-Lenia moves mass by F = grad(G(U)) — a GRADIENT flow, which
  is curl-free, so it can only RELAX mass toward a stationary equilibrium. That is WHY every motion attempt (R25
  plain Lenia, R27/R28 Flow-Lenia) lands on a compact STATIONARY creature. Sustained locomotion needs a NON-gradient
  flow = MULTI-CHANNEL Flow-Lenia (the combined flow of several channels is not a pure gradient — the paper's glider
  mechanism). The recurring negative is now EXPLAINED, not just observed.

### What did NOT work / honest notes
- No mobile creature (the headline hope). But the round converts a string of "it didn't move" results into one clean
  mechanistic explanation + a concrete next step. That is the honest, valuable form of this negative.
- This is the diminishing-returns boundary on the motion frontier: the next real attempt (multi-channel) is a
  multi-round speculative build. Per protocol, surface the decision to the user rather than autonomously sinking rounds.

### Next-round seed
Round 29 = a DECISION POINT (surface to user): (a) build MULTI-CHANNEL Flow-Lenia (ungated, big, the path to gliders +
rich ecology), or (b) CONSOLIDATE (the project is comprehensive; update the capstone and call it), or (c) gated leaps.

## Round 29 — multi-channel Flow-Lenia: built; motion still walled off (DONE, committed)

The user chose (via AskUserQuestion) to build MULTI-CHANNEL Flow-Lenia — the paper's glider mechanism.
- BUILT (genesis/flowlenia_mc.py): FlowWorldMC — C channels, each advected by its own affinity gradient, COUPLED via
  cross-channel kernels (offset/asym/weight); each channel conserves its OWN mass. Structured 2-channel coupled
  creatures form + conserve mass (unit-tested). The substrate the user asked for is delivered and works.
- MOTION (the negative): a serious GA over the coupling reached only 0.11R; probes of offset + attract/repel chase
  couplings gave ~0.06R; and a less-diffusive REINTEGRATION-TRACKING advection (the suspected lever) ALSO gave ~0.
  -> motion is the SAME wall with or without multi-channel. The binding constraint is the GRADIENT FLOW (F=grad(G)
  relaxes to a stationary equilibrium); translation needs a self-sustaining asymmetric uniform-drift attractor that
  these numpy formulations don't reach. This makes the mobile-creature negative EXHAUSTIVE: plain Lenia (R25) +
  Flow-Lenia single-channel (R27/R28) + multi-channel (R29), across two advection schemes, all stationary.
- HONEST: the round delivered the substrate but not motion; that's the real result, cleanly explained. Diminishing
  returns on numpy motion are now PROVEN, not asserted. The remaining real path is GATED (differentiable Lenia / JAX).

### Next-round seed
Round 30 = the ungated motion frontier is EXHAUSTED -> surface to user: (a) GATED differentiable Lenia (torch/JAX to
gradient-FIND a glider, the biggest real shot, needs a dependency) vs (b) CONSOLIDATE (comprehensive + every negative
now exhaustively explained). Do NOT keep grinding numpy motion variants.

## Round 30 — emergent communication: two agents evolve a shared code (DONE, committed)

### What worked (a clean positive, back on the core goal)
- "继续" after the motion wall -> pivoted from the (proven-walled) motion frontier back to the CORE goal (real
  intelligence), into a dimension the single-agent mind arc never touched: SOCIAL / multi-agent.
- genesis/communicate.py: a Lewis signaling / referential game. A SPEAKER sees a hidden referent (1 of K=4) and emits a
  continuous 2-D SIGNAL; a LISTENER sees only the signal and must name the referent. Neither is given a code; both nets
  are evolved JOINTLY (OpenAI-ES, pure numpy). A shared language EMERGES: accuracy 0.25 (chance) -> 1.00; the K signals
  separate into distinct "words"; I(referent; listener-action) reaches 2.00 bits = log2(K) ceiling. ABLATION (random
  signal): accuracy -> 0.23, I -> 0.01 bits -> the channel is genuinely USED.
- Measured in bits like the earlier mind rounds (R12/R15/R16); a striking, rigorous emergent-communication result.

### What did NOT work / honest notes
- Clean round, no negative. (The K=4, 2-D-signal game is deliberately simple — a clear demonstration, not a claim of
  rich/compositional language. Compositionality is the natural next push.)

### Next-round seed
Round 31 = push social intelligence (COMPOSITIONAL communication — signal encodes 2 attributes; does the code factorise?
or grounded comm in the foraging world), OR a milestone review (30 rounds; last review R26). Motion in numpy stays a proven wall.

## Round 31 — compositional communication: does the language factorise? (DONE, committed)

### What worked
- genesis/communicate_comp.py: referents now have TWO attributes (3 shapes x 3 colours); the speaker conveys both, the
  listener decodes both (two heads); some combos are HELD OUT for a zero-shot test. NAIVE emergent comm (the round-30
  trick) is HOLISTIC: train accuracy 1.00 but held-out zero-shot 0.00 and topographic similarity only ~0.25 — it
  MEMORISES each meaning rather than building from reusable parts. Add a STRUCTURAL pressure (reward topographic
  similarity) and the language becomes COMPOSITIONAL: topo 0.25 -> 0.79, with partial zero-shot generalisation
  (0.00 -> 0.33). Compositionality — the hallmark that lets finite parts express infinite meanings — is NOT free from
  communicative success alone; it emerges under a learnability/structure pressure. A recognised emergent-language
  result (cf. iterated-learning / Kirby), replicated cleanly in numpy.

### What did NOT work / honest notes
- Held-out zero-shot generalisation rose only to ~0.33 (not 1.0) and is noisy across seeds — the SPEAKER's code becomes
  compositional (topo 0.79) but the LISTENER's zero-shot decoding is the bottleneck. So "partial generalisation," stated
  honestly. The 2-D PCA scatter of the pressured (3-D) code shows increased structure but not a textbook-crisp grid;
  topographic similarity (0.25 -> 0.79) is the robust quantitative claim, the scatter the qualitative illustration.

### Next-round seed
Round 32 = MILESTONE REVIEW (overdue; last review R26) or GROUNDED communication in the foraging world. Motion stays a proven wall.

## Round 32 — MILESTONE REVIEW + capstone refresh (DONE, committed)

Adversarial re-verification of the post-R26 rounds on FRESH seeds (last full review was R26).
- R27 Flow-Lenia: mass conserved EXACTLY (ratio 1.00000) in 2D AND 3D on a fresh config -> holds.
- R30 emergent communication (fresh seeds 5-7): accuracy 1.00, 2.00 bits (ceiling), ablated 0.25 (chance) -> robust, not cherry-picked.
- R31 compositional (fresh seeds 5-6): topo naive 0.33 -> pressured 0.80 -> compositionality emerges robustly.
- 93 tests green; all README images resolve; no orphan modules; every honest negative/caveat intact.
- REFRESHED docs/CAPSTONE.md (was stale at "25 rounds"): now 31 rounds, three threads (mind loop / open-endedness /
  SOCIAL), adds the Flow-Lenia substrate + the EXHAUSTIVELY-explained motion negative (gradient-flow wall + gated
  diff-Lenia path) + the social rounds; README capstone pointer (25->31) + intro arc updated to include R30/31.
- NOTE: repo .git 80M / outputs 55M (growing; inherent to the visual README — ~33 round figures + GIFs; acceptable).
VERDICT: the recent rounds stand up on unseen seeds; docs are back in sync with the (now 31-round) reality. Resume the social vein.

## Round 33 — grounded communication: a signal that drives foraging (DONE, committed)

### What worked
- genesis/communicate_grounded.py: communication GROUNDED in action. A SCOUT sees the food direction and emits a
  signal; a BLIND FORAGER sees only the signal and must navigate to the food; the pair is evolved JOINTLY (ES).
- RESULT (catch rate, unseen episodes): sighted upper bound 1.00 | comm pair 0.58 | ablated (random signal) 0.05.
  The signal carries ACTIONABLE SPATIAL information: with the channel the blind forager reaches food, ablated it is
  lost. The trajectory visual makes it vivid — WITH comm the forager steers straight to the food star; ablated it
  scatters. This FUSES the embodied track (navigation) with the social track (signalling) — the first communication
  that DOES something in the world (not an abstract label game).

### What did NOT work / honest notes
- Catch rate with comm is 0.58 (not ~1.0): encoding a continuous direction and decoding it into a heading is lossy
  (the catch radius is tight), so the pair forages a majority of the time but not always. The comm-vs-ablated gap
  (0.58 vs 0.05, ~12x) is the robust claim; 0.58 < sighted 1.00 is the honest ceiling gap. Stated, not hidden.

### Next-round seed
Round 34 = ITERATED LEARNING (does compositionality emerge from a transmission bottleneck, no hand-added pressure?) or
multi-agent COORDINATION / theory-of-mind. Motion stays a proven wall.

## Round 34 — iterated learning: compositionality from cultural transmission (DONE, committed)

### What worked (the principled "why" behind R31)
- genesis/communicate_iterate.py: Kirby-style iterated learning. Each generation LEARNS the language (a meaning->signal
  map, a tanh-MLP trained by hand-coded numpy backprop) from only a SUBSET of meanings (the learnability BOTTLENECK),
  then PRODUCES the whole language (generalising to unseen meanings); a light expressivity rescale prevents the trivial
  all-signals-collapse. Over ~45 generations: under the BOTTLENECK (5/9) topographic similarity rises ~0 -> ~0.3 (peak
  0.4); under FULL transmission (9/9) it stays flat ~0.0. So compositional structure emerges from TRANSMISSION ALONE
  (holistic codes can't be reconstructed from few examples and degrade; systematic ones survive) — NO hand-added
  structure term (unlike R31). This is the recognised mechanism, replicated cleanly in numpy with manual backprop.

### What did NOT work / honest notes
- The emergent topo is ~0.3 (more modest than R31's hand-pressured 0.79) and FLUCTUATES across seeds/generations (the
  re-learning from a random subset each generation is stochastic). The robust claim is the bottleneck-vs-full CONTRAST
  (~0.3 vs ~0.0), not a precise level. The expressivity rescale is a (mild) hand-added anti-collapse term — without it
  pure transmission collapses to one signal (degenerate); stated honestly.

### Next-round seed
Round 35 = THEORY-OF-MIND (infer another agent's hidden goal from behaviour) or multi-agent COORDINATION. Motion stays a proven wall.

## Round 35 — theory of mind: infer a hidden goal from behaviour (DONE, committed)

### What worked
- genesis/theory_of_mind.py: a social axis DISTINCT from communication — mentalising. An ACTOR moves toward one of K=4
  goals with heavy noise; an OBSERVER (recurrent net, ES-evolved) sees only the actor's step-by-step MOTION (displacements,
  no absolute position) and must infer WHICH goal. Result: accuracy 0.48 -> 0.84 as it watches; belief SHARPENS on the
  true goal (mean P(true) 0.43->0.81, P(best-wrong) 0.56->0.19); ABLATED (random motion) -> 0.29 ~ chance (0.25). The
  observer integrates noisy motion into a belief about intent — reading another agent's mind from behaviour.

### What did NOT work / honest notes
- Inferring "which target an agent walks toward" is PARTLY GEOMETRIC: a position oracle (nearest target to current
  position) also solves it well (~0.95), because the actor cooperatively reveals its goal. So this is NOT "modeling beats
  the optimal heuristic"; the honest result is (1) the observer LEARNS to read intent from motion-only, (2) the belief-
  updating dynamic, (3) ablation -> chance. The harder, genuine ToM (behaviour that MISLEADS — detours/obstacles, where
  surface motion points the wrong way) is a future frontier; stated, not hidden. Also tuned noise up to 1.5 so single
  steps are ambiguous (the belief sharpens gradually) — at low noise it's certain after 1 step (trivial).

### Next-round seed
Round 36 = multi-agent COORDINATION (a task only solvable together) or comm-enabled coordination; OR consolidate (6 social
rounds done -> a "social intelligence" summary + milestone review). Motion stays a proven wall.

## Round 47 — FINAL WHOLE-PROJECT CAPSTONE + honest edge + forks surfaced (DONE, committed)

Re-verified R46 on FRESH seeds and closed the capstone for the whole 46-round arc.
- R46 HOLDS WITH A CAVEAT: spatial cooperation persists (0.19-0.37) / well-mixed collapses to 0 on most seeds, but the
  showcase b=1.62 is NEAR-CRITICAL (~1.7) and SEED-SENSITIVE -- on seeds (7,8) spatial cooperation collapses too. The
  ROBUST result is the PHASE TRANSITION (b-sweep): space sustains cooperation up to a critical b, well-mixed never; at
  lower b~1.5 spatial is robustly ~0.85. Added the seed-sensitivity caveat to the README + STATUS (cf. R18 nuance).
- REFRESHED docs/CAPSTONE.md (44 -> 46 rounds) as a CLOSING summary: 3 capstoned threads + 3 post-summit dims + the
  cooperation transition; reviews -> 11; substrate framed at its HONEST EDGE. 113 tests green; all images resolve; no orphans.
- SURFACED THE FORK to the user via AskUserQuestion (round 47): (A) open the GATE (differentiable Lenia/torch -> mobile
  creature / 3D scale), (B) ungated open research (open-ended ratchet of new KINDS / N-agent symmetry / embodied cooperation),
  (C) recognise completeness. The clearly-distinct ungated dimensions are exhausted; the open direction is delivered broad+deep.

### What this means for the loop
- The autonomous /loop has reached genuinely diminishing returns on NEW dimensions. Per protocol, at the edge the move is
  to PRESENT the forks (done) rather than invent same-shaped increments or cross a dependency gate autonomously. Future
  rounds without a steer = light polish/review only; a gated leap (A) needs explicit user go-ahead.

### Next-round seed
Round 48 = await the user's fork choice; if none, light polish/review. Do NOT cross the gate (A) autonomously.

## Round 46 — the evolution of cooperation: a spatial major transition (DONE, committed)

### What worked (the last clearly-distinct ungated dimension; closes the eco thread R6-8)
- genesis/cooperation.py: a spatial prisoner's dilemma on a 60x60 grid (Nowak & May 1992). A cooperator earns 1 per
  cooperating neighbour; a defector earns b=1.62 per cooperating neighbour (free-riding); each cell imitates the
  highest-scoring strategy in its Moore neighbourhood. Result: WELL-MIXED (positions shuffled each step) -> defectors win,
  cooperation COLLAPSES to 0.00; SPATIAL (real neighbours) -> cooperators CLUSTER (clustering ~2.2x random) and cooperation
  PERSISTS at ~0.37 (dynamic coexistence). A b-sweep shows space sustains cooperation up to a critical temptation (~1.7),
  then collapse; well-mixed is 0 for all b. The mechanism is NETWORK RECIPROCITY -- same game + update, the only difference
  is a 2D neighbourhood vs a shuffled one. A genuine major evolutionary transition, fittingly spatial/2D, distinct from
  every net-vs-ablation round; it answers the arc's "推演世界的演化 / derive the world's evolution" for cooperation.

### What did NOT work / honest notes
- It's a classic abstract game (PD + imitate-best), not embodied in Lenia bodies -- the cleanest demonstration of the
  spatial-cooperation transition; an embodied version (Lenia foragers that share/withhold) would be heavier and noisier.
- The result is a faithful replication of a known model (Nowak-May), not a novel mechanism -- but it closes a real gap
  in the world-evolution arc that the project had not shown. Stated.

### Next-round seed
Round 47 = FINAL WHOLE-PROJECT CAPSTONE (re-verify R46, refresh capstone to 46 rounds) + present the remaining forks to the
user; the substrate is at its honest edge (further big leaps are gated or open research questions). Motion stays a proven wall.

## Round 45 — MILESTONE REVIEW: R43/R44 re-verified + capstone refresh + edge assessment (DONE, committed)

Adversarial re-verification of the post-summit rounds on FRESH unseen seeds (last full review R42).
- R43 open-ended ratchet (seeds 5-6): cumulative height 127-160 vs individual 5 vs transmit 1 -> unbounded accumulation reproduces.
- R44 symmetry-breaking (seeds 5-6): split 1.00 vs no-interaction 0.01-0.45 vs no-trigger 0.00 -> roles differentiate only with both ingredients.
- 111 tests green; all README images resolve; no orphan modules.
- REFRESHED docs/CAPSTONE.md (41 -> 44 rounds): notes the two post-summit dims (open-ended ratchet R43, emergent roles R44);
  reviews list -> 10; arc one-line updated; summit paragraph corrected ("two further" post-summit dims, not three).
- ASSESSMENT: the genuinely-new-dimension vein is THIN. 3 post-summit dims done (R41 cumulative culture, R43 open-ended
  ratchet, R44 symmetry-breaking). The ONE clearly-distinct ungated dimension still untouched is ECO-EVOLUTION major
  transitions (evolution of cooperation / niche construction on the R6-8 ecology). After that the substrate is at its honest
  edge -- remaining big leaps are gated on a dependency (differentiable Lenia for motion; torch for scale).
- NOTE: repo .git 107M / outputs 58M (growing; the one real watch-item).
VERDICT: post-summit rounds hold; docs in sync with 44-round reality; round 46 = the last distinct ungated dimension
(eco-evolution / major transitions), then the substrate's honest edge + a final capstone.

## Round 44 — emergent roles from scratch: symmetry-breaking with no id (DONE, committed)

### What worked (closes the honest gap in R37/R40)
- genesis/symmetry_break.py: in R37/R40 the role distinction was GIVEN (each agent had an id). Here two IDENTICAL agents
  (ONE shared policy, no id, the same near-symmetric start) must end in COMPLEMENTARY roles. A learned MUTUAL-INHIBITION
  dynamic + a tiny symmetry-breaking trigger make their leanings DIVERGE to opposite roles (a pitchfork bifurcation).
  Result: split rate 1.00 full | 0.00 no-interaction | 0.00 no-trigger (perfect symmetry). BOTH ingredients necessary --
  spontaneous symmetry breaking (a balanced pencil needs instability AND a perturbation). Roles emerge from interaction,
  not a handed-out label. The bifurcation visual (identical leanings fanning to +-1) makes the mechanism vivid.

### What did NOT work / honest notes
- The trigger turned out to live in the INITIAL near-symmetric state (0.02 jitter), not the per-step noise -- so the
  honest "no-trigger" ablation is a PERFECTLY symmetric start (exactly 0,0) with no noise -> the deterministic symmetric
  dynamics never break (split 0.00). Corrected the ablation to be honest about what the trigger is.
- It's a 2-agent / 2-role abstract settling dynamic (scalar leanings), not embodied in Lenia; the symmetry-breaking
  principle is substrate-independent and this is the cleanest demonstration. N-agent / N-role from scratch is a harder extension.

### Next-round seed
Round 45 = MILESTONE REVIEW (re-verify R43/R44, refresh capstone to 44 rounds; the new-dimension vein is thinning) OR
eco-evolution major transitions. Motion stays a proven wall.

## Round 43 — open-ended ratchet: cumulative complexity with no target (DONE, committed)

### What worked (deepens R41 into open-endedness)
- genesis/open_ended.py: cumulative culture that INVENTS its own complexity, with NO target (R41 ratcheted toward a given
  star). The artifact is a TOWER of stacked blocks; the only rule is physics -- the running centre of mass above every
  joint must stay supported, else it topples. Complexity = HEIGHT, unbounded. Each generation INHERITS the tower and adds
  a BOUNDED few blocks (one lifetime). Result: CUMULATIVE (transmit + innovate) accumulates a ~148-block spire; INDIVIDUAL
  (restart each gen) is capped at ~5 (one lifetime's budget); TRANSMIT-ONLY (copy) stays at 1. The tower keeps reaching
  with no ceiling but stability -- open-ended complexity no single lifetime could build. First round with NO target/fitness
  target (the complexity is intrinsic, not distance-to-a-goal); the tall gradient spire vs stub makes it legible.

### What did NOT work / honest notes
- The strict CoM-stability rule caps horizontal REACH/overhang at <1 block-width (a tall tower must be near-vertical to
  stay stable), so the open-ended axis is HEIGHT, not the harmonic-overhang reach. Stated; height is genuinely unbounded.
- "Open-ended" here = unbounded accumulation of one complexity axis (height), not unbounded NOVELTY of kind. A ratchet
  that invents qualitatively new structure-types is a deeper frontier; this is the honest, clean first step.

### Next-round seed
Round 44 = emergent roles WITHOUT pre-given ids (symmetry-breaking from scratch) OR eco-evolution major transitions. Motion stays a proven wall.

## Round 42 — WHOLE-PROJECT CAPSTONE/REVIEW + frontier menu (DONE, committed)

Adversarial re-verification of the latest rounds on FRESH unseen seeds (last review R39), at a broad/deep summit.
- R40 unified social (seeds 5-6): full 1.00 vs no-comm 0.38-0.56 vs no-coord 0.50 -> both faculties load-bearing.
- R41 cumulative culture (seeds 5-6): cumulative 0.91-0.92 vs individual ~0 vs transmit ~0 -> the ratchet holds.
- 107 tests green; all README images resolve; no orphan modules.
- REFRESHED docs/CAPSTONE.md (38 -> 41 rounds): now framed as a SUMMIT -- 3 threads each carried to its own capstone
  (mind UNIFIED R21, open-endedness ILLUMINATED R22/24, social UNIFIED R40 + cumulative RATCHET R41); reviews -> 9.
- Added a FRONTIER MENU surfaced for the human: GATED (mobile creature via differentiable Lenia -> needs torch/JAX);
  UNGATED NEW DIMENSIONS (open-ended ratchet that invents its own complexity; emergent roles from scratch; eco-evolution
  major transitions); or RECOGNISE COMPLETENESS. At this summit the direction is the human's to set.
- NOTE: repo .git 100M / outputs 57M (growing; the one real watch-item; inherent to the visual README).
VERDICT: latest rounds hold; docs in sync with the 41-round summit; the menu is surfaced -- round 43 should be a
genuinely-new ungated dimension (lean open-ended ratchet) rather than another same-shaped increment.

## Round 41 — cumulative culture: the ratchet (DONE, committed)

### What worked (a genuinely new dimension beyond R34)
- genesis/cumulative_culture.py: the Tomasello ratchet. Each generation INHERITS the previous artifact (a point cloud),
  innovates a BOUNDED amount within one lifetime (hill-climbing), and passes on a slightly better one -> skill RATCHETS
  UP past any single lifetime. The ratchet needs BOTH transmission AND innovation: CUMULATIVE (both) climbs quality 0 ->
  0.92, building a sharp 5-pointed STAR; INDIVIDUAL (innovate, restart each gen) is stuck at the single-lifetime ceiling
  (~0.05); TRANSMIT-ONLY (copy, no innovation) never improves (~0.00). The accumulated artifact is something no single
  lifetime here could build -- the hallmark of human cumulative culture. The artifact-emerging visual (blob -> partial ->
  star) makes the ratchet legible; distinct from R34 (which transmitted a FIXED language, no improvement).

### What did NOT work / honest notes
- The task is an abstract point-cloud-matching artifact (not a Lenia body) -- the cumulative principle is substrate-
  independent and this is the cleanest demonstration; a Lenia-embodied ratchet would be heavier. Stated.
- "Quality" is shape-match to a fixed target; the ratchet is real (accumulation past single-lifetime) but the target is
  given, not open-ended-invented. A truly open-ended ratchet (no fixed target) is a deeper future frontier.

### Next-round seed
Round 42 = whole-project capstone/review (41 rounds, 3 threads + cumulative culture) OR one more new dimension. A rich summit.

## Round 40 — unified social world: communicate AND coordinate (the social-arc capstone) (DONE, committed)

### What worked
- genesis/unified_social.py: the multi-agent analogue of R21's unification of the single mind. The isolated social rounds
  R30-38 each showed ONE faculty; here they must work TOGETHER. Each round a few sites are RICH but only a SCOUT sees
  which; it signals a team of foragers (N=2 < K=4 sites) who must COVER the rich sites without piling up. Full yield needs
  BOTH faculties: COMMUNICATION (which sites are rich) and COORDINATION (split across them). Result: team yield FULL 1.00
  | NO-COMM 0.52 | NO-COORD 0.50 -> ablate EITHER and the yield HALVES; each is LOAD-BEARING, exactly as R21 proved
  memory/prediction/planning each necessary. The vignettes make it legible: FULL covers both rich sites, NO-COMM forages
  blind and misses one, NO-COORD piles both foragers on one. The social arc now has its capstone, integrating R30-38.

### What did NOT work / honest notes
- First attempt used N=K=4 (foragers==sites): communication became UNNECESSARY because the team could blanket ALL sites and
  cover the rich ones for free (no-comm = 1.0). Fixed by making FEWER foragers than sites (N=2 < K=4) so they must CHOOSE
  which to cover -> communication becomes load-bearing. Documented the design fix.
- The no-comm vignette needed a representative (failing) draw, not a lucky one, to match the 0.52 average -- handled.

### Next-round seed
Round 41 = a WHOLE-PROJECT CAPSTONE pass (the social arc reached its R21-analogue summit; 40 rounds, 3 mature threads) OR a
genuinely new dimension (cumulative culture; emergent roles from scratch). A natural plateau -- worth surfacing the frontier menu.

## Round 39 — MILESTONE REVIEW: R37/R38 re-verified + capstone refresh + ambition critic (DONE, committed)

Adversarial re-verification of the latest rounds on FRESH unseen seeds (last full review R36).
- R37 coordination (seeds 5-7): coverage 1.00 vs ablated 0.25 -> perfect division of labour reproduces.
- R38 harder ToM (seed 4): mid-detour observer 0.99 vs oracle 0.63, mean 0.77 vs 0.70 -> modelling beats the position
  oracle through misleading behaviour; ablated 0.33 chance -> holds.
- 103 tests green; all README images resolve; no orphan modules.
- REFRESHED docs/CAPSTONE.md (35 -> 38 rounds): the social arc now spans all 8 social rounds; reviews list -> 8.
- AMBITION CRITIC: the social vein is now DEEP + BROAD (8 rounds) and the pattern (evolve a net, beat an ablation) is
  becoming repetitive. Strongest IN-KIND next move = a UNIFIED SOCIAL WORLD (multi-agent analogue of R21's unification:
  agents must COMMUNICATE AND COORDINATE to forage/survive under selection, each social faculty load-bearing). Else a
  genuinely new dimension, or recognise broad completeness. This integrates R30-38 rather than adding a 9th isolated demo.
- NOTE: repo .git 94M / outputs 57M (growing; inherent to the visual README; watch).
VERDICT: latest rounds hold on unseen seeds; docs in sync with 38-round reality; round 40 = the unified social world (capstone of the social arc).

## Round 38 — harder theory of mind: reading intent when behaviour misleads (DONE, committed)

### What worked (addresses the R35 honest gap)
- genesis/tom_obstacle.py: the ACTOR must DETOUR around a central obstacle to reach goals behind it (potential-field /
  go-around navigation), so its early motion points AWAY from its goal. A POSITION ORACLE (nearest goal to current
  position) is FOOLED while the actor is on the wrong side; only an OBSERVER (recurrent, ES) that has learned the obstacle
  + the go-around policy infers the true goal mid-detour. Result: at the misleading moment (3/4 through) observer ~0.97 vs
  oracle ~0.67; mean over the trajectory observer 0.80 > oracle 0.73 (the oracle only catches up at the very end when the
  actor arrives); ablated 0.35 ~ chance (0.33). Unlike R35 (where a position oracle did as well), the modelling observer
  BEATS the position oracle here -- it reads intent THROUGH misleading behaviour. The vivid detour trajectories + the
  mid-detour snapshot (observer right, oracle fooled) make the mechanism legible.

### What did NOT work / honest notes
- Potential-field navigation traps at the directly-OPPOSITE goal (forces balance) -> needed explicit "blocked -> steer
  tangentially around (consistent side)" logic + a longer window (T=34) so all detours complete. Documented.
- The oracle still wins at the FINAL step (1.00) once the actor arrives -- the observer's edge is reading intent EARLY /
  mid-detour (0.97 vs 0.67) and on AVERAGE (0.80 vs 0.73), not at the trivial endpoint. Stated honestly.

### Next-round seed
Round 39 = MILESTONE REVIEW (8 social rounds; last review R36) OR a genuinely new dimension. Motion stays a proven wall.

## Round 37 — multi-agent coordination: division of labour (DONE, committed)

### What worked
- genesis/coordinate.py: ACTING TOGETHER — a team axis distinct from communication. N=4 agents must COVER N sites;
  the team's yield is the number of DISTINCT sites occupied. With distinct ROLES the team evolves a DIVISION OF LABOUR
  (ES) — a PERMUTATION assigning each agent its own site -> coverage 1.00; an independent-random team gets ~0.69 (some
  collide by chance); an ABLATED identical team collapses to 0.25 (all pile on one site). Coordination needs BROKEN
  SYMMETRY; the assignment is an emergent convention (a different permutation each seed). The embodied figure makes it
  vivid: evolved agents fan out to a site each; ablated agents pile onto one.

### What did NOT work / honest notes
- First tried a COMM-RENDEZVOUS variant (only A sees the target, signals B, both must meet on it). Two issues: (1) with
  a static meeting site the agents found a fixed CONVENTION that needed no channel (ablated stayed high); (2) with a
  varying target only A sees, B decoded perfectly (match 1.00) but A failed to reliably go to the target (0.35-0.64) —
  it kept collapsing toward one-way communication (already covered in R30/R33). Division-of-labour is the cleaner,
  genuinely-distinct coordination result; documented the dead-end rather than forcing the comm variant.
- The role distinction is GIVEN (each agent has an id); symmetry-breaking FROM SCRATCH (identical agents differentiating
  without pre-given ids) is a harder future frontier. Stated honestly.

### Next-round seed
Round 38 = consolidation/review soon (7 social rounds) OR harder theory-of-mind (misleading behaviour) OR a new non-social frontier. Motion stays a proven wall.

## Round 36 — MILESTONE REVIEW: social rounds re-verified + capstone refresh (DONE, committed)

Adversarial re-verification of the SOCIAL rounds (R33/34/35) on FRESH unseen seeds (last full review R32).
- R33 grounded comm (seeds 5-6): catch WITH comm 0.71 vs ablated 0.03 (committed 0.58/0.05 -> even stronger) -> holds.
- R34 iterated learning (seeds 7-9): topo bottleneck 0.28 vs full 0.10 (the bottleneck-vs-full gap holds) -> holds.
- R35 theory of mind (seed 3): observer 0.78 vs ablated 0.25 = chance -> holds.
- 99 tests green; all README images resolve; no orphan modules.
- CORRECTED an over-claim: theory_of_mind.py docstring claimed the observer "beats a position-based naive heuristic"
  -> it does NOT (a position oracle is as good, since the actor cooperatively reveals its goal); fixed to honest scope.
- REFRESHED docs/CAPSTONE.md (25 -> 31 -> NOW 35 rounds): the SOCIAL thread now spans all 5 social rounds; reviews -> 7;
  README capstone pointer 31->35. NOTE: repo .git 88M / outputs 56M (growing; inherent to the visual README; watch).
VERDICT: the social rounds stand up on unseen seeds; one docstring over-claim corrected; docs back in sync with 35-round reality.

## Frontier (durable ambition horizon — what ORIENT is pulled up by)

- CURRENT CEILING (after 46 rounds + TEN reviews): the mind's core loop is COMPLETE + INTEGRATED + SOCIAL (communication:
  emergent/compositional/grounded/iterated, + theory of mind x2 incl. misleading-behaviour, + COORDINATION/division-of-labour, + UNIFIED social world where comm+coord are each load-bearing, + CUMULATIVE CULTURE/the ratchet, + OPEN-ENDED RATCHET/unbounded complexity no target, + EMERGENT ROLES from scratch/symmetry-breaking) + ECO-EVOLUTION (the EVOLUTION OF COOPERATION via spatial network reciprocity); the substrate LEVELED UP (Flow-Lenia + multi-channel); the mobile creature is an exhaustively-explained negative — a continuous-CA world with ONE engine across 1D/2D/3D; an
  embodied creature that emerges, moves, senses+forages (agency), forages-to-survive (metabolism);
  a social ECOLOGY with stabilizing selection and EVOLUTION RUNNING (discovers the optimum); a
  two-species predator-prey world (top-down regulation); a creature that LEARNS within its life and
  re-learns under reversal; AND (round 11) that plastic brain now lives INSIDE a Lenia creature —
  body+agency+learning united in one organism; AND (round 12) intelligence MEASURED — I(brain;world)
  = 0.69 bits, with an operating envelope; AND (round 13) learning shown ADAPTIVE under selection —
  learners win in a changing world, lose in a stable one (Baldwin). The arc emergence->locomotion->
  agency->survival->3D->ecology->evolution->predation->learning->embodied-learning->measured-mind->
  learning-selected->baldwin-rate->memory->prediction->embodied-memory->PLANNING->UNIFICATION->OPEN-ENDEDNESS(bodies)
  ->OPEN-ENDED-MINDS->3D-CREATURE(compact yes, mobile no)->FLOW-LENIA(mass-conserving substrate; robust 3D + multi-creature).
  (Round 27: a mass-conserving substrate in pure numpy — 3D creatures are now ROBUST where plain Lenia 3D was knife-edge,
  and multiple creatures coexist; mobile creature reopened because mass can no longer dissipate.)
- NEXT FRONTIER(S), ranked by ambition x feasibility:
  1. EVOLVE a MOBILE Flow-Lenia creature (ungated): search rule + asymmetric/multi-channel kernel + seed on the conserving
     substrate -> the real shot at the mobile 2D/3D glider (a found mover won't dissipate). The reopened frontier.
  2. An open-ended ECOLOGY on Flow-Lenia (multi-creature + food + selection; conserved mass = natural resource competition).
  3. GATED (ask the user): torch/GPU (scale, bigger 3D), differentiable Lenia (gradient-find the glider).
- FIDELITY / STACK ESCALATION LADDER:
  numpy CPU (now) -> vectorised batch search -> torch + MPS/GPU for large 2D/3D and
  for differentiable/neural controllers -> real-time interactive viewer.
- RADICAL IDEAS WEIGHED: Flow-Lenia (mass-conserving, lets multiple species share a
  world — strong for open-ended ecology + foraging); Particle-Lenia (particle substrate,
  cheap agency); differentiable Lenia (gradient-evolve creatures / learned controllers).
  Flow-Lenia is now the leading candidate substrate for round 3+ (food + ecology).
- AMBITION CRITIC (after round 19 — the mind's core loop is complete): react (3-13), remember (15), predict
  (16), and now INTEND (19) are all present, measured, and where it matters shown to PAY. The "make real
  intelligence" arc is, in kind, WHOLE — modest in scale, but a genuine perceive->model->predict->act-to-achieve
  loop grown from local rules. Honest framing of what this is and is NOT: it is NOT one creature that does all of
  these at once (the capabilities are demonstrated across separate, focused settings — embodied where it counts,
  abstract where that isolates the result); it is NOT large-scale or competitive with engineered agents; and the
  3D *creature* remains an open negative. The next ambition is therefore UNIFICATION (one organism with body +
  memory + prediction + planning in the ecology) and TELLING (a capstone README that makes the whole arc legible),
  not another isolated capability. Adding verbs is done; integrating them into one mind, and proving the integrated
  whole, is the remaining frontier — plus the long-parked stable 3D creature for anyone returning to the substrate.
- AMBITION CRITIC (after round 20 — review + capstone): TELLING is now done (whole-arc README + honest-scope box),
  and the review honestly corrected R18's headline (real but moderate/seed-sensitive, not 2x). What remains is the
  single most valuable next step: UNIFICATION — stop demonstrating verbs in isolation and build ONE creature that
  must use body + memory + prediction + planning together to survive a task none of them alone solves. That is the
  honest answer to "it's not one organism doing it all," and it is the natural climax of the arc. Second: open-ended
  novelty (a ZOO + diversity metric) to show the world keeps GENERATING, not just converging. The 3D creature stays
  parked (would need a heavier specialised search). Build, don't add: integrate and prove the whole.
- AMBITION CRITIC (after round 21 — unification done): the climax landed — ONE creature integrates all four faculties
  and ablation proves each is load-bearing (full 263 > memory_only 188 > no_memory 140). Honest dent: the integrated
  controller is HAND-coded, not evolved (the faculties were each shown evolvable earlier; ES couldn't learn them jointly
  — a real, documented limitation, the R18/R19/R21 tension). So "grown not coded" holds for the faculties and the
  substrate, but the final wiring is engineered. What's genuinely NOT yet shown across 21 rounds: OPEN-ENDEDNESS /
  creativity — every result so far CONVERGES to an optimum or a fixed skill; nothing demonstrates the world keeps
  GENERATING new behaviours. That is the most expert-unimpressive gap now ("nice, but does it keep surprising you?").
  Round 22+ ratchet: novelty search / a diversity metric / a ZOO showing open-ended generation — or honestly report
  that this numpy-CA substrate converges and open-endedness needs Flow-Lenia / a richer substrate (a real finding either
  way). Secondary: evolve the unified controller; the long-parked 3D creature.
- AMBITION CRITIC (after round 22 — open-endedness shown, with caveats): the creativity gap is now addressed — MAP-Elites
  fills 54/64 niches (a real zoo) where fitness converges to one, so the substrate IS generative across a behaviour space.
  Honest dents an expert would still press: (1) the zoo is bodies, not MINDS — diversity of move/size patterns, and many
  high-mass niches are foam textures, not discrete organisms; open-ended BEHAVIOUR/intelligence (a zoo of distinct
  foraging strategies) is the deeper, unshown claim. (2) The map is BOUNDED — coverage saturates toward 64; this is
  illumination of a finite space, not UNBOUNDED open-endedness (which needs a coevolving/Flow-Lenia substrate). (3) 22
  rounds in, R21-22 are unreviewed and the last full review was R20. Ratchet: a MILESTONE REVIEW (re-verify R21-22,
  refresh the arc), THEN either open-ended MINDS (novelty over foraging behaviour) or finally attack the 3D creature.
  The arc is now broad AND deep AND (mostly) honest; the remaining moves are consolidation and the few real negatives.
- AMBITION CRITIC (after round 23 — review): R21-22 verified robust on fresh seeds (no correction needed — the project's
  honesty discipline is holding). 23 rounds + 4 reviews in, the work is comprehensive; the two genuinely-unshown things an
  expert would still name are: (1) open-ended MINDS, not just bodies — R22 illuminated a zoo of body shapes, but a zoo of
  distinct FORAGING STRATEGIES (behaviour, not morphology) would be the stronger creativity claim and reuses the existing
  foraging machinery; (2) the stable mobile 3D CREATURE — the one hard negative parked since R5, the only place the
  "1D->2D->3D" promise is incomplete (3D gives self-organisation, not a glider). Round 24 ratchet: open-ended minds (higher
  feasibility, extends R22) OR a real attempt at the 3D creature (higher ambition, may stay negative — but worth one honest
  push with the heavier search). Either is a legitimate frontier; neither is busywork. The substrate itself (numpy CA) is
  near its ceiling — bigger leaps (Flow-Lenia, torch/GPU, differentiable) are gated on a new dependency = ask the user first.
- AMBITION CRITIC (after round 24 — open-ended minds done): the creativity claim is now BOTH bodies (R22) and minds (R24) —
  a zoo of foraging strategies with visibly distinct trajectories. Of the two gaps the R23 critic named, one is closed; the
  remaining headline negative is the STABLE MOBILE 3D CREATURE (parked since R5). It is the single most honest, most
  expert-legible thing still missing — "you scaled the ENGINE to 3D but never grew a 3D creature." Round 25 should make ONE
  serious, well-resourced attempt (multi-ring/shell 3D kernels, CMA-ES or a big random screen over rule+seed, viability =
  localized+persistent+mobile in 3D) and REPORT THE RESULT HONESTLY — a found 3D glider would be the capstone; a rigorous
  negative ("here is exactly how hard it is, here's the search, here's why it's knife-edge") is itself a real result and
  closes the question. After that, the numpy-CA substrate is genuinely near its ceiling: further leaps (Flow-Lenia, torch/GPU)
  need a dependency = a GATE (ask the user). So R25 = the 3D push; then likely a capstone/曲终 unless the user redirects.
- AMBITION CRITIC (after round 25 — the 3D push, mixed result): the last big negative is now SHARPLY characterised — a
  stable COMPACT 3D creature is found (upgrade on R5), but the mobile 3D glider remains the knife-edge intersection of
  compact-stationary and diffuse-moving. This is the most honest possible state for that question on this substrate: not
  "we never tried hard enough" but "here is exactly the trade-off that makes it rare." An expert would now respect the
  negative rather than dock it. WHAT'S LEFT, honestly: the project is comprehensive (25 rounds: full mind arc + integration
  + open-endedness of bodies AND minds + a rigorous 3D result). Every remaining BIG leap needs a new substrate/dependency
  (Flow-Lenia for unbounded open-endedness + real 3D creatures; torch/GPU for scale; differentiable Lenia to gradient-find
  the 3D glider) — all GATED on the user. So the honest next move is CONSOLIDATION: a capstone review + the definitive
  "what is this" telling, making the repo the artifact. Continuing to add small numpy-CA rounds would be busywork / diminishing
  returns; the high-value moves from here are either (a) capstone, or (b) a user-approved substrate leap. Round 26 = capstone,
  and surface the gated substrate options to the user rather than silently grinding.
- AMBITION CRITIC (after round 27 — Flow-Lenia, substrate leveled up): the R26 "everything left is gated" read was WRONG —
  Flow-Lenia is an algorithm (pure numpy), so the substrate leap was ungated and got done. Real progress: a mass-conserving
  substrate where 3D creatures are now ROBUST (the R5/R25 knife-edge is gone) and multiple creatures coexist. The frontier has
  REOPENED: the mobile creature, which looked terminal on plain Lenia, is now plausibly reachable because mass can't dissipate.
  So returns are NOT diminishing — there is a concrete, high-value, ungated next round: EVOLVE a mobile Flow-Lenia creature
  (round-2-style search over rule + asymmetric/multi-channel kernel + seed, on the conserving substrate). That is the right
  round 28 — keep pushing, don't consolidate prematurely. Genuinely gated leaps (torch/GPU, differentiable Lenia) remain for
  later if the user wants them. Lesson recorded: check whether a "frontier" actually needs a dependency before declaring a plateau.
- AMBITION CRITIC (after round 28 — motion frontier resolved as an EXPLAINED negative): the mobile creature was NOT cracked,
  but R28 did the honest, valuable thing — a proper GA (round-2-equivalent) confirmed single-channel Flow-Lenia can't locomote,
  AND diagnosed WHY (gradient flow = curl-free = relaxational). That turns three rounds of "it won't move" into one clean
  mechanism + a concrete next step (multi-channel = non-gradient flow). This is now genuine diminishing returns on the motion
  frontier within reach of one round: the next real attempt (multi-channel Flow-Lenia) is a MULTI-ROUND speculative build with
  uncertain payoff. Per the loop protocol ("stop only when returns genuinely diminish after real pushing — and even then,
  present the remaining frontier options"), round 29 = SURFACE THE DECISION to the user: (a) commit to multi-channel (big, the
  real path to gliders + rich ecology), or (b) consolidate — the project is comprehensive and every negative is now explained.
  Do NOT autonomously sink many rounds into speculative multi-channel work without the user's call. The mind arc is whole, the
  substrate is leveled up, the last negative is understood; this is a legitimate place to ask "keep climbing, or call it?"
- AMBITION CRITIC (after round 29 — multi-channel built, motion still walled): the user chose to build multi-channel and it
  was built honestly + searched hard — and it DIDN'T crack motion, which is itself decisive: the same 0.1R wall with or without
  multi-channel PROVES the constraint is the gradient flow, not the channel count. Two advection schemes (upwind + reintegration)
  ruled out diffusion too. So the mobile creature is now an EXHAUSTIVE numpy negative — there is no more ungated numpy lever to
  pull (I tested the obvious ones: substrate, kernel asymmetry, rotated flow, multi-channel coupling, less-diffusive advection,
  proper GAs). This is REAL diminishing returns, demonstrated not asserted. The honest next move is to surface the binary to the
  user: (a) GATED differentiable Lenia (gradient-descend through the dynamics to FIND a glider — the method the Flow-Lenia paper
  actually used, needs torch/JAX = a dependency), or (b) CONSOLIDATE and call the project complete-in-kind. Continuing to grind
  numpy motion variants would now be busywork against a proven wall. The whole arc (mind + integration + open-endedness +
  leveled-up substrate + an exhaustively-explained creature negative) stands as a defensible, honest body of work.
- AMBITION CRITIC (after round 30 — social intelligence opened): the right read of the motion wall was NOT "consolidate or
  go gated" — it was "the MOTION sub-goal is walled, but the CORE goal (real intelligence) has whole unexplored dimensions
  that are ungated." R30 proved it: emergent communication (a recognised hallmark of intelligence) landed cleanly in numpy,
  reopening the intelligence frontier into SOCIAL/multi-agent. Lesson: when one sub-frontier walls, don't default to gated
  leaps or consolidation — check whether the PRIMARY goal has other ungated dimensions. It did (and likely still does:
  compositional language, grounded/embodied communication, multi-agent coordination, theory-of-mind, curiosity/intrinsic
  motivation, cultural transmission). So returns are NOT diminishing on the core goal; round 31 should push social
  intelligence further (compositionality or grounding) — a richer, ungated frontier. The mobile creature stays a gated,
  honestly-parked negative; it is not the only way forward and should not gate the whole project.
- AMBITION CRITIC (after round 31 — social vein deepening, review overdue): R30 opened social intelligence; R31 deepened
  it (compositionality emerges under structural pressure) — both clean, ungated, on the CORE goal, confirming the core goal
  has a rich ungated vein far from exhausted. But 31 rounds in with the last FULL review at R26, the project is overdue a
  CONSOLIDATION pass (re-verify recent rounds, refresh the capstone to include the Flow-Lenia substrate + the social rounds
  + the exhaustively-explained motion negative; prune). So round 32 = MILESTONE REVIEW, then resume the social vein
  (grounded comm, iterated learning, coordination). Motion stays parked-gated. Balance: push the live frontier, but
  consolidate before the docs/claims drift from the (now large, 31-round) reality.
