# STATUS — genesis

GOAL: in a sandbox world, grow real intelligence; derive the world's evolution;
scale the same engine 1D -> 2D -> 3D. (open /loop direction, evolving mode)

MODE: evolving / frontier. A cleared bar is a floor, not the finish.

ROUND: 36 (MILESTONE REVIEW — social rounds re-verified + capstone refresh; complete, committed)

REVIEW (round 36): re-verified the SOCIAL rounds (R33/34/35) on FRESH unseen seeds -> ALL HOLD. R33 grounded comm
(seeds 5-6): catch WITH comm 0.71 vs ablated 0.03 (committed 0.58/0.05 -> even stronger). R34 iterated learning
(seeds 7-9): topo bottleneck 0.28 vs full 0.10 (the bottleneck-vs-full gap holds). R35 theory of mind (seed 3):
observer 0.78 vs ablated 0.25 = chance (holds). 99 tests green; all images resolve; no orphan modules. CORRECTED an
over-claim: theory_of_mind.py docstring said the observer "beats a position-based naive heuristic" -> it does NOT
(a position oracle is as good); fixed to the honest scope. REFRESHED docs/CAPSTONE.md (25->31->NOW 35 rounds): the
SOCIAL thread now spans all 5 social rounds (emergent/compositional/grounded/iterated comm + theory of mind);
review list updated to 7 (10/17/20/23/26/32/36); README capstone pointer 31->35. NOTE: repo .git 88M / outputs 56M
(growing; inherent to the visual README; watch it).

REVIEW (round 32): re-verified the post-R26 rounds on FRESH seeds -> ALL HOLD. R27 Flow-Lenia: mass conserved
EXACTLY (ratio 1.00000) 2D AND 3D on a fresh config. R30 emergent comm (fresh seeds 5-7): accuracy 1.00, 2.00
bits (ceiling), ablated 0.25 (chance) -> robust. R31 compositional (fresh seeds 5-6): topo naive 0.33 ->
pressured 0.80 -> compositionality emerges robustly. 93 tests green; all images resolve; no orphan modules;
honest negatives intact. REFRESHED docs/CAPSTONE.md (was stale at 25 rounds): now 31 rounds, adds the social
dimension + Flow-Lenia substrate + the EXHAUSTIVELY-explained motion negative; README pointer + intro updated.
NOTE: repo .git 80M / outputs 55M (growing; inherent to the visual README; ~33 round figures + GIFs).

REVIEW (round 26): capstone consolidation. R25 re-verified on a FRESH seed (11): the NEGATIVE is ROBUST —
even the motion search's best is DIFFUSE (conc 0.40), no compact-AND-moving creature appears. The POSITIVE
(compact 3D creature) has SEARCH VARIANCE — committed seeds 0/7 reach conc 1.00 (and the saved genome
reproduces conc 1.00), but a SHORTER run on seed 11 reached only conc 0.39, so the compact creature is
FINDABLE but not on every seed/budget (honest nuance). Wrote docs/CAPSTONE.md (the definitive "what is this"
+ honest negatives + the gated next-step decision); README points to it. 80 tests green; all images resolve;
no orphans. Repo .git 67M / outputs 52M (growing; inherent to the visual README).

REVIEW (round 23): re-verified R21-22 on FRESH unseen seeds -> BOTH HOLD (not cherry-picked). R21 ablation
ladder on seeds 30-49: full 288 > memory_only 192 > no_memory 157 (committed 263/188/140 on seeds 0-19 —
reproduces directionally). R22 on fresh seed 2: MAP-Elites fills 54/64 niches, fitness-GA diversity collapses
to 8 (reproduces exactly). 75 tests green; all images resolve; no orphan modules; honest negatives/caveats
(R18 seed-sensitivity, R21 hand-wired + bounce overshoot, R22 foam + bounded) all intact. NOTE: repo growing
(.git 62M, outputs 51M) from committed GIFs/PNGs — inherent to the visual README; acceptable, watch it.

REVIEW (round 20): re-verified R18-19. R19 reproduces EXACTLY (pursuit 54 vs planner 24; recurrent 0.45 vs
feedforward 0.05). R18 HOLDS ON AVERAGE but is the project's MOST SEED-SENSITIVE result: a quick under-
trained probe FLIPPED it (recurrent 0.2 < feedforward 1.0); careful re-check (4 ES seeds, committed settings)
gives recurrent mean 1.90 vs feedforward 1.21, recurrent wins 3/4 -> real moderate effect, NOT the clean 2x
the single figure showed. CORRECTED overclaims: montage caption ("collects nothing" -> "~1.2") + README
(added honest seed-sensitivity caveat) + whole-arc capstone intro with honest-scope box. 69 tests green; all
images resolve. Earlier reviews (R10: rounds 1-9; R17: rounds 11-16) still hold.

REVIEW (round 10): re-verified rounds 1-9 -> all hold. REVIEW (round 17): re-verified rounds 11-16 ->
ALL HOLD. R11 93%/0.89-vs-0.52; R12 0.69 bits + envelope; R13 changing->1.00 / stable->0.44; R14 lr
0.48->0.59; R15 1.00-vs-0.53 (1 bit); R16 1.00-vs-0.50 (1 bit). R15/R16 robust on FRESH unused seeds
(recurrent ~1.0, feedforward ~chance -> not cherry-picked). 62 tests green; no orphan modules; all README
images resolve; honest negatives intact. NOTE: repo growing (.git ~50M, outputs ~46M) from committed
GIFs/PNGs — inherent to the visual README; acceptable, watch it.

CURRENT STATE:
- Dimension-agnostic Lenia engine (genesis/world.py): N-D field, FFT radial-kernel
  convolution + growth; same code 1D/2D/3D. 99 tests green.
- Emergence metrics (genesis/metrics.py): alive/localized/persistent/locomotion +
  scale-aware mass-concentration & gyration (creature vs soup), wrap-aware centroid.
- Evolution (genesis/evolve.py): co-evolves RULE + evolvable SEED MORPHOLOGY (patch).
- Foraging+metabolism (genesis/foraging.py): ForagingWorld = Lenia body + food field +
  sensorimotor reflex (sense food gradient -> rigid mass-exact np.roll drift) + eating +
  a scalar ENERGY reserve (metabolism drains it, eating refills it, 0 -> body dissipates).
- ROUND 1 (1D): self-organisation into a persistent homeostatic lattice; motion transient.
- ROUND 2 (2D): EVOLVED a genuine glider DE NOVO (travel 3.78w, straightness 0.99).
- ROUND 3 (2D): AGENCY. Evolved forager eats 85% vs 18% ablated vs 18% random on UNSEEN
  layouts -> the win is food-directed sensing, not motion.
- ROUND 4 (2D): SURVIVAL. Metabolism on; food spawns; the creature DIES without eating.
  On UNSEEN schedules the evolved forager lives ~832 vs 158 ablated vs 299 random (cap 900).
- ROUND 5 (3D, VERIFIED): DIMENSIONAL GENERALITY. The IDENTICAL engine runs a 3D world and
  self-organises a seed into a ROBUST, persistent 3D structure (tail mass_cv=0.0008, final
  mass ~22150 reproducible across 3 seeds) — the 3D analogue of round 1's lattice. Completes
  the 1D->2D->3D engine arc. HONEST NEGATIVE (ADVANCED in R25): a compact MOBILE 3D creature is NOT
  delivered. (R25 found a stable COMPACT 3D creature — conc 1.00 — but it does not move; the mobile 3D
  glider remains the open negative.) Evidence: outputs/round5_3d.png + .gif; genesis/evolve3d.py.
- ROUND 6 (2D, VERIFIED): ECOLOGY. Many creatures (each a Lenia body in its own channel)
  compete for ONE shared, finite, replenishing food field; each has energy/metabolism +
  the foraging reflex (genesis/ecology.py). Result: STABILIZING SELECTION — survival peaks
  at INTERMEDIATE foraging gain (~4), with under-foragers (gain 0) starving first and
  over-foragers (gain 16) overshooting; robust across food-scarcity regimes (spawn 38/50/65,
  optimum=4 in all). Evidence: outputs/round6_ecology.png + .gif. run6.py.
- ROUND 7 (2D, VERIFIED): EVOLUTION RUNNING. genesis/evo_ecology.py — foraging gain is a
  HERITABLE gene; well-fed creatures REPRODUCE (offspring inherit gain + mutation), starving
  ones die. Under scarce food (food-limited population), from RANDOM gains the population
  self-tunes: mean gain 9.2 -> 3.5 across 3 runs, MATCHING round 6's independently-swept
  optimum (~4). Selection is happening, not measured. Evidence: outputs/round7_evolution.png
  (convergence curves + start/evolved gain histograms) + .gif. run7.py.
- ROUND 8 (2D, VERIFIED): PREDATOR-PREY. genesis/predprey.py — two species in one world:
  prey (forage food + flee predators) and predators (hunt+eat prey via a field interaction);
  both have energy/metabolism/reproduction + a heritable strategy gene. Results: (1) TOP-DOWN
  REGULATION — predators hold prey ~14 vs ~55 without predators (4x suppression), prey shows
  boom-bust fluctuations; (2) COEXISTENCE for the full run; (3) CO-EVOLUTION (honest surprise)
  — prey evolve LOWER evasion (7->3), not higher: fleeing costs more foraging than it saves, so
  "forage hard, out-breed predation" wins; predator pursuit stable (~10). HONEST NEGATIVE: no
  clean escalating arms race — across 5 regimes evasion was always selected down and the
  balance is knife-edge (strong predation -> prey extinction; moderate -> predator-cap-pinned
  coexistence). Evidence: outputs/round8_predprey.png + .gif.
- ROUND 9 (VERIFIED): WITHIN-LIFETIME LEARNING. genesis/learning.py — a creature with a PLASTIC
  value-learning brain on a REVERSAL-learning task (two food sources, only one nutritious, the
  rule flips every 320 steps). The learner tracks the rewarding source and RE-LEARNS after every
  reversal: accuracy 0.87 vs 0.49 for the SAME creature with plasticity off (lr=0, chance). This
  is adaptation within ONE life, not across generations — the missing ingredient. Pure numpy (no
  torch, avoided the dependency gate). Evidence: outputs/round9_learning.png (accuracy-over-life
  with reversal dips+recoveries + learner-vs-ablated bars) + .gif. run9.py.
- ROUND 11 (VERIFIED): EMBODIED LEARNING. genesis/embodied_learning.py — the plastic brain now lives
  INSIDE a Lenia creature: it forages two food types, its drift is a PLASTIC policy
  w_A*grad(F_A)+w_B*grad(F_B), only one type nutritious, rule flips mid-life, food evaporates if
  uneaten (so it must actively forage). The drift weights flip with the rule (argmax(w)==active 93%)
  and the embodied learner eats 0.89 nutritious vs 0.52 for the same body with plasticity off. Body
  (R2) + agency (R3) + learning (R9) united in ONE creature. Evidence: outputs/round11_embodied.png
  (weight-flip trajectory + nutritious-fraction curve) + .gif. run11.py.
- ROUND 12 (VERIFIED): MEASURING THE MIND. genesis/measure.py — quantify intelligence as the mutual
  information I(brain;world) in BITS between the creature's internal state (which food its brain
  prefers) and the world's hidden variable (which food is nutritious). Result: learner 0.69 bits vs
  ablated 0.00 bits. OPERATING ENVELOPE (MI vs reversal interval): 0.17 bits (fast world, flip 100)
  -> 0.88 bits (slow world, flip 1600) -> the rate of change the mind can track, approaching the
  1-bit ceiling. The mind MEASURED, not asserted. Evidence: outputs/round12_measure.png (MI bar +
  envelope) + .gif. run12.py.
- ROUND 13 (VERIFIED): LEARNING UNDER SELECTION. genesis/evo_learning.py — learners (plastic brains)
  and fixed-reflex creatures (non-plastic inherited strategy) COMPETE in one world for two food types
  with 'is_learner' heritable; the nutritious type reverses at a tunable rate. Result (evolution of
  learning / Baldwin): in a CHANGING world the learner fraction RISES 0.5 -> 1.00 (learning wins); in
  a STABLE world it FALLS 0.5 -> 0.39 (fixed reflexes win — learning's startup cost buys nothing). The
  change-rate sweep shows a sharp transition (~interval 1000-2000): learning pays ONLY in worlds that
  change within a lifetime. Knowing more (R12) makes a creature win — conditionally. Evidence:
  outputs/round13_selection.png + .gif. run13.py.
- ROUND 14 (VERIFIED, mixed): THE BALDWIN EFFECT. genesis/baldwin.py — the LEARNING RATE is a
  heritable gene; creatures are born naive and learn at their own inherited rate, mutated+selected.
  POSITIVE: from RANDOM rates the population self-tunes to ~0.57 (concentrated high) -> evolution
  discovers a good amount of plasticity (favouring fast learning in a changing world); the Baldwin
  effect operates on the rate. HONEST NEGATIVE: the evolved rate does NOT track the world's change-
  rate (tested flip 80..3000, +/- reward noise -> ~flat / no clean gradient). The clean Bayesian
  volatility->learning-rate law is washed out by the embodied foraging dynamics. Evidence:
  outputs/round14_baldwin.png + .gif. run14.py.
- ROUND 15 (VERIFIED): A BRAIN WITH MEMORY. genesis/memory_brain.py — a tiny RECURRENT controller
  (hidden state, evolved by an ES, pure numpy) on a cue-recall task: a cue appears, vanishes, and
  after a random delay the creature must act on it. Result: recurrent 1.00 accuracy + holds 1.00 BIT
  of memory across the delay; feedforward (memory ablated) 0.53 (chance) + 0.00 bits — it literally
  cannot, the cue is gone when it must act. Memory unlocks a task class impossible for a reflex, and
  the memory is both MEASURED (1 bit, via measure.py MI) and VISUALISED (the held hidden-state trace).
  Fast (~6s, no FFTs). Evidence: outputs/round15_memory.png (ES curves + memory trace) + .gif
  (watch the cue set the memory ... fire the correct action). run15.py.
- ROUND 16 (VERIFIED): A BRAIN THAT PREDICTS. genesis/predict_brain.py — a recurrent net (ES, numpy)
  must predict the next state of a periodic AMBIGUOUS world (pattern [0,0,1,1]: a 0 leads to 0 at one
  phase and 1 at another, so you must track phase + model the pattern). Result: recurrent 1.00 accuracy
  + 1.00 BIT of PREDICTIVE INFORMATION I(brain;next-state), anticipating EVERY flip; feedforward
  (reactive) 0.50 + 0.00 bits. The brain builds a model and FORESEES the world. run16.py.
- ROUND 18 (VERIFIED): EMBODIED MEMORY — the two tracks reunited + payoff proven. genesis/embodied_memory.py
  — a recurrent brain (ES, numpy) drives a LENIA forager's drift; the world FLASHES (food gradient visible
  briefly, then dark while the food persists). Result (food collected/episode, averaged over 2 ES seeds,
  unseen eval seeds): recurrent-flashing 1.9 vs feedforward-flashing 1.0; feedforward-STEADY control 3.4.
  So flashing CRUSHES the memoryless forager (3.4 -> 1.0, -70%) and memory roughly DOUBLES it (-> 1.9),
  recovering much of the loss -> memory PAYS exactly when the world hides information across time. The
  embodied (rounds 3-13) and deep-mind (rounds 15-16) tracks are now ONE creature. Evidence:
  outputs/round18_embodied_memory.png + .gif. run18.py.
- ROUND 19 (VERIFIED): PLANNING — acting on foresight (the mind's capstone). genesis/planning.py — a
  target ORBITS on a circle; an agent must intercept. REACTIVE pursuit heads at where the target IS and
  tail-chases around the circle; a model-based PLANNER heads where it WILL be and cuts across. Result:
  planner catches in 24 steps vs reactive pursuit 54 (~2.2x faster), with the classic pursuit-curve-vs-
  interception trajectories. And an EVOLVED RECURRENT controller (sees only relative position, must infer
  the motion) LEARNS to anticipate: catch rate 0.45 vs 0.05 for a feedforward (reaction-only) controller.
  Completes the mind's core loop perceive->model->predict->ACT-TO-ACHIEVE. (Honest: ES doesn't reach the
  hand-coded planner; interception is hard to learn from scratch, but recurrent >> feedforward 9x.) run19.py.
- ROUND 21 (VERIFIED): UNIFICATION — one creature, all four faculties, proven load-bearing by ablation.
  genesis/unified.py — a Lenia creature must stay on food that MOVES (constant-velocity, bouncing) and FLASHES
  (visible briefly then dark) under METABOLISM, so tracking it is survival. ONE integrated controller:
  perceive -> REMEMBER (hold the estimate across dark) -> PREDICT (dead-reckon by remembered velocity) ->
  PLAN/ACT (drive the Lenia drift to the lead/intercept). Ablation ladder (steps survived, max 320, 20 seeds):
  full 263 > memory_only 188 (no prediction -> lags) > no_memory 140 (stalls in the dark). Each faculty earns
  its place. Body (3-13)+memory (15/18)+prediction (16)+planning (19) are now ONE organism. HONEST: controller
  is hand-integrated (faculties were shown EVOLVABLE earlier; this proves they COMPOSE + each is necessary);
  the lead occasionally overshoots at a bounce so full isn't best on every seed (best on avg + 10/16 seeds).
  Fast (~4s, deterministic). Evidence: outputs/round21_unified.png (ablation bars + energy-over-life) + .gif. run21.py.
- ROUND 22 (VERIFIED): OPEN-ENDEDNESS — does the world keep GENERATING, or converge? genesis/openended.py —
  MAP-Elites ILLUMINATES a 2-D behaviour space (drift-speed x body-mass), keeping the most VIABLE creature in
  each niche; vs a fitness-GA baseline. Result: MAP-Elites fills 54/64 niches (84%) across the FULL move x size
  range, and diversity ACCUMULATES (coverage climbs 24->54 over evaluations); the fitness GA's population
  diversity COLLAPSES/stays ~8 as selection converges to one body type. So the substrate is GENERATIVE across
  this behaviour space, not single-attractor. HONEST: the "zoo" spans a clean small glider (mass ~140) through
  large multi-blob "foam" textures (high mass) — the behaviour space is filled with viable PATTERNS, not all
  discrete single organisms; and the map is bounded (no claim of UNBOUNDED open-endedness). Evidence:
  outputs/round22_openended.png + .gif. run22.py.
- ROUND 24 (VERIFIED): OPEN-ENDED MINDS — a zoo of foraging STRATEGIES, not body shapes. genesis/openmind.py —
  same body (the evolved glider), but the MIND is a 4-parameter foraging policy (sensing range, gain, exploration,
  momentum); the creature forages scattered food. MAP-Elites illuminates the foraging-BEHAVIOUR space (how much it
  ROAMS x how WIDELY its path SPREADS), keeping the best-eating forager per niche; vs a fitness GA. Result:
  MAP-Elites fills 28 niches (of the reachable ~30; low-roam can't be high-spread) with VISIBLY DISTINCT
  trajectories — compact efficient foragers, sprawling rovers, columnar pacers — while the fitness GA's population
  diversity collapses to ~5. So open-endedness holds for MINDS too, not just morphologies. HONEST: the reachable
  behaviour region is a triangle (~30/64 is near max, not 64); and a compact path can eat as much as a long rover
  (small world) -> the diversity is in STYLE, not all in success. Evidence: outputs/round24_openmind.png (strategy
  map + coverage curve + TRAJECTORY gallery) + .gif (4 minds foraging). run24.py.
- ROUND 25 (VERIFIED, mixed): THE 3D CREATURE — a serious push at the R5 negative. genesis/creature3d.py upgrades
  R5's search with (1) MULTI-RING kernels (kernel_peaks), (2) a SHAPED viability gradient (partial credit for healthy
  mass + compactness + persistence, so the GA can climb where the round-2 fitness hard-gates on `alive` and gives no
  gradient), (3) a motion reward. POSITIVE: a STABLE COMPACT 3D creature is found — a single localised body,
  concentration 1.00, mass ~3162 (an upgrade on R5's diffuse self-organisation). HONEST NEGATIVE (sharpened): a
  MOBILE 3D glider is still NOT found. The compactness x motion scatter shows exactly why — compact bodies are
  STATIONARY attractors, the structures that MOVE (travel ~0.5) are DIFFUSE (conc ~0.3); the glider is the unmet
  INTERSECTION (compact AND moving), and that corner is empty. Evidence: outputs/round25_creature3d.png (3 views of
  the compact creature + the compact-OR-moving trade-off scatter) + .gif (rotating 3D creature). run25.py.
- ROUND 27 (VERIFIED): FLOW-LENIA — a mass-conserving substrate, pure numpy (genesis/flowlenia.py). Plain Lenia
  grows/clips mass in place (not conserved -> moving bodies dissipate, 3D is knife-edge). Flow-Lenia MOVES mass
  along the affinity gradient (F = grad(growth(U)), clipped) with mass-conserving UPWIND advection. Results: mass
  conserved EXACTLY (ratio 1.0000) in 2D AND 3D; a seed self-organises into a COMPACT creature with an emergent
  orbium-like ring in 2D, and a ROBUST compact 3D creature (conc ~0.98 — where plain Lenia 3D died/foamed); and
  4 creatures COEXIST in one world sharing the conserved mass. HONEST: a MOBILE creature is NOT delivered here
  either (symmetric radial kernel -> stationary attractors; an offset kernel did not unlock motion); but it is now
  on a substrate where a moving body cannot dissipate. Additive module — plain Lenia (world.py) untouched.
  Evidence: outputs/round27_flowlenia.png (2D creature + mass-conservation curve + 3D creature + multi-creature) +
  .gif (4 creatures, mass conserved). run27.py.
- ROUND 28 (VERIFIED, negative w/ diagnosis): WHY THE CREATURE WON'T MOVE. genesis/creature_flow.py — a proper
  round-2-style GA over Flow-Lenia rule + kernel asymmetry + EVOLVED SEED SHAPE, rewarding directed motion of a
  compact body. Result: evolution pushes net travel up from ~0.06R (random) to ~0.2R (evolved) but PLATEAUS far
  below locomotion (round-2 plain-Lenia glider: 3.78 widths ~ 7.5R). Also ruled out (probes): asymmetric kernels,
  rotated (chiral) flow, and multi-creature competition (creatures COEXIST inertly, no mass transfer). DIAGNOSIS:
  single-channel Flow-Lenia moves mass by F = grad(G(U)) — a GRADIENT flow, which is curl-free, so it can only
  RELAX to a stationary equilibrium; sustained locomotion needs a NON-gradient flow (= MULTI-CHANNEL Flow-Lenia,
  the paper's glider mechanism). This resolves the recurring motion negative (R25/R27/R28) with a precise mechanism.
  Evidence: outputs/round28_motion_diagnosis.png (GA curve vs locomotion bar + compact-but-stationary creature +
  centroid-stays-put trajectory) + .gif. run28.py.
- ROUND 29 (VERIFIED, negative): MULTI-CHANNEL FLOW-LENIA (the user's chosen direction, via AskUserQuestion).
  genesis/flowlenia_mc.py — FlowWorldMC: C channels, each advected by its OWN affinity gradient, COUPLED via
  cross-channel kernels (offsets/asym/weights); EACH channel conserves its own mass. POSITIVE: built + works —
  structured 2-channel coupled creatures form and conserve mass (unit-tested). NEGATIVE: a serious GA over the
  coupling (22 gens) reached only 0.11R travel, AND a less-diffusive REINTEGRATION-TRACKING advection (tested) also
  gave ~0. So motion is the SAME wall with or without multi-channel -> the binding constraint is the GRADIENT FLOW
  (F=grad(G) relaxes to a stationary equilibrium), NOT the channel count or advection scheme. Translation needs a
  self-sustaining asymmetric (uniform-drift) attractor these numpy formulations don't reach. The mobile creature is
  now an EXHAUSTIVELY-tested, well-explained negative (plain Lenia + Flow-Lenia single/multi-channel + 2 advections).
  Evidence: outputs/round29_multichannel.png + .gif. run29.py.
- ROUND 30 (VERIFIED): EMERGENT COMMUNICATION — the first SOCIAL-intelligence round (the mind arc was single-agent).
  genesis/communicate.py — a SPEAKER sees a hidden referent (1 of K=4) and emits a continuous 2-D SIGNAL; a LISTENER
  sees only the signal and must name the referent. Neither is given a code; both are evolved JOINTLY (OpenAI-ES, numpy).
  A shared language EMERGES: listener accuracy climbs 0.25 (chance) -> 1.00; the K signals separate into distinct
  "words" in signal space; realised information I(referent; listener-action) reaches 2.00 bits = log2(K) ceiling.
  ABLATION (feed the listener a random signal): accuracy -> 0.23 (chance), I -> 0.01 bits -> the channel is genuinely
  USED. Pivots from the walled motion frontier back to the CORE goal (real intelligence), extending the arc into the
  SOCIAL dimension. Evidence: outputs/round30_communication.png (accuracy curve + evolved code scatter + bits bars) +
  .gif (the 4 signals separating into a code over evolution). run30.py.
- ROUND 31 (VERIFIED): COMPOSITIONAL COMMUNICATION — does the emerged language factorise? genesis/communicate_comp.py
  — referents now have TWO attributes (3 shapes x 3 colours); speaker conveys both, listener decodes both (two heads);
  some combos HELD OUT for a zero-shot test. NAIVE emergent comm (round-30 trick) is HOLISTIC: train accuracy 1.00 but
  held-out zero-shot 0.00 and topographic similarity only ~0.25 (it MEMORISES each meaning, doesn't build from parts).
  Add a STRUCTURAL pressure (reward topographic similarity) and the language becomes COMPOSITIONAL: topo 0.25 -> 0.79,
  with partial zero-shot generalisation (0.00 -> 0.33). So compositionality (the hallmark that lets finite parts express
  infinite meanings) is NOT free from communicative success alone — it emerges under a learnability/structure pressure.
  A recognised emergent-language result, replicated in numpy. Evidence: outputs/round31_compositional.png (topo bars +
  naive-scrambled vs pressured-structured signal scatter) + .gif (signals organising under pressure). run31.py.
- ROUND 33 (VERIFIED): GROUNDED COMMUNICATION — a signal that drives ACTION (embodied + social fused).
  genesis/communicate_grounded.py — a SCOUT sees where food is (direction theta) and emits a signal; a BLIND
  FORAGER sees only the signal and must navigate to the food. The pair is evolved JOINTLY (ES). Result (catch
  rate, unseen episodes): sighted upper bound 1.00 | comm pair 0.58 | ablated (random signal) 0.05. So the
  emitted signal carries ACTIONABLE SPATIAL information the body uses to forage — with the channel the blind
  forager reaches food, ablated it is lost. Trajectory visual: WITH comm the forager steers to the food star;
  ablated it scatters randomly. Fuses the embodied track (navigation) with the social track (signalling) — the
  first communication that DOES something in the world. Evidence: outputs/round33_grounded.png (catch-rate bars
  + with-comm vs ablated trajectories) + .gif (foragers navigating, comm vs ablated side by side). run33.py.
- ROUND 34 (VERIFIED): ITERATED LEARNING — compositionality from cultural transmission (the principled "why" behind
  R31). genesis/communicate_iterate.py — each "generation" LEARNS the language (a meaning->signal map, a tanh-MLP
  trained by hand-coded numpy backprop) from only a SUBSET of meanings (the learnability BOTTLENECK), then PRODUCES
  the whole language (generalising to unseen meanings); a light expressivity rescale prevents collapse. Result over
  ~45 generations (mean of seeds): under the BOTTLENECK (5 of 9) topographic similarity rises ~0 -> ~0.3 (peak 0.4);
  under FULL transmission (9 of 9) it stays flat ~0.0. So compositional structure emerges from transmission ALONE
  (holistic codes can't be reconstructed from few examples and degrade) — NO hand-added structure term (unlike R31).
  HONEST: the effect is ~0.3 (more modest than R31's hand-pressured 0.79) and fluctuates across seeds (a stochastic
  process); the bottleneck-vs-full contrast (~0.3 vs ~0.0) is the robust claim. Evidence: outputs/round34_iterated.png
  (topo-over-generations bottleneck vs full + the structured transmitted language) + .gif (language organising over generations). run34.py.
- ROUND 35 (VERIFIED): THEORY OF MIND — inferring another agent's hidden GOAL from its behaviour (a social axis
  distinct from communication). genesis/theory_of_mind.py — an ACTOR moves toward one of K=4 goals with HEAVY noise;
  an OBSERVER (recurrent net, evolved by ES) sees only the actor's step-by-step MOTION (displacements, no absolute
  position) and must infer WHICH goal. Result: goal-inference accuracy rises 0.48 -> 0.84 as it watches (reads intent
  PROGRESSIVELY); the belief SHARPENS on the true goal (mean P(true) 0.43 -> 0.81, P(best-wrong) 0.56 -> 0.19);
  ABLATED (random motion) -> 0.29 ~ chance (0.25). So the observer mentalises intent from behaviour, integrating noisy
  motion into a belief. HONEST: inferring 'which target an agent walks toward' is partly geometric (a position oracle
  also solves it); the genuine result is the LEARNED belief-updating from motion-only + the ablation dependence (the
  harder 'behaviour that MISLEADS', e.g. detours/obstacles, is a future frontier). Evidence: outputs/round35_theory_of_mind.png
  (accuracy-by-step + belief-sharpening + trajectories) + .gif (actor moving while the observer's belief bars update). run35.py.

NEXT ROUND SEED (round 37 — review done, resume building): social vein ungated. Ranked:
  (a) MULTI-AGENT COORDINATION (leading): a task only solvable TOGETHER (it-takes-two food / division of labour) -> coordinated
      pair beats solo; or COMBINE threads: comm-enabled coordination (agents agree where to meet via the channel, ablate -> fail).
  (b) HARDER theory-of-mind where behaviour MISLEADS (actor detours around an obstacle; modeling beats the position oracle) ->
      this is the genuine ToM the R35 review flagged as the future frontier.
  (c) Iterated/cultural dynamics over the GROUNDED language; or a new non-social frontier if the social vein thins.
  NOTE: numpy motion is a PROVEN wall; the mobile creature is a gated, honestly-parked negative -> do NOT grind it.

HOW TO RUN (drivers verified in round-10 review):
  cd ~/evolab/genesis
  .venv/bin/python -m genesis.run35 --gif     # round-35 theory of mind (infer hidden goal from behaviour) + gif (~15s)
  .venv/bin/python -m genesis.run34 --gif     # round-34 iterated learning (compositionality from transmission) + gif (~12s)
  .venv/bin/python -m genesis.run33 --gif     # round-33 grounded communication (scout guides blind forager) + gif (~10s)
  .venv/bin/python -m genesis.run31 --gif     # round-31 compositional communication (structure under pressure) + gif (~8s)
  .venv/bin/python -m genesis.run30 --gif     # round-30 emergent communication (two agents evolve a code) + gif (~6s)
  .venv/bin/python -m genesis.run29 --gif     # round-29 multi-channel Flow-Lenia (built; motion walled off) + gif (~10s)
  .venv/bin/python -m genesis.run28 --gif     # round-28 Flow-Lenia motion diagnosis (gradient flow relaxes) + gif (~20s)
  .venv/bin/python -m genesis.run27 --gif     # round-27 Flow-Lenia (mass-conserving substrate) + gif (~3s)
  .venv/bin/python -m genesis.run25 --gif     # round-25 the 3D creature (compact found, mobile negative) + gif (~7s)
  .venv/bin/python -m genesis.run24 --gif     # round-24 open-ended minds (foraging-strategy zoo) + gif (~6 min)
  .venv/bin/python -m genesis.run22 --gif     # round-22 open-endedness (MAP-Elites zoo) + gif (~5 min)
  .venv/bin/python -m genesis.run21 --gif     # round-21 unification (ablation of faculties) + gif (~4s)
  .venv/bin/python -m genesis.run19 --gif     # round-19 planning / interception + gif (~90s)
  .venv/bin/python -m genesis.run18 --gif     # round-18 embodied memory (recurrent brain in Lenia) + gif (~4 min)
  .venv/bin/python -m genesis.run16 --gif     # round-16 a brain that predicts + gif (~5s)
  .venv/bin/python -m genesis.run15 --gif     # round-15 a brain with memory + gif (~6s)
  .venv/bin/python -m genesis.run14 --gif     # round-14 Baldwin effect + gif (~4 min)
  .venv/bin/python -m genesis.run13 --gif     # round-13 learning under selection + gif (~6 min)
  .venv/bin/python -m genesis.run12 --gif     # round-12 measuring the mind (bits) + gif
  .venv/bin/python -m genesis.run11 --gif     # round-11 embodied learning + gif
  .venv/bin/python -m genesis.run9 --gif      # round-9 within-lifetime learning + gif
  .venv/bin/python -m genesis.run8 --gif      # round-8 predator-prey + gif
  .venv/bin/python -m genesis.run7 --gif      # round-7 evolution inside the ecology + gif
  .venv/bin/python -m genesis.run6 --gif      # round-6 ecology / competition + gif
  .venv/bin/python -m genesis.run5_3d --gif   # round-5 3D self-organisation + rotating gif
  .venv/bin/python -m genesis.run4 --gif      # round-4 survival (metabolism) + benchmark
  .venv/bin/python -m genesis.run3 --gif      # round-3 foraging agency + benchmark
  .venv/bin/python -m genesis.run2d --gif     # evolve a 2D glider, render strip+gif
  .venv/bin/python -m genesis.run1d           # round-1 1D self-organisation
  .venv/bin/python -m genesis.overview        # rebuild the progress montage
  .venv/bin/python -m pytest -q

VERIFICATION BAR: never trust tests/metrics alone — run the sim, LOOK at the field /
trajectory, confirm the metric matches the picture. (Round 2's soup proved why.)
