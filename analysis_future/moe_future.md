# MoE routing -> physical future, per-chunk resolution

Question: does the MoE routing latent predict the *continuous physical future* at
per-chunk resolution, beyond (tier 1) what a deployed system already has?
Tier 2 (privileged 47-dim sim_state) is recorded but demoted to a footnote: routing is
a function of the observation, so the Shannon answer there is known.

## Design (locked)
- Unit = (branch, query index k).  n = 16,158 (corpus A) / 22,883 (corpus B).
- Strict causality: every feature uses chunk k and k-1/k-2 only.
- CV = leave-one-group-out; group = worker (A, 4) / init_state_id (B, 16).
  Metric computed WITHIN each held-out group, pooled with equal weight (the four
  rolling-star workers have success rates 0/6/26/91%, so pooling raw is invalid).
- Same model class and capacity for every block: standardise -> PCA to 128 whitened
  components -> ridge with exact-LOO alpha over a fixed 21-point grid.
  Routing is pre-reduced (PCA 256 of the 2816-dim denoise-9 prob tensor, EVR = 0.983)
  which can only lose routing information.
- Soft probabilities only; no hard top-4 IDs anywhere.
- Controls get relative geometry, not raw coordinates:
  tier 1 = policy_state, aperture, eef->goal-landmark distance and signed offset,
  full 10x7 action chunk A_k and A_{k-1} with derived path/net/rotation/gripper terms,
  planned chunk endpoint and its distance to the landmark, 1- and 2-step deltas (251 dims).
  tier 2 = tier 1 + full 47-dim sim_state + both pot positions + all pairwise
  eef/pot/pot and pot/goal distances and signed offsets, aperture x distance
  interactions, planned-endpoint-to-pot distances, 1- and 2-step deltas (584 dims).

## Corpus fact that constrains the "vision-blind" argument
The scene layout is near-fixed across held-out groups.  Initial pot 1 sits at
(-0.05, 0.25) +- 0.013 m and pot 2 at (0.04, 0.05) +- 0.015 m in every one of the 16
corpus-B init states, while the pots are 0.17-0.25 m apart.  So a proprio-only model
CAN infer which pot the arm is near from eef position via a scene prior that
generalises across held-out init states.  "Which pot moves" is therefore NOT
structurally blind early in an episode.  It becomes blind once the pots have left
their initial poses, which is what `which_moves_late` isolates (valid only when some
pot has moved >5 cm from its own start).

## Target prevalences (m = 4)
| target | A | B |
|---|---|---|
| which_moves | n=5744, 69.4% pot2 | n=9843, 58.5% |
| which_moves_late | n=3509, 49.9% (balanced) | n=7069, 42.2% |
| obj_nc (pot moves, gripper open, arm >8cm away) | 647/14750 = 4.4% | 1119/20835 = 5.4% |
| obj_ncfar (>12cm from both pots) | 32 pos | 15 pos -> untestable |
| obj_gt_arm (pot moves further than arm) | 1789 = 12.1% | 4407 = 21.2% |
| undo | 19 pos | 19 pos -> untestable |
| undo_nc | 11 pos | 12 pos -> untestable |

Status: model grid running.
