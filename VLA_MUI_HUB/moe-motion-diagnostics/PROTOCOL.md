# Motion and MoE diagnostics on the VLA hub

This is a retrospective descriptive experiment. The corpus and earlier reports
have already been inspected. The thresholds below are fixed before running this
new analysis, but this is not a preregistered study or a new blind holdout.

## Scope and clocks

Use every episode in the five `right-16x32` tasks, with exactly the episode order
of `moe-v4-0904/results/cache16x32_v4/episode_alarms.csv`: 2,560 episodes,
307 failures, 51,308 queries. Exclude duplicate `pin-base`, interventions, and
partial CALVIN captures. Group results by task and initial state; repeated noise
draws and repeated queries are not independent experimental replications.

`q` is a zero-based inference index. `state[q,:3]` is the measured end-effector
position before executing chunk q. `actions[q]` contains ten predicted commands.
The final chunk may execute only a prefix, so command descriptors are explicitly
prediction descriptors. Never treat all saved commands as executed observations.
The ten denoising iterations are a computation axis, not physical time.

No seconds or hertz are inferred. Physical pose has only one observation per ten
actions. Dense mechanical vibration cannot be established from these samples;
the command spectrum uses cycles per predicted action.

## Descriptors and fixed rules

All windows end at the current query. No event is backdated to the beginning of
its confirmation run. All flags require two consecutive qualifying queries and
can overlap. None is a failure annotation or a task-progress predicate.

* Stillness: mean xyz displacement over six transitions <= 1 mm/query AND
  <= 1/4 of the mean of the first four transitions. A zero baseline abstains.
  This measures end-effector stillness; gripper and object progress are unobserved.
* Backtracking: six-transition direction-reversal fraction >= 1/2 (adjacent
  displacements have cosine <= -0.3), net displacement / path <= 0.35,
  mean displacement >= 1 mm/query, and bounding-box diagonal >= 5 mm.
* Periodic return: in twelve transitions (thirteen observations), define
  D(k) = RMS Euclidean distance between observations k queries apart, k=1..6.
  Choose the minimum D(k)/D(1), k=2..5. Require both D(k)/D(1) <= 1/2 and
  D(k)/mean(D(k-1),D(k+1)) <= 1/2, plus the physical movement floors above.
  The window spans at least two candidate cycles. This requires actual return,
  not merely a curved distance-versus-lag curve. It does not test periods >5.
* Predicted-command jitter: xyz-command centered RMS >= 0.05 command units,
  Hann-periodogram power fraction in [0.3,0.5] cycles/action >= 1/2, and
  first-difference reversal fraction >= 1/2. Ten samples give coarse frequency
  resolution. This is neither measured vibration nor a calibrated failure rule.
* MoE periodic return: use the same return rule separately on front/back groups
  of final-denoising action-token square-root probabilities. Retain token and
  expert alignment. Distances are RMS Hellinger distances across tokens/layers.
  Require nonzero mobility and >=1/4 initial mobility so freezing is not a cycle.

Physical thresholds are transparent descriptive choices, not calibrated safety
limits. Report both successes and failures, all denominators, observation-window
eligibility, fixed-query controls, and missed cases. Do not optimize thresholds
using these outcomes or assert that failure implies oscillation.

## Existing methods

Compare frozen v7, v8, v8.2, and history-only back-layer half-mobility/K4.
Keep their fitted profiles and selected thresholds unchanged. Recompute both v8
heads on each unpadded episode prefix using existing repository functions, then
verify equality to cached first alarms after rejecting q >= episode length.
Record invalid cache counts. `np.nansum` on padding can otherwise invent
front/back inversion after the episode has finished.

Report all valid successful-episode false alarms. For failure lead >=4, use
`length - alarm_q >= 4` to retain the existing convention (including the action
chunk about to execute); do not apply that lead filter to the headline false
alarm denominator. Also provide the historical lead-filtered FP count for an
exact comparison. Do not call remaining time before timeout a physical warning.

## Timing and controls

For every phenotype, report alarms strictly before its first confirmation and
strictly before the earliest sample in that confirmation's evidence window.
The first quantity is only lead to recognition, not lead to physical onset; the
second is a conservative temporal comparison, still not proof of causality.
Use all phenotype episodes as the denominator, including those never alarmed.

At fixed q=7,13,19,31, compare only trajectories with an observed query q.
Compute within-task/initial-state AUC for specified continuous descriptors, with
larger scores always oriented toward failure except physical/MoE mobility, whose
direction is fixed negative. Give pair counts and eligible failures/successes.
These are descriptive multiple comparisons without significance claims.

Plot measured xyz, predicted commands, routing mobility, return lag curves and
valid alarm times. Select examples deterministically from phenotype-positive
episodes (median first-confirmation time, separately by outcome), and include a
same-task/same-initial-state opposite-outcome control where available.

After inspecting the first phenotype counts, add two descriptive diagnostics:
show the strongest sustained joint-threshold fraction when a phenotype has no
positive examples, and compute fixed-query within-task/init rank correlations
between pose step distance and route mobility, and between command high-frequency
power and v8 curvature. Center ranks within groups before pooling. These are
post-hoc visualization/association checks; they do not change the fixed rules,
are not independent confirmation, and do not establish causal direction.

## Follow-up requiring new evidence

To confirm physical vibration, replay saved simulator states with recorded
executed action prefixes and verify the next checkpoint before trusting dense
pose traces. Record per-action xyz, orientation, joint velocity, contacts and
object/goal progress. A dense replay is a new measurement, not provided by this
analysis. Only after separate data confirms a useful warning should an online
intervention compare unchanged policy, action smoothing, hysteresis, and v8-gated
replanning at matched triggers and initial simulator states.
