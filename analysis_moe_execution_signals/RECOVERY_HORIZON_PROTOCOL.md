# MoE-triggered short-horizon recovery pilot protocol

Frozen on 2026-09-05 before observing any horizon-intervention outcome.

Pre-formal runtime amendment: the implementation smoke was excluded after a
shared-cgroup OOM-counter increase. It also showed that reusing and hard-resetting
one LIBERO environment can change render-only model state. Formal collection
therefore creates a fresh seed-7 environment for every frozen state, matching the
source corpus's one-environment-per-episode path. This amendment was made for the
predeclared input-hash gate; the arms, states, endpoint, threshold, noise, and
analysis were not changed.

Invalidated-attempt note (2026-09-05 11:12 UTC): the first formal collection at
`horizon_recovery_runs/shard0` was stopped and excluded after unrelated concurrent
workers raised the shared-cgroup `oom_kill` counter from 18 to 20 and terminated
the policy server/client. It had written 29 complete states plus one incomplete
state. Those outcomes were not analyzed and did not change any arm, state,
endpoint, threshold, random seed, or statistical test. Clean collection is split
deterministically into three sequential shards on one maximum-power GPU as a
failure-isolation measure; sharding does not change any state or potential
outcome.

## Question

Can a local increase in observation and replanning frequency keep HiMoE-VLA out
of a task-8 static Trap, and does the routed/shared convergence signal identify
the states that benefit?

This is a recovery pilot, not a new detector fit. The MoE score, threshold,
states, restoration gates, intervention, budget, and tests below are fixed
before collecting either intervention arm.

## Frozen states and detector

- Use only `relative_query = -4` rows in `rich_event_plan.json`.
- A matched `pair_id` is eligible only when both its static-event row and its
  same-initial-state/same-query healthy-control row passed the restoration gates
  in the rich functional capture. This excludes pair 4 and leaves 23 pairs (46
  reconstructed states).
- The frozen score is `routed_shared_cosine`, averaged over the last four stored
  HB layers, all ten denoise steps, and action tokens 1-10.
- The alarm threshold is `score > 0.2945967216491699`, the linear empirical
  90th percentile of the 23 eligible healthy-control scores. It alarms on 19/23
  event rows and 3/23 control rows. This threshold is descriptive and
  task-specific because it was calibrated on these controls; recovery outcomes
  are not used to set it.

## State reconstruction and pairing

For each row, create a fresh seed-7 LIBERO environment at its task initial state,
execute the standard ten dummy settling actions, and replay every recorded
action before the selected query. This reconstructs the controller's hidden
goal/interpolator state. Then pin MuJoCo to the saved query-boundary state and
save a full branch snapshot, including controller state, episode clock, and
`qacc_warmstart`.

Camera observables are disabled only between policy-query boundaries during
settling, replay, and action execution, then re-enabled and force-refreshed at
every policy query. This avoids rendering unused intermediate frames. The
query-boundary images must match the frozen rich inputs bitwise; simulator
dynamics, task predicates, controller updates, and executed actions are not
skipped.

Both intervention arms restore that same full snapshot. Initial simulator,
controller-snapshot, policy-observation, and first-query noise hashes must match
between arms. A state that is already successful at the branch point, cannot be
restored exactly in MuJoCo, or cannot complete both arms is excluded as a whole
matched state; no exclusion may depend on an arm's outcome.

The source archives do not contain the original controller state, so replay is
the closest reconstructable controller state rather than a bitwise historical
snapshot. The causal comparison between the two newly run arms remains exact.

## Intervention and common random numbers

Every state receives both potential outcomes in a deterministic, balanced arm
order:

| Arm | Executed actions per policy query |
|---|---|
| `h10` | 10 throughout |
| `h2_burst10` | 2 while physical offset is below 10, then 10 |

Thus `h2_burst10` makes five observations/plans over the first ten physical
steps and then returns to the released policy's horizon. It changes neither
weights nor router decisions and uses exactly the same 200-physical-step
opportunity window as `h10`.

At offset zero, both arms use the row's original recorded flow noise so the
frozen MoE score and initial action are aligned. Later noise is generated from
`SeedSequence([20260905, pair_id, physical_offset])`. Noise at offsets shared by
two arms is therefore identical; the extra offsets 2, 4, 6, and 8 exist only in
the short-horizon arm. Event and control rows within a pair also share future
noise tensors. There is one common-random-number repeat in this pilot.

## Endpoints

The primary binary endpoint is LIBERO task success within 200 physical actions
after the branch. Success terminates an arm immediately. Query count and wall
time are costs, not alternative opportunity budgets.

The two primary estimands are:

1. event-state effect: `success(h2_burst10) - success(h10)`;
2. matched difference in differences: the event-state effect minus the
   healthy-control-state effect within each `pair_id`.

Report raw paired means and exact two-sided sign-flip/randomization p-values,
with a maxT correction across these two estimands. Report 95% pair-bootstrap
confidence intervals using 20,000 resamples and seed 20260906. With only 23
pairs, effect sizes and uncertainty take priority over a thresholded claim.

Secondary endpoints are actions and queries to success; success by 50, 100,
150, and 200 actions; and entry into the frozen physical recurrence rule after
subsampling both trajectories on the same 10-physical-action grid. Secondary
analyses are descriptive.

## Detector-to-recovery analyses

Using the already frozen alarm and the two observed potential outcomes, report:

- the event-row outcome of an alarm-gated policy (`h2_burst10` on alarm,
  otherwise `h10`) versus always `h10`;
- the same quantity on healthy controls, including false-alarm harm;
- always-short versus always-`h10`;
- whether selecting the alarmed event rows beats all equally sized random
  subsets of event rows (exact enumeration when feasible).

These analyses test treatment targeting, but remain internal pilot evidence:
the threshold and state set require prospective validation on new rollouts.

## Decision rule

- Positive event effect with positive difference in differences supports local
  replanning as a Trap-specific recovery candidate.
- Similar gains on event and control states mean shorter horizon is a generic
  policy improvement, not MoE-targeted recovery.
- No event gain means this ten-step local intervention is insufficient; it does
  not rule out a longer alarm cooldown or another recovery action.
- Control harm or query cost is reported explicitly and cannot be hidden by
  reporting only alarmed event rows.

## Runtime validity gates

- Checkpoint SHA-256 must be
  `cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256` and
  wrist layout must be `paper-right`.
- Selected GPUs must report power limit equal to their hardware maximum before
  model loading and after collection.
- No increase in cgroup `oom_kill` is allowed.
- Every completed state must have both arms, identical initial full-state and
  policy-input hashes, identical offset-zero noise, and bitwise-identical first
  predicted action chunks.
- Collection is resumable only at a whole-arm boundary; partial arm files are
  never analyzed.
