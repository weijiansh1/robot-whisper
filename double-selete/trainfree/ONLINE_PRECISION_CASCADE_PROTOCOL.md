# Train-free high-precision routing alarm

Status: frozen before routing scores or outcome labels from
`right-50x8b-20260903` are read.

## Objective and evidence boundary

The deployment target is one live rollout. The detector emits an internal
`suspect` state and a latched binary alarm using only current and past HB-MoE
routing. It does not intervene.

The detector family and operating point were selected after outcomes from the
earlier `right-50x8-20260903` cohort had been inspected. That cohort is the
development set. The procedure is train-free at runtime and uses no outcome in
feature extraction or threshold calibration, but method selection is
outcome-informed. Its main evidential test is therefore the untouched `8b`
flow-seed replication below.

## Frozen external cohort

- Historical calibration: all 400 trajectories per task from
  `right-50x8-20260903`, flow seeds 1000--1007.
- External test: every complete task from `right-50x8b-20260903`, flow seeds
  1008--1015.
- Both campaigns use 50 initial states and eight flow draws per state. A test
  rollout is never part of its historical profile.
- One `8b` task, `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, has
  a non-empty `error` field in `meta.json` and is excluded before reading any outcome.
  The frozen test cohort is consequently 39 tasks and 15,600 rollouts.
- Completeness is decided only from run metadata (including absence of an
  `error` field) and route/index consistency.

The scorer may read only `episode_index`, `init_state_id`, `flow_noise_seed`,
and `inference_calls` from client summaries. It may not read `success`, final
labels, simulator state, images, or future queries.

## Route-mobility confirmation

For query `q`, use final-flow action-token routing from HB layers 12--15. Let

```text
m(q) = mean Hellinger(p(q), p(q-1))
       over four layers and ten action-token positions

u_W(q) = mean(m(q-W+1:q))

E_WK(q) = -max(u_W(q-K+1:q))
```

Thus `E_WK(q)` is high only when every one of the latest `K` smoothed mobility
values is low. All windows are trailing. No score is available until its full
history exists.

Frozen detectors are:

```text
mobility_w4_k1 = E_4,1
mobility_w4_k2 = E_4,2
mobility_w4_k4 = E_4,4        # primary hard-alarm evidence
mobility_w8_k3 = E_8,3        # legacy control
```

For each task and detector, compute one maximum causal score per historical
trajectory. Thresholds are higher empirical quantiles of these 400 maxima at
q95, q97.5, q99, and q99.5. Strict exceedance triggers and then latches the
alarm. The primary high-precision operating point is
`mobility_w4_k4@q99`; `q97.5` is the frozen balanced operating point.
If fewer than 32 historical trajectories are long enough to define a legacy
control, that control is unavailable and cannot trigger for that task. Every
task must have a finite primary `mobility_w4_k4` threshold.

At runtime, the q99 `mobility_w4_k4` threshold also defines a state machine:

```text
normal  : current -u_4 does not exceed the frozen threshold
suspect : current -u_4 exceeds it for one to three consecutive queries
alarm   : it exceeds it for four consecutive queries; alarm is latched
```

This is exactly equivalent to thresholding `E_4,4`; `suspect` is not counted
as an externally emitted alarm.

## Typed gates

The four causal three-query-smoothed route heads from the earlier protocol are
retained: `instability`, `lock_in`, `flat_narrow_support`, and
`feedback_decoupling`. Their components are converted to same-task,
same-query empirical percentiles using only the 400 historical trajectories.

After the primary mobility condition is met, frozen diagnostic gates are:

```text
loop_gate     = instability exceeded 0.80 at least once in q-6:q-1
static_gate   = max(lock_in, flat_narrow_support) exceeded 0.80 twice in a row
feedback_gate = feedback_decoupling exceeded 0.80 twice in a row

typed_gate = loop_gate OR static_gate OR feedback_gate
```

`loop_confirmed`, `static_confirmed`, `feedback_confirmed`, and
`typed_confirmed` require both `mobility_w4_k4` threshold crossing and their
gate. They are nested precision diagnostics, not OR branches. The primary alarm
remains `mobility_w4_k4` because development data showed that typed gating
discarded substantially more failures than false alarms.

## Sealing and evaluation

Before loading any `8b` outcomes, write and hash:

- the external route-feature cache;
- historical task profiles and thresholds;
- every per-query detector score, gate, and latched alarm;
- one-row-per-rollout first-alarm records;
- protocol, scorer, and runtime-monitor source.

The manifest must state `labels_used: []`, the exact summary-field whitelist,
and that test trajectories were not used for calibration. Only after the seal
is complete may a separate evaluator read `success`.

For every detector and operating point report TP, FP, FN, TN, failure recall,
successful-rollout FPR, alarm precision, recall at least 2/4/8/12 queries before
termination, and median lead. Report task-bootstrap intervals and task-level
concentration. No AUC is a primary result.

The q99 detector is considered high-precision only if external success FPR is
at most 1%. Precision is interpreted together with the external failure base
rate. No causal, intervention, or pre-physical-onset claim is allowed because
the external cohort has no independently frozen physical onset labels.
