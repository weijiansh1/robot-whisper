# Single-rollout streaming Trap alarm report

## Answer

The requested deployment setting is now implemented: one stateful monitor is
created for one live rollout, and one `update` call is made after each VLA
forward. Runtime scoring uses only the current/past routing tensors, current/past
robot states, issued action chunks, and a frozen task profile. It does not know
the rollout's final length, timeout, success, future, simulator state, or other
flow-noise draws.

The best predeclared detector is not the more complicated physical phenotype
state machine. It is `persistent_route`: require high route evidence on two
consecutive queries, calibrating the whole detector to the same trajectory-level
false-alarm budget.

At q95:

| strictly online detector | failed rollouts alarmed | successful rollouts alarmed | precision | at least 8q early | median failure lead |
|---|---:|---:|---:|---:|---:|
| instantaneous route | 19.92% (97/487) | 5.32% (762/14,313) | 11.29% | 18.28% | 18q |
| **persistent route** | **22.79% (111/487)** | **5.07% (725/14,313)** | **13.28%** | **20.53%** | **17q** |
| phenotype state machine, no persistence | 19.30% (94/487) | 5.15% (737/14,313) | 11.31% | 17.66% | 17.5q |
| frozen primary phenotype state machine | 18.89% (92/487) | 5.13% (734/14,313) | 11.14% | 16.84% | 17.5q |

Thus persistence moves both requested quantities in the correct direction:

- 14 additional failures alarmed relative to the matched instantaneous route
  detector;
- 37 fewer successful rollouts alarmed;
- failure-recall change +2.87 percentage points, task-bootstrap 95% CI
  +0.24 to +5.81 points, two-sided p=0.038;
- success-FPR change -0.26 points, 95% CI -0.49 to -0.03 points, p=0.023.

Against the earlier v1 `multi_max` q95 result, persistent route alarms 28 more
failures and 48 fewer successes. The paired changes are +5.75 recall points
(95% CI +2.84 to +10.03) and -0.34 FPR points (95% CI -0.57 to -0.11).

This is a real but modest improvement. It does not justify saying that routing
alone recognizes most Trap rollouts.

## Operating points

| threshold | failure recall | success FPR | precision | 8q-early recall |
|---|---:|---:|---:|---:|
| q90 | 36.96% | 10.03% | 11.14% | 33.47% |
| **q95** | **22.79%** | **5.07%** | **13.28%** | **20.53%** |
| q97.5 | 14.58% | 2.60% | 16.03% | 13.55% |

q95 is the default balance. q97.5 is available when an unnecessary recovery is
more costly than a missed Trap; q90 is available for a low-cost diagnostic or
reversible intervention.

## Why persistence helps

Successful rollouts also contain route peaks, valleys, expert narrowing, and
short corrective changes. A trajectory-maximum alarm treats one healthy
correction as a failure signal. Requiring two consecutive abnormal queries
suppresses some of those transients while retaining failures with sustained
routing abnormality.

The improvement is not caused by waiting near timeout: no clock, completion
duration, eventual length, censoring filter, or outcome is present in the
detector. Detected failures are alarmed a median 17 queries before termination.

The improvement also holds at physical proxy onsets. Across 325 failures with a
usable onset, persistent route has:

| onset measure | rate |
|---|---:|
| any alarm | 29.85% |
| first alarm before onset | 11.38% |
| strict precursor `[-4,-1]` | 6.46% |
| timely `[-2,+2]` | 12.92% |
| reaction `[0,+4]` | 12.00% |

The subtype alarm rates remain heterogeneous: 52.63% for gripper cycling,
30.83% for regrasp/drop, 23.15% for stagnation, and only 8.64% for the untyped
`other` group. This explains the low global recall.

## Why the complex state machine did not help

Combining MoE routes with EEF motion, command magnitude, reversals, gripper
flips, and state recurrence sounded mechanically appropriate, but healthy
manipulation contains the same short patterns. The frozen primary state machine
alarms fewer failures than persistent route at nearly the same successful
false-alarm rate. Its result is negative and it is retained only for phenotype
analysis.

The available 8-D client state describes EEF pose and gripper state. It does not
say whether the object moved, slipped, reached the goal, or was the correct
object. Consequently it cannot resolve false grasp, wrong goal, and many
stagnation cases. A stronger single-rollout detector needs deployable visual
object/task progress or contact feedback.

## Runtime use

The frozen deployment profile contains only historical feature reference banks
and thresholds. It is loaded once; no historical trajectory is accessed by the
live update loop after loading.

```python
from pathlib import Path

from single_rollout_monitor import SingleRolloutMonitor, TaskProfile

profile = TaskProfile.load(
    Path("results/online_single_rollout/deployment_profiles.npz"),
    task_name,
)
monitor = SingleRolloutMonitor(
    profile,
    quantile=0.95,
    alarm_detector="persistent_route",
)

for query in rollout:
    result = monitor.update(
        query.hb_router_probs,  # [8, 10, 11, 32]
        query.hb_expert_ids,    # [8, 10, 11, 4]
        query.robot_state,      # [8]
        query.action_chunk,     # [10, 7]
    )
    if result["alarm"]:
        recovery_type = result["phenotype"]
        break
```

On CPU, an end-to-end update took 1.08 ms median and 1.31 ms p95 over 390
measured updates. Raw streaming feature extraction matched the cached replay to
within `3.9e-4`; the state machine and batch replay use identical equations.

## Experimental procedure

1. Freeze the runtime contract and detector equations.
2. Hold out one initial state and all eight of its rollouts.
3. Build an outcome-blind task profile from all 392 other-initial-state
   trajectories. Successful and failed references are both retained.
4. Replay each held-out rollout one query at a time using only its prefix.
5. Calibrate every composite independently from historical trajectory maxima;
   latch its first strict threshold exceedance.
6. Hash and seal profiles, scores, thresholds, alarms, code, and protocol with
   `labels_used: []`.
7. Reveal outcomes for all 14,800 trajectories and compute the complete
   confusion matrix. Physical behavior labels enter only the secondary onset
   analysis.

The result remains exploratory because the detector family was motivated after
earlier inspection of this cohort. New tasks and seeds should be frozen before
the next confirmation run.

No intervention performance is claimed.
