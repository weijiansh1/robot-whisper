# Censor-aware closed-loop online alarm v2

## Bottom line

The data structure yields a strong train-free operational alarm, but it is not
a new MoE signature.

In the frozen 14,800-rollout cohort, outcome and termination censoring are
exactly equivalent:

| outcome | ended before suite limit | reached suite limit |
|---|---:|---:|
| success | 14,313 | 0 |
| failure | 0 | 487 |

Consequently, the recommended binary trigger is the censor-aware
`completion_delay` head. At q95 it alarms on all 487 failures, alarms on 421 of
14,313 successes, and fires a median 9 queries before failure termination.

The tested MoE/state confirmation should not be OR-ed into that binary trigger.
It did not improve any predeclared early-recall endpoint and increased false
alarms.

## Main results

### Completion-delay operating points

| calibration | failure alarm | success false alarm | precision | at least 4q early | at least 8q early | median lead |
|---|---:|---:|---:|---:|---:|---:|
| q90 | 100.00% (487/487) | 5.06% (724/14,313) | 40.21% | 100.00% | 97.54% | 10q |
| **q95** | **100.00% (487/487)** | **2.94% (421/14,313)** | **53.63%** | **100.00%** | **90.35%** | **9q** |
| q97.5 | 100.00% (487/487) | 1.87% (267/14,313) | 64.59% | 100.00% | 62.83% | 9q |

Task-bootstrap 95% intervals at q95 are 100.00--100.00% for failure recall and
2.36--3.51% for success false-alarm rate. The detected-failure lead has mean
12.31q, median 9q, and interquartile range 8--19q.

q95 is the balanced choice when at least eight queries of recovery time matter.
q97.5 is preferable when minimizing interventions on eventual successes is
more important.

### What each added head contributes at q95

| detector | failure alarm | success false alarm | precision | 8q-early recall |
|---|---:|---:|---:|---:|
| completion delay | **100.00%** | **2.94%** | **53.63%** | **90.35%** |
| route only | 23.61% | 6.06% | 11.71% | 23.61% |
| MoE + physical mechanism only | 18.89% | 5.74% | 10.08% | 18.89% |
| delay + route | 100.00% | 3.38% | 50.15% | 90.76% |
| delay + confirmed mechanism | 100.00% | 3.16% | 51.86% | 90.35% |

The frozen primary `delay_confirmed` detector is therefore worse than the
simpler completion-delay detector. Relative to completion delay it:

- adds no detected failure and loses no detected failure;
- changes none of the 2q-, 4q-, 8q-, or 12q-early recall endpoints;
- fires one query earlier on only 12 failures from 6 task-initial-state cells;
- advances 42 successful trajectories and creates 31 additional successful
  false alarms;
- increases success FPR by 0.22 percentage points, with task-bootstrap 95% CI
  0.00--0.56 points.

Of the 12 advanced failures, 10 carry the heuristic `gripper_cycling` label, 1
`goal_regression`, and 1 `regrasp_or_drop`. This narrow concentration is useful
for phenotype diagnosis, not for the global trigger.

## Improvement over v1

The q95 censor-aware completion detector changes the v1 result materially:

| detector | TP | FP | failure recall | success FPR |
|---|---:|---:|---:|---:|
| v1 MoE `multi_max` q95 | 83 | 773 | 17.04% | 5.40% |
| v1 all-outcome `clock` q95 | 194 | 301 | 39.84% | 2.10% |
| **v2 completion delay q95** | **487** | **421** | **100.00%** | **2.94%** |

Against v1 `multi_max`, this is 404 more failed rollouts detected and 352 fewer
successful rollouts alarmed. Against v1 `clock`, it is 293 more failed rollouts
detected at a cost of 120 additional successful alarms. Operationally,
completion-delay q97.5 also dominates v1 clock q95 here: 100% versus 39.84%
recall and 1.87% versus 2.10% FPR, although these are different calibration
quantiles.

The reason v1 clock missed hard tasks is structural: its reference bank mixed
timeouts into the nominal duration distribution. On tasks with more than 5%
timeouts, the all-outcome q95 threshold can move to the task limit itself. The
v2 reference bank treats those observations as right-censored instead.

## Replay procedure

1. Freeze the 37 tasks and 14,800 trajectories. Each task contains 50 initial
   states and 8 flow-noise draws per state.
2. Hide `success`, failure subtype, `sim_state`, and all future queries from the
   scorer.
3. Hold out one initial state and all eight of its draws. Use the other 392
   historical rollouts as references.
4. Mark a historical rollout ending at the configured suite limit as
   right-censored. Build the healthy completion-time and feature distributions
   only from trajectories ending before that limit.
5. At online query `q`, score elapsed completion delay, the causal MoE prefix,
   and deployable EEF/gripper state plus the previously issued action chunks.
6. Calibrate each detector on maxima from completed historical references and
   freeze q90/q95/q97.5 thresholds. An exceedance latches the alarm.
7. Write and hash scores, thresholds, alarms, features, source identities, code,
   and protocol. The sealed manifest records `labels_used: []` and separately
   discloses use of historical termination times for censoring.
8. Only then join outcomes for all 14,800 trajectories at once and form TP, FP,
   FN, and TN. Failure subtypes are examined only as a secondary diagnostic;
   they do not decide which failures count.

This is not “alarm first, inspect only the alarming failures.” Every rollout is
scored before outcome reveal, and both alarming and non-alarming successes and
failures enter the confusion matrix.

## Why successful rollouts still alarm

The 421 q95 false alarms are the long tail of successful execution time. At the
alarm query, the robot has exceeded the duration in which 95% of comparable
completed references normally finish, but some trajectories genuinely need
more time and later succeed.

Current MoE and proprioception cannot reliably identify that benign tail:

- normal correction and failed retry share instability/cycle signatures;
- stable success phases and static failure share lock-in signatures;
- the deployable `state` array contains EEF pose and gripper state but not object
  pose, contact, or task-goal progress;
- 447 of 487 failures occur in task-initial-state cells containing both success
  and failure, so initial-state difficulty is not a sufficient discriminator.

This is why the next useful information source is object- and goal-conditioned
progress from observations, not another OR over routing heads. `sim_state` can
measure that offline as an oracle, but using it in the deployed detector would
be leakage. A deployable version needs visual object tracking, contact sensing,
or an equivalent online progress estimator.

## Recommended online structure

Use a two-stage interface, but only the first stage decides whether to alarm:

```text
binary alarm = completion_delay exceeds q95 (or q97.5)

recovery type after alarm = argmax(loop_confirm,
                                   static_confirm,
                                   feedback_confirm)
```

MoE-only suspicion can be logged before the binary alarm, but the present data
does not justify intervention on that signal. This preserves the large
TP/FP improvement from the censor-aware data structure while retaining typed
signals as scaffolding for later correction policies.

## Validity boundary

The 100% failure recall is not evidence that MoE recognizes every Trap. On this
cohort, “failure” is exactly “still running at the administrative limit,” so the
termination schema is an outcome proxy. The detector is useful for operation,
but its performance must be revalidated on failures that can terminate early,
successful rollouts allowed to run to the horizon, and new tasks/seeds frozen
before outcomes are inspected.

No intervention result is claimed.
