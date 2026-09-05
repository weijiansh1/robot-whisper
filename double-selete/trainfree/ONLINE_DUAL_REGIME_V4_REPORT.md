# MoE-only Online Dual-Regime Alarm: Core Result

## Verdict

The new detector improves recall and precision together.  On the 15,600-rollout
external 8B replay it detects 410 of 564 original-horizon risks and falsely
alarms on 81 of 15,036 timely successes:

| Detector | Risk recall | Precision | Timely-success FPR | TP | FP | Median lead |
|---|---:|---:|---:|---:|---:|---:|
| Old back-layer mean, q95, K2 | 44.68% | 70.79% | 0.692% | 252 | 104 | 8 queries |
| Lock-in head only | 67.73% | 83.41% | 0.506% | 382 | 76 | 13 queries |
| Dual-regime v4 | **72.70%** | **83.50%** | **0.539%** | **410** | **81** | **12 queries** |

Thus 83.5% of v4 alarms correspond to a late success or persistent failure.
Relative to the old alarm, v4 adds 158 true alarms while removing 23 false
alarms.  The task-cluster bootstrap intervals for v4 are 64.47-84.02% recall and
71.85-91.39% precision.

Development results are consistent: 76.39% recall, 84.74% precision, and 0.468%
timely-success FPR on 14,800 rollouts.

## Method

Both heads use only final-flow action-token HB router probabilities from the
current rollout.  For every layer and query, they compute adjacent-query
Hellinger route mobility.

1. The lock-in head takes the median mobility across all eight layers, a causal
   four-query mean, a task-reference q75 threshold, and four consecutive low
   crossings.
2. The instability head takes L5 mobility, a causal four-query mean, a
   task-reference q80 threshold, and eight consecutive high crossings.
3. The online alarm is the latched OR of the two heads.

Thresholds are empirical same-task reference order statistics.  Reference
outcomes are not loaded or filtered, and no classifier or fusion weight is fit.
Runtime has no image, robot state, progress, horizon, action, outcome, or other
rollout input.

The key change is to separate anomaly magnitude from duration.  The old q95
instantaneous threshold followed by short confirmation was too strict.  A lower
q75 lock threshold with longer persistence rejects transient normal deviations
while retaining sustained failures.  Taking the median across layers also avoids
diluting one locked layer or letting one noisy layer control the signal.

## What Worked

- All-layer median was more robust than the old four-back-layer mean.
- Lower instantaneous threshold plus four-query persistence raised recall without
  increasing successful-trajectory alarms.
- The opposite, sustained-high L5 head was genuinely complementary.  Externally
  it adds 28 TP and 5 FP beyond the lock-in head.
- The detector covers 35 of the 36 external tasks that contain risk rollouts.
  Sixteen tasks gain true detections relative to the old rule.

## What Did Not Work

- Individual mobility-derivative heads had low standalone coverage.  The useful
  second regime was sustained high mobility, not another numerical derivative.
- The earlier generic four-head phenotype OR improved recall only modestly and
  reduced precision.
- Simply loosening the old short-persistence alarm raised too many timely-success
  alarms.  Duration and threshold must be changed together.

## Limits

- Overall external recall is 72.70%, not 80%.  Recall with at least four queries
  of lead is 58.51%, and with at least eight is 40.60%; these are the relevant
  numbers for a recovery policy that needs reaction time.
- Late-success recall is 65.85%, below persistent-failure recall of 73.23%.
- One moka-pot task contributes 64 of the 158 additional external detections.
  The other 94 additions span other tasks, so the gain is not single-task-only,
  but task concentration remains material.
- The external cohort was already inspected during v3 development.  The complete
  v4 rule was frozen before its candidate alarm replay, but this is not a pristine
  never-seen holdout.  A new route cohort is required for a final confirmation.

No intervention is performed in this experiment; it measures recognition and
online alarm behavior only.
