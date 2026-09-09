# Fixed-horizon physical progress experiment

This retrospective experiment was specified before fitting its new progress
models. The corpus has been studied previously, so it is not a prospective blind
test. The original success and budget-audit artifacts remain separate.

## Target and timing

- Prediction time: after inference q, before executing its already sampled
  ten-action chunk. Input: only the last eight MoE queries, including q.
- Horizon: the next five chunks (50 actions), identical across all tasks.
- A positive physical milestone is any of:
  1. The collector records task success within those 50 actions.
  2. Two consecutive future saved checkpoints both have more satisfied original
     BDDL goal predicates than the current checkpoint.
  3. An unfinished goal object that is not currently both grasped and lifted is
     both grasped and lifted at two consecutive future saved checkpoints. Grasp
     uses the simulator's two-finger contact check; lift is at least 0.025 m above
     that object's first recorded height. The goal-predicate count must not fall
     below its current value at either confirming checkpoint.
- Both confirming checkpoints must be strictly after q and at or before q+5.
  A transient single-checkpoint event is not positive. Two observations do not
  establish continuous contact between them.
- These are conservative observed milestones, not an exhaustive measure of
  useful progress, independent human trap labels, or proof of escape. Approaches,
  partial drawer movement, and unsampled short events can be missed.

## Observation support

Use q >= 7 and (q+5)*10 < the preconfigured action cap. This outcome-independent
schedule rule ensures a complete future checkpoint window unless success stops
the episode sooner. Exact-cap endpoints are excluded because the collector does
not save the terminal failure state. Early success is positive, never dropped.
No actual episode length, cap, absolute query, task ID, physical state, or alarm
history enters the online feature vector. Fixed caps still affect which active
prefixes exist in this retrospective corpus; arbitrary longer-horizon behavior
is not identified.

Extraction audit amendment, before model fitting: some collector-active states
satisfy the complete goal after restoring qpos/qvel and recomputing simulator
geometry. Record this discrepancy and exclude already-complete starting states
using their current restored predicates. Do not discard their earlier prefixes
or replace the collector's terminal outcome/time. This is a local-state filter;
no future outcome determines this exclusion.

## Labels and controls

Restore all saved simulator states for both complete 50x8 natural cohorts, using
the pinned LIBERO/robosuite environment. Validate task identity, joint layout,
robot-state alignment, and original predicate/check_success agreement. Source
files are read-only. Retain primitive physical measurements and source hashes.
No new VLA actions are sampled or executed.

Use the existing 18 local MoE features, fixed learner settings, and disjoint
initial-state train/calibration/test split. No class weighting, terminal-length
weighting, or hyperparameter search. Freeze the existing five task folds; each
held-out task is absent from that fold's training and calibration data.

Compare a shared MoE-only model against a constant prior and diagnostic models
using task/clock and current physical stage. Physical-stage inputs are offline
controls only. Also compare stage+MoE against stage, and the equivalent controls
with task/clock. Report raw and sigmoid-calibrated probabilities without choosing
between them from test results.

Report Brier, log loss, AUROC, calibration, and within-task/query AUROC. A stricter
comparison also matches the current goal bitmask, goal-object grasp bitmask,
lift bitmask, and reach bitmask (distance <= 0.12 m). This is coarse stage matching,
not identical physical states. Matched AUROC weights each eligible positive /
negative pair equally. Retain each stratum's support and per-task results.
Uncertainty resamples initial-state groups within task, preserving all seeds and
queries; matched AUROC uses the same cluster resampling. These intervals condition
on the fitted models and sampled tasks, not model retraining.

All existing half-k4 first alarms are inspected descriptively, including the 42
alarm-then-success episodes. The rule and cases were already explored. A future
milestone after an alarm does not establish that the alarm was a true trap.

## Deliverables

Reproducible extraction, label/model pipeline, a MoE-only streaming progress
monitor, focused boundary/leakage tests, saved physical traces and predictions,
task splits, uncertainty, plots, source audits, and a Chinese report under
`results_progress/`.
