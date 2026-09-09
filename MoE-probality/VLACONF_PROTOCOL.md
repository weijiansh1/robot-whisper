# VLAConf-inspired scalar probability calibration

This protocol is fixed before fitting this experiment. It is a retrospective
extension on an already explored corpus, not a new blind test or a reproduction
of the VLAConf network. Paper source: `../safe&vlaconf/arXiv-2605.29605v2.tar.gz`,
`approach.tex` and `appendix.tex`; https://arxiv.org/html/2605.29605v2.

## Question and target

Can a fixed success-support score plus a two-parameter monotone sigmoid estimate
the final natural success probability from causal MoE observations?

The target is original-deadline success in the recorded policy/task mixture.
There are no online task, suite, step, budget, physical-state, or endpoint inputs.
Removing budget inputs does not remove the deadlines from the outcome definition.
No new policy rollouts, true trap labels, or branch escape probabilities are claimed.

## Fixed score

Use the existing 18 local MoE features, each computed from at most eight router
queries. Keep q >= 7, exactly as in the no-budget audit. The first seven queries
abstain; no future episode length or fraction-of-completion selects a checkpoint.

Use every q >= 7 prefix of successful TRAIN episodes as a reference library.
Fit coordinate means and standard deviations on that library only, with the
standard scaler's handling of constant coordinates. Fix k = 20 and define

`s(q) = log1p(mean Euclidean distance to the 20 nearest reference prefixes)`.

Reference membership uses successful outcomes offline. The representation,
distance, k, and aggregation do not use calibration or test outcomes. There is
no new neural or outcome-trained feature model. This replaces the paper's
success-demonstration CFN with a nonparametric support proxy on policy rollouts;
it cannot be called CFN or an exact inverse pseudocount. Nearby reference rows
can belong to the same trajectory. Query density and redundant coordinates can
affect distance; neither distance nor success-only density is itself a posterior.

## Aggregation and calibration

Predeclare four aggregations starting with the first ready score at q = 7:

1. `current`: current scalar only.
2. `window4`: mean of the last four available scores, the PRIMARY method.
3. `prefix_mean`: mean of all ready scores so far.
4. `prefix_max`: maximum of all ready scores so far.

The first three ready queries use partial score windows. The primary method
uses at most 11 router queries once ready. Prefix aggregations have causal but
unbounded history; they may encode elapsed duration implicitly. Prefix maximum
with the monotone calibrator cannot increase the predicted success probability.

For each aggregation fit `p = sigmoid(-alpha * u + beta)`, alpha >= 0, with
unweighted Bernoulli log loss on all eligible CALIBRATION prefixes and their
parent rollout's final success. Only these two parameters use both success and
failure supervision. Do not choose the score direction or aggregation on test
results. A zero optimal slope is retained and reported. Scaling a scalar inside
the optimizer is a numerical reparameterization, not an extra predictive input.

As a fixed existing-score control, also calibrate
`-log(max(freeze_back_quarter_score, 1e-8))`, without extra aggregation.
This control uses the detector's causal initial baseline, unlike the primary
bounded-window method. No detector threshold is tuned.

## Splits and comparison

Reuse the exact original train/calibration/primary unseen-init/secondary noise
splits. Only train success prefixes populate reference libraries. No training
prefix predictions are evaluated against their own reference library.

Repeat all scalar methods in the existing five task folds: exclude held tasks
from the scaler, reference library, and probability calibration. No target-task
recalibration. All fitted models remain global, without suite selection.

Compare the saved no-budget supervised MoE model and clock+cap baseline on the
identical q >= 7 test rows. Include a calibration-set constant success prior,
and a separate prior excluding held tasks for task-holdout comparisons. Do not
compare losses with the old all-prefix 0.9146 experiment or the different
five-chunk progress target.

## Evaluation and audit

- Both test cohorts: Brier, log loss, AUROC, equal-width reliability bins/ECE;
  also fixed q = 7, 15, 25 and suite-specific diagnostics.
- Primary test: pair-weighted AUROC within task/q and task/q/current physical
  stage. The existing stage label uses only current or earlier physical states;
  it is evaluation metadata, not a model input. Do not filter by future progress.
- Task/init-cluster bootstrap intervals, fixed models and tasks, for matched
  AUROC and paired loss differences. These do not measure training or new-task
  population uncertainty.
- Save raw scalar rankings as well as calibrated probabilities. A monotone
  global sigmoid cannot repair incorrect rankings. Cross-task folds can have
  different maps, so their pooled rankings need not be preserved.
- Describe adjacent probability increases, including successful alarmed
  episodes. Increases are not verified physical recovery or true trap evidence.
- Verify original artifact hashes, no split overlap, and streaming/batch parity
  using the same 80 outcome-independent raw-route episode samples as before.
- Save scores, calibration coefficients, reference keys, held-task exclusions,
  row predictions, figures, report, online bundle, and artifact hashes separately
  under `results_vlaconf/`. Preserve previous experiment outputs.
