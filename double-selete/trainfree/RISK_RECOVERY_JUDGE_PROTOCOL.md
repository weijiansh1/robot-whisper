# Train-free risk and recoverability judge protocol

Status: frozen before the formal risk/recovery replay is executed. The design is
post-hoc exploratory because both cohorts' outcomes were inspected while forming
the hypothesis. Runtime scoring and calibration remain label-free.

## Target

Keep three outcomes separate:

- `timely`: succeeds inside the original suite horizon;
- `late`: reaches the original horizon but succeeds within ten extra VLA queries;
- `persistent`: does not succeed inside that extension.

The risk target is `late OR persistent`. The recoverability target is `late`,
conditional on having reached the original horizon. A late success is therefore
a correct risk alarm, not a normal-success false alarm.

## Online inputs

The judge may use only information available in a single live rollout:

- current and historical MoE router probabilities / expert IDs;
- current and historical deployable policy state (EEF pose and gripper state);
- action chunks already issued;
- elapsed query count and configured task horizon.

It may not use `sim_state`, object oracle state, the future continuation, final
outcome, other concurrently evaluated rollouts, or the eventual trajectory
length.

## Head 1: deadline-risk alarm

Reuse the shared-threshold route-mobility cascade at q95:

- two consecutive exceedances: `warning`;
- four consecutive exceedances: `hard_alarm`.

The score is the negative four-query mean of adjacent-query action-route
Hellinger mobility. K2 and K4 share the K1 episode-maximum threshold. Thresholds
are task-specific and calibrated without outcomes. This head runs throughout
the rollout and detects persistent route lock-in.

## Head 2: recoverability evidence

At any query with at least four transitions, compute six causal components:

1. mean EEF translation during the previous four transitions;
2. latest EEF translation;
3. realized EEF motion divided by prior commanded translation magnitude;
4. current MoE route mobility;
5. current state/action feedback split;
6. negative lag recurrence.

For each task, build a terminal reference profile from historical trajectories
that completed before the configured horizon. Each reference sample is measured
at its final available query. Convert every current component to an empirical
midrank against the corresponding completed-reference component. The recovery
score is the unweighted mean of the six ranks:

```text
recovery_score = mean(component empirical ranks)
```

High values mean that an unfinished rollout still looks active, responsive, and
less recurrent in the way normally seen just before successful completion. Low
values mean physical and computational lock-in. No weights are fitted.

The formal allocation decision is made at the last planned query, before its
action chunk is issued, so an intervention still has one original chunk of lead:

- score above the historical censored-score q90: `strong_extend`;
- score above q75: `extend_watch`;
- otherwise: `intervene_first`.

The q75/q90 cutoffs are calibrated from historical trajectories that reached
their horizon, using only their prefix scores and censoring status. They control
the fraction of deadline-risk cases receiving extra budget; their later success
under extension is not used.

## Replay separation

For `development_main`, hold out one initial-state ID and all eight flow-noise
draws. Completed terminal profiles exclude that initial state. Every trajectory
gets a cross-fitted score. Allocation thresholds exclude the held-out
initial-state ID.

For `external_8b`, profiles and allocation thresholds come only from the prior
`right-50x8-20260903` reference run. The evaluated
`right-50x8b-20260903` trajectories never enter calibration.

Only after all scores and decisions are fixed are the three outcome labels
joined for evaluation.

## Reported endpoints

- warning/hard-alarm recall for late and persistent outcomes separately;
- timely-success false-alarm rate and risk precision;
- recoverability ROC-AUC and average precision at the budget boundary;
- q75/q90 late-success recall, extension precision, lift over extending a
  random censored trajectory, and extra-query cost per recovered rollout;
- per-task counts to expose task concentration.

The experiment does not claim causal intervention benefit. It estimates which
already-observed deadline-risk trajectories would have been recovered by the
frozen ten-query extension.
