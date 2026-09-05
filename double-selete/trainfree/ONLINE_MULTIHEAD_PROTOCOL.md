# Train-free multi-head online Trap alarm protocol

Status: frozen before the new per-query alarm scores and thresholds are generated.

## Question

Can causal MoE routing signals raise a useful alarm before a rollout terminates in
failure, and can separate heads expose a failure phenotype that is useful for a
later recovery policy?

This is an alarm experiment, not a ranking experiment. A trajectory is either
alerted or not alerted at each query. No intervention is applied.

## Cohort

- Source: complete `right-50x8-20260903` runs below
  `VLA_MUI_HUB/cache_new/HiMoE-VLA`.
- A run must have `status=complete`, complete sampling, matching designed and
  actual episode counts, and a readable `server/routes.zarr`.
- The expected frozen cohort is the same 37-task, 14,800-trajectory cohort used
  by `HUB_BINARY_AUDIT.md`. A changed cohort is an error rather than an implicit
  update.
- `episode_id` in `routes.zarr` defines trajectory boundaries. The scorer may
  read only `episode_index`, `init_state_id`, `flow_noise_seed`, and
  `inference_calls` from summaries. It must ignore `success` and every physical
  behavior field.

The head definitions below were motivated by the earlier retrospective audit on
this cohort. Consequently this is an exploratory causal re-analysis, not an
independent confirmation set. Label-free calibration prevents direct outcome
fitting but does not erase that hypothesis-selection history.

## Causal query features

At query `q`, all features use only the current MoE forward and queries `<= q`.
No centered window, future query, eventual length, success flag, or physical
label is available to the online update.

The HB action tokens are tokens 1--10, HB front layers are 2--5, HB back layers
are 12--15, and the final denoising step is 9. Router probabilities are
renormalized defensively. Distances are Hellinger distance; route recurrence is
weighted Jaccard similarity.

Current-forward features:

- `late_flow_volatility`: mean route distance between adjacent denoising steps
  6--9 over back-layer action tokens;
- `route_acceleration`: mean second difference of square-root router
  probabilities over denoising time;
- `gate_entropy`: normalized final-step action-token gate entropy;
- `top12_margin`: top-1 minus top-2 gate probability;
- `top4_union`: fraction of the 32 experts present in the recorded top-4 union;
- `token_disagreement`: action-token route disagreement;
- `front_state_action_gap` and `layer5_state_action_gap`: state/action route
  disagreement.

Past-dependent features:

- `route_mobility(q)`: route distance from `q-1` to `q`;
- `lag_recurrence(q)`: maximum route similarity to lags 1--4;
- `lag_periodicity(q)`: maximum lag-2--4 similarity minus lag-1 similarity;
- `front_feedback_split(q)`: state-route jump minus action-route jump.

Queries 0--3 are warm-up. The first eligible alarm query is 4, when all four
lags and a three-query trailing smoother are defined.

## Four typed heads

Raw signals are converted to directed empirical percentiles at the same query
index using the unlabeled reference bank described below. Every component and
every head has fixed equal weight.

```text
instability = mean(high late_flow_volatility,
                   high route_acceleration,
                   high route_mobility,
                   high lag_periodicity)

lock_in = mean(low route_mobility,
               high lag_recurrence,
               low top4_union,
               low token_disagreement)

flat_narrow_support = mean(high gate_entropy,
                           low top12_margin,
                           low top4_union)

feedback_decoupling = mean(high front_state_action_gap,
                           high layer5_state_action_gap,
                           high front_feedback_split)
```

The online head value is the arithmetic mean of its raw value at `q-2:q`. The
detectors evaluated are:

- `instability`, `lock_in`, `flat_narrow_support`, and
  `feedback_decoupling`: the four typed heads;
- `dual_mean`: mean of the smoothed instability and lock-in heads;
- `dual_max`: maximum of those two heads;
- `multi_max`: maximum of all four smoothed heads, the primary multi-head alarm;
- `instant_multi_max`: maximum of all four unsmoothed heads, a persistence
  ablation;
- `clock`: elapsed query divided by the task's configured query limit, a
  routing-free duration baseline.

At the first `multi_max` alarm, the largest smoothed typed head is emitted as the
alarm phenotype. This type output is descriptive and does not affect whether an
alarm fires.

## Label-free calibration and sealing

Calibration is leave-one-initial-state-out within each task. For the eight test
rollouts of one initial state, the reference bank consists of all 392 rollouts
from the other 49 initial states. Outcomes are not filtered: successful and
failed reference trajectories are both retained.

For a raw signal at query `q`, its reference CDF uses reference trajectories that
actually reached the same `q`. A query is unscorable if fewer than 32 reference
trajectories reached it. This avoids extrapolating a phase baseline into a
region without support.

For each detector, take the maximum causal score over each reference trajectory.
The q90, q95, and q97.5 quantiles of these unlabeled trajectory maxima are the
three frozen operating thresholds. The primary operating point is q95. An alarm
fires on the first strict threshold exceedance and remains latched. Thresholds
are separately calibrated for composites, so an OR/max detector cannot obtain
extra alarm budget merely by adding heads.

The scoring program writes scores, thresholds, booleans, source hashes, and an
explicit `labels_used: []` manifest. Only after those artifacts exist may the
evaluation program read outcomes or physical labels.

## Outcome evaluation

Primary endpoint at q95:

- failure recall: fraction of eventual failed trajectories with any alarm;
- success false-positive rate: fraction of successful trajectories with any
  alarm;
- precision among alarmed trajectories;
- early recall at least 4 and at least 2 queries before trajectory termination;
- median first-alarm lead among detected failures.

All metrics are reported for q90/q95/q97.5. Differences between `multi_max` and
`lock_in`, `dual_mean`, `instant_multi_max`, and `clock` use paired bootstrap
resampling of tasks. The duration baseline is mandatory because failed episodes
are usually longer and an online detector can otherwise appear useful by merely
detecting that the rollout is still running.

## Physical-onset evaluation

Physical labels enter only here and remain heuristic proxies. Onsets are frozen
as follows:

- stagnation: start of the longest pre-90% static run;
- gripper cycling: third gripper-command sign flip;
- active retry / EEF oscillation: third eligible EEF direction reversal;
- goal regression: the closest-to-goal query before regression;
- goal approach-leave: first hysteretic goal-region exit;
- subtask undo: first exit after reaching a goal;
- regrasp/drop: first off-goal lift loss, or the second lift onset if earlier.

Report any-head recall in `[-4,-1]` (strict precursor), `[-2,+2]` (timely), and
`[0,+4]` (reaction) around onset. Also report which typed head won at the first
timely alarm. These are not semantic ground truth for false grasp, contact jam,
or stale chunks.

## Decision rule

Call the overall multi-head alarm an improvement only if, at the primary q95
operating point, it raises failure recall over both the best typed single head
and `clock` without increasing realized success FPR by more than one percentage
point, and the task-bootstrap 95% interval of each recall difference excludes
zero. Otherwise the overall result is negative or mixed, even if phenotype
routing is qualitatively useful.

No claim about corrective intervention is allowed in this experiment.
