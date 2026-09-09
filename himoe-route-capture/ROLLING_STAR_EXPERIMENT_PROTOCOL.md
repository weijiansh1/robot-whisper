# Rolling-star K=16 MoE experiment protocol

The core collection design, q0/q0-q2 windows, physical thresholds, controls,
and grouped evaluation were frozen after validating the first few branches of
snapshot 0 and before any complete K=16 snapshot was available. Later
exploratory amendments are logged explicitly below.

## Question

Can HiMoE routing identify an early risky action sample, or a specific physical
failure mechanism, when task identity, initial simulator state, observation,
and rollout time are held fixed?

## Collection

- Task: `libero_10`, task 8, `put_both_moka_pots_on_the_stove`.
- At every trunk query, save the exact simulator/controller state and restore it
  before each of 16 terminal branches.
- Siblings differ only in a deterministic, recorded flow-noise stream.
- Execute each branch until success or the shared 520-action horizon.
- Record full server-side HB and AS routing at every query. Do not store model
  hidden states.
- Record dense simulator state, EEF position, gripper position, and executed
  action at every environment action for physical outcome labeling.
- Write atomic branch artifacts and a snapshot manifest only after all 16
  branches and the next trunk action are durable.
- Incrementally sync the run to the local workstation throughout collection.

Four environment workers share one model server. Workers 0 and 1 were planned
initially. Workers 2 and 3 were added adaptively after telemetry showed large
GPU idle gaps and unchanged memory use; this adaptation changes sample volume,
not the within-snapshot comparison.

| worker | init state | historical role |
|---:|---:|---|
| 0 | 0 | mixed success/failure |
| 1 | 20 | enrich non-stagnation failures |
| 2 | 3 | enrich stagnation failures |
| 3 | 7 | easier success-side control |

## Physical labels

MoE routing is not read when labels are constructed. Success terminal object
positions define empirical goal references. Failures receive multi-label
physical annotations and one priority label:

- `stagnation`: long low-motion windows using EEF, both objects, and gripper;
- `loop_or_cycling`: a nonlocal physical state return after at least three
  queries and at least 12 cm intervening EEF path;
- `drop_or_regrasp`: off-goal height loss after lifting, or repeated lifts;
- `subtask_undo`: an object enters the empirical goal set and later exits it;
- `goal_regression`: distance from the best historical goal approach grows by
  at least 6 cm;
- `goal_contact_near_miss`: all object centers finish near successful stove
  placements but at least one original BDDL `On` predicate remains false;
- `single_subtask_omission`: exactly one original BDDL `On` predicate is true
  and the missing object remains nearly unmoved;
- `stove_not_on`: the original terminal `Turnon` predicate is false;
- `active_retry`: residual failure with substantial EEF travel and low
  goal-progress efficiency after no more specific mechanism applies;
- `no_progress` and `timeout_other` as residual categories.

Thresholds are constants in `analyze_rolling_star_experiment.py` and are also
exported with every analysis run. They were checked against the first available
success/failure videos to ensure units and action-rate interpretation were
correct; they were not selected from MoE effects.

## MoE tests

Primary route windows are q0 and q0-q2 only. No final-query, remaining-time,
timeout-sentinel, or last-30%-of-rollout feature is eligible.

1. Compare top and bottom within-snapshot quartiles for entropy, top-1 mass,
   action-token dispersion, and sibling-consensus distance across action and
   state token families, front/back HB layer groups, and all 10 denoise steps.
2. Permute outcomes only within snapshots. Use max-statistic correction across
   the full layer x denoise x metric screen and bootstrap whole snapshots for
   confidence intervals.
   Separately screen all 32 expert probabilities with one 1,280-cell maxT
   family per early prefix.
3. Hold out entire snapshots for all predictive checks, then repeat the core
   model set while holding out an entire worker/initial-state trajectory.
   Candidate-level snapshot checks use five grouped folds (4-5 complete K=16
   snapshots per test fold); worker checks use four folds, one worker each.
   Report Brier error, log-loss, and best-of-16 selected success rate, not AUC.
4. Compare route-only models with raw flow noise, first action chunks, and the
   combined noise + action + route model. Incremental MoE information requires
   the combined model to improve over noise + action.
5. Repeat prediction and route screens within failed branches for every
   physical failure label with adequate positive and negative support.
   Include the derived target `non_stagnation_non_loop`, which directly tests
   whether routing separates failures missed by the original two-pattern
   detector.
6. Run both absolute-feature models and models with each unlabeled K=16
   snapshot mean subtracted. The latter remove common initial-state/phase
   information and test sibling ranking directly.
7. Predict each snapshot's K=16 failure rate with both leave-one-snapshot-out
   and leave-one-worker-out models of initial simulator state, policy state,
   low-dimensional initial object/EEF geometry, mean output action, AS routing,
   and mean HB routing. Bootstrap the latter by worker. This is the separate
   task-difficulty question.
8. Use exact-zero AS within-snapshot variation as the routing negative control,
   audit state-token variation without assuming it is zero, and verify exact
   snapshot/input and branch-noise hashes. Permute outcomes within snapshots to
   verify that deterministic candidate IDs 0-15 do not carry an outcome bias.

## Interpretation

An association means routing accompanies an early risky sample. It does not by
itself establish that an expert caused the failure. A causal expert claim would
require a later routing intervention experiment.

## Pre-manifest implementation audit

Before the first complete K=16 manifest was available, dry runs found and fixed
the following definition details:

- empirical stove-goal references pool terminal positions across both pots,
  because both BDDL predicates use the same cook region;
- `single_subtask_omission` requires an unfinished object itself to remain
  unmoved, rather than any object;
- sibling-consensus distance uses the other 15 candidates (leave one out);
- the scalar stove-button joint is audited separately; pilot trajectories show
  no meaningful within-branch button motion.
- a terminal-state replay of failure candidate 9 showed pot 1 `On=True`, stove
  `Turnon=True`, and pot 2 `On=False` despite its center lying near successful
  terminals; this anchors the physical `goal_contact_near_miss` category.

These changes used simulator semantics and pilot trajectory/video inspection,
not any route/outcome effect. Statistical feature definitions and corrections
remained unchanged at that pre-manifest stage; later additions are listed
below rather than presented as part of the frozen core.

## Logged exploratory amendments

After four complete snapshots (64 branches) were available, the analysis dry
run exposed two coverage/confounding issues. Before the remaining rolling
snapshots completed, the following additions were made:

- state-token routing was expanded from one denoise step to all 10 steps;
- within-snapshot-centered models were added to distinguish sibling risk from
  the common rolling-state difficulty signal;
- snapshot-level leave-one-out difficulty models were added with a fixed ridge
  penalty;
- an all-expert probability screen was added, retaining all 32 experts and
  correcting the full 1,280-cell family rather than selecting expert IDs;
- terminal states were replayed against the original LIBERO BDDL predicates,
  and exact predicates were made mandatory for omission, near-miss, stove, and
  success consistency checks.
- the broad active-motion/low-efficiency condition was retained as a physical
  descriptor but `active_retry` was restricted to residual failures, because
  the condition is expected for most non-successful 520-step trajectories and
  is not a useful mechanism when a specific label is already present.
- failure-subtype prediction was expanded to leave-one-worker-out evaluation
  with CV-unit bootstrap intervals after the pilot showed that an absolute
  omission model could otherwise be mistaken for a phase/initial-state effect.

No layer, denoise step, metric direction, branch, or worker was removed based
on an observed route effect. The max-statistic family was enlarged to include
both token families. Because these additions followed pilot outcome review,
their results are reported as exploratory rather than independent confirmatory
evidence.

After the first 12 committed snapshots (three per worker, 192 branches) had
been analyzed, two exploratory coordinates were frozen as a family for later,
directed validation:

- at `2026-08-29T01:17:01Z`, q0 action tokens, layers 2--5, denoise step 0,
  expert 0 probability, with higher probability predicting failure;
- at `2026-08-29T01:22:04Z`, q0 state tokens, layers 2--5, denoise step 1,
  top-1 mass, with a higher top-vs-bottom-quartile value predicting failure.

The discovery set is snapshot indices 0--2 for every worker. Snapshots already
complete, in flight, or with candidate outcomes visible at freeze time form a
transition set and are excluded from the prospective test. The validation set
starts at snapshot 4 for workers 0--2 and snapshot 5 for worker 3, the first
respective snapshot boundaries that had not started. The two directed primary
tests use within-snapshot permutations and are Bonferroni-corrected as a family
of size two. A best-of-16 ranking test is reported separately as an operational
effect, not substituted for the primary statistic. No more frozen signals will
be added. The full generic and 1,280-cell screens remain exploratory even if
they agree with these tests.
