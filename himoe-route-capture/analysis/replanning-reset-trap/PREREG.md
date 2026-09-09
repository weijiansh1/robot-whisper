# Replanning-reset trap: frozen offline analysis

Frozen before running `analyze_replanning_reset_trap.py` on 2026-08-27.

## Question

When a chunked VLA returns to a previously visited physical configuration, do
its aligned HB-MoE routes and emitted action chunk also return, without task
progress?  The proposed failure mechanism is

`physical return -> routing return -> action replay -> no progress`.

This is different from the existing fixed-lag periodicity audit.  For each
query `k`, this analysis first chooses the most physically similar eligible
history query `j*`, then measures route and action return at that pair.

## Available and unavailable observations

- Physical proxy `X`: recorded MuJoCo `qpos`, excluding simulation time and
  velocity, standardized within task.  With fixed cameras and deterministic
  rendering this determines the visual scene, but it is not a stored RGB or
  vision-encoder embedding.
- Routing `R_state`: actual recorded top-4 HB experts at the state token,
  aligned by layer and denoise position.
- Routing `R_action`: actual recorded top-4 HB experts at all action tokens,
  aligned by layer, denoise position, and token position.
- Action `A`: the recorded 10 x 7 action chunk, standardized within task.
- Progress `P`: distance of task-object positions from held-out successful
  terminal poses, using the same target-object construction as the existing
  failure-mode audit.
- The flow-noise seed is fixed within each rollout, so every within-episode
  return comparison is a same-seed comparison.
- RGB frames were not stored.  Exact snapshot re-inference under multiple
  seeds and vision-embedding recurrence are not identifiable from this cache.

## Pair construction

Primary history search: lag 2 through lag 8 control queries.  For every
eligible `k`, choose

`j* = argmin_j RMS(z(qpos[k]) - z(qpos[j]))`.

Lag 1 is excluded because it is the already-established stasis/persistence
case.  The following frozen sensitivity searches are also run:

- lag 3 through 8;
- lag 2 through 16;
- lag 2 through all available history.

No label is used to choose `j*`.

## Pair metrics

- `X similarity = exp(-0.5 * physical_rms_distance^2)`.
- `R_state similarity`: mean top-4 set overlap over aligned HB layers at the
  state token and denoise position 0.
- `R_action similarity`: mean top-4 set overlap over aligned HB layer x
  denoise x action-token cells.
- `A similarity = exp(-0.5 * standardized_action_rms_distance^2)`.
- `P similarity = exp(-0.5 * (abs(progress_delta) / 0.01 m)^2)`.
- `joint return score`: geometric mean of X, R_action, A, and P similarity.

The episode score is the 90th percentile of query-level joint return scores.
The percentile, rather than the maximum, is frozen to reduce single-pair
winner's-curse sensitivity.

## Primary comparison

For each task, use the longest prefix shared by every one of its 512 episodes.
Success and failure are compared only within `task x initial_state` strata.
The primary endpoint is failure AUC for `joint_return_p90`, with a one-sided
within-stratum permutation test against AUC 0.5 and an initial-state cluster
bootstrap confidence interval.

Interpretation:

- AUC above 0.5 supports enrichment of replanning-reset geometry in failures.
- A null does not exclude loops after the shared prefix.
- A positive endpoint remains associative; it does not establish that routing
  caused the repeated action.

Seven component/control metrics are reported with a family-wise max-statistic:
physical return, state-route return, action-route return, action return,
route-reset score, action-replay score, and joint return score.

## Secondary comparisons

1. Same-length terminal windows.  This can see late slips but is phase
   confounded and is descriptive.
2. Failure-only full trajectories, grouped by the frozen failure-mode rules.
   `reached_then_lost` is the most direct loop prediction; small counts are
   reported without being promoted to the primary endpoint.
3. Slip events: a target object transitions from more than 1 cm above its
   episode-start height to not lifted, while farther than 5 cm from its
   successful terminal goal.  Its post-slip query is matched to all earlier
   history with lag >=2.
4. Repeated lift attempts: a second or later false-to-true lift transition for
   the same target object.  The query immediately before it is compared with
   the corresponding earlier attempt query.
5. Four-way counts among physically recurrent pairs, with thresholds learned
   only from successful equal-prefix pairs: route repeat/different x action
   repeat/different.  These counts are descriptive; the continuous score is
   primary.

## Guards

- Real routing comes only from `server/routes.zarr`; client NPZ routing stubs
  are never read.
- Recorded expert IDs are authoritative; top-4 is not reconstructed from fp16
  probabilities.
- All Zarr stores are opened in read-only mode.
- Equal-prefix results are kept separate from terminal and failure-only
  results.
- State-token routing is not called visual-token routing.  This model has no
  independently recorded visual-token MoE stream.
