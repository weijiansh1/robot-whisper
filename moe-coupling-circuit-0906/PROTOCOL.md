# Coupling-evidence circuit protocol

Frozen on 2026-09-06, before rendering any counterfactual observation and before
running any policy query in this study.

## Question

The routing interface can tell a coupled object from an object that stayed
behind (`moe-progress-ratio-v12-0906/results/object_readability`, AUC 0.84 at
L15 on action tokens by relative query 3, against a coupled-versus-coupled null
centred on 0.500). That is a readout claim about an observational contrast. It
does not say the policy *uses* the evidence, and it does not say which
computation would carry it.

This study asks the computational question directly:

> When object-interaction evidence changes, which computation turns it into a
> change in the action chunk, and where does that computation stop working when
> the object is lost?

## Why the corpus cannot answer this by selection

The behaviour "notices the object is gone, revises the action" is absent from
the corpus:

- `analysis_trap_taxonomy/results/belief_mismatch_summary.json`:
  `belief_mismatch_successes` is 0 in both 16,000-episode runs. 0 of ~229
  phantom-grasp events sits in a successful episode.
- `himoe-route-capture/analyze_failure_behavior_taxonomy.py`: `active_retry`
  0/307 failures, `eef_oscillation` 0/307.
- `analysis_moe_execution_signals/RECOVERY_HORIZON_PROTOCOL.md`: 0/23 event
  states recovered under either arm.

So there is no observed correct revision to use as a reference, and none can be
manufactured from the policy's own outputs. The reference computation has to
come from a behaviour the policy *does* perform. This protocol supplies it as a
**graded pre-closure positive control**, and treats post-closure coupling loss
as the case to be explained.

## Cohort

Primary task: `libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`,
capture `right-50x8-20260903`, 400/400 successes.

Replication task, run only after the primary is complete and analysed:
`libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
same capture label, 400/400 successes.

Episodes are taken in `episode_index` order. Every episode in these captures
succeeded, so ordering carries no outcome information. Target object joint from
`analysis_trap_taxonomy/results/candidate_target_audit.json`; state slices from
the capture's own `sim_layout.json` (`[time] + qpos + qvel`, `state_dim` 92 for
the replication task).

Two anchor queries per episode, both derived from the gripper aperture in
`state[:, 6:8]` and the target's position in `sim_state[:, lo:lo+3]`, with no
reference to any policy internal:

- `q_pre`: the last query at which the aperture is still open and the
  end-effector is within 0.10 m of the target. The approach.
- `q_post`: the first query after aperture closure at which the target has
  lifted at least 0.02 m. Coupling is established.

## Conditions

Each condition restores one `sim_state` row, writes the target's free-joint
slice, calls `forward()`, and re-renders both cameras.

| id | anchor | edit | role |
|---|---|---|---|
| `held` | either | none | reference |
| `shift2`, `shift4`, `shift8` | `q_pre` | target translated 2 / 4 / 8 cm in the table plane, away from the end-effector, quaternion kept, qvel zeroed | graded positive control |
| `uncoupled` | `q_post` | target written to its pose at query 0, qvel zeroed | the case to be explained |
| `light` | either | camera light position and table texture perturbed, object untouched | "vision changed, action should not" control |

The shift is **perpendicular**, in the table plane, to the horizontal component
of (target minus end-effector). Amended 2026-09-06 before any rendering: the
first draft shifted the target along that component, which changes the distance
to the target but barely changes its bearing, and `B_dir` is a bearing readout.
A perpendicular shift is the displacement that `B_dir` is able to see. Sign is
the counter-clockwise rotation in the table plane; if that sign fails a validity
gate the opposite sign is tried, and the realised sign is recorded per row. If
the horizontal component is under 1 cm the bearing is undefined and the episode
is dropped, recorded with a reason. Never dropped silently.

The `light` condition must clear the same visibility floor as the object edits.
A lighting control that changes no pixels is not a control.

## Frozen validity gates

A row is admitted only if all of these hold. Thresholds are not revisited after
rendering.

1. **Restoration.** Same tolerances as `render_rich_event_inputs.py`: position
   2e-3 m, rotation 1e-2 rad, gripper 1e-4, and the unedited coordinates of the
   restored sim state within 1e-9 of the source row.
2. **No interpenetration.** After `forward()`, the minimum distance between the
   target geom and every robot geom is at least 5 mm in `held`; in `uncoupled`
   and the shifts, the target must not overlap any other body.
3. **Support.** In `uncoupled` and the shifts the target rests on a surface: at
   least one contact with a non-robot geom, and vertical position within 5 mm
   of its query-0 height.
4. **Rendering.** Both cameras render at 224x224x3 and neither is constant.
5. **Visibility.** The edit must be present in the observation: at least 1% of
   pixels differ by at least 8/255 in at least one camera, against `held`. The
   full distribution is reported, not only the pass rate.

Gate 5 is the one the conclusion depends on. Without it a null response is
uninterpretable, because "the evidence never entered the observation" and "the
evidence entered and was not used" are different claims and this study exists to
separate them.

### Amendments to the gates, all made before any policy query

- **q_post** now also requires the end effector to have departed
  `EEF_DEPARTURE_M` = 0.10 m from the closure pose, matching the taxonomy's
  phantom-grasp definition. At the first query with a 2 cm lift the closed
  gripper is still 2 cm above the table, so writing the object back to its
  resting pose puts it inside the fingers; that rejected every post anchor.
- **Gate 2 applies only to conditions that move the object.** `light` renders
  the unedited state, so after closure it inherits the gripper's contacts with
  the carried object. Gating it as an object edit rejected every post anchor a
  second time.
- **Gate 1's gripper tolerance is 1e-3 m, not the 1e-4 copied from
  `render_rich_event_inputs.py`.** Measured over 60 episodes of this capture:
  with the gripper closed and static the restore error is about 1e-6; with the
  fingers open and moving it is median 1.14e-4, p90 2.02e-4, max 3.41e-4. At
  1e-4 the gate rejected 49 of 208 approach anchors over a 0.1 mm discrepancy in
  finger position, three orders below the 20-80 mm displacement under test and
  half the already-accepted position tolerance. The threshold was deciding on
  quantisation, not on fidelity. This is the one amendment made after seeing a
  gate bind; it is recorded here with the measurement that motivated it, and it
  touches input fidelity only -- no policy had been run and no outcome existed.
- **Budgets are per anchor.** Post anchors admit at close to 100% and were
  exhausting a shared budget before the approach anchors, which are the stage-1
  gate, could fill.

## Readout

For a chunk `A` of shape (10, 7), all three components are reported for every
condition; the primary is named per anchor.

- `B_dir`: cosine between the cumulative commanded translation and the unit
  vector from the end-effector to the target's **current** position.
  **Primary at `q_pre`.**
- `B_grip`: mean gripper command over the chunk, and chunks to first reopen.
  **Primary at `q_post`.**
- `B_mag`: norm of the cumulative commanded translation.

Registered secondary, added 2026-09-06 before any policy query: `B_track`, the
cosine between the *difference* in cumulative commanded translation
(`edited` minus `held`, horizontal components) and the realised shift direction.
Its null is exactly 0 and it does not depend on how the approach was already
aimed, so it is the more sensitive test of "did the command follow the object".
It is secondary because `B_dir` was registered first; it is not promoted later
whatever the two of them do.

`B_dir` against `B_mag` is what separates "adjusted toward the expected place"
from "moved more, or more randomly". `||A_held - A_edited||` is reported as a
diagnostic and is never the effect size.

Flow noise is `_noise_at(flow_noise_seed, query)`, identical across conditions
within a row. Two replay modes, reported separately:

- **local**: one forward pass at a fixed flow time and fixed noisy latent; the
  quantity is the predicted vector field. This is the patching substrate.
- **full chunk**: fixed initial noise, complete denoising; the quantity is
  `B(A)`. This is where every behavioural claim lives.

Recorded in advance: once a patch is applied, a shared initial seed does not
imply shared intermediate latents, so any downstream difference is propagation
and is not attributed to a later module reading the observation directly.

## Staged gates

Each stage may end the study. A stage that ends it produces a written negative
result, not a relaxed threshold.

**Stage 1 — is there a working evidence-to-action computation at all?**
At `q_pre`, `B_dir` under `shift8` must differ from `held` by an
episode-clustered bootstrap interval excluding zero, in the direction of the
displaced object, while `light` does not. If `shift8` produces no directional
response, the premise of the study fails and stages 3 and 4 are not run.

**Stage 2 — the phenomenon.** At `q_post`, the `uncoupled` response is measured
on the same scale. The prediction registered here is that it is small relative
to the `shift` dose-response curve; the interpretable statement is the ratio,
not a significance test on a null.

**Stage 3 — coarse localisation, at `q_pre` only.** Patch between the `held` and
`shift8` forward passes at the same latent, in both directions, at: residual
stream in and out of each of the 8 HB blocks; attention output against MoE
output within a block; routed branch total against shared branch; state token
against action tokens separately. The last of these tests, and does not assume,
the path `V -> state token -> action token`.
Controls: a norm-matched random direction, and a `light` donor.
**Gate**: some module must recover at least 25% of the `held`-to-`shift8` gap in
`B_dir` while the random-direction control recovers under 10%. If none does, the
harness is not sensitive enough to support a mechanism claim, and stage 4 is not
run.

**Stage 4 — gate against content**, at the modules that passed stage 3.
Recipient background is the `shift8` pass; donor is `held`.

| arm | what is taken from the donor |
|---|---|
| `mask_only` | the selected expert set, recipient weights renormalised over it |
| `weight_only` | the gate weights, restricted to the recipient's own set and renormalised |
| `route_full` | set and weights, experts recomputed on the recipient's input |
| `content_only` | the expert outputs, recipient's own set and weights |
| `shared_only` | the shared-branch output |
| `route_and_content` | both |
| `magnitude` / `direction` | routed vector's norm, or its unit direction |

`content_only` requires evaluating, on the donor input, experts the donor did
not select. They are evaluated explicitly; no other expert is substituted for a
missing one.

## Predictions registered in advance

Stage 4 is run to separate these. Each predicts a different pattern, and the
patterns are mutually exclusive on the primary readout.

| hypothesis | registered prediction |
|---|---|
| gating schedules the behaviour | `route_full` recovers at least 50% of the gap; `content_only` under 20%; `mask_only` carries most of `route_full` |
| experts transform the input | `content_only` at least 50%; `route_full` under 20% |
| the routed branch is a gain | `magnitude` recovers more than `direction`, and the effect lands on `B_mag` and `B_grip` rather than on `B_dir` |
| the evidence bypasses routing | `shared_only` at least matches the routed arms, or no MoE arm passes stage 3 while attention-output patching does |
| no specific mechanism found | every arm produces undirected perturbation: `B_mag` moves, `B_dir` does not |

## Stage 5 — where it breaks

The stage-3 and stage-4 arms are then re-run at `q_post` with `uncoupled` as the
edited condition, to place the failure at one of four stages. These are
candidate mechanisms to be tested, not findings:

1. the evidence is not present at the MoE input;
2. it is present at the input but the gate does not read it;
3. the gate reads it but the experts write nothing that differs;
4. it is written and cancelled downstream.

## What this study does not claim

It does not test closed-loop recovery, does not estimate any success-rate
change, and does not identify a unique circuit. Redundancy is not resolvable
here: a module whose ablation does nothing may be unimportant or may be
compensated for, and a large effect from removing several modules may be damage
rather than a path. Patched activations are off the natural distribution, and
every effect is relative to the chosen background and donor.

Round 1 is a single task plus one replication task. Nothing here supports a
task-general statement.
