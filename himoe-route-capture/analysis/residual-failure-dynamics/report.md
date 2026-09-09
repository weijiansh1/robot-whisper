# Residual failure routing dynamics

## Direct answer

The consensus core contains 213 of 307 failures. The residual contains 94: 62 with one method vote and 32 with zero votes. 89/94 residual failures come from the long moka-pot task.

Within that task, the residual failures are not routing-normal successes:

- The strongest organization is a continuous activity-to-stasis axis, not another set of discrete failure classes.
- At 0 votes routing remains active or restarts late; at 3 votes its largest change happens early and it then slows strongly. Votes 1 and 2 mostly lie between them.
- Independent robot-state tapes agree: among failures at the same partial-completion stage, low-vote rollouts move the EEF farther and reverse the gripper more often in the second half.

The remaining failures are therefore best read as active retrying or differently timed variants along the same broad noncompletion process. Small subgroups remain candidates, not validated new failure types.

## Composition

| group | all n | long moka-pot n | mean queries |
|---|---:|---:|---:|
| success | 2253 | 296 | 16.8 |
| failure core (2-3 votes) | 213 | 127 | 41.4 |
| failure, 1 vote | 62 | 59 | 50.7 |
| failure, 0 votes | 32 | 30 | 50.1 |

## Long-task dynamics

All phases are relative phases. Speed is divided by each rollout's median speed; no absolute control step is used.

| group | n | late/early speed | return frequency | terminal return | early-mid low speed | late low speed | layer peak sync | change contrast | distance to success speed template |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| success | 296 | 1.120 | 0.119 | 0.226 | 0.019 | 0.002 | 0.615 | 0.122 | 0.129 |
| failure core (2-3 votes) | 127 | 0.691 | 0.165 | 0.433 | 0.010 | 0.066 | 0.923 | 0.548 | 0.616 |
| failure, 1 vote | 59 | 1.054 | 0.275 | 0.424 | 0.124 | 0.107 | 0.725 | 0.217 | 0.260 |
| failure, 0 votes | 30 | 1.437 | 0.217 | 0.533 | 0.422 | 0.078 | 0.725 | 0.256 | 0.275 |

The speed-template distance gives a useful gradient: success is compact, the consensus core is far away, and the one-/zero-vote failures are intermediate. They are therefore weaker or differently timed deviations, not simply missed copies of the core.

## Initial-state-matched check

For every initial state containing both successes and the target failure group, the table subtracts the success mean first, then averages states equally.

| failure group | matched states | delta late/early speed | delta return frequency | delta terminal return | delta early-mid low speed | delta late low speed | delta layer peak sync | delta change contrast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure core (2-3 votes) | 10 | -0.501 | 0.024 | 0.159 | -0.063 | 0.189 | 0.213 | 0.429 |
| failure, 1 vote | 11 | -0.136 | 0.096 | 0.132 | 0.041 | 0.106 | 0.094 | 0.094 |
| failure, 0 votes | 9 | 0.210 | 0.098 | 0.439 | 0.257 | 0.075 | 0.065 | 0.105 |

The zero-vote terminal-return excess is especially large after this control. The pause/restart observation is therefore not only due to difficult initial states being overrepresented.

## Exact-vote axis at identical length

All 216 failed moka-pot rollouts contain exactly 52 queries. The same full-window vote membership is frozen below; `truncate90` recomputes route dynamics on phase 0.50-0.90 and excludes phase 0.90-1.00.

| votes | n | full late/early | trunc late/early | full peak phase | trunc peak phase | full late stasis | trunc late stasis |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 1.437 | 1.440 | 0.827 | 0.770 | 0.067 | 0.100 |
| 1 | 59 | 1.054 | 1.075 | 0.713 | 0.713 | 0.085 | 0.068 |
| 2 | 27 | 0.969 | 0.734 | 0.741 | 0.591 | 0.333 | 0.593 |
| 3 | 100 | 0.617 | 0.571 | 0.547 | 0.522 | 0.880 | 0.900 |

| ordinal trend | full Spearman rho | truncate90 Spearman rho |
|---|---:|---:|
| late/early speed | -0.729 | -0.805 |
| route-speed peak phase | -0.699 | -0.764 |
| late stasis | 0.735 | 0.740 |

The ordering is strong but not perfectly stepwise in every full-window group mean. After terminal truncation, late/early speed and peak timing become strictly ordered across 0/1/2/3 votes. This is more consistent with one severity/timing axis than four discrete mechanisms.

## Independent physical check

The physical measurements below are read from client `state`, `actions`, and `sim_state`; none participates in the routing vote or grouping. EEF path uses query-boundary positions. A gripper flip is a sign change in the mean command of consecutive action chunks.

The terminal proxy labels 198/216 failures as `partial`: exactly one of the two moka pots is placed and the other is incomplete. This supplies a same-task, same-length, same-coarse-stage comparison.

| votes | partial n | EEF path total (m) | EEF path late half (m) | gripper flips | late-half flips | targets lifted | targets placed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 2.197 | 0.901 | 3.926 | 1.926 | 1.037 | 1.000 |
| 1 | 51 | 2.467 | 1.133 | 4.157 | 2.157 | 1.235 | 1.000 |
| 2 | 20 | 1.628 | 0.281 | 2.450 | 0.450 | 1.150 | 1.000 |
| 3 | 100 | 1.576 | 0.253 | 2.080 | 0.080 | 1.030 | 1.000 |

Pooling 0/1 votes, the late-half EEF path is 1.052 m and late gripper flips are 2.077; for 2/3 votes they are 0.258 m and 0.142.

After subtracting 2/3-vote rollouts within each of 7 shared initial states, the 0/1-vote excess remains 0.622 m of late EEF travel and 1.725 late gripper flips.

This supports the plain-language interpretation: the residual tail is often still moving and reopening/reclosing, while the high-consensus core has largely stopped changing physically. It does not show that every retry is useful or that routing causes the retry.

## Why one vote is not one class

Vote bit order is aligned-kernel / event / lag-spectrum.

| pattern | interpretation | n | late/early speed | return frequency | terminal return | early-mid low speed | layer sync | change contrast |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | aligned only; mostly flat-speed recurrence | 42 | 0.982 | 0.345 | 0.595 | 0.056 | 0.664 | 0.112 |
| 010 | event only; terminal slowdown | 9 | 0.721 | 0.097 | 0.000 | 0.000 | 0.875 | 0.437 |
| 001 | lag only; middle pause then reactivation | 8 | 1.806 | 0.109 | 0.000 | 0.625 | 0.875 | 0.519 |

The 59 long-task one-vote failures split into 42/9/8 across these three signatures. Averaging them into one curve hides that structure.

## Zero-vote candidates

The 30 long-task zero-vote failures are already split by the independent event partition into C1 n=6, C2 n=15, C3 n=9.

- Event C2 is the clearest pause/restart candidate: the normalized speed trough is in the early-middle phase, followed by late reactivation.
- Event C3 has repeated nonlocal returns, especially near the end, without a sustained terminal slowdown.
- Event C1 is flatter and has no single dominant route event.

These are hypotheses for targeted replay or physical-event labeling. With n=15/9/6 they should not yet be presented as stable failure types.

## Limits

- Consensus membership and these summaries come from the same routing data, so the analysis is descriptive rather than an independent validation.
- Relative phase removes rollout-length scale but still uses the final episode horizon; it does not establish prediction before the first physical error.
- The residual is dominated by one task. Cross-task claims for its candidate subtypes are not supported.
- A nonlocal route return means an older routing state is closer than the immediately previous state. It does not by itself prove that the robot repeated the same physical action.

## Artifacts

- `summary.json`: counts, curves, scalar profiles, matched-state contrasts, and subgroup profiles.
- `assignments.csv`: episode-level consensus membership, selected dynamics, and independent physical measurements for long-task failures.
- `long_task_dynamics.png`: speed, return, low-speed, and scalar comparisons.
- `vote_axis_and_physics.png`: full/truncated routing trends and partial-failure physical activity.
