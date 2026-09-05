# HB-MoE token dynamics pilot

## Protocol

- One rollout is one sample. Control rows and denoise iterations are repeated measurements.
- State 0 is discovery; state 42 is untouched confirmation. Both contain 14 successes and 18 timeout failures.
- Horizons: t7 (before the displayed pose split), t12 (before the displayed route split), and t34 (before the length/survivor split).
- Every temporal score uses an 8-control trailing window and aligned token positions. Raw lags 1-5 and recurrence residuals for periods 2-5 are tested.
- Discovery uses a two-sided maxT permutation correction over the complete MoE feature family. The chosen direction is frozen for the one-sided confirmation test.

## Primary result

Feature: `route_top4|h34|lag1|T7`

- Discovery failure AUC: `0.0159`; unadjusted p `0.000050`; maxT p `0.000100`.
- Confirmation AUC in the discovery direction: `0.7262`; one-sided p `0.014099`.

## Discovery-selected family leads

| family | feature | discovery AUC | maxT p | confirmation oriented AUC | confirmation p |
|---|---|---:|---:|---:|---:|
| token_lag | `route_top4|h34|lag1|T7` | 0.0159 | 0.000100 | 0.7262 | 0.015049 |
| descriptor_lag | `route_top4|h34|lag2|action_std` | 0.0159 | 0.000100 | 0.9206 | 0.000100 |
| periodicity | `hidden|h34|period2|T0` | 0.9444 | 0.002550 | 0.9444 | 0.000050 |
| structure_level | `route_top4|h34|level|endpoint_delta_norm` | 0.0714 | 0.006650 | 0.6508 | 0.079346 |

## Source leads

| source | feature | discovery AUC | maxT p | confirmation oriented AUC | confirmation p |
|---|---|---:|---:|---:|---:|
| route_soft | `route_soft|h34|lag1|T0` | 0.0198 | 0.000150 | 0.8452 | 0.000300 |
| route_top4 | `route_top4|h34|lag1|T7` | 0.0159 | 0.000100 | 0.7262 | 0.014749 |
| hidden | `hidden|h34|lag4|T10` | 0.0198 | 0.000150 | 0.8532 | 0.000250 |

## Horizon-specific tests

| horizon | selected feature | discovery AUC | horizon maxT p | all-MoE maxT p | confirmation oriented AUC | confirmation p |
|---:|---|---:|---:|---:|---:|---:|
| t7 | `hidden|h07|lag4|T6` | 0.7976 | 0.654567 | 0.914504 | 0.5000 | 0.511224 |
| t12 | `route_top4|h12|lag4|T2` | 0.1548 | 0.161642 | 0.373081 | 0.4683 | 0.621719 |
| t34 | `route_top4|h34|lag1|T7` | 0.0159 | 0.000050 | 0.000100 | 0.7262 | 0.013999 |

## Behavior baseline

Best discovery behavior feature: `behavior|h34|lag1|action_mean`. Discovery AUC `0.0159` (behavior-family maxT p `0.000050`); confirmation oriented AUC `0.7460` (p `0.009300`).

## Post-hoc behavior residual audit

These checks are diagnostic, not additional confirmatory tests: the targets were chosen after inspecting the raw discovery results.

| target | behavior R2 discovery / confirmation | residual discovery AUC (p) | residual confirmation oriented AUC (p) |
|---|---:|---:|---:|
| `route_top4|h34|lag2|action_std` | 0.951 / 0.889 | 0.456 (0.6984) | 0.659 (0.0664) |
| `hidden|h34|period2|T0` | 0.577 / 0.732 | 0.567 (0.5399) | 0.778 (0.0034) |
| `route_soft|h34|lag1|T0` | 0.972 / 0.958 | 0.456 (0.6901) | 0.484 (0.5735) |

## Lag-curve check for the selected period-2 score

The hidden T0 lag curve remains monotone in both outcomes. The period-2 residual is therefore curvature of a slowing trajectory, not a return to a previous state.

| state | outcome | lag1 | lag2 | lag3 | lag4 | lag5 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | success | 0.2367 | 0.5167 | 0.7222 | 0.8543 | 0.9250 |
| 0 | timeout_failure | 0.0799 | 0.1542 | 0.2145 | 0.2668 | 0.3006 |
| 42 | success | 0.2545 | 0.5848 | 0.8097 | 0.9181 | 0.9466 |
| 42 | timeout_failure | 0.1572 | 0.3390 | 0.4815 | 0.5709 | 0.6404 |

## Interpretation guardrails

- A replicated hidden feature is an HB-block-input marker, not by itself a MoE-specific or causal trap.
- The same 32 noise seed IDs occur in both initial states; state holdout does not create unseen noise seeds.
- t34 is only before episode-length separation. It is after the displayed pose/route separation and cannot establish an early precursor.
