# t34 closure audit

All 512 long-task rollouts still have a query at absolute index 34. The outcome is eventual failure, and every model is scored only on success/failure pairs within the same held-out initial state.

## Risk-set audit

- Median success length: 39.0; median failure length: 52.0.
- t34 is phase 0.895 for the median success and 0.667 for the median failure.
- Exact future-remaining-time overlap contains 1 successes and 216 failures: Gate FAIL.

## Model ladder

| model | within-init AUC | pooled AUC |
|---|---:|---:|
| M0_proprio | 0.765 | 0.843 |
| M1_full_state | 0.848 | 0.939 |
| M2_full_state_action | 0.828 | 0.931 |
| M3_M2_state_route | 0.860 | 0.939 |
| M4_M2_action_route | 0.857 | 0.944 |
| M5_M2_hidden | 0.843 | 0.931 |

## Increment over full state + action

| addition | delta | 95% init-bootstrap CI | approximate MDE80 | passes +0.05 |
|---|---:|---:|---:|---|
| M3_M2_state_route | +0.032 | [-0.015, +0.083] | 0.069 | no |
| M4_M2_action_route | +0.029 | [-0.008, +0.077] | 0.063 | no |
| M5_M2_hidden | +0.015 | [-0.021, +0.053] | 0.053 | no |

## Post-hoc sentinels

`remaining_queries_posthoc` AUC is 0.998 and `-phase_episode_posthoc` AUC is 0.998. These variables are unavailable online; they diagnose the corpus geometry only.

## Verdict

The active-at-t34 condition is satisfied, but future remaining time has essentially no two-outcome overlap. The routing MDE80 values (0.063-0.069) also exceed the +0.05 practical threshold, so the finite-sample negative does not exclude a +0.05 effect. Independently of power, the failed overlap gate means this cache can test concurrent state decodability at t34 but cannot turn it into a phase-matched early-warning claim.
