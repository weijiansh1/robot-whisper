# Lead-Aware Task-Conditional Guard (v5 candidate)

## Scope

This candidate treats first-alarm lead time as the primary objective. It keeps
the selected long-suite guard frozen and searches only the non-long lock head.
The L5 instability head is the frozen v4 head everywhere. Runtime input remains
the current rollout's dense `hb_router_probs`; no outcome or future query is an
online input.

## Development-only family

The candidate lock representation is either the all-layer median or one of the
eight individual HB layers. Only low mobility is considered. Causal mean widths
are 1, 2, or 4; confirmation counts are 1, 2, 4, 6, or 8; thresholds use the 14
quantiles in the complete layerwise development sweep.

For each candidate, `libero_long` uses the frozen long guard. Every non-long
trajectory uses the OR of the candidate lock and frozen L5 instability alarm.
Thresholds are same-task, outcome-blind quantiles of maximum instantaneous
oriented score. Development thresholds leave out all eight seeds belonging to
the evaluated initial state.

## Selection

A final-system candidate is eligible only when development:

- overall timely-success FPR is at most 0.5%;
- every suite timely-success FPR is at most 1%;
- every task timely-success FPR is at most 2.5%;
- precision is at least 75%.

Eligible candidates are ranked by risk recall with at least four queries of
lead, then at least eight queries of lead, total recall, lower overall FPR,
higher precision, and finally the less aggressive threshold/configuration.

External 8B is a previously inspected cohort. Its replay is useful iteration
evidence but is not a pristine confirmation result.
