# Long-Suite Guard and Lead-Aware Lock Selection (v5 candidate)

## Motivation

The frozen v4 detector has stable aggregate behavior, but its false alarms are
concentrated in `libero_long`: 60 of 67 development false alarms and 71 of 81
external false alarms.  Non-long suites retain the frozen v4 detector unchanged.

## Development-only candidate family

Only the long-suite lock head is varied.  The candidate family is restricted to
three low-mobility representations already present in the frozen layerwise sweep:

- all-eight-layer median (`all_median`);
- layer L12 alone (`L12`);
- minimum of the four back-layer mobilities (`back_min`).

The causal mean width is 2 or 4, the consecutive-crossing count is 4, 6, 8, 10,
or 12, and the threshold quantile uses the existing fixed development grid.  The
threshold remains an outcome-blind same-task quantile of the per-trajectory
maximum instantaneous oriented score.  The frozen L5 instability head remains
enabled and unchanged.

For every candidate, non-long trajectories use their original sealed v4 alarm.
Long trajectories use the OR of the candidate lock alarm and the sealed v4
instability alarm.

## Selection rule

A development candidate is eligible only when:

- aggregate timely-success FPR is at most 0.5%;
- long-suite timely-success FPR is at most 1%;
- every long-task timely-success FPR is at most 2.5%;
- long-suite precision is at least 75%.

Candidates are ranked by overall recall with at least four queries of lead, then
overall recall with at least eight queries of lead, long-suite early recall,
overall recall, worst long-task recall among tasks with at least five risks, and
finally lower false-alarm rates.

The selected configuration is frozen before its external route replay is joined
to outcomes.  This remains post-hoc iteration on a previously inspected external
cohort and requires a new pristine cohort for confirmation.
