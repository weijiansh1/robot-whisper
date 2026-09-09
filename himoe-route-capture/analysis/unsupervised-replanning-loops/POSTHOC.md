# Post-unblinding robustness additions

Added after the first outcome-unblinded run on 2026-08-27. These checks are
not part of the frozen primary analysis in `PREREG.md`.

The first run found an overall association for q99 candidate presence, but
inspection showed two risks:

1. the association could be driven by one task;
2. a positive return gain can be arbitrarily small and need not represent a
   substantive loop.

The following diagnostics were therefore added without changing the original
score, q99 threshold, candidate assignments, or cluster assignments:

- per-task outcome validation;
- leave-one-failing-task-out validation;
- cluster-to-task purity;
- absolute recovery fractions for physical state, routing, and action;
- candidate-presence sensitivity requiring the minimum of those three
  recovery fractions to reach 1%, 2.5%, 5%, 10%, or 25%.

The recovery fractions are

- physical: the preregistered physical closure fraction;
- routing: return-overlap gain divided by the full departure from overlap 1;
- action: return-distance gain divided by the maximum interior action
  distance.

All five recovery thresholds are reported as one post-hoc family with a
max-statistic family-wise p-value. They diagnose effect size; they must not be
presented as preregistered endpoints.
