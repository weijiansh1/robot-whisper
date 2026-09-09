# History-only MoE pattern audit

This analysis implements the user's no-training constraint strictly: only the
current episode's HB router probabilities through the decision query are used.
No success reference, failure labels, fitted threshold, task identity, initial
state, action, image, physical state, total length, or horizon cap enters a rule.
Outcome labels are read for evaluation only. There is no parameter optimization.

The primary rule is declared before running this analysis: back-layer mobility
falls to at most one quarter of its own initial mobility, smoothed over two
post-baseline queries and confirmed at two consecutive queries. The baseline is
the mean of transitions q1..q4. The first possible primary alarm is q7 (zero
based). An alarm at q is emitted after inference q and before its actions; an
additional q+1 evaluation reports the stricter previous-inference interpretation.

Mobility is the mean action-token Hellinger distance between consecutive final
denoising-step distributions. Ratios are formed per layer before taking the
median over L12..15. Distances use sqrt-probability Euclidean norms to avoid
catastrophic cancellation near identical routes. A zero baseline abstains.

Sensitivity rules are fixed in monitor.fixed_rules(): three layer groups,
half/quarter/eighth mobility, one/two/four confirmations, two/four/eight baseline
transitions; four-transition displacement/path ratios 0.15/0.25/0.35; token
dispersion contraction by half/quarter; joint contraction of mobility and
dispersion. These are descriptive sensitivity analyses, not a labeled search
that selects a winner. Low geometric progress is not assumed to mean physical
failure. A constant route with no measurable path abstains from the loop test.

All complete available LIBERO episodes are indexed directly from source client
summaries and matched against server episode_id/control_step. pin-base is a
known duplicate and pin-on/off are interventions: they are scored separately.
CALVIN subtask boundaries may be reconstructed from the evaluator's fresh action
buffer per subtask, ceiling(environment_steps/replan_steps), and verified against
each recorded chain's inference_calls and the full Zarr row count. These partial
chain collections are reported separately, without treating them as independent
replications or complete benchmark coverage.

Evaluation reports episode false alarms, precision, failure recall, first alarm,
lead to the recorded endpoint, and coverage at fixed query/relative deadlines.
Relative phase and lead are retrospective measurements only. At each fixed
query, only still-running episodes are compared within source run and optionally
initial state. A clock-only diagnostic and matched survival baseline expose the
length confound. Physical failure categories are joined only after scoring.

Existing experiments in this repository have already inspected these cohorts.
This is a reproducible retrospective audit, not a fresh blinded validation.
Finite parameter sweeps cannot establish an information-theoretic upper bound,
nor can an observational alarm establish an irreversible or causal failure point.

## Follow-up sensitivity, after the initial 20-rule replay

The half-threshold rule produced many brief false alarms on successful object
tasks. An exploratory follow-up applies the already considered persistence axis
to the half threshold: K=1,4,8, with baseline=4, width=2, and threshold=0.5
unchanged. All three follow-up results are retained, including the loss of early
recall. These three rules were not preregistered with the original 20. No model
is trained and no numeric threshold is fitted, but choosing to emphasize K=4
after this sweep is exploratory method selection, not independent confirmation.
