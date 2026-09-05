# External online-alarm validation seal

This secondary validation was declared after evaluating the frozen 14,800-episode
main cohort but before reading outcomes from the three tasks that completed after
the earlier retrospective audit.

- Cohort: the three task names in `online_multihead_alarm.py::EXTERNAL_TASKS`,
  1,200 trajectories total.
- Method: replay the already frozen v1 scorer without changing features, heads,
  smoothing, calibration, thresholds, or operating points.
- Primary detector and operating point: `multi_max`, q95.
- Baselines: the same typed heads, `dual_mean`, `dual_max`,
  `instant_multi_max`, and `clock` already present in the v1 artifact.
- Evaluation: eventual failure recall, success false-positive rate, precision,
  and alarms at least four queries before termination.
- No physical subtype/onset claim is planned for this small three-task holdout.

Frozen scorer SHA-256:
`64f5390f935beb11ea6d9be830c6e9f50aefe876cd185008386d7a5a84facb8a`

The score-generation program may parse only its whitelisted non-outcome summary
fields. Outcome fields are revealed by a separate evaluator after the external
score manifest is written.
