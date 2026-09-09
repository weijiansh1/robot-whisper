# Calibration-Free Ordinal MDL Experiment

## Verdict

Implemented and evaluated one alternative that needs no threshold calibration,
reference corpus, or outcome-guided configuration search. It is **not an
accuracy improvement and is not recommended as a replacement for v7**.
The original v7 implementation and results were left unchanged.

This is a completed negative experiment, not a claim that every calibration-free
method must fail. Its continuous code-gain and candidate-change outputs remain
available for investigation. Do not treat either as a calibrated failure risk.

## What changed

The [monitor](../method/ordinal_mdl_guard.py) reuses v7's raw MoE mobility,
denoising acceleration, and lag-recurrence extraction. It replaces the global
profile, numerical score thresholds, fixed warm-up baselines, smoothing widths,
and persistence counts with causal ordinal symbols and a fixed universal code.

At each query it compares stable, freeze-change, and joint-turbulence-change
explanations. A change has to pay the cost of identifying its location and
mechanism. The best shorter explanation supplies a candidate; the actual
detection query and estimated earlier change query are recorded separately.
The two turbulence features must support the same split, unlike v7's separately
latched threshold crossings. This is a new decision mechanism and representation,
NOT an ablation that changes only numerical threshold values.

The [protocol](../method/ORDINAL_MDL_GUARD_PROTOCOL.md) specifies the complete
formula. The principle follows [minimum description length](https://web.mit.edu/6.433/www/handouts/minimumdescriptionlength.pdf)
and a [KT integrated code](https://users.cecs.anu.edu.au/~kee/jair-aixi-ctw.pdf).
These sources motivate coding, not the validity of this MoE detector.

## What "no threshold" means here

| Property | New candidate |
| --- | --- |
| q97.5 / q70 / q65 alarm quantiles | Removed |
| Global reference profile and periodicity scale | Removed |
| Outcome-guided parameter grid | None in this experiment |
| Learned failure classifier or offline fitting | None |
| Runtime input | Current completed MoE tensor and its episode history |
| Fixed design choices | Feature directions, back layers, lags, ordinal reference, code convention |
| Online adaptation | Historical order statistics and symbol counts |
| Binary decision boundary | Still implicit in selecting the shorter code |
| Controlled false-positive rate | Not guaranteed |

Thus this is **calibration-free, not assumption-free or boundary-free**.
The historical lower median supplies an ordinal reference, not a selected alarm
cutoff. An interpretation of "no training" that also forbids all adaptive
statistical estimation would exclude the symbol-count decoder as well.

## Evaluation

All new streams and decisions were sealed at `2026-09-06T13:30:47.521137+00:00`
before opening outcome CSVs. Exactly one candidate was evaluated: no subsequent
search over its code priors, feature subsets, or decision rule was performed.
The author had already seen related results on these cohorts; this is not
outcome-blind method discovery, and external_8b is not a pristine holdout.

External_8b, 15,600 trajectories, including 564 original-horizon failures:

| Method | TP | FP | Precision | Failure recall | Success false-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing v7 | 439 | 80 | 84.59% | 77.84% | 0.532% |
| Ordinal MDL | 234 | 1,731 | 11.91% | 41.49% | 11.512% |

Development_main, 14,800 trajectories, including 487 original-horizon failures:

| Method | TP | FP | Precision | Failure recall | Success false-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing v7 | 382 | 67 | 85.08% | 78.44% | 0.468% |
| Ordinal MDL | 199 | 1,839 | 9.76% | 40.86% | 12.848% |

External recall before the midpoint of the recorded episode increased from
46/564 to 79/564, but with 1,731 rather than 80 false alarms. Overall recall at
least four queries before the endpoint decreased from 58.69% to 34.93%.
The detected-failure median endpoint lead decreased from 12 to 8.5 queries.
These results do not support an improved early-warning claim. No physical
failure-onset annotations were used in this experiment.

On external object and spatial tasks alone the new detector produced 623 and
689 false alarms. A plausible explanation is that ordinal structural changes
also accompany successful execution, and ordinal encoding discards the size of
the change. That explanation is a hypothesis, not an identified causal effect.
The results establish that this rule's structural-change decision is insufficient
as a reliable failure decision, not that successful MoE states never change.

Complete results, task-clustered intervals, and episode-level timing are in
[outcome_metrics.csv](../results/ordinal_mdl_guard/outcome_metrics.csv),
[suite metrics](../results/ordinal_mdl_guard/outcome_metrics_by_suite.csv), and
[episode_alarms.csv](../results/ordinal_mdl_guard/episode_alarms.csv).
`mdl_estimated_keypoint_at_alarm` must never be used as the actual alarm time.

## Verification and use

- 41 tests passed across the v7 test suite, including 24 new tests.
- The ternary integrated code sums to probability one on enumerated sequences.
- Streaming and batch results agree; future mutation, truncation, and invalid
  suffix padding cannot change earlier decisions.
- Constant features and consistent monotonic drift do not alone force an alarm.
- Seal-time outcome access is blocked in a test; changed outputs fail verification.
- 16 label-blind uniformly sampled raw episodes, 248 queries, exactly reproduce
  cached decisions and keypoints. These include 4 episodes with a new-method
  alarm. All raw replay and future-prefix checks passed on CPU.

The sample size is not exhaustive raw-corpus verification. Full evaluation uses
30,400 aligned cached trajectories. The
[raw audit](../results/ordinal_mdl_guard/raw_replay_verification.json) records
source paths, raw hashes, numerical errors, and exact decision checks.

```python
from ordinal_mdl_guard import OrdinalMDLGuard

monitor = OrdinalMDLGuard()  # New instance for each episode; no profile to load.
record = monitor.update(hb_router_probs)
gain = record["gain_bits"]
candidate = record["candidate_query"]
decision_query = record["first_alarm_query"]
```

`gain_bits` can be negative or negative infinity when there is no eligible
alternative. A candidate with nonpositive gain is not a selected change.
The first-alarm fields latch; `alarm_now` and current evidence can recover.

Run from the workspace root:

```bash
python -m pytest -q moe-v7-0905/tests
python moe-v7-0905/experiments/evaluate_ordinal_mdl_guard.py --stage evaluate
python moe-v7-0905/experiments/verify_ordinal_mdl_raw.py
```

For a new full replay, give `--output` a new directory. Sealing refuses to
overwrite an existing directory, and evaluation verifies source/output hashes
before loading labels. Both new score archives retain all query-level gains,
selected mechanisms, candidate splits, and first-alarm fields.
