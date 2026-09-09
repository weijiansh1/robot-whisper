# v7 With Automatic Unlabeled Threshold Calibration

## Result

Implemented a threshold-calibration replacement that keeps the existing v7
monitor unchanged. The operator supplies one episode-alarm workload budget;
the three numerical decision thresholds are calculated from unlabeled
historical MoE trajectories, without consulting outcomes or the published
thresholds. The primary run uses the declared default budget of 0.03.

This removes outcome-guided threshold selection while retaining most of v7's
observed detection performance. It is not an accuracy improvement over the
label-tuned published profile, and it is not proof of statistical equivalence.
The original v7 and earlier ordinal-MDL experiment were not modified.

## External comparison

Same external_8b cohort: 15,600 episodes, 564 original-horizon failures.

| Method | TP | FP | Precision | Failure recall | Success FPR | Total alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Published v7 | 439 | 80 | 84.59% | 77.84% | 0.532% | 3.327% |
| Automatic calibration | 419 | 92 | 82.00% | 74.29% | 0.612% | 3.276% |

Precision decreased by 2.59 percentage points and recall by 3.55 points. There
were 20 fewer detected failures and 12 more false alarms. The point estimate
tradeoff is substantially smaller than the earlier ordinal replacement, but
it should not be advertised as a detection improvement.

The reference alarm rate is exactly 3%; the external alarm rate is 3.276%.
**The budget is an empirical reference-set ceiling, not an external hard quota
and not a class-conditional false-positive-rate guarantee.** It is calibrated
on all reference outcomes mixed together. A distribution shift can change it.

Development_main contains 14,800 episodes and overlaps the reference corpus:

| Method | TP | FP | Precision | Failure recall | Total alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published v7 | 382 | 67 | 85.08% | 78.44% | 3.034% |
| Automatic calibration | 350 | 76 | 82.16% | 71.87% | 2.878% |

Development performance is not out-of-reference generalization evidence.
External_8b has also been inspected in earlier research, so it is not a fresh
holdout. The inherited mechanism and feature design used outcome feedback;
only the NEW threshold-calibration procedure is label-free.

## What stayed unchanged

- The current `[8, 10, 11, 32]` routing tensor is the only runtime observation.
- Continuous, per-layer initial normalization and amplitude information.
- Freeze W6; acceleration W3/K8; recurrence W6/K4; original baseline windows.
- `freeze OR (acceleration AND recurrence)` with independently latched heads.
- One global four-scalar profile, no task identity, future observations, horizon
  input, fitted failure model, learned weights, or per-task thresholds.
- The historical q75 periodicity scale; its value exactly matches published v7.

The [calibrator](../method/unlabeled_budget_calibration.py) returns a standard
`GlobalIntrinsicProfile`. The original `IntrinsicGuardMonitor` loads and uses
the new profile without any runtime code changes.

## How the thresholds are determined

Use all 16,000 historical reference trajectories. For a budget of 0.03, allow
at most 480 reference episodes to trigger at least once. Full-trajectory peaks
retain the observed opportunity for repeated crossings, rather than treating
each query as a separate episode.

The two Boolean branches share a marginal alarm allowance m. The freeze
threshold is an empirical order statistic. The turbulence branch uses a
common percentile for acceleration and recurrence, selected by the observed
count where both persistent streams cross. A monotone integer search finds
the largest m whose actual UNION of branch alarm sets stays within 480.
No branch-independence assumption is used.

The selected allowance was 283 per branch:

```text
283 freeze alarms + 283 turbulence alarms - 86 overlaps = 480 total alarms
```

The thresholds produced by this deterministic procedure are:

| Constant | Published v7 | Automatic calibration |
| --- | ---: | ---: |
| Freeze threshold | 0.57406306 | 0.76633531 |
| Acceleration threshold | 0.10750750 | 0.07495879 |
| Recurrence threshold | 0.37491593 | 0.24855959 |
| Periodicity scale | 0.02070767 | 0.02070767 |

These are outputs, not hand-selected settings. The freeze empirical level is
approximately q98.23; the common turbulence level is q51.68. The complete
[calibration audit](../results/unlabeled_budget_guard/calibration_audit.json)
records all nine integer search probes, overlaps, thresholds, and the exact
post-serialization reference replay count.

Equal branch allowances and a shared turbulence percentile are explicit design
conventions. They do not optimize precision or guarantee that all feasible
three-threshold combinations are searched. The procedure and conventions are
specified in the [frozen protocol](../method/UNLABELED_BUDGET_CALIBRATION_PROTOCOL.md).

## Diagnostic ablations

These were all sealed in advance. The primary method was NOT selected from
their outcome metrics.

| External variant | TP | FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Only freeze threshold replaced | 377 | 46 | 89.13% | 66.84% |
| Only turbulence thresholds replaced | 469 | 124 | 79.09% | 83.16% |
| All thresholds replaced, primary | 419 | 92 | 82.00% | 74.29% |

The automatic freeze boundary is stricter, reducing both detections and false
alarms. The automatic turbulence boundaries are looser, increasing both.
Partial replacements still use label-selected published thresholds elsewhere;
their higher precision or recall cannot be presented as a fully label-free
calibration result.

## Timing and remaining limitations

External recall at least four queries before the original endpoint changes
from 58.69% to 56.21%. Recall at least eight queries before the endpoint is
44.50% versus 45.57%; these are different detected subsets, not a paired timing
improvement. Median lead among detected failures is 12 versus 13 queries.
Recall before the recorded midpoint is 46/564 versus 47/564.

None of these endpoint-relative numbers identifies the first physical failure
or establishes that intervention would rescue the episode. The calibration
continues to depend on the reference corpus and its horizon distribution.
Existing v7 survival/length limitations remain; this change does not solve them.

External per-suite TP/FP for the automatic profile are goal 62/1, long 237/85,
object 22/3, and spatial 98/3. Relative to v7, the main recall loss is in goal
and spatial, while the extra false alarms are concentrated in long tasks.
Full task/suite tables and task-clustered intervals are saved with the results.

## Verification

- All 66 tests passed, including 25 new calibration/integration checks.
- Budget limits, tie handling, missing peaks, monotonicity, and exact agreement
  between the integer search and brute-force enumeration are tested.
- Row permutation and invalid suffix padding do not change calibration.
- New profiles round-trip through the existing v7 loader, with reference
  replay reproducing the exact calibration count.
- Different thresholds leave v7's continuous score streams unchanged; future
  mutation cannot change earlier online results.
- A full seal succeeds with CSV outcome access explicitly blocked in a test;
  changing a sealed profile is detected before evaluation.
- All five published v7 first-alarm arrays exactly reproduce their prior seals
  on both cohorts. Original source cache hashes also match.
- Sixteen label-blind raw episodes, eight alarming and eight non-alarming,
  cover 406 queries. All raw scores, exact first-alarm queries, and future-prefix
  checks passed on CPU. This is sampled, not exhaustive raw-corpus verification.

The new profile and all fixed-variant alarm arrays were sealed at
`2026-09-06T14:09:18.182713+00:00`, before outcome CSV access in the evaluator.
The [seal](../results/unlabeled_budget_guard/sealed_manifest.json) includes
source/input/output hashes; [raw verification](../results/unlabeled_budget_guard/raw_replay_verification.json)
records individual episodes and numerical errors.

## Use and reproduce

From an environment where `moe-v7-0905/method` is on the Python import path:

```python
from pathlib import Path
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

profile = GlobalIntrinsicProfile.load(
    Path("moe-v7-0905/results/unlabeled_budget_guard/global_profile.npz")
)
monitor = IntrinsicGuardMonitor(profile)  # New instance per episode.
result = monitor.update(hb_router_probs)
```

There is no per-head threshold choice in this deployment path. The generated
profile has not silently replaced the published v7 profile.

```bash
python -m pytest -q moe-v7-0905/tests
python moe-v7-0905/experiments/evaluate_unlabeled_budget.py --stage evaluate
python moe-v7-0905/experiments/verify_unlabeled_budget_raw.py
```

For a new reference calibration, choose a workload budget BEFORE inspecting its
outcome metrics and supply a new output directory, for example:

```bash
python moe-v7-0905/experiments/evaluate_unlabeled_budget.py \
  --alarm-budget 0.03 --stage all \
  --output moe-v7-0905/results/unlabeled_budget_reproduction
```

Sealing refuses to overwrite an existing directory. Evaluation verifies the
seal and cannot change its budget. Do not search budgets using failure labels
and then call the chosen result label-free calibration.
