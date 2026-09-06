# Leave-One-Suite-Out 任务无关性验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 4 折 leave-one-suite-out 重新校准 v7 的四个语料常量，测出它在完全未见 suite 上的真实指标，并把退化分解成阈值数值（L1）与 operating point（L2）两个来源。

**Architecture:** 两个新模块加进 `moe-v7-0905/experiments/`。`loso_folds.py` 是纯函数层——折掩码、校准、报警、每折 640 点选择，不做任何产物 I/O；`evaluate_loso_suite.py` 是编排层——跑 4 折 × 2 级、按 v7 的封存纪律先写 profile/alarm/哈希再打开 external outcome、落盘全部 CSV/JSON。计分、指标、约束逻辑一律从已有的 `intrinsic_guard_monitor.py` / `evaluate_intrinsic_guard_v7.py` / `select_operating_point.py` 导入复用，不复制实现——这是"退化来自校准而非实现差异"这一结论成立的前提。

**Tech Stack:** Python 3.13、numpy、pandas、pytest。纯 CPU，无 GPU，无 rollout 重跑，全部特征读已有 npz 缓存。

Spec: `docs/superpowers/specs/2026-09-05-loso-task-agnostic-validation-design.md`

---

## 背景：新工程师必须先知道的五件事

1. **数据流**。三个 cohort 缓存都是 `[episode, query, ...]` 对齐的 npz：
   - 路由 mobility：`moe-v4-0904/results/layerwise_mobility/{main_reference,extra_reference,external_8b}.npz`，
     字段 `task_names / task_index / episode / init_state_id / length / valid / mobility`，
     `mobility` 形状 `[n, T, 8]` float32，`valid` 形状 `[n, T]` bool。
   - 标量路由特征：`double-selete/trainfree/results/.../unlabeled_query_features.npz`，
     用 `feature(cache, "route_acceleration")` 和 `feature(cache, "lag_periodicity")` 取出 `[n, T]`。
   - 标签 CSV：`double-selete/trainfree/results/timeout_extension_plus10/`。
   `reference = main(14,800) ⊕ extra(1,200)` 沿 axis 0 拼接 = 16,000，所以 **reference 的前 14,800 行就是 main 行、顺序一致**。这个事实后面要用。

2. **task 名格式**是 `"<suite>/<task>"`，例如 `libero_long/put_the_yellow_and_white_mug...`。suite 靠 `split("/", 1)[0]` 取。

3. **校准与报警用的不是同一个分数流**。`intrinsic_score_arrays` 返回 `freeze / acceleration / periodicity`（平滑后）和
   `acceleration_persistent / periodicity_persistent`（再经 K 次确认的 min-over-window）。
   v7 的阈值标定用**平滑流**的 trajectory peak，报警判定用**persistent 流**。见
   `evaluate_intrinsic_guard_v7.py:158-167` 与 `:182-188`。必须原样照搬，不要"顺手改成一致"。

4. **`periodicity_scale` 是分数的除数，不是阈值**（`intrinsic_guard_monitor.py:182-184`）。它随折变化，
   所以每折的分数数组必须整个重算，不能复用 v7 的分数缓存。这是本实验唯一容易写错的地方。

5. **精度陷阱**：`global_profile.npz` 里存的是 float32，但 evaluate 运行时传给
   `intrinsic_score_arrays` 的 `periodicity_scale` 是 Python float（float64）。要复现 sealed 报警，
   必须用**重算的 float64 值**，不能用存盘的 float32 值。

---

## File Structure

| 文件 | 责任 |
|---|---|
| `moe-v7-0905/experiments/loso_folds.py`（新建） | 纯函数：suite 推导、折掩码、常量校准、报警、每折 640 点选择。无产物 I/O。 |
| `moe-v7-0905/experiments/evaluate_loso_suite.py`（新建） | 编排：4 折 × 2 级、封存纪律、全部产物落盘。 |
| `moe-v7-0905/tests/test_loso_suite.py`（新建） | 单元测试（折隔离、校准复现）+ 产物断言（漂移表、隔离、provenance）。 |
| `moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md`（新建） | 中文报告。 |
| `moe-v7-0905/results/loso_validation/`（新建目录） | 全部产物。 |

**不修改任何已有文件。** `results/intrinsic_guard_v7/` 下的封存产物只读。

---

## Task 1: 折构造

**Files:**
- Create: `moe-v7-0905/experiments/loso_folds.py`
- Create: `moe-v7-0905/tests/test_loso_suite.py`

- [ ] **Step 1: 写失败测试**

创建 `moe-v7-0905/tests/test_loso_suite.py`：

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import loso_folds  # noqa: E402
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_LAYER,
    EXTRA_LAYER,
    MAIN_LAYER,
    load_npz,
)


EXPECTED_CALIBRATION = {
    "libero_goal": 12_000,
    "libero_long": 12_000,
    "libero_object": 12_000,
    "libero_spatial": 12_000,
}
EXPECTED_DEVELOPMENT = {
    "libero_goal": 10_800,
    "libero_long": 12_000,
    "libero_object": 10_800,
    "libero_spatial": 10_800,
}
EXPECTED_EVALUATION = {
    "libero_goal": 4_000,
    "libero_long": 4_000,
    "libero_object": 3_600,
    "libero_spatial": 4_000,
}


def test_fold_sizes_match_the_spec_table() -> None:
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    external_layer = load_npz(EXTERNAL_LAYER)
    reference_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    main_suite = loso_folds.suite_of(main_layer)
    external_suite = loso_folds.suite_of(external_layer)

    assert set(loso_folds.SUITES) == set(np.unique(reference_suite))
    for held_out in loso_folds.SUITES:
        assert int((reference_suite != held_out).sum()) == EXPECTED_CALIBRATION[held_out]
        assert int((main_suite != held_out).sum()) == EXPECTED_DEVELOPMENT[held_out]
        assert int((external_suite == held_out).sum()) == EXPECTED_EVALUATION[held_out]


def test_calibration_never_contains_a_held_out_task() -> None:
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    reference_task = np.concatenate(
        (loso_folds.task_of(main_layer), loso_folds.task_of(extra_layer))
    )
    reference_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    for held_out in loso_folds.SUITES:
        calibration = set(reference_task[reference_suite != held_out])
        held = set(reference_task[reference_suite == held_out])
        assert calibration & held == set()
        assert len(held) > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'loso_folds'`

- [ ] **Step 3: 写最小实现**

创建 `moe-v7-0905/experiments/loso_folds.py`：

```python
#!/usr/bin/env python3
"""Leave-one-suite-out folds and calibration for the v7 intrinsic routing guard."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Per-trajectory `<suite>/<task>` name."""
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def suite_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Per-trajectory suite name."""
    return np.asarray([name.split("/", 1)[0] for name in task_of(cache)])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: PASS，2 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/loso_folds.py moe-v7-0905/tests/test_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: add leave-one-suite-out fold construction

Derives per-trajectory suite names and asserts the four fold sizes match
the design table: 12,000 calibration trajectories per fold, 10,800-12,000
development trajectories, and 3,600-4,000 held-out evaluation trajectories.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 常量校准与全语料复现锚点

四个常量的校准逻辑。锚点：用**全语料**（掩码全 True）跑本模块，必须复现 `global_profile.npz` 里的四个数。

**Files:**
- Modify: `moe-v7-0905/experiments/loso_folds.py`
- Modify: `moe-v7-0905/tests/test_loso_suite.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_loso_suite.py` 顶部的 import 块追加：

```python
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_FEATURE,
    EXTRA_FEATURE,
    MAIN_FEATURE,
    combine_reference,
    feature,
)
```

并把已有的 `from evaluate_intrinsic_guard_v7 import (...)` 合并成一个 import（保留
`EXTERNAL_LAYER, EXTRA_LAYER, MAIN_LAYER, load_npz` 四项）。

在文件末尾追加：

```python
RESULT = BUNDLE / "results/intrinsic_guard_v7"


def full_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    extra_feature = load_npz(EXTRA_FEATURE)
    return combine_reference(main_layer, extra_layer, main_feature, extra_feature)


def test_full_corpus_calibration_reproduces_the_published_profile() -> None:
    mobility, acceleration, periodicity = full_reference()
    scale = loso_folds.periodicity_scale_of(periodicity)
    peaks = loso_folds.reference_peaks(mobility, acceleration, periodicity, scale)
    constants = loso_folds.constants_from_peaks(
        peaks, scale, *loso_folds.PUBLISHED_QUANTILES
    )
    with np.load(RESULT / "global_profile.npz", allow_pickle=False) as published:
        np.testing.assert_array_equal(
            np.float32(constants.freeze_threshold), published["freeze_threshold"]
        )
        np.testing.assert_array_equal(
            np.float32(constants.acceleration_threshold),
            published["acceleration_threshold"],
        )
        np.testing.assert_array_equal(
            np.float32(constants.periodicity_threshold),
            published["periodicity_threshold"],
        )
        np.testing.assert_array_equal(
            np.float32(constants.periodicity_scale), published["periodicity_scale"]
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py::test_full_corpus_calibration_reproduces_the_published_profile -v`
Expected: FAIL，`AttributeError: module 'loso_folds' has no attribute 'periodicity_scale_of'`

- [ ] **Step 3: 写最小实现**

在 `loso_folds.py` 的 `SUITES = (...)` 之后、`task_of` 之前插入 import 与常量：

```python
from evaluate_intrinsic_guard_v7 import PERIODICITY_SCALE_QUANTILE  # noqa: E402
from intrinsic_guard_monitor import (  # noqa: E402
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)

PUBLISHED_QUANTILES = (0.975, 0.70, 0.65)
PEAK_STREAMS = ("freeze", "acceleration", "periodicity")


@dataclass(frozen=True)
class FoldConstants:
    """The four corpus constants plus the quantile levels that produced them."""

    freeze_threshold: float
    acceleration_threshold: float
    periodicity_threshold: float
    periodicity_scale: float
    freeze_quantile: float
    acceleration_quantile: float
    periodicity_quantile: float
```

在文件末尾追加：

```python
def periodicity_scale_of(periodicity: np.ndarray) -> float:
    """Robust scale that divides the recurrence score. float64 on purpose."""
    finite = np.abs(periodicity[np.isfinite(periodicity)])
    if len(finite) == 0:
        raise ValueError("no finite periodicity values in the calibration slice")
    return float(np.quantile(finite, PERIODICITY_SCALE_QUANTILE, method="linear"))


def reference_peaks(
    mobility: np.ndarray,
    acceleration: np.ndarray,
    periodicity: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """Per-trajectory peaks of the three smoothed streams, as v7 calibrates them."""
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, scale)
    return {name: row_max(scores[name]) for name in PEAK_STREAMS}


def constants_from_peaks(
    peaks: dict[str, np.ndarray],
    scale: float,
    freeze_quantile: float,
    acceleration_quantile: float,
    periodicity_quantile: float,
) -> FoldConstants:
    return FoldConstants(
        freeze_threshold=quantile_higher(peaks["freeze"], freeze_quantile),
        acceleration_threshold=quantile_higher(
            peaks["acceleration"], acceleration_quantile
        ),
        periodicity_threshold=quantile_higher(
            peaks["periodicity"], periodicity_quantile
        ),
        periodicity_scale=scale,
        freeze_quantile=freeze_quantile,
        acceleration_quantile=acceleration_quantile,
        periodicity_quantile=periodicity_quantile,
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: PASS，3 passed

如果这一条不过，**先停下来查**：说明本模块的校准路径和 v7 evaluator 不等价，后面所有 LOSO
数字都不可信。最可能的原因是把 `periodicity_scale` 提前转成了 float32（见背景第 5 条）。

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/loso_folds.py moe-v7-0905/tests/test_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: add fold calibration with a published-profile anchor

Calibrating on the full 16,000-trajectory corpus with the published
quantiles reproduces global_profile.npz exactly, so any later LOSO
degradation comes from the calibration slice rather than from a
reimplementation difference.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 报警与 sealed 外部报警回归锚点

**Files:**
- Modify: `moe-v7-0905/experiments/loso_folds.py`
- Modify: `moe-v7-0905/tests/test_loso_suite.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_loso_suite.py` 末尾追加：

```python
def test_published_constants_reproduce_the_sealed_external_alarms() -> None:
    mobility, acceleration, periodicity = full_reference()
    scale = loso_folds.periodicity_scale_of(periodicity)
    peaks = loso_folds.reference_peaks(mobility, acceleration, periodicity, scale)
    constants = loso_folds.constants_from_peaks(
        peaks, scale, *loso_folds.PUBLISHED_QUANTILES
    )

    external_layer = load_npz(EXTERNAL_LAYER)
    external_feature = load_npz(EXTERNAL_FEATURE)
    alarms = loso_folds.guard_alarms(
        loso_folds.cohort_scores(
            external_layer["mobility"],
            feature(external_feature, "route_acceleration"),
            feature(external_feature, "lag_periodicity"),
            scale,
        ),
        external_layer["valid"].astype(bool),
        constants,
    )
    with np.load(RESULT / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard"):
            np.testing.assert_array_equal(alarms[name], sealed[f"external_{name}"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py::test_published_constants_reproduce_the_sealed_external_alarms -v`
Expected: FAIL，`AttributeError: module 'loso_folds' has no attribute 'guard_alarms'`

- [ ] **Step 3: 写最小实现**

在 `loso_folds.py` 的 `intrinsic_guard_monitor` import 块中补上三个函数：

```python
from intrinsic_guard_monitor import (  # noqa: E402
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)
```

在文件末尾追加：

```python
def cohort_scores(
    mobility: np.ndarray,
    acceleration: np.ndarray,
    periodicity: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """All prefix-causal score streams for one cohort slice under one fold scale."""
    return intrinsic_score_arrays(mobility, acceleration, periodicity, scale)


def guard_alarms(
    scores: dict[str, np.ndarray],
    valid: np.ndarray,
    constants: FoldConstants,
) -> dict[str, np.ndarray]:
    """First-alarm query per trajectory, -1 when the rollout never alarms.

    Thresholds are calibrated on the smoothed streams but applied to the
    persistent streams, exactly as evaluate_intrinsic_guard_v7 does.
    """
    first_freeze = first_from_score(
        scores["freeze"], constants.freeze_threshold, valid
    )
    first_acceleration = first_from_score(
        scores["acceleration_persistent"], constants.acceleration_threshold, valid
    )
    first_periodicity = first_from_score(
        scores["periodicity_persistent"], constants.periodicity_threshold, valid
    )
    first_turbulence = first_and(first_acceleration, first_periodicity)
    return {
        "freeze": first_freeze,
        "acceleration": first_acceleration,
        "periodicity": first_periodicity,
        "turbulence": first_turbulence,
        "guard": first_or(first_freeze, first_turbulence),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: PASS，4 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/loso_folds.py moe-v7-0905/tests/test_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: add guard alarm evaluation with a sealed-alarm regression anchor

The published constants reproduce all five sealed external first-alarm
arrays element by element, pinning the LOSO scoring path to the v7
evaluator.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 每折 operating point 选择（L2）

复用 `select_operating_point.py` 的网格、约束与排序，只换输入数据。infeasible 时返回 `None`，
**不放宽约束**。

**Files:**
- Modify: `moe-v7-0905/experiments/loso_folds.py`
- Modify: `moe-v7-0905/tests/test_loso_suite.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_loso_suite.py` 末尾追加：

```python
def test_fold_selection_grid_is_640_points_and_uses_only_calibration_suites() -> None:
    import pandas as pd

    from evaluate_intrinsic_guard_v7 import LABEL_ROOT, aligned_labels

    held_out = "libero_object"
    main_layer = load_npz(MAIN_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    mobility, acceleration, periodicity = full_reference()
    main_suite = loso_folds.suite_of(main_layer)
    keep = main_suite != held_out

    scale = loso_folds.periodicity_scale_of(periodicity)
    peaks = loso_folds.reference_peaks(mobility, acceleration, periodicity, scale)
    development = loso_folds.cohort_scores(
        main_layer["mobility"][keep],
        feature(main_feature, "route_acceleration")[keep],
        feature(main_feature, "lag_periodicity")[keep],
        scale,
    )
    labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    ).loc[keep].reset_index(drop=True)

    candidates, primary = loso_folds.select_fold_operating_point(
        peaks, development, main_layer["valid"].astype(bool)[keep], labels
    )
    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == 640
    assert int(labels.shape[0]) == 10_800
    assert set(labels["suite"]) == set(loso_folds.SUITES) - {held_out}
    assert primary is None or set(
        ["freeze_quantile", "acceleration_quantile", "periodicity_quantile"]
    ) <= set(primary.index)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py::test_fold_selection_grid_is_640_points_and_uses_only_calibration_suites -v`
Expected: FAIL，`AttributeError: module 'loso_folds' has no attribute 'select_fold_operating_point'`

- [ ] **Step 3: 写最小实现**

在 `loso_folds.py` 的 import 区追加（放在 `intrinsic_guard_monitor` import 之后）：

```python
import pandas as pd  # noqa: E402

from select_operating_point import (  # noqa: E402
    ACCELERATION_QUANTILES,
    FREEZE_QUANTILES,
    PERIODICITY_QUANTILES,
    choose,
    metrics,
)
```

在文件末尾追加：

```python
def select_fold_operating_point(
    peaks: dict[str, np.ndarray],
    development: dict[str, np.ndarray],
    valid: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Audit the 640-point quantile grid on the calibration suites only.

    Returns the full candidate table and the primary pick, or None when no
    candidate satisfies the predeclared constraints. The constraints are never
    relaxed: an infeasible fold is a result, not a failure to retry.
    """
    first_freeze = {
        quantile: first_from_score(
            development["freeze"], quantile_higher(peaks["freeze"], quantile), valid
        )
        for quantile in FREEZE_QUANTILES
    }
    first_acceleration = {
        quantile: first_from_score(
            development["acceleration_persistent"],
            quantile_higher(peaks["acceleration"], quantile),
            valid,
        )
        for quantile in ACCELERATION_QUANTILES
    }
    first_periodicity = {
        quantile: first_from_score(
            development["periodicity_persistent"],
            quantile_higher(peaks["periodicity"], quantile),
            valid,
        )
        for quantile in PERIODICITY_QUANTILES
    }
    rows: list[dict[str, float | int]] = []
    for freeze_q, freeze_alarm in first_freeze.items():
        for acceleration_q, acceleration_alarm in first_acceleration.items():
            for periodicity_q, periodicity_alarm in first_periodicity.items():
                guard = first_or(
                    freeze_alarm, first_and(acceleration_alarm, periodicity_alarm)
                )
                rows.append(
                    {
                        "freeze_quantile": freeze_q,
                        "acceleration_quantile": acceleration_q,
                        "periodicity_quantile": periodicity_q,
                        **metrics(guard, labels),
                    }
                )
    candidates = pd.DataFrame(rows).sort_values(
        ["freeze_quantile", "acceleration_quantile", "periodicity_quantile"],
        kind="stable",
    )
    try:
        primary = choose(candidates, conservative=False)
    except RuntimeError:
        primary = None
    return candidates, primary
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: PASS，5 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/loso_folds.py moe-v7-0905/tests/test_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: add per-fold operating point selection

Reuses the 640-point grid, constraints, and ranking from
select_operating_point so the L2 fold pick is byte-identical logic on a
different slice. An infeasible fold returns None instead of relaxing the
predeclared constraints.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 编排器与封存纪律

**Files:**
- Create: `moe-v7-0905/experiments/evaluate_loso_suite.py`

- [ ] **Step 1: 写编排器**

创建 `moe-v7-0905/experiments/evaluate_loso_suite.py`：

```python
#!/usr/bin/env python3
"""Leave-one-suite-out validation of the task-agnostic intrinsic guard."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import loso_folds  # noqa: E402
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_FEATURE,
    EXTERNAL_LAYER,
    EXTRA_FEATURE,
    EXTRA_LAYER,
    LABEL_ROOT,
    MAIN_FEATURE,
    MAIN_LAYER,
    aligned_labels,
    assert_aligned,
    combine_reference,
    feature,
    load_npz,
    metric_row,
    plain,
    sha256,
)

DEFAULT_OUTPUT = BUNDLE / "results/loso_validation"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
FIRST_FREEZE_QUERY = 6
FIRST_TURBULENCE_QUERY = 10
LEVELS = ("published", "loso_l1", "loso_l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def eligible_counts(length: np.ndarray) -> dict[str, int]:
    """Episodes long enough for each branch to be able to fire at all."""
    return {
        "eligible_freeze_n": int((length >= FIRST_FREEZE_QUERY + 1).sum()),
        "eligible_turbulence_n": int((length >= FIRST_TURBULENCE_QUERY + 1).sum()),
    }


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    external_layer = load_npz(EXTERNAL_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    extra_feature = load_npz(EXTRA_FEATURE)
    external_feature = load_npz(EXTERNAL_FEATURE)
    assert_aligned(main_layer, main_feature, "main")
    assert_aligned(extra_layer, extra_feature, "extra")
    assert_aligned(external_layer, external_feature, "external")

    ref_mobility, ref_acceleration, ref_periodicity = combine_reference(
        main_layer, extra_layer, main_feature, extra_feature
    )
    ref_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    is_main = np.arange(len(ref_suite)) < len(main_layer["task_index"])
    main_suite = loso_folds.suite_of(main_layer)
    main_valid = main_layer["valid"].astype(bool)
    external_suite = loso_folds.suite_of(external_layer)
    external_valid = external_layer["valid"].astype(bool)
    external_mobility = external_layer["mobility"]
    external_acceleration = feature(external_feature, "route_acceleration")
    external_periodicity = feature(external_feature, "lag_periodicity")

    # Development outcomes for the calibration suites only; each fold slices
    # this frame before it is used and never reads the held-out rows.
    main_labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    )
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        published_alarms = {
            name: np.asarray(sealed[f"external_{name}"])
            for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard")
        }

    profiles: dict[str, np.ndarray] = {}
    fold_alarms: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {}
    drift_rows: list[dict[str, Any]] = []
    fold_state: dict[str, dict[str, Any]] = {}

    for held_out in loso_folds.SUITES:
        calibration = ref_suite != held_out
        scale = loso_folds.periodicity_scale_of(ref_periodicity[calibration])
        calibration_scores = loso_folds.cohort_scores(
            ref_mobility[calibration],
            ref_acceleration[calibration],
            ref_periodicity[calibration],
            scale,
        )
        peaks = {
            name: loso_folds.row_max(calibration_scores[name])
            for name in loso_folds.PEAK_STREAMS
        }

        # Development rows are exactly the calibration rows that came from main,
        # in main order, so they are sliced rather than recomputed.
        development_rows = is_main[calibration]
        development_scores = {
            name: values[development_rows]
            for name, values in calibration_scores.items()
        }
        development_keep = main_suite != held_out
        development_labels = main_labels.loc[development_keep].reset_index(drop=True)
        candidates, primary = loso_folds.select_fold_operating_point(
            peaks,
            development_scores,
            main_valid[development_keep],
            development_labels,
        )
        candidates.to_csv(output / f"fold_candidates_{held_out}.csv", index=False)

        constants = {
            "loso_l1": loso_folds.constants_from_peaks(
                peaks, scale, *loso_folds.PUBLISHED_QUANTILES
            )
        }
        if primary is not None:
            constants["loso_l2"] = loso_folds.constants_from_peaks(
                peaks,
                scale,
                float(primary["freeze_quantile"]),
                float(primary["acceleration_quantile"]),
                float(primary["periodicity_quantile"]),
            )
        selection[held_out] = {
            "development_episodes": int(len(development_labels)),
            "calibration_episodes": int(calibration.sum()),
            "calibration_suites": sorted(set(loso_folds.SUITES) - {held_out}),
            "candidate_count": int(len(candidates)),
            "feasible": primary is not None,
            "primary": None if primary is None else plain(primary.to_dict()),
        }

        evaluation = external_suite == held_out
        alarms: dict[str, dict[str, np.ndarray]] = {
            "published": {
                name: values[evaluation] for name, values in published_alarms.items()
            }
        }
        for level, fold_constants in constants.items():
            scores = loso_folds.cohort_scores(
                external_mobility[evaluation],
                external_acceleration[evaluation],
                external_periodicity[evaluation],
                fold_constants.periodicity_scale,
            )
            alarms[level] = loso_folds.guard_alarms(
                scores, external_valid[evaluation], fold_constants
            )
            for field, value in asdict(fold_constants).items():
                profiles[f"{held_out}__{level}__{field}"] = np.asarray(
                    value, dtype=np.float64
                )
        for level, level_alarms in alarms.items():
            for name, values in level_alarms.items():
                fold_alarms[f"{held_out}__{level}__{name}"] = values

        fold_state[held_out] = {
            "evaluation": evaluation,
            "alarms": alarms,
            "constants": constants,
        }

        with np.load(PUBLISHED / "global_profile.npz", allow_pickle=False) as published:
            baseline = {
                "freeze_threshold": float(published["freeze_threshold"]),
                "acceleration_threshold": float(published["acceleration_threshold"]),
                "periodicity_threshold": float(published["periodicity_threshold"]),
                "periodicity_scale": float(published["periodicity_scale"]),
            }
        for level, fold_constants in constants.items():
            for field, value in baseline.items():
                fold_value = float(getattr(fold_constants, field))
                drift_rows.append(
                    {
                        "held_out_suite": held_out,
                        "level": level,
                        "constant": field,
                        "published_value": value,
                        "fold_value": fold_value,
                        "absolute_drift": fold_value - value,
                        "relative_drift_percent": 100.0 * (fold_value - value) / value
                        if value != 0.0
                        else float("nan"),
                    }
                )
        print(f"fold {held_out}: sealed {len(alarms)} levels", flush=True)

    np.savez_compressed(
        output / "fold_profiles.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loso.profiles.v1"),
        **profiles,
    )
    np.savez_compressed(
        output / "fold_first_alarms.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loso.alarms.v1"),
        **fold_alarms,
    )
    pd.DataFrame(drift_rows).to_csv(output / "threshold_drift.csv", index=False)
    (output / "fold_selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.loso.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "folds": list(loso_folds.SUITES),
        "held_out_outcomes_used_for_calibration": False,
        "held_out_outcomes_used_for_selection": False,
        "external_outcomes_loaded_after_seal": True,
        "external_cohort_pristine_holdout": False,
        "artifacts": {
            "fold_profiles_sha256": sha256(output / "fold_profiles.npz"),
            "fold_first_alarms_sha256": sha256(output / "fold_first_alarms.npz"),
            "threshold_drift_sha256": sha256(output / "threshold_drift.csv"),
            "fold_selection_sha256": sha256(output / "fold_selection.json"),
            "folds_module_sha256": sha256(HERE / "loso_folds.py"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "loso_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("sealed all folds; opening held-out outcomes now", flush=True)

    write_metrics(args, output, fold_state, external_layer, external_suite)


if __name__ == "__main__":
    main()
```

`write_metrics` 在 Task 6 实现。本步骤先让编排器停在封存结束处。

- [ ] **Step 2: 加一个临时桩让脚本能跑通封存段**

在 `evaluate_loso_suite.py` 的 `def main()` 之前插入：

```python
def write_metrics(
    args: argparse.Namespace,
    output: Path,
    fold_state: dict[str, dict[str, Any]],
    external_layer: dict[str, np.ndarray],
    external_suite: np.ndarray,
) -> None:
    raise NotImplementedError("implemented in Task 6")
```

- [ ] **Step 3: 跑封存段，确认四折都产出且在桩处停下**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python experiments/evaluate_loso_suite.py`
Expected: 打印四行 `fold <suite>: sealed N levels` 与 `sealed all folds; opening held-out outcomes now`，
然后 `NotImplementedError: implemented in Task 6`。

检查产物已落盘：

Run: `ls moe-v7-0905/results/loso_validation/`
Expected: `fold_profiles.npz  fold_first_alarms.npz  threshold_drift.csv  fold_selection.json  loso_manifest.json  fold_candidates_*.csv`

- [ ] **Step 4: 看一眼漂移表，确认数值不是 NaN**

Run: `python -c "import pandas as pd; d=pd.read_csv('moe-v7-0905/results/loso_validation/threshold_drift.csv'); print(d.to_string(index=False)); assert d['fold_value'].notna().all()"`
Expected: 打印全部漂移行，无断言错误

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/evaluate_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: add LOSO orchestrator with seal-before-outcomes discipline

Runs four folds at two leakage levels, writes fold profiles, first alarms,
the threshold drift table, the selection audit, and a hashed manifest
before any held-out outcome file is opened.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 指标、eligible 分母与三级对照

**Files:**
- Modify: `moe-v7-0905/experiments/evaluate_loso_suite.py`

- [ ] **Step 1: 用真实实现替换桩**

把 Task 5 Step 2 插入的 `write_metrics` 桩整体替换为：

```python
def write_metrics(
    args: argparse.Namespace,
    output: Path,
    fold_state: dict[str, dict[str, Any]],
    external_layer: dict[str, np.ndarray],
    external_suite: np.ndarray,
) -> None:
    """Open held-out outcomes and score the three levels side by side."""
    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []

    for held_out, state in fold_state.items():
        evaluation = state["evaluation"]
        labels = external_labels.loc[evaluation].reset_index(drop=True)
        counts = eligible_counts(labels["length"].to_numpy(int))
        frame = labels.copy()
        frame["held_out_suite"] = held_out
        for level in LEVELS:
            if level not in state["alarms"]:
                rows.append(
                    {
                        "held_out_suite": held_out,
                        "level": level,
                        "feasible": False,
                        "episodes": int(len(labels)),
                        **counts,
                    }
                )
                continue
            guard = state["alarms"][level]["guard"]
            row = metric_row(
                "external_8b",
                f"{held_out}__{level}",
                guard,
                labels,
                args.bootstrap,
                rng,
                group=held_out,
            )
            row.update({"held_out_suite": held_out, "level": level, "feasible": True})
            row.update(counts)
            if level in state["constants"]:
                constants = state["constants"][level]
                row.update(
                    {
                        "freeze_quantile": constants.freeze_quantile,
                        "acceleration_quantile": constants.acceleration_quantile,
                        "periodicity_quantile": constants.periodicity_quantile,
                    }
                )
            rows.append(row)
            frame[f"first_{level}_query"] = guard

            tasks = labels["task"].to_numpy(str)
            for task in np.unique(tasks):
                take = tasks == task
                task_row = metric_row(
                    "external_8b",
                    f"{held_out}__{level}",
                    guard[take],
                    labels.loc[take].reset_index(drop=True),
                    args.bootstrap,
                    rng,
                    group=task,
                    intervals=False,
                )
                task_row.update(
                    {"held_out_suite": held_out, "level": level, "task": task}
                )
                task_rows.append(task_row)
        episode_frames.append(frame)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "loso_metrics.csv", index=False)
    pd.DataFrame(task_rows).to_csv(output / "loso_metrics_by_task.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(
        output / "episode_alarms.csv", index=False
    )

    summary = {
        "schema": "himoe.intrinsic_guard_v7.loso.evaluation.v1",
        "macro": {
            level: macro_summary(metrics, level) for level in LEVELS
        },
        "micro": {
            level: micro_summary(metrics, level) for level in LEVELS
        },
        "infeasible_folds": sorted(
            metrics.loc[~metrics["feasible"].astype(bool), "held_out_suite"].unique().tolist()
        ),
    }
    (output / "loso_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        metrics[
            [
                "held_out_suite",
                "level",
                "feasible",
                "tp",
                "fp",
                "risk_recall",
                "precision",
                "timely_fpr",
                "early4_risk_recall",
            ]
        ].to_string(index=False),
        flush=True,
    )
```

- [ ] **Step 2: 加两个汇总函数**

在 `write_metrics` 之前插入：

```python
def macro_summary(metrics: pd.DataFrame, level: str) -> dict[str, float | int]:
    """Equal weight per fold. Infeasible folds are excluded and counted."""
    block = metrics[(metrics["level"] == level) & metrics["feasible"].astype(bool)]
    if block.empty:
        return {"folds": 0}
    return {
        "folds": int(len(block)),
        "risk_recall": float(block["risk_recall"].mean()),
        "precision": float(block["precision"].mean()),
        "timely_fpr": float(block["timely_fpr"].mean()),
        "early4_risk_recall": float(block["early4_risk_recall"].mean()),
    }


def micro_summary(metrics: pd.DataFrame, level: str) -> dict[str, float | int]:
    """Episode-weighted pooling across folds."""
    block = metrics[(metrics["level"] == level) & metrics["feasible"].astype(bool)]
    if block.empty:
        return {"folds": 0}
    tp = int(block["tp"].sum())
    fp = int(block["fp"].sum())
    risk = int(block["risk_n"].sum())
    timely = int(block["timely_n"].sum())
    return {
        "folds": int(len(block)),
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / risk if risk else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "timely_fpr": fp / timely if timely else float("nan"),
    }
```

- [ ] **Step 3: 把敏感度列回写进漂移表**

spec §3.4 要求漂移表带上"该折常量相对基线常量造成的留出 suite 指标变化"。在
`write_metrics` 里 `pd.DataFrame(task_rows).to_csv(...)` 之后、`summary = {...}` 之前插入：

```python
    # spec 3.4: the drift table carries the metric consequence of each drift,
    # so a threshold move can be read against what it actually cost.
    drift = pd.read_csv(output / "threshold_drift.csv")
    keyed = metrics[metrics["feasible"].astype(bool)].set_index(
        ["held_out_suite", "level"]
    )
    baseline = keyed.xs("published", level="level")
    for column in ("risk_recall", "precision", "timely_fpr"):
        drift[f"{column}_vs_published"] = [
            float(keyed.loc[(suite, level), column] - baseline.loc[suite, column])
            if (suite, level) in keyed.index
            else float("nan")
            for suite, level in zip(drift["held_out_suite"], drift["level"], strict=True)
        ]
    drift.to_csv(output / "threshold_drift.csv", index=False)
```

注意这一步会重写 `threshold_drift.csv`，所以 Task 5 写进 `loso_manifest.json` 的
`threshold_drift_sha256` 会与磁盘不符。不要改 Task 5 的封存段，改为在 `write_metrics`
末尾（`loso_summary.json` 写完之后）重新哈希并覆盖：

```python
    manifest_path = output / "loso_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["threshold_drift_sha256"] = sha256(
        output / "threshold_drift.csv"
    )
    manifest["artifacts"]["loso_metrics_sha256"] = sha256(output / "loso_metrics.csv")
    manifest_path.write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
```

封存纪律不受影响：漂移表的**数值**在打开 outcome 之前就已确定，这里只追加它造成的指标后果，
并重新哈希。

- [ ] **Step 4: 跑完整评估**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python experiments/evaluate_loso_suite.py`
Expected: 四折 × 三级的指标表打印出来，无异常；`results/loso_validation/` 下新增
`loso_metrics.csv`、`loso_metrics_by_task.csv`、`episode_alarms.csv`、`loso_summary.json`

- [ ] **Step 5: 核对 published 级复现 v7 的分 suite 数字**

`published` 级是 sealed 报警的切片，所以它按 suite 汇总必须等于 v7 报告的
`outcome_metrics_by_suite.csv`。

Run:
```bash
python -c "
import pandas as pd
new = pd.read_csv('moe-v7-0905/results/loso_validation/loso_metrics.csv')
new = new[new['level']=='published'].set_index('held_out_suite')
old = pd.read_csv('moe-v7-0905/results/intrinsic_guard_v7/outcome_metrics_by_suite.csv')
old = old[(old.cohort=='external_8b')&(old.detector=='intrinsic_guard_v7')].set_index('group')
for suite in new.index:
    assert int(new.loc[suite,'tp']) == int(old.loc[suite,'tp']), suite
    assert int(new.loc[suite,'fp']) == int(old.loc[suite,'fp']), suite
print('published level matches the v7 per-suite table')
"
```
Expected: `published level matches the v7 per-suite table`

- [ ] **Step 6: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/experiments/evaluate_loso_suite.py
git commit -m "$(cat <<'EOF'
feat: score the three LOSO levels with eligible-alarm denominators

Reuses metric_row so the published / LOSO-L1 / LOSO-L2 columns share the
v7 metric definitions, reports macro and micro pooling separately, and
records how many episodes are long enough for each branch to fire.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 隔离与 provenance 测试

**Files:**
- Modify: `moe-v7-0905/tests/test_loso_suite.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_loso_suite.py` 末尾追加：

```python
LOSO = BUNDLE / "results/loso_validation"


def test_fold_selection_used_only_calibration_suite_episodes() -> None:
    selection = json.loads((LOSO / "fold_selection.json").read_text())
    expected = {
        "libero_goal": 10_800,
        "libero_long": 12_000,
        "libero_object": 10_800,
        "libero_spatial": 10_800,
    }
    for held_out, record in selection.items():
        assert record["development_episodes"] == expected[held_out]
        assert record["calibration_episodes"] == 12_000
        assert held_out not in record["calibration_suites"]
        assert record["candidate_count"] == 640


def test_manifest_declares_the_seal_order() -> None:
    manifest = json.loads((LOSO / "loso_manifest.json").read_text())
    assert manifest["held_out_outcomes_used_for_calibration"] is False
    assert manifest["held_out_outcomes_used_for_selection"] is False
    assert manifest["external_outcomes_loaded_after_seal"] is True
    assert manifest["external_cohort_pristine_holdout"] is False
    assert sorted(manifest["folds"]) == sorted(loso_folds.SUITES)


def test_drift_table_covers_every_fold_level_and_constant() -> None:
    drift = pd.read_csv(LOSO / "threshold_drift.csv")
    constants = {
        "freeze_threshold",
        "acceleration_threshold",
        "periodicity_threshold",
        "periodicity_scale",
    }
    assert set(drift["constant"]) == constants
    assert set(drift["held_out_suite"]) == set(loso_folds.SUITES)
    assert drift["fold_value"].notna().all()
    for held_out in loso_folds.SUITES:
        block = drift[(drift["held_out_suite"] == held_out) & (drift["level"] == "loso_l1")]
        assert set(block["constant"]) == constants


def test_published_level_reproduces_the_v7_per_suite_counts() -> None:
    new = pd.read_csv(LOSO / "loso_metrics.csv")
    new = new[new["level"] == "published"].set_index("held_out_suite")
    old = pd.read_csv(RESULT / "outcome_metrics_by_suite.csv")
    old = old[
        (old["cohort"] == "external_8b") & (old["detector"] == "intrinsic_guard_v7")
    ].set_index("group")
    for suite in loso_folds.SUITES:
        assert int(new.loc[suite, "tp"]) == int(old.loc[suite, "tp"])
        assert int(new.loc[suite, "fp"]) == int(old.loc[suite, "fp"])


def test_provenance_hashes_match_the_files_on_disk() -> None:
    from evaluate_intrinsic_guard_v7 import sha256

    manifest = json.loads((LOSO / "loso_manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert artifacts["fold_profiles_sha256"] == sha256(LOSO / "fold_profiles.npz")
    assert artifacts["threshold_drift_sha256"] == sha256(LOSO / "threshold_drift.csv")
    assert artifacts["fold_selection_sha256"] == sha256(LOSO / "fold_selection.json")
    assert artifacts["folds_module_sha256"] == sha256(
        BUNDLE / "experiments/loso_folds.py"
    )
```

并把测试文件顶部的 import 补成：

```python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests/test_loso_suite.py -v`
Expected: 新增的 5 个测试中至少 `test_provenance_hashes_match_the_files_on_disk` 通过，
其余若失败会指出具体缺失字段。若 Task 5/6 已正确落盘，5 个应全部 PASS。

若 `test_provenance_hashes_match_the_files_on_disk` 失败且提示 `evaluator_sha256` 不符，
说明 Task 6 修改脚本后没有重跑——重跑 `python experiments/evaluate_loso_suite.py` 再测。

- [ ] **Step 3: 跑全套测试**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests -q`
Expected: v7 原有 5 个测试 + 本文件 10 个测试全部 PASS

- [ ] **Step 4: 钉住 LOSO 关键数值**

跑完拿到真实数字后，在测试文件末尾追加一个固定指标测试，把 `loso_summary.json` 的
macro `loso_l1` 与 `loso_l2` 的 `risk_recall` / `precision` 四舍五入到小数点后四位写死：

```python
def test_loso_macro_summary_is_pinned() -> None:
    summary = json.loads((LOSO / "loso_summary.json").read_text())
    macro = summary["macro"]
    assert round(macro["loso_l1"]["risk_recall"], 4) == PINNED_L1_RECALL
    assert round(macro["loso_l1"]["precision"], 4) == PINNED_L1_PRECISION
    assert round(macro["loso_l2"]["risk_recall"], 4) == PINNED_L2_RECALL
    assert round(macro["loso_l2"]["precision"], 4) == PINNED_L2_PRECISION
```

`PINNED_*` 四个常量定义在文件顶部 `LOSO = ...` 之后，取值直接从跑出的
`loso_summary.json` 抄写。若 `loso_l2` 的 `folds` 为 0（四折全 infeasible），
则把后两条断言换成 `assert macro["loso_l2"]["folds"] == 0`。

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/tests/test_loso_suite.py
git commit -m "$(cat <<'EOF'
test: pin LOSO fold isolation, drift coverage, and provenance

Asserts each fold's selection touched only calibration-suite episodes,
that the published level reproduces the v7 per-suite counts, that the
drift table covers every fold/level/constant, and that manifest hashes
match the files on disk.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: 报告

**Files:**
- Create: `moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md`
- Modify: `moe-v7-0905/README.md`

- [ ] **Step 1: 取齐报告要用的数字**

Run:
```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python -c "
import json, pandas as pd
s=json.load(open('results/loso_validation/loso_summary.json'))
print(json.dumps(s, indent=2, ensure_ascii=False))
m=pd.read_csv('results/loso_validation/loso_metrics.csv')
print(m[['held_out_suite','level','feasible','tp','fp','risk_recall','precision','timely_fpr','early4_risk_recall','eligible_freeze_n','eligible_turbulence_n']].to_string(index=False))
print(pd.read_csv('results/loso_validation/threshold_drift.csv').to_string(index=False))
"
```

- [ ] **Step 2: 写报告**

创建 `moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md`，按以下章节写，数字全部来自 Step 1 的输出，
不得手改：

1. **结论**——一句话回答 spec §6.3：v7 在未见 suite 上的 macro precision/recall 为 X/Y，
   相对全语料下降 Z 个百分点，其中 operating-point 泄漏（L1−L2 差）占 W 个百分点。
2. **三级对照表**——4 折 × {published, loso_l1, loso_l2}，含 eligible 分母两列。
3. **阈值漂移表**——四常量 × 四折，含相对漂移百分比。
4. **预注册判读**——照 spec §3.4：若 macro precision < 70%，明确写出"v7 不得宣称任务无关，
   必须改述为运行时不读任务身份、但校准依赖任务语料"。判读结果照实写，不论正负。
5. **F-long 单独一节**——它是校准集变成纯短 horizon 语料的那一折，按 spec §1.2 预期最差。
   照实报，不得剔除。
6. **限制**——照抄 spec §7 全部六条，其中 external 非 pristine holdout 一条必须保留。
7. **对实验 B 的结论**——根据漂移幅度给出"秩统计量重构是否必要、要做到多狠"的判断。
8. **复现命令**：
   ```bash
   cd /home/jovyan/work/himoe-vla/moe-v7-0905
   python experiments/evaluate_loso_suite.py
   pytest -q tests
   ```

- [ ] **Step 3: 更新 README**

在 `moe-v7-0905/README.md` 的 `See REPORT_ZH.md ...` 段落之后插入一段，指向新报告，
并把 LOSO macro 数字与 77.84% 并列，明确标注后者是**见过任务**上的结果。

- [ ] **Step 4: 全套验证**

Run: `cd /home/jovyan/work/himoe-vla/moe-v7-0905 && python -m pytest tests -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md moe-v7-0905/README.md moe-v7-0905/results/loso_validation/
git commit -m "$(cat <<'EOF'
docs: report leave-one-suite-out validation results

All 39 external tasks were already inside the 40-task calibration corpus,
so the published 77.84% recall was never an unseen-task result. This
reports what the guard does when an entire suite is held out of
calibration, separating threshold-value drift from operating-point
leakage.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 完成标准

- [ ] `pytest -q moe-v7-0905/tests` 全绿（v7 原有 5 个 + 新增 11 个）
- [ ] `results/loso_validation/` 含 spec §3.6 列出的全部产物
- [ ] 报告能用一句话回答 spec §6.3
- [ ] 报告写明了预注册判读的结果，以及实验 B 的必要性判断
