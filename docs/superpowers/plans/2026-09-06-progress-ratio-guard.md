# 进度率保护器 v12 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并评估进度率 `R = 净路由位移 / 路径长度`，检验它能否在严格任务无关、train-free 的约束下，同时提高低先验区间的召回与精确度。

**Architecture:** 新建自包含 bundle `moe-progress-ratio-v12-0906/`，结构与 `moe-v7-0905/` 对齐。`method/` 放纯函数与在线 monitor，`experiments/` 放缓存构建、开发集选点、证否闸门、external 评估。判据只有一条：`R(q,W) < r*` 连续确认 `K` 次。评估口径是逐 chunk 超额 hazard，不是 episode 二值精确度。

**Tech Stack:** Python 3.13、numpy、pandas、zarr、pytest；GPU 部分用 torch（`CUDA_VISIBLE_DEVICES=6,7`，H20-3e）。

**Spec:** `docs/superpowers/specs/2026-09-06-progress-ratio-guard-design.md`

---

## 关键背景（实现者必读）

你可能对这个代码库没有上下文。三件事决定了本计划的形状：

1. **`risk` 的定义是"没能在 horizon 上限前完成"**，所以"还在跑"本身就是强预测量。存活先验从
   chunk 0 的 0.07 涨到接近上限的 0.95+。因此**不能用 episode 级 precision 作为主指标**——晚报警
   会免费拿到高 precision。主指标是相对存活先验的提升倍数。
2. **数据已经缓存好，不需要重跑任何 rollout。** 路由原始张量在 Zarr 里，逐 query 形状
   `[8, 10, 11, 32]` = `[层, 去噪步, token, 专家]`。token 0 是 state，1–10 是 action。
   只用 `FINAL_FLOW = 9` 的 action token。
3. **`R` 的路径长度 `L` 可以完全由已有的 v4 mobility 缓存算出**（它就是相邻 query 的 Hellinger），
   只有净位移 `D`（lag-W 距离）需要回到 Zarr 重算。这是本计划最省事的一步，别重算 `L`。

### 文件结构

| 文件 | 职责 |
|---|---|
| `method/progress_ratio.py` | 纯函数：概率归一化、Hellinger、lag 距离、`L`/`D`/`R`、持续确认、下尾分位数。无 I/O |
| `method/progress_monitor.py` | 在线 monitor：环形缓冲，逐 query 吐 `R` 与报警状态 |
| `method/ONLINE_PROGRESS_GUARD_V12_PROTOCOL.md` | 协议声明，随 manifest 一起哈希 |
| `experiments/build_progress_cache.py` | 从 Zarr 批量算 lag 距离并缓存，可 resume |
| `experiments/select_operating_point_v12.py` | 开发集扫 `W` × 层组 × `K` × 分位数，两种阈值模式 |
| `experiments/evaluate_phenotype_2x2.py` | K1 / K3 闸门诊断 |
| `experiments/evaluate_baseline_dnf.py` | 方案 B 对照基线 |
| `experiments/evaluate_hazard.py` | 逐 chunk 超额 hazard 评估，三个先验切点，K4 / K5 / K6 |
| `experiments/evaluate_loso_v12.py` | LOSO 三级对照 |
| `experiments/verify_raw_causal_gpu.py` | 原始 Zarr 因果回放 |
| `tests/` | 度量性质、因果性、与 v4/v7 的锚点一致性 |

### 数据源常量（照抄，不要改）

```python
WORKSPACE = <repo root>                                   # /home/jovyan/work/himoe-vla
ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
COHORTS = {
    "development_main": {"cache": V4 / "main_reference.npz", "root": ROUTE_ROOT},
    "development_extra": {"cache": V4 / "extra_reference.npz", "root": ROUTE_ROOT},
    "external_8b":      {"cache": V4 / "external_8b.npz",      "root": ROUTE_ROOT},
}
LABEL_ROOT = WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
```

每个 cohort 的 npz 含：`task_names`、`task_index`、`episode`、`init_state_id`、`length`、
`valid [E, 52]`、`layer_names (8,)`、`mobility [E, 52, 8]`。
Zarr 路径为 `root / task / run_id / "server/routes.zarr"`，含 `episode_id` 与 `hb_router_probs`。
`run_id` 取自 cohort npz 的 `run_id` 字段。

---

## Task 1: 建立 bundle 骨架

**Files:**
- Create: `moe-progress-ratio-v12-0906/README.md`
- Create: `moe-progress-ratio-v12-0906/requirements.txt`
- Create: `moe-progress-ratio-v12-0906/method/__init__.py`（空文件）
- Create: `moe-progress-ratio-v12-0906/tests/__init__.py`（空文件）

- [ ] **Step 1: 创建目录与文件**

```bash
cd /home/jovyan/work/himoe-vla
mkdir -p moe-progress-ratio-v12-0906/{method,experiments,tests,results,docs}
touch moe-progress-ratio-v12-0906/method/__init__.py
touch moe-progress-ratio-v12-0906/tests/__init__.py
```

`requirements.txt`：

```text
numpy>=1.26
pandas>=2.1
zarr>=2.16
torch>=2.1
pytest>=7.4
```

`README.md`：

```markdown
# 进度率保护器 v12

判据：`R(q, W) = d(z_q, z_{q-W}) / sum_{k=1..W} d(z_{q-k+1}, z_{q-k}) < r*`，连续确认 K 次。

`R` 是同单位之比，无量纲，因此不需要 episode 基线也不需要语料常数定尺度。
低 `R` 同时覆盖冻结（路径短且净位移小）与兜圈（路径长但净位移小）；
高 `R` 否决慢速成功，那是当前误报的主体。

设计见 `docs/superpowers/specs/2026-09-06-progress-ratio-guard-design.md`。
主指标是相对存活先验的提升倍数，不是 episode 级精确度。

复现顺序见 `docs/REPRODUCE.md`（由 Task 13 写入）。
```

- [ ] **Step 2: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906
git commit -m "chore: scaffold the progress-ratio v12 bundle"
```

---

## Task 2: 概率原语与 v7 锚点

**Files:**
- Create: `moe-progress-ratio-v12-0906/method/progress_ratio.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_progress_primitives.py`

各 bundle 按本仓库既有惯例自包含（`moe-v7-0905` 与 `moe-hb-front-back-0905` 各有一份
`normalize`/`hellinger`）。因此这里复制这两个小函数，并用一条锚点测试强制它与 v7 数值一致。

- [ ] **Step 1: 写失败测试**

`tests/test_progress_primitives.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(WORKSPACE / "moe-v7-0905/method"))

import progress_ratio as pr  # noqa: E402


def test_hellinger_matches_v7_bitwise():
    """本 bundle 自带的 hellinger 必须与 v7 的实现数值一致。"""
    import intrinsic_guard_monitor as v7

    rng = np.random.default_rng(20260906)
    left = rng.random((7, 32), dtype=np.float32)
    right = rng.random((7, 32), dtype=np.float32)
    np.testing.assert_allclose(
        pr.hellinger(left, right), v7.hellinger(left, right), rtol=0, atol=0
    )


def test_hellinger_is_a_metric_on_simple_cases():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert pr.hellinger(a, a) == pytest.approx(0.0, abs=1e-6)
    assert pr.hellinger(a, b) == pytest.approx(1.0, abs=1e-6)


def test_action_route_selects_final_flow_and_action_tokens():
    rng = np.random.default_rng(1)
    probs = rng.random((8, 10, 11, 32)).astype(np.float32)
    routes = pr.action_route(probs)
    assert routes.shape == (8, 10, 32)
    np.testing.assert_allclose(routes.sum(axis=-1), 1.0, rtol=1e-6)
    # token 0 是 state，必须被排除
    expected = probs[:, 9, 1:11, :]
    expected = expected / expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(routes, expected, rtol=1e-5)


def test_action_route_rejects_wrong_shape():
    with pytest.raises(ValueError, match="router probability"):
        pr.action_route(np.zeros((8, 10, 11, 16), dtype=np.float32))


def test_route_distance_reduces_tokens():
    rng = np.random.default_rng(2)
    left = rng.random((8, 10, 32)).astype(np.float32)
    right = rng.random((8, 10, 32)).astype(np.float32)
    assert pr.route_distance(left, right).shape == (8,)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_primitives.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'progress_ratio'`

- [ ] **Step 3: 写实现**

`method/progress_ratio.py`：

```python
#!/usr/bin/env python3
"""Train-free, task-agnostic progress ratio over HiMoE routing trajectories.

R(q, W) = d(z_q, z_{q-W}) / sum_{k=1..W} d(z_{q-k+1}, z_{q-k})

The numerator is net displacement and the denominator is path length, both in
Hellinger units, so the ratio is dimensionless.  Task scale cancels by
construction rather than by an episode baseline.
"""

from __future__ import annotations

import numpy as np


SCHEMA = "himoe.progress_ratio_v12.profile.v1"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
LAYER_GROUPS = {"front": slice(0, 4), "back": slice(4, 8), "all": slice(0, 8)}
FINAL_FLOW = 9
ACTION = slice(1, 11)
N_EXPERTS = 32
N_ACTION_TOKENS = 10
EPSILON = 1e-6

W_GRID = (2, 3, 4, 6, 8, 10, 12)
LAGS = (1,) + W_GRID


def normalize_probability(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def action_route(hb_router_probs: np.ndarray) -> np.ndarray:
    """[..., 8, 10, 11, 32] -> [..., 8, 10, 32] final-flow action routes."""
    probability = np.asarray(hb_router_probs, dtype=np.float32)
    expected = (len(LAYER_NAMES), 10, 11, N_EXPERTS)
    if probability.shape[-4:] != expected:
        raise ValueError(
            f"router probability must end with shape {expected}, got {probability.shape}"
        )
    return normalize_probability(probability[..., FINAL_FLOW, ACTION, :])


def route_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Mean-over-action-token Hellinger.  [..., 8, 10, 32] x same -> [..., 8].

    A positive-coefficient mean of metrics is a metric, so the triangle
    inequality survives the token reduction.  That is what guarantees D <= L.
    """
    return hellinger(left, right).mean(axis=-1, dtype=np.float32)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_primitives.py -v
```

Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/method/progress_ratio.py moe-progress-ratio-v12-0906/tests/test_progress_primitives.py
git commit -m "feat: add routing distance primitives anchored to the v7 implementation"
```

---

## Task 3: L、D、R 与度量性质

**Files:**
- Modify: `moe-progress-ratio-v12-0906/method/progress_ratio.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_progress_geometry.py`

**索引约定（写错会静默毁掉全部结果，务必对齐）：**
`adjacent[e, q, l] = d(q, q-1)`，`q = 0` 处为 NaN。
`L(q, W) = sum_{k=1..W} d(q-k+1, q-k) = sum(adjacent[e, q-W+1 : q+1, l])`，因此 `q >= W` 才有效。
`D(q, W) = d(q, q-W) = lag_distance[e, q, index_of(W), l]`，同样 `q >= W` 才有效。

- [ ] **Step 1: 写失败测试**

`tests/test_progress_geometry.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402


def _random_walk_routes(n_query: int, seed: int) -> np.ndarray:
    """[Q, 8, 10, 32] 的一条平滑路由轨迹。"""
    rng = np.random.default_rng(seed)
    base = rng.random((8, 10, 32)).astype(np.float32)
    routes = np.empty((n_query, 8, 10, 32), dtype=np.float32)
    current = base
    for q in range(n_query):
        current = np.abs(current + 0.05 * rng.standard_normal(current.shape).astype(np.float32))
        routes[q] = current / current.sum(axis=-1, keepdims=True)
    return routes


def test_lag_distances_lag1_equals_adjacent_hellinger():
    routes = _random_walk_routes(12, seed=3)
    lags = pr.lag_distances(routes, (1, 4))
    expected = pr.route_distance(routes[1:], routes[:-1])
    np.testing.assert_allclose(lags[1:, 0], expected, rtol=1e-6)
    assert np.isnan(lags[0, 0]).all()
    assert np.isnan(lags[:4, 1]).all()


def test_displacement_never_exceeds_path_length():
    """三角不等式：D <= L。这条一旦破了，R 就不在 [0, 1] 里。"""
    routes = _random_walk_routes(30, seed=4)
    lags = pr.lag_distances(routes, pr.LAGS)
    adjacent = lags[None, :, 0, :]
    for window in pr.W_GRID:
        length = pr.path_length(adjacent, window)
        displacement = lags[None, :, pr.LAGS.index(window), :]
        both = np.isfinite(length) & np.isfinite(displacement)
        assert both.any()
        assert np.all(displacement[both] <= length[both] + 1e-5)


def test_path_length_validity_starts_at_window():
    routes = _random_walk_routes(20, seed=5)
    adjacent = pr.lag_distances(routes, (1,))[None, :, 0, :]
    length = pr.path_length(adjacent, 4)
    assert np.isnan(length[0, :4]).all()
    assert np.isfinite(length[0, 4:]).all()


def test_progress_ratio_within_unit_interval():
    routes = _random_walk_routes(30, seed=6)
    lags = pr.lag_distances(routes, pr.LAGS)
    adjacent = lags[None, :, 0, :]
    length = pr.path_length(adjacent, 4)
    displacement = lags[None, :, pr.LAGS.index(4), :]
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-4)
    finite = np.isfinite(ratio)
    assert finite.any()
    assert np.all(ratio[finite] >= 0.0)
    assert np.all(ratio[finite] <= 1.0)


def test_frozen_prefix_gives_zero_ratio_not_nan():
    """L -> 0 时 D <= L 也 -> 0；这是 0/0，正确的极限值是 0（毫无净运动）。"""
    displacement = np.array([[[1e-9]]], dtype=np.float32)
    length = np.array([[[2e-9]]], dtype=np.float32)
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-6)
    assert ratio[0, 0, 0] == 0.0


def test_straight_line_gives_ratio_one():
    """单调朝一个方向走满窗口时 D == L，R == 1。"""
    displacement = np.array([[[0.4]]], dtype=np.float32)
    length = np.array([[[0.4]]], dtype=np.float32)
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-6)
    assert ratio[0, 0, 0] == pytest.approx(1.0)


def test_group_ratio_takes_median_within_group():
    ratio = np.full((1, 1, 8), np.nan, dtype=np.float32)
    ratio[0, 0, :4] = [0.1, 0.2, 0.3, 0.4]
    ratio[0, 0, 4:] = [0.6, 0.7, 0.8, 0.9]
    assert pr.group_ratio(ratio, "front")[0, 0] == pytest.approx(0.25)
    assert pr.group_ratio(ratio, "back")[0, 0] == pytest.approx(0.75)
    assert pr.group_ratio(ratio, "all")[0, 0] == pytest.approx(0.5)


def test_group_ratio_rejects_unknown_group():
    with pytest.raises(ValueError, match="unknown layer group"):
        pr.group_ratio(np.zeros((1, 1, 8), dtype=np.float32), "middle")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_geometry.py -v
```

Expected: FAIL，`AttributeError: module 'progress_ratio' has no attribute 'lag_distances'`

- [ ] **Step 3: 写实现**

追加到 `method/progress_ratio.py`：

```python
def lag_distances(routes: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """[Q, 8, 10, 32] -> [Q, len(lags), 8]; 元素 [q, i, l] = d_l(q, q - lags[i]).

    q < lag 处为 NaN。
    """
    routes = np.asarray(routes, dtype=np.float32)
    if routes.ndim != 4 or routes.shape[1:] != (
        len(LAYER_NAMES),
        N_ACTION_TOKENS,
        N_EXPERTS,
    ):
        raise ValueError(f"routes must be [query, 8, 10, 32], got {routes.shape}")
    n_query = len(routes)
    output = np.full((n_query, len(lags), len(LAYER_NAMES)), np.nan, dtype=np.float32)
    for index, lag in enumerate(lags):
        if lag < 1:
            raise ValueError("lags must be positive")
        if n_query > lag:
            output[lag:, index] = route_distance(routes[lag:], routes[:-lag])
    return output


def path_length(adjacent: np.ndarray, window: int) -> np.ndarray:
    """[E, Q, 8] 相邻距离 -> [E, Q, 8] 窗口路径长。q < window 处为 NaN。"""
    values = np.asarray(adjacent, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"adjacent distances must be [episode, query, layer], got {values.shape}")
    if window < 1:
        raise ValueError("window must be positive")
    output = np.full_like(values, np.nan)
    for query in range(window, values.shape[1]):
        block = values[:, query - window + 1 : query + 1]
        good = np.isfinite(block).all(axis=1)
        output[:, query] = np.where(good, block.sum(axis=1, dtype=np.float32), np.nan)
    return output


def progress_ratio(
    displacement: np.ndarray, length: np.ndarray, eps_length: float
) -> np.ndarray:
    """R = D / L，其中 L < eps_length 时取极限值 0（没有任何净运动）。"""
    numerator = np.asarray(displacement, dtype=np.float32)
    denominator = np.asarray(length, dtype=np.float32)
    if numerator.shape != denominator.shape:
        raise ValueError("displacement and path length do not align")
    if not np.isfinite(eps_length) or eps_length <= 0.0:
        raise ValueError("eps_length must be finite and positive")
    output = np.full_like(numerator, np.nan)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    frozen = finite & (denominator < eps_length)
    moving = finite & ~frozen
    output[frozen] = 0.0
    output[moving] = np.clip(numerator[moving] / denominator[moving], 0.0, 1.0)
    return output


def group_ratio(ratio: np.ndarray, group: str) -> np.ndarray:
    """[E, Q, 8] -> [E, Q]，组内取中位数。"""
    if group not in LAYER_GROUPS:
        raise ValueError(f"unknown layer group: {group}")
    values = np.asarray(ratio, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != len(LAYER_NAMES):
        raise ValueError(f"ratio must be [episode, query, 8], got {values.shape}")
    subset = values[:, :, LAYER_GROUPS[group]]
    output = np.full(subset.shape[:2], np.nan, dtype=np.float32)
    good = np.isfinite(subset).all(axis=2)
    output[good] = np.median(subset[good], axis=1).astype(np.float32)
    return output
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_geometry.py -v
```

Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/method/progress_ratio.py moe-progress-ratio-v12-0906/tests/test_progress_geometry.py
git commit -m "feat: add net displacement over path length with its metric guarantees"
```

---

## Task 4: 下尾阈值与持续确认

**Files:**
- Modify: `moe-progress-ratio-v12-0906/method/progress_ratio.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_progress_thresholds.py`

这四个函数是 v7 上尾工具的**严格对偶**。v7 控制"至少向上穿越一次"的概率，这里控制"至少向下
穿越一次"的概率。这一点是刻意保留的：`CALIBRATION_VARIANTS` §2.1 证明了逐轨迹极值分位数隐式
承担逐轨迹多重检验校正，换成逐 query 分位数会把 FPR 抬高约 100 倍。

- [ ] **Step 1: 写失败测试**

`tests/test_progress_thresholds.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402


def test_row_min_ignores_nan_and_empty_rows():
    values = np.array([[0.5, np.nan, 0.2], [np.nan, np.nan, np.nan]], dtype=np.float32)
    result = pr.row_min(values)
    assert result[0] == pytest.approx(0.2)
    assert np.isnan(result[1])


def test_quantile_lower_controls_per_trajectory_crossing_rate():
    """r* 应使约 alpha 比例的参考轨迹曾经跌破。"""
    minima = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    threshold = pr.quantile_lower(minima, 0.05)
    assert (minima < threshold).mean() == pytest.approx(0.05, abs=0.02)


def test_quantile_lower_is_the_dual_of_v7_quantile_higher():
    import sys as _sys

    _sys.path.insert(0, str(BUNDLE.parent / "moe-v7-0905/method"))
    import intrinsic_guard_monitor as v7

    rng = np.random.default_rng(11)
    values = rng.standard_normal(500).astype(np.float32)
    # v7.quantile_higher(w, p) 的 p 是"低于阈值的比例"，不是"尾部概率"；
    # 因此取负后要在 p 上做 1-q 才能对齐 quantile_lower(v, q) 的下尾比例 q。
    # 恒等式 m - ceil(p*m) = floor((1-p)*m)（m = n-1）保证该等式在实数上恒成立。
    assert pr.quantile_lower(values, 0.1) == pytest.approx(
        -v7.quantile_higher(-values, 1 - 0.1), abs=1e-6
    )


def test_quantile_lower_requires_enough_references():
    with pytest.raises(ValueError, match="at least 32"):
        pr.quantile_lower(np.zeros(8, dtype=np.float32), 0.05)


def test_persistent_low_is_a_rolling_max():
    """跌破持续 K 次 <=> 窗口内最大值也跌破。"""
    values = np.array([[0.9, 0.1, 0.1, 0.1, 0.8]], dtype=np.float32)
    result = pr.persistent_low(values, 3)
    assert np.isnan(result[0, :2]).all()
    assert result[0, 2] == pytest.approx(0.9)
    assert result[0, 3] == pytest.approx(0.1)
    assert result[0, 4] == pytest.approx(0.8)


def test_persistent_low_with_one_confirmation_is_identity():
    values = np.array([[0.4, 0.6]], dtype=np.float32)
    np.testing.assert_allclose(pr.persistent_low(values, 1), values)


def test_first_below_returns_first_valid_crossing():
    values = np.array([[0.9, 0.2, 0.1]], dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    assert pr.first_below(values, 0.5, valid)[0] == 1
    valid[0, 1] = False
    assert pr.first_below(values, 0.5, valid)[0] == 2


def test_first_below_returns_minus_one_when_never_crossing():
    values = np.array([[0.9, 0.8]], dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    assert pr.first_below(values, 0.5, valid)[0] == -1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_thresholds.py -v
```

Expected: FAIL，`AttributeError: module 'progress_ratio' has no attribute 'row_min'`

- [ ] **Step 3: 写实现**

追加到 `method/progress_ratio.py`：

```python
def row_min(values: np.ndarray) -> np.ndarray:
    """每条轨迹的全程最小值；全 NaN 行返回 NaN。"""
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmin(values[good], axis=1)
    return output


def quantile_lower(values: np.ndarray, quantile: float) -> float:
    """逐轨迹最小值的下尾分位数，v7 quantile_higher 的对偶。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = np.sort(finite[np.isfinite(finite)])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if len(finite) < 32:
        raise ValueError(f"at least 32 finite references are required, got {len(finite)}")
    position = int(np.floor(quantile * (len(finite) - 1)))
    return float(finite[position])


def persistent_low(values: np.ndarray, confirmations: int) -> np.ndarray:
    """滚动最大值。低于阈值即代表窗口内每一个 query 都低于阈值。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"persistent_low expects [episode, query], got {values.shape}")
    if confirmations < 1:
        raise ValueError("confirmations must be positive")
    if confirmations == 1:
        return values.copy()
    output = np.full_like(values, np.nan)
    for query in range(confirmations - 1, values.shape[1]):
        window = values[:, query - confirmations + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].max(axis=1)
    return output


def first_below(
    values: np.ndarray, threshold: float, valid: np.ndarray
) -> np.ndarray:
    """首次跌破阈值的 query 下标；从未跌破返回 -1。"""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("score and validity masks do not align")
    trigger = np.isfinite(values) & (values < threshold) & valid
    any_trigger = trigger.any(axis=1)
    first = np.full(len(values), -1, dtype=np.int16)
    first[any_trigger] = trigger[any_trigger].argmax(axis=1).astype(np.int16)
    return first
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_thresholds.py -v
```

Expected: 8 passed

`test_quantile_lower_is_the_dual_of_v7_quantile_higher` 是真实的跨 bundle 数值对偶检查。
两个函数的 `quantile` 参数**都表示"低于阈值的比例"**，不是尾部概率——v7 用
`FREEZE_QUANTILE = 0.975` 取一条保守的**高**阈值正是这个含义。所以取负之后必须在参数上做
`1 - q`。它失败说明实现有 bug，不是测试有问题。**不要削弱任何测试。**

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/method/progress_ratio.py moe-progress-ratio-v12-0906/tests/test_progress_thresholds.py
git commit -m "feat: add the lower-tail duals of the v7 per-trajectory threshold tools"
```

---

## Task 5: lag 距离缓存构建

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/build_progress_cache.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_progress_cache.py`

**输出契约**（每个 cohort 一个 npz，写到 `results/progress_cache/<cohort>.npz`）：

| 键 | 形状 | 说明 |
|---|---|---|
| `schema` | `()` | `"himoe.progress_ratio_v12.cache.v1"` |
| `cohort` | `()` | cohort 名 |
| `lags` | `(8,)` int16 | 即 `progress_ratio.LAGS` |
| `layer_names` | `(8,)` | 与 v4 缓存逐位一致 |
| `task_names` / `task_index` / `episode` / `init_state_id` / `length` / `valid` | 同源 cohort npz 原样透传 | |
| `lag_distance` | `[E, 52, 8, 8]` float32 | `[episode, query, lag, layer]` |
| `source_cache_sha256` | `()` | 源 cohort npz 的 SHA-256 |

`lag_distance[..., 0, :]`（lag 1）必须与源 npz 的 `mobility` 在 `atol=2e-5` 内一致——这是
**强制锚点**，不一致即抛错退出，不要放宽。

- [ ] **Step 1: 写失败测试**

`tests/test_progress_cache.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(BUNDLE / "experiments"))

import build_progress_cache as bpc  # noqa: E402
import progress_ratio as pr  # noqa: E402


def test_episode_lag_distances_matches_reference_mobility():
    """lag 1 必须复现 v4 缓存里 mobility 的定义。"""
    rng = np.random.default_rng(21)
    probs = rng.random((9, 8, 10, 11, 32)).astype(np.float32)
    routes = pr.action_route(probs)
    computed = bpc.episode_lag_distances(probs, pr.LAGS)
    expected = pr.route_distance(routes[1:], routes[:-1])
    np.testing.assert_allclose(computed[1:, 0], expected, rtol=0, atol=2e-5)


def test_episode_lag_distances_pads_short_episodes():
    rng = np.random.default_rng(22)
    probs = rng.random((3, 8, 10, 11, 32)).astype(np.float32)
    computed = bpc.episode_lag_distances(probs, pr.LAGS)
    assert computed.shape == (3, len(pr.LAGS), 8)
    assert np.isnan(computed[:, pr.LAGS.index(4)]).all()


def test_anchor_check_rejects_mismatched_mobility():
    lag_distance = np.zeros((2, 5, len(pr.LAGS), 8), dtype=np.float32)
    mobility = np.ones((2, 5, 8), dtype=np.float32)
    valid = np.ones((2, 5), dtype=bool)
    with pytest.raises(ValueError, match="lag-1 distances disagree"):
        bpc.assert_mobility_anchor(lag_distance, mobility, valid)


def test_anchor_check_accepts_matching_mobility():
    lag_distance = np.zeros((2, 5, len(pr.LAGS), 8), dtype=np.float32)
    mobility = np.zeros((2, 5, 8), dtype=np.float32)
    valid = np.ones((2, 5), dtype=bool)
    valid[:, 0] = False
    bpc.assert_mobility_anchor(lag_distance, mobility, valid)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_cache.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'build_progress_cache'`

- [ ] **Step 3: 写实现**

`experiments/build_progress_cache.py`。结构照抄
`moe-hb-front-back-0905/experiments/extract_layer_graphs_gpu.py`（同样的 cohort 遍历、
zarr 打开、`--batch` / `--resume`、逐 task 进度打印）。两个必须自己写对的函数：

```python
def episode_lag_distances(
    hb_router_probs: np.ndarray, lags: tuple[int, ...]
) -> np.ndarray:
    """一个 episode 的 [Q, 8, 10, 11, 32] -> [Q, len(lags), 8]。"""
    routes = progress_ratio.action_route(np.asarray(hb_router_probs, dtype=np.float32))
    return progress_ratio.lag_distances(routes, lags)


def assert_mobility_anchor(
    lag_distance: np.ndarray, mobility: np.ndarray, valid: np.ndarray
) -> None:
    """lag-1 必须复现 v4 的 mobility，否则本缓存与既往全部结果不可比。"""
    computed = lag_distance[:, :, 0, :]
    mask = valid[:, :, None] & np.isfinite(computed) & np.isfinite(mobility)
    if not mask.any():
        raise ValueError("no overlapping finite lag-1 entries to anchor against")
    deviation = float(np.abs(computed[mask] - mobility[mask]).max())
    if deviation > 2e-5:
        raise ValueError(
            f"lag-1 distances disagree with the v4 mobility cache by {deviation:.3e}"
        )
```

**本任务用 CPU，不用 GPU。** 实测 Zarr 读取为 11,800 行/秒，单 task 5,064 行约 0.4 秒，
瓶颈完全在 I/O 而非算力；抽出 `final_action = probs[:, :, 9, 1:11, :]` 后每行只剩 10 KB，
numpy 足够。（另外本机 8 张卡当前被外部进程占满，各 139 GB，不可占用。）

处理方式：按 `--batch` 取 `[batch, 8, 10, 11, 32]`，立刻抽出并归一化成
`[batch, 8, 10, 32]`，**不要**把整个 `hb_router_probs` 留在内存里，再按 episode 切片调用
`episode_lag_distances`。

主流程逐 cohort：

1. 读 cohort npz，取 `task_names` / `task_index` / `episode` / `length` / `valid` / `mobility`。
2. 逐 task 打开 `root / task / run_id / "server/routes.zarr"`，按 `--batch` 取
   `hb_router_probs`，抽出 final-flow action routes。
3. 按 `episode_id` 切成 episode，校验 `len(positions) == length[row]` 且下标连续，否则抛错。
4. 逐 episode 算 lag 距离写入 `lag_distance[row, :length]`。
5. 全部 task 完成后调用 `assert_mobility_anchor`，通过才写盘。
6. `--resume` 时若目标 npz 已存在且形状与 `lags` 一致则跳过。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_cache.py -v
```

Expected: 4 passed

- [ ] **Step 5: 在真实语料上构建缓存**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/build_progress_cache.py \
  --output results/progress_cache --batch 512 --resume
```

Expected: 三个 cohort 各打印逐 task 进度，最后 `anchor ok: max lag-1 deviation = <2e-5`。
产物约 208 MB / cohort。任一 cohort 抛出 `lag-1 distances disagree` 都必须停下排查，
不要调大容差。

- [ ] **Step 6: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/build_progress_cache.py moe-progress-ratio-v12-0906/tests/test_progress_cache.py
git commit -m "feat: cache multi-lag routing displacements anchored to the v4 mobility cache"
```

---

## Task 6: 在线 monitor 与因果性

**Files:**
- Create: `moe-progress-ratio-v12-0906/method/progress_monitor.py`
- Create: `moe-progress-ratio-v12-0906/method/ONLINE_PROGRESS_GUARD_V12_PROTOCOL.md`
- Create: `moe-progress-ratio-v12-0906/tests/test_progress_monitor.py`

接口对齐 `moe-v7-0905/method/intrinsic_guard_monitor.py`：`Profile.load(path)` 类方法 +
`Monitor(profile).update(hb_router_probs) -> dict`。

- [ ] **Step 1: 写失败测试**

`tests/test_progress_monitor.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_monitor as pm  # noqa: E402
import progress_ratio as pr  # noqa: E402


def _profile(window=4, confirmations=2, threshold=0.5, group="back"):
    return pm.GlobalProgressProfile(
        window=window,
        confirmations=confirmations,
        ratio_threshold=threshold,
        eps_length=1e-5,
        layer_group=group,
    )


def _episode(n_query, seed, step=0.05):
    rng = np.random.default_rng(seed)
    current = rng.random((8, 10, 11, 32)).astype(np.float32)
    out = np.empty((n_query, 8, 10, 11, 32), dtype=np.float32)
    for q in range(n_query):
        current = np.abs(current + step * rng.standard_normal(current.shape).astype(np.float32))
        out[q] = current / current.sum(axis=-1, keepdims=True)
    return out


def test_monitor_emits_no_ratio_before_the_window_fills():
    monitor = pm.ProgressGuardMonitor(_profile(window=4))
    episode = _episode(6, seed=31)
    rows = [monitor.update(query) for query in episode]
    assert all(not np.isfinite(row["ratio"]) for row in rows[:4])
    assert np.isfinite(rows[4]["ratio"])


def test_monitor_alarm_latches_and_records_first_query():
    # R <= 1 恒成立，阈值 1.0 使随机游走必然触发
    monitor = pm.ProgressGuardMonitor(_profile(threshold=1.0, confirmations=1))
    episode = _episode(8, seed=32)
    rows = [monitor.update(query) for query in episode]
    assert rows[-1]["alarm"] is True
    first = rows[-1]["first_alarm_query"]
    assert first == next(i for i, row in enumerate(rows) if row["alarm"])
    assert all(row["alarm"] for row in rows[first:])


def test_monitor_never_alarms_above_threshold():
    # R >= 0 恒成立，阈值 0.0 使 R < 0 永不成立
    monitor = pm.ProgressGuardMonitor(_profile(threshold=0.0))
    episode = _episode(10, seed=33)
    rows = [monitor.update(query) for query in episode]
    assert not any(row["alarm"] for row in rows)
    assert rows[-1]["first_alarm_query"] == -1


def test_rewriting_the_future_does_not_change_an_existing_prefix():
    """严格因果：q 的分数只由 P(0..q) 决定。"""
    profile = _profile()
    episode = _episode(12, seed=34)
    baseline = [pm.ProgressGuardMonitor(profile).update(q) for q in episode][:7]

    mutated = episode.copy()
    rng = np.random.default_rng(99)
    tail = np.abs(rng.random(mutated[7:].shape).astype(np.float32))
    mutated[7:] = tail / tail.sum(axis=-1, keepdims=True)
    replayed = [pm.ProgressGuardMonitor(profile).update(q) for q in mutated][:7]

    for left, right in zip(baseline, replayed):
        assert left["alarm"] == right["alarm"]
        np.testing.assert_allclose(left["ratio"], right["ratio"], equal_nan=True)


def test_online_matches_batch_computation():
    """在线 monitor 与批量路径必须给出同一条 R 序列。"""
    profile = _profile(window=4, group="back")
    episode = _episode(15, seed=35)
    online = np.array(
        [pm.ProgressGuardMonitor(profile).update(q)["ratio"] for q in episode]
    )
    # 重新用批量路径算一遍
    lags = pr.lag_distances(pr.action_route(episode), pr.LAGS)
    adjacent = lags[None, :, 0, :]
    length = pr.path_length(adjacent, profile.window)
    displacement = lags[None, :, pr.LAGS.index(profile.window), :]
    ratio = pr.progress_ratio(displacement, length, profile.eps_length)
    batch = pr.group_ratio(ratio, profile.layer_group)[0]
    np.testing.assert_allclose(online, batch, rtol=1e-5, equal_nan=True)


def test_profile_rejects_task_metadata(tmp_path):
    path = tmp_path / "bad_profile.npz"
    np.savez(
        path,
        schema=pr.SCHEMA,
        window=4,
        confirmations=2,
        ratio_threshold=0.5,
        eps_length=1e-5,
        layer_group="back",
        task_names=np.array(["a"]),
    )
    with pytest.raises(ValueError, match="must not contain task metadata"):
        pm.GlobalProgressProfile.load(path)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_monitor.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'progress_monitor'`

- [ ] **Step 3: 写实现**

`method/progress_monitor.py`：

```python
#!/usr/bin/env python3
"""Task-agnostic, train-free online progress-ratio monitor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from progress_ratio import (
    LAGS,
    LAYER_GROUPS,
    SCHEMA,
    action_route,
    group_ratio,
    path_length,
    progress_ratio,
    route_distance,
)


@dataclass(frozen=True)
class GlobalProgressProfile:
    window: int
    confirmations: int
    ratio_threshold: float
    eps_length: float
    layer_group: str

    def __post_init__(self) -> None:
        if self.window < 1 or self.confirmations < 1:
            raise ValueError("window and confirmations must be positive")
        if self.window not in LAGS:
            raise ValueError(f"window {self.window} is not a cached lag")
        if not np.isfinite(self.ratio_threshold) or not 0.0 <= self.ratio_threshold <= 1.0:
            raise ValueError("ratio threshold must lie in [0, 1]")
        if not np.isfinite(self.eps_length) or self.eps_length <= 0.0:
            raise ValueError("eps_length must be finite and positive")
        if self.layer_group not in LAYER_GROUPS:
            raise ValueError(f"unknown layer group: {self.layer_group}")

    @classmethod
    def load(cls, path: Path) -> "GlobalProgressProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                raise ValueError(f"unknown progress profile schema in {path}")
            if "task_names" in archive.files or "task_index" in archive.files:
                raise ValueError("a global profile must not contain task metadata")
            return cls(
                window=int(archive["window"]),
                confirmations=int(archive["confirmations"]),
                ratio_threshold=float(archive["ratio_threshold"]),
                eps_length=float(archive["eps_length"]),
                layer_group=str(archive["layer_group"]),
            )


class ProgressGuardMonitor:
    """Stateful one-rollout monitor using no task identity or learned weights."""

    def __init__(self, profile: GlobalProgressProfile) -> None:
        self.profile = profile
        self.query = -1
        self._routes: deque[np.ndarray] = deque(maxlen=profile.window + 1)
        self._adjacent: deque[np.ndarray] = deque(maxlen=profile.window)
        self._below_run = 0
        self.alarm = False
        self.first_alarm_query = -1

    def _current_ratio(self) -> float:
        if self.query < self.profile.window:
            return float("nan")
        displacement = route_distance(self._routes[-1], self._routes[0])
        length = np.sum(np.stack(tuple(self._adjacent)), axis=0, dtype=np.float32)
        ratio = progress_ratio(
            displacement[None, None, :], length[None, None, :], self.profile.eps_length
        )
        return float(group_ratio(ratio, self.profile.layer_group)[0, 0])

    def update(self, hb_router_probs: np.ndarray) -> dict[str, Any]:
        self.query += 1
        routes = action_route(hb_router_probs)
        if self._routes:
            self._adjacent.append(route_distance(routes, self._routes[-1]))
        self._routes.append(routes)

        ratio = self._current_ratio()
        below = np.isfinite(ratio) and ratio < self.profile.ratio_threshold
        self._below_run = self._below_run + 1 if below else 0
        alarm_now = self._below_run >= self.profile.confirmations
        if alarm_now and not self.alarm:
            self.first_alarm_query = self.query
        self.alarm |= alarm_now
        return {
            "query": self.query,
            "ratio": ratio,
            "below_run": self._below_run,
            "alarm": self.alarm,
            "first_alarm_query": self.first_alarm_query,
        }
```

`method/ONLINE_PROGRESS_GUARD_V12_PROTOCOL.md` 写明：运行时输入只有当前 query 的
`hb_router_probs`；profile 只含 `window`、`confirmations`、`ratio_threshold`、`eps_length`、
`layer_group` 五个标量；明确不含 task/suite ID、任务原型、每任务阈值、outcome、horizon、
模拟器状态或未来 query；最早可能报警在 `q = window + confirmations - 1`。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_progress_monitor.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/method/progress_monitor.py moe-progress-ratio-v12-0906/method/ONLINE_PROGRESS_GUARD_V12_PROTOCOL.md moe-progress-ratio-v12-0906/tests/test_progress_monitor.py
git commit -m "feat: add the causal online progress-ratio monitor"
```

---

## Task 7: 开发集选点与 K2 闸门

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/select_operating_point_v12.py`
- Create: `moe-progress-ratio-v12-0906/experiments/hazard_common.py`（本任务只实现 `survival_prior`
  与 `matched_prior`；Task 9 追加 `decompose` / `band` 与完整测试）
- Create: `moe-progress-ratio-v12-0906/tests/test_select_operating_point.py`

**网格**：`W ∈ W_GRID`(7) × `layer_group ∈ {front, back, all}`(3) × `K ∈ {1,2,3,4,6}`(5) ×
`quantile ∈ {0.005,0.01,0.025,0.05,0.075,0.10,0.15,0.20}`(8) = **840 个候选**。
两种阈值模式各跑一遍（`global` 与 `per_task`），仅用于 K2 对照；**`global` 是主模式**。

**阈值数值不读 outcome**：对每条 reference trajectory 取 `row_min(persistent_low(R, K))`，
再对这些逐轨迹最小值取 `quantile_lower(·, quantile)`。参考语料是 development_main +
development_extra 的全部 16,000 条（成功与失败样本全部保留）。

`eps_length` 取 development 参考语料全部有限 `L` 值的 **1% 分位数**（不是第一四分位数），随 profile 冻结。

**选择规则（预注册，照抄，不得事后修改）**：在候选中取满足
`timely FPR <= 0.005` 且 `低先验 TP >= 20` 的子集，最大化**低先验 TP**；并列时取低先验
precision 更高者，再并列取 `W` 更小者。若子集为空则记录 `feasible: false` 并保留完整候选表。

**输出**：`results/operating_point/development_candidates.csv`（840 × 2 模式全部行，
列含 `window,layer_group,confirmations,quantile,threshold_mode,ratio_threshold,tp,fp,recall,precision,timely_fpr,low_prior_tp,low_prior_fp,low_prior_precision`）、
`selection.json`（选中点、feasibility、K2 判定）。

- [ ] **Step 1: 写失败测试**

`tests/test_select_operating_point.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(BUNDLE / "experiments"))

import select_operating_point_v12 as sel  # noqa: E402


def _candidates(**overrides):
    base = pd.DataFrame(
        {
            "window": [4, 6, 8, 4],
            "layer_group": ["back", "back", "all", "front"],
            "confirmations": [2, 2, 3, 2],
            "quantile": [0.05, 0.05, 0.05, 0.05],
            "ratio_threshold": [0.3, 0.3, 0.3, 0.3],
            "timely_fpr": [0.004, 0.004, 0.010, 0.004],
            "low_prior_tp": [25, 25, 90, 12],
            "low_prior_precision": [0.6, 0.7, 0.9, 0.9],
        }
    )
    return base.assign(**overrides)


def test_choose_maximises_low_prior_tp_under_constraints():
    chosen = sel.choose(_candidates())
    # 第 3 行低先验 TP 最高但 FPR 超标；第 4 行 TP 不足 20
    # 第 1、2 行并列 TP=25，取 precision 更高的第 2 行
    assert chosen["window"] == 6
    assert chosen["low_prior_precision"] == pytest.approx(0.7)


def test_choose_breaks_precision_ties_with_smaller_window():
    candidates = _candidates(low_prior_precision=[0.7, 0.7, 0.9, 0.9])
    assert sel.choose(candidates)["window"] == 4


def test_choose_returns_none_when_no_candidate_is_feasible():
    assert sel.choose(_candidates(low_prior_tp=[1, 2, 3, 4])) is None


def test_choose_rejects_candidates_over_the_fpr_cap():
    assert sel.choose(_candidates(timely_fpr=[0.02, 0.02, 0.02, 0.02])) is None


def test_k2_gate_requires_half_the_per_task_yield():
    assert sel.k2_gate(global_tp=50, per_task_tp=90) is True
    assert sel.k2_gate(global_tp=40, per_task_tp=90) is False
    assert sel.k2_gate(global_tp=3, per_task_tp=83) is False  # flow_settling 的对照点


def test_calibrate_threshold_ignores_outcomes():
    """阈值只能是无标签 order statistic。"""
    rng = np.random.default_rng(41)
    scores = rng.random((200, 30)).astype(np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    threshold = sel.calibrate_threshold(scores, valid, confirmations=2, quantile=0.05)
    assert 0.0 <= threshold <= 1.0
    # 打乱"标签"不改变阈值：函数签名里根本没有 outcome
    assert threshold == sel.calibrate_threshold(scores, valid, confirmations=2, quantile=0.05)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_select_operating_point.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'select_operating_point_v12'`

- [ ] **Step 3: 写实现**

`experiments/select_operating_point_v12.py`。总体结构参照
`moe-v7-0905/experiments/select_operating_point.py`。四个必须写对的函数：

```python
FPR_CAP = 0.005
MIN_LOW_PRIOR_TP = 20
QUANTILE_GRID = (0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
CONFIRMATION_GRID = (1, 2, 3, 4, 6)


def calibrate_threshold(
    ratio: np.ndarray, valid: np.ndarray, confirmations: int, quantile: float
) -> float:
    """无标签阈值：逐轨迹最小值的下尾分位数。签名里没有 outcome，这是刻意的。"""
    persistent = progress_ratio.persistent_low(ratio, confirmations)
    persistent = np.where(valid, persistent, np.nan)
    return progress_ratio.quantile_lower(
        progress_ratio.row_min(persistent), quantile
    )


def choose(candidates: pd.DataFrame) -> pd.Series | None:
    """预注册选择规则。不满足约束返回 None，不放宽重试。"""
    feasible = candidates[
        (candidates["timely_fpr"] <= FPR_CAP)
        & (candidates["low_prior_tp"] >= MIN_LOW_PRIOR_TP)
    ]
    if feasible.empty:
        return None
    ordered = feasible.sort_values(
        ["low_prior_tp", "low_prior_precision", "window"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0]


def k2_gate(global_tp: int, per_task_tp: int) -> bool:
    """global 模式的 TP 不得低于 per_task 模式的一半。"""
    if per_task_tp <= 0:
        return False
    return global_tp >= 0.5 * per_task_tp
```

`per_task` 模式的阈值改为按 `task_index` 分组各自调用 `calibrate_threshold`，其余一致。

低先验区间的判定需要存活先验。本任务同时创建 `experiments/hazard_common.py`，**只写
`survival_prior` 与 `matched_prior` 两个函数**（实现见 Task 9 Step 3，逐字照抄那两段）。
Task 9 会在同一文件上追加 `decompose` 与 `band`，并补齐全部测试。这样 Task 7 与 Task 11
共用同一份先验估计，不会出现两套口径。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_select_operating_point.py -v
```

Expected: 6 passed

- [ ] **Step 5: 在开发集上跑选点**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/select_operating_point_v12.py --output results/operating_point
```

Expected: 打印 840 × 2 候选完成、选中点、以及 `K2: global_tp=... per_task_tp=... -> PASS/FAIL`。

- [ ] **Step 6: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments moe-progress-ratio-v12-0906/tests/test_select_operating_point.py moe-progress-ratio-v12-0906/results/operating_point
git commit -m "feat: select the progress-ratio operating point under a pre-registered rule"
```

---

## Task 8: K1 / K3 表型闸门

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/evaluate_phenotype_2x2.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_phenotype.py`

K1 与 K3 都在**开发集**上判定，且必须在接触 external 之前完成。

- **K1**：mobility 按开发集中位数切高/低。要求**至少 20 个 risk episode** 的首次低-`R` 报警
  落在（高 mobility, 低 R）格。少于 20 即"兜圈"模态不存在，**停止项目并写否定结果**。
- **K3**：`R` 检出的 risk episode 集合中，**至少 10%** 是 v7 relative_freeze 未检出的。
  v7 的 freeze 首报数组直接读 `moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz`。

- [ ] **Step 1: 写失败测试**

`tests/test_phenotype.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import evaluate_phenotype_2x2 as ph  # noqa: E402


def test_cell_assignment_uses_the_development_median():
    mobility = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    ratio = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)
    cells = ph.assign_cells(mobility, ratio, mobility_median=0.5, ratio_threshold=0.5)
    assert list(cells) == ["low_mob_low_R", "low_mob_high_R", "high_mob_low_R", "high_mob_high_R"]


def test_k1_gate_counts_only_risk_episodes_in_the_high_mobility_cell():
    cells = np.array(["high_mob_low_R"] * 25 + ["low_mob_low_R"] * 40)
    risk = np.array([True] * 19 + [False] * 6 + [True] * 40)
    assert ph.k1_gate(cells, risk) == (19, False)
    risk[19] = True
    assert ph.k1_gate(cells, risk) == (20, True)


def test_k3_gate_requires_ten_percent_novel_detections():
    ratio_first = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=np.int16)
    freeze_first = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=np.int16)
    risk = np.ones(10, dtype=bool)
    assert ph.k3_gate(ratio_first, freeze_first, risk) == (0.0, False)
    freeze_first[:2] = -1
    share, passed = ph.k3_gate(ratio_first, freeze_first, risk)
    assert share == pytest.approx(0.2)
    assert passed is True


def test_k3_gate_ignores_non_risk_episodes():
    ratio_first = np.array([3, 3], dtype=np.int16)
    freeze_first = np.array([-1, -1], dtype=np.int16)
    risk = np.array([False, False])
    assert ph.k3_gate(ratio_first, freeze_first, risk) == (0.0, False)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_phenotype.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'evaluate_phenotype_2x2'`

- [ ] **Step 3: 写实现**

```python
K1_MIN_RISK_EPISODES = 20
K3_MIN_NOVEL_SHARE = 0.10


def assign_cells(
    mobility: np.ndarray, ratio: np.ndarray, mobility_median: float, ratio_threshold: float
) -> np.ndarray:
    """报警时刻的 2x2 表型标签。"""
    high_mobility = np.asarray(mobility, dtype=np.float32) >= mobility_median
    low_ratio = np.asarray(ratio, dtype=np.float32) < ratio_threshold
    labels = np.empty(len(high_mobility), dtype=object)
    labels[~high_mobility & low_ratio] = "low_mob_low_R"
    labels[~high_mobility & ~low_ratio] = "low_mob_high_R"
    labels[high_mobility & low_ratio] = "high_mob_low_R"
    labels[high_mobility & ~low_ratio] = "high_mob_high_R"
    return labels


def k1_gate(cells: np.ndarray, risk: np.ndarray) -> tuple[int, bool]:
    count = int(((cells == "high_mob_low_R") & np.asarray(risk, dtype=bool)).sum())
    return count, count >= K1_MIN_RISK_EPISODES


def k3_gate(
    ratio_first: np.ndarray, freeze_first: np.ndarray, risk: np.ndarray
) -> tuple[float, bool]:
    risk = np.asarray(risk, dtype=bool)
    detected = risk & (np.asarray(ratio_first) >= 0)
    if not detected.any():
        return 0.0, False
    novel = detected & (np.asarray(freeze_first) < 0)
    share = float(novel.sum() / detected.sum())
    return share, share >= K3_MIN_NOVEL_SHARE
```

主流程：读开发集缓存、用 Task 7 选中的 profile 算 `R` 与首报、取报警时刻的 mobility 与 `R`、
写 `results/phenotype/cells.csv`（逐 episode）、`results/phenotype/gates.json`
（K1 计数与判定、K3 份额与判定、mobility 中位数）。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_phenotype.py -v
```

Expected: 4 passed

- [ ] **Step 5: 跑闸门**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/evaluate_phenotype_2x2.py --output results/phenotype
```

Expected: 打印 2×2 计数表与 `K1: n=<count> -> PASS/FAIL`、`K3: share=<share> -> PASS/FAIL`。

- [ ] **Step 6: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/evaluate_phenotype_2x2.py moe-progress-ratio-v12-0906/tests/test_phenotype.py moe-progress-ratio-v12-0906/results/phenotype
git commit -m "feat: gate the progress ratio on its development-set phenotype"
```

---

## ⛔ 硬闸门：在此停下

**K1、K2、K3 必须全部 PASS 才允许继续。** 这不是建议。

- **K1 FAIL**（高 mobility + 低 R 格的 risk episode < 20）：本语料没有"兜圈"模态，`R` 相对 v7
  freeze 的新增内容是空的。**停止实施，把否定结果写成报告**，保留 `results/phenotype/` 全部产物。
- **K2 FAIL**（global 模式 TP < per_task 的 50%）：`R` 的无量纲性没有兑现成任务无关性，
  与 `flow_settling` 同样的失败。**停止实施**，报告须直接对照 `FRAME_SURVEY_REPORT_ZH.md` §2。
- **K3 FAIL**（新增检出 < 10%）：`R` 只是 freeze 的重写，不是新轴。**停止实施**。

任一失败都**不得放宽阈值重试**，也不得跳过继续跑 external。`CALIBRATION_VARIANTS` 的先例是：
不可行的记录连同完整候选表一并保留，那就是结论本身。

向用户报告闸门结果并等待确认，再执行 Task 9 及之后。

---

## Task 9: 补齐逐 chunk 超额 hazard 评估器

**Files:**
- Modify: `moe-progress-ratio-v12-0906/experiments/hazard_common.py`（Task 7 已创建，
  含 `survival_prior` 与 `matched_prior`；本任务追加 `decompose` 与 `band`）
- Create: `moe-progress-ratio-v12-0906/tests/test_hazard.py`

**存活先验必须在 development 上估计、应用到 external。** `moe-v7-0905` 的现版本是同 cohort
估计（报告自标"对基线略有偏袒"），本 bundle 修正这一点。

- [ ] **Step 1: 写失败测试**

`tests/test_hazard.py`：

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import hazard_common as hz  # noqa: E402


def _labels():
    return pd.DataFrame(
        {
            "suite": ["libero_long"] * 6,
            "length": [10, 10, 20, 20, 30, 30],
            "original_failure": [False, False, False, True, True, True],
        }
    )


def test_survival_prior_is_conditional_on_still_running():
    prior = hz.survival_prior(_labels())["libero_long"]
    assert prior[0] == pytest.approx(3 / 6)
    assert prior[10] == pytest.approx(3 / 4)
    assert prior[20] == pytest.approx(2 / 2)


def test_survival_prior_is_estimated_on_the_given_frame_only():
    """development 估计的先验必须能整体套用到另一批 episode。"""
    prior = hz.survival_prior(_labels())
    assert set(prior) == {"libero_long"}


def test_lift_divides_precision_by_the_matched_prior():
    alarms = pd.DataFrame(
        {
            "suite": ["libero_long"] * 4,
            "alarm_chunk": [0, 0, 20, 20],
            "correct": [True, False, True, True],
        }
    )
    prior = {"libero_long": {0: 0.25, 20: 0.90}}
    summary = hz.decompose(alarms, prior)
    assert summary["precision"] == pytest.approx(0.75)
    assert summary["matched_prior"] == pytest.approx(0.575)
    assert summary["lift"] == pytest.approx(0.75 / 0.575)


def test_low_prior_band_selects_only_informative_alarms():
    alarms = pd.DataFrame(
        {
            "suite": ["libero_long"] * 3,
            "alarm_chunk": [0, 10, 20],
            "correct": [True, True, True],
        }
    )
    prior = {"libero_long": {0: 0.10, 10: 0.50, 20: 0.95}}
    band = hz.band(alarms, prior, upper=0.25)
    assert len(band) == 1
    assert band.iloc[0]["alarm_chunk"] == 0


def test_missing_chunk_in_prior_raises_rather_than_silently_dropping():
    alarms = pd.DataFrame(
        {"suite": ["libero_long"], "alarm_chunk": [99], "correct": [True]}
    )
    with pytest.raises(KeyError, match="no survival prior"):
        hz.decompose(alarms, {"libero_long": {0: 0.1}})
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_hazard.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'hazard_common'`

- [ ] **Step 3: 写实现**

```python
PRIOR_BANDS = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.01))
PRIOR_CUTS = (0.10, 0.25, 0.40)


def survival_prior(labels: pd.DataFrame) -> dict[str, dict[int, float]]:
    """P(risk | still running at chunk q)，按 suite 估计。"""
    priors: dict[str, dict[int, float]] = {}
    for suite, block in labels.groupby("suite"):
        risk = block["original_failure"].to_numpy(bool)
        length = block["length"].to_numpy(int)
        priors[str(suite)] = {
            chunk: float(risk[length > chunk].mean())
            for chunk in range(int(length.max()))
        }
    return priors


def matched_prior(alarms: pd.DataFrame, prior: dict[str, dict[int, float]]) -> np.ndarray:
    values = np.empty(len(alarms), dtype=np.float64)
    for position, (suite, chunk) in enumerate(
        zip(alarms["suite"], alarms["alarm_chunk"])
    ):
        table = prior.get(str(suite))
        if table is None or int(chunk) not in table:
            raise KeyError(f"no survival prior for suite {suite} at chunk {chunk}")
        values[position] = table[int(chunk)]
    return values


def decompose(alarms: pd.DataFrame, prior: dict[str, dict[int, float]]) -> dict[str, float]:
    matched = matched_prior(alarms, prior)
    precision = float(alarms["correct"].to_numpy(bool).mean())
    base = float(matched.mean())
    return {
        "alarms": int(len(alarms)),
        "precision": precision,
        "matched_prior": base,
        "net_gain": precision - base,
        "lift": precision / base if base > 0 else float("nan"),
    }


def band(
    alarms: pd.DataFrame, prior: dict[str, dict[int, float]], upper: float
) -> pd.DataFrame:
    return alarms.loc[matched_prior(alarms, prior) < upper].copy()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_hazard.py -v
```

Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/hazard_common.py moe-progress-ratio-v12-0906/tests/test_hazard.py
git commit -m "feat: score alarms against a development-estimated survival prior"
```

---

## Task 10: 方案 B 对照基线

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/evaluate_baseline_dnf.py`

K4 要求 `R` 严格超过这条基线，所以它必须先存在。基线是推论 3 的直接实现：把
`moe-hb-front-back-0905/results/layer_graphs/` 里的十二个量做 episode 内自归一化
（每个量除以该 rollout 自身 q1–q4 的中位数），在 `global` 阈值模式下重扫，再按误报依赖矩阵
挑近正交配对组成析取式。

- [ ] **Step 1: 实现**

```python
SELF_NORM_BASELINE = slice(1, 5)   # 与 v7 的 q1..q4 相对基线一致


def self_normalize(metric: np.ndarray) -> np.ndarray:
    """[E, Q, 8] -> [E, Q, 8]，除以自身 q1..q4 中位数，消掉任务尺度。"""
    base = np.nanmedian(metric[:, SELF_NORM_BASELINE], axis=1, keepdims=True)
    return metric / np.maximum(np.abs(base), 1e-8)


def false_alarm_dependence(first_a: np.ndarray, first_b: np.ndarray, timely: np.ndarray) -> float:
    """观测共现 / 边际率乘积。等于 1 表示独立。"""
    fa = timely & (first_a >= 0)
    fb = timely & (first_b >= 0)
    total = int(timely.sum())
    expected = (fa.sum() / total) * (fb.sum() / total)
    if expected <= 0:
        return float("nan")
    return float(((fa & fb).sum() / total) / expected)
```

流程：十二个量各自扫 `global` 阈值网格选最佳单检测器 → 对入围量两两算误报依赖比 →
取依赖比最低的若干对做 AND → 这些 AND 项做 OR。项数上限 3，**在开发集上确定后冻结**，
不得在 external 上调整（`FLOW_AXIS` §6.3 的教训）。

输出 `results/baseline_dnf/development_candidates.csv`、`dependence_matrix.csv`、
`baseline_selection.json`、`external_first_alarms.npz`。

- [ ] **Step 2: 跑基线**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/evaluate_baseline_dnf.py --output results/baseline_dnf
```

Expected: 打印每个量的最佳 global 配置、依赖矩阵、选中的 DNF、以及它的低先验 TP。
这个低先验 TP 就是 K4 的门槛。

- [ ] **Step 3: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/evaluate_baseline_dnf.py moe-progress-ratio-v12-0906/results/baseline_dnf
git commit -m "feat: build the self-normalized DNF baseline that the progress ratio must beat"
```

---

## Task 11: 封存与 external 评估（K4 / K5 / K6）

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/evaluate_hazard.py`
- Create: `moe-progress-ratio-v12-0906/tests/test_gates.py`

**顺序是强制的**：先写入并哈希 profile 与逐 episode first-alarm，**再**打开 outcome 文件。
`sealed_manifest.json` 必须记录代码 SHA-256、协议 SHA-256、输入缓存 SHA-256、first-alarm 数组
SHA-256 与封存时间戳。

- [ ] **Step 0a: 先写闸门的失败测试**

三条闸门编码了预注册阈值，必须单测锁死，否则很容易在跑出难看结果后被"顺手"放宽。

`tests/test_gates.py`：

```python
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import evaluate_hazard as eh  # noqa: E402


def test_k4_requires_strictly_beating_the_baseline():
    assert eh.k4_gate(ratio_low_prior_tp=41, baseline_low_prior_tp=40) is True
    assert eh.k4_gate(ratio_low_prior_tp=40, baseline_low_prior_tp=40) is False
    assert eh.k4_gate(ratio_low_prior_tp=39, baseline_low_prior_tp=40) is False


def test_k5_treats_a_spatial_breakout_as_failure_not_success():
    assert eh.k5_gate(1.00) is True
    assert eh.k5_gate(0.95) is True
    assert eh.k5_gate(1.15) is True
    assert eh.k5_gate(0.80) is False
    assert eh.k5_gate(2.50) is False   # "变好"同样是 FAIL


def test_k6_requires_a_third_of_the_slow_success_false_alarms_vetoed():
    assert eh.k6_gate(vetoed=17, freeze_false_alarms=51) is True
    assert eh.k6_gate(vetoed=16, freeze_false_alarms=51) is False
    assert eh.k6_gate(vetoed=0, freeze_false_alarms=0) is False


def test_gate_constants_match_the_spec():
    assert eh.K5_LIFT_BAND == (0.9, 1.15)
    assert eh.K6_MIN_VETO_SHARE == pytest.approx(1 / 3)
```

- [ ] **Step 0b: 运行测试确认失败**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_gates.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'evaluate_hazard'`

- [ ] **Step 1: 实现**

流程：

1. 载入 Task 7 选中的 profile，写 `results/hazard/global_profile.npz`（五个常量，
   `GlobalProgressProfile.load` 必须能读回，且不含任何 task 字段）。
2. 在 external 缓存上算 `R` 与首报，写 `sealed_first_alarms.npz`。
3. 计算全部哈希，写 `sealed_manifest.json`。
4. **此时才**读 `LABEL_ROOT`。
5. 用 development 估计的存活先验评估：
   - 总体与分 suite 的 TP/FP/precision/matched_prior/lift；
   - 三个先验切点 `PRIOR_CUTS = (0.10, 0.25, 0.40)` 各自的低先验 TP/precision/lift/召回；
   - episode 二值口径同时记录，**仅用于与 v4/v6/v7 对表**。
6. 三条闸门：

```python
K5_LIFT_BAND = (0.9, 1.15)
K6_MIN_VETO_SHARE = 1 / 3


def k4_gate(ratio_low_prior_tp: int, baseline_low_prior_tp: int) -> bool:
    """R 必须严格超过方案 B 基线。"""
    return ratio_low_prior_tp > baseline_low_prior_tp


def k5_gate(spatial_lift: float) -> bool:
    """spatial 应保持在 [0.9, 1.15]。超出更可能是泄漏而非发现。"""
    low, high = K5_LIFT_BAND
    return low <= spatial_lift <= high


def k6_gate(vetoed: int, freeze_false_alarms: int) -> bool:
    """libero_long 的 freeze 分支误报中至少 1/3 被 R 否决。"""
    if freeze_false_alarms <= 0:
        return False
    return vetoed >= K6_MIN_VETO_SHARE * freeze_false_alarms
```

K6 的 `freeze_false_alarms` 集合来自 `moe-v7-0905/results/survival_baseline/alarm_priors.csv`
中 `suite == "libero_long"`、freeze 分支、`correct == False` 的行（报告记录为 51 个）；
`vetoed` 是这些 episode 中 `R` 规则未报警的数量。

7. `results/hazard/` 写出：`hazard_metrics.csv`、`hazard_by_suite.csv`、
   `prior_cut_sensitivity.csv`、`gates.json`、`comparison_v4_v6_v7.csv`。

- [ ] **Step 2: 运行闸门测试确认通过**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest tests/test_gates.py -v
```

Expected: 4 passed

- [ ] **Step 3: 跑评估**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/evaluate_hazard.py --output results/hazard
```

Expected: 先打印封存哈希，再打印指标，最后
`K4 ... PASS/FAIL`、`K5 spatial lift=... PASS/FAIL`、`K6 vetoed=.../51 PASS/FAIL`。

**K5 FAIL 时不要当作好消息。** 按 spec §6，spatial 提升显著超出 `[0.9, 1.15]` 应触发泄漏审计：
检查 `R` 是否间接读到了 episode 长度或 outcome，并向用户报告后再继续。

- [ ] **Step 4: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/evaluate_hazard.py moe-progress-ratio-v12-0906/tests/test_gates.py moe-progress-ratio-v12-0906/results/hazard
git commit -m "feat: seal and evaluate the progress ratio against per-chunk excess hazard"
```

---

## Task 12: LOSO 三级对照

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/evaluate_loso_v12.py`

`r*` 仍是语料分位数，所以 `LOSO_VALIDATION` §4 的 horizon 配方漂移依然适用，必须重跑。
直接复用 `moe-v7-0905/experiments/loso_folds.py` 的折定义。

- [ ] **Step 1: 实现**

三级对照，与 v7 同口径：

- `published`：Task 11 的全语料 profile；
- `loso_l1`：分位数水平锁死，只在校准 suite 上重算 `ratio_threshold` 与 `eps_length`；
- `loso_l2`：额外在校准 suite 的 development outcomes 上重跑 840 点选择。

输出 `results/loso/loso_metrics.csv`（三级 × 四折，含 cluster bootstrap CI）、
`threshold_drift.csv`（两个常量 × 四折的漂移百分比）、`fold_selection.json`、
`loso_summary.json`（macro 与 micro 分别汇总）。

**macro 与 micro 必须同时报告。** v7 的 LOSO 显示两者分歧很大（macro precision 几乎不掉，
micro 掉 6.33 pp），只报 macro 会掩盖误报向 `libero_long` 集中的事实。

回归锚点测试：全语料校准必须复现 Task 11 的 `global_profile.npz` 两个常量，
已发布常量必须复现 `sealed_first_alarms.npz`。两条都过，才能说退化来自校准切片而非重实现差异。

- [ ] **Step 2: 跑 LOSO**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
python experiments/evaluate_loso_v12.py --output results/loso
pytest tests/ -q
```

Expected: 四折全部输出，回归锚点通过。

- [ ] **Step 3: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906/experiments/evaluate_loso_v12.py moe-progress-ratio-v12-0906/results/loso
git commit -m "feat: re-run leave-one-suite-out validation for the progress-ratio cut"
```

---

## Task 13: 原始因果回放与报告

**Files:**
- Create: `moe-progress-ratio-v12-0906/experiments/verify_raw_causal_gpu.py`
- Create: `moe-progress-ratio-v12-0906/REPORT_ZH.md`
- Create: `moe-progress-ratio-v12-0906/docs/REPRODUCE.md`

- [ ] **Step 1: 实现因果回放**

照抄 `moe-v7-0905/experiments/verify_raw_causal_gpu.py` 的结构，把 monitor 换成
`ProgressGuardMonitor`。取 24 个 episode，均匀覆盖 alarm / no-alarm，且**不看 outcome** 抽样。

必须核对的项：`first alarm 与 sealed cache 完全一致 24/24`、
`改写未来 query 后既有前缀不变 24/24`、`GPU vs cache 最大原始特征误差 < 2e-5`、
`stream vs batch 最大 score 误差 < 3e-5`、两张卡都实际执行。

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
CUDA_VISIBLE_DEVICES=6,7 python experiments/verify_raw_causal_gpu.py
```

- [ ] **Step 2: 写报告**

`REPORT_ZH.md` 必须包含，且按此顺序：

1. **结论**：`R` 是否通过 K4，给出低先验 TP / precision / lift 与方案 B 的对照。
2. **六条闸门的实测值与判定**，包括未通过的。
3. **主表**：三个先验切点下的低先验 TP / precision / lift / 召回，分 suite。
   **每个 precision 数字必须同时给出匹配存活先验与提升倍数**——单独引用 precision 会把
   存活基线算进检测器的功劳。
4. **2×2 表型表**，含四格的 episode 数与结局分布。
5. **LOSO 三级 macro 与 micro**。
6. **限制**，至少覆盖：external 8B 不是 pristine holdout（v3–v7 与 HB bundle 已反复查看）；
   存活先验虽在 development 估计但两批 cohort 任务重叠；`r*` 仍依赖语料 horizon 配方；
   `W` 与 `K` 的网格边界；`libero_spatial` 的结论边界；本实验不回答"报警后干预是否有用"。
7. **复现命令**。

`docs/REPRODUCE.md` 按 Task 5 → 7 → 8 →（闸门）→ 10 → 11 → 12 → 13 的顺序列出全部命令与
预期耗时。

- [ ] **Step 3: 全量测试与提交**

```bash
cd /home/jovyan/work/himoe-vla/moe-progress-ratio-v12-0906
pytest -q
```

Expected: 全部通过。

```bash
cd /home/jovyan/work/himoe-vla
git add moe-progress-ratio-v12-0906
git commit -m "docs: report the progress-ratio guard against its pre-registered gates"
```

---

## 自查记录

- **Spec 覆盖**：§2 量定义 → Task 2/3；§3 检测规则 → Task 3/4/6；§4 评估协议 → Task 9/11；
  §5 防泄漏封存 → Task 11（封存顺序）/Task 12（LOSO）/Task 13（因果回放）；
  §6 六条闸门 → K1/K3 Task 8、K2 Task 7、K4/K5/K6 Task 11；§7 文件结构 → 全部任务；
  §8 明确不做 → 计划中无对应实现，正确。
- **`eps_length` 校准**：spec 要求取 pooled `L` 的 q1 分位数，在 Task 7 的 profile 冻结中完成。
- **命名一致性**：`progress_ratio` / `path_length` / `group_ratio` / `persistent_low` /
  `quantile_lower` / `row_min` / `first_below` 在 Task 3、4、6、7 中签名一致；
  `GlobalProgressProfile` 的五个字段在 Task 6、7、11、12 中一致。
- **W_GRID 与 LAGS**：Task 2 定义 `LAGS = (1,) + W_GRID`，Task 5 缓存、Task 6 profile 校验
  （`window not in LAGS` 抛错）、Task 7 网格三处引用同一常量。
- **自查中修掉的四处**：(1) Task 6 两处测试阈值 `1.01` / `-1.0` 会被
  `GlobalProgressProfile.__post_init__` 的区间校验拒绝，改为 `1.0` / `0.0`——利用
  `0 <= R <= 1` 的边界达到同样的"必触发 / 永不触发"效果；(2) Task 7 代码里
  `progress_ratio_module` 与其余任务的 `progress_ratio` 命名不一致；(3) `hazard_common.py`
  原先在 Task 9 创建却被 Task 7 使用，现改为 Task 7 创建两个函数、Task 9 追加两个函数；
  (4) Task 11 的三条闸门原先无测试，现补 `tests/test_gates.py` 并前置为 Step 0a/0b——
  闸门常数不锁死，跑出难看结果时最容易被"顺手"放宽。
