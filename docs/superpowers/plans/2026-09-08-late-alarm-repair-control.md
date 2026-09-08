# 报警后修复控制实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GPU 4/5 上，对 Long Pro/Plus 失败主轨迹在早/中/晚三个分叉点施加六种强度递增的无训练修复控制，测量修复后固定窗口与原上限内的救回率，并记录切回策略后第一次前向的 MoE 内部结构读数。

**Architecture:** 复用 `moe-trap-control` 的精确 C0 重放、快照、配对随机流、分块存储与编排器；新增一个纯函数的修复控制器模块、一个新的协议/臂注册模块、一个派生的采集会话，以及编排器的一个新计划模式。REPAIR 阶段不调用模型，只走 `env.step`；切回后沿用原生 10 步 chunk 与新噪声流。

**Tech Stack:** Python 3.8（LIBERO 环境，`/home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python`），NumPy，MuJoCo/robosuite OSC_POSE（单位动作 = 5 cm 位移目标），冻结的 kNN-20 与 v7 monitor，pytest（仓库常规 Python 3 环境跑 CPU 测试）。

设计文档：`docs/superpowers/specs/2026-09-08-late-alarm-repair-control-design.md`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `moe-trap-control/repair_controller.py`（新） | 目标解析、物理类别、比例控制 chunk 生成、REPAIR 状态机、脚本化抓取；不 import LIBERO，靠鸭子类型 env |
| `moe-trap-control/repair_control.py`（新） | 协议常量、GPUS=(4,5)、臂注册、三种分叉事件、计划加载/作业调度/存储上界 |
| `moe-trap-control/collect_repair_control.py`（新） | `RepairSession`：C0 重放并在三种分叉点存快照与物理量，分支执行 REPAIR 加 VLA 续跑 |
| `moe-trap-control/prepare_repair_experiment.py`（新） | 抽样 50 失败 + 15 成功主轨迹，写冻结计划 |
| `moe-trap-control/audit_repair_experiment.py`（新） | 独立审计：C0、快照、臂完整性、守卫日志、配对 |
| `moe-trap-control/analyze_repair_experiment.py`（新） | 终点、分层、结构读数、图与中文报告 |
| `moe-trap-control/test_repair_control.py`（新） | CPU 单测 |
| `moe-trap-control/run_collection_preflight.py`（改） | 新增 `--repair-plan` 模式 |

不改 `adaptive_control.py`、`collect_adaptive_control.py` 与任何冻结产物。

---

### Task 1: 修复控制器纯函数模块

**Files:**
- Create: `moe-trap-control/repair_controller.py`
- Test: `moe-trap-control/test_repair_control.py`

- [ ] **Step 1: 写失败的测试**

```python
# moe-trap-control/test_repair_control.py
import numpy as np
import pytest

from repair_controller import (retract_chunk, RepairPhase, physical_class, aperture,
                               eef_position, ProportionalRetract)


def test_retract_chunk_moves_toward_target_and_opens_gripper():
    eef = np.array([0.0, 0.0, 1.0])
    target = np.array([0.10, 0.0, 1.10])
    chunk = retract_chunk(eef, target, gain=0.8, unit_metres=0.05)
    assert chunk.shape == (10, 7)
    assert np.all(chunk[:, 6] == -1.0)                 # 张开
    assert chunk[0, 0] > 0 and chunk[0, 2] > 0         # 朝目标
    assert np.all(np.abs(chunk[:, :3]) <= 1.0)          # 饱和
    assert np.all(chunk[:, 3:6] == 0.0)                 # 不转动


def test_retract_chunk_is_zero_when_at_target():
    eef = np.array([0.3, -0.1, 0.9])
    chunk = retract_chunk(eef, eef.copy(), gain=0.8, unit_metres=0.05)
    assert np.all(chunk[:, :6] == 0.0)


def test_phase_guard_completes_when_close_for_five_steps():
    phase = RepairPhase(target=np.array([0.0, 0.0, 1.0]), tolerance=0.02, settle_steps=5, max_chunks=6)
    for _ in range(4):
        assert not phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.reason == "reached"


def test_phase_guard_times_out_after_max_chunks():
    phase = RepairPhase(target=np.array([1.0, 0.0, 1.0]), tolerance=0.02, settle_steps=5, max_chunks=2)
    for chunk in range(2):
        phase.begin_chunk()
        for _ in range(10):
            phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.exhausted and phase.reason == "max_chunks"


def test_physical_class_untouched_displaced_dropped():
    initial = np.array([0.0, 0.0, 0.90])
    assert physical_class(np.array([0.005, 0.0, 0.90]), initial, np.array([0.0, 0.0, 1.0]), table_z=0.85, tilt_deg=0.0, grasped=False) == "untouched"
    assert physical_class(np.array([0.05, 0.0, 0.90]), initial, np.array([0.0, 0.0, 1.0]), table_z=0.85, tilt_deg=5.0, grasped=False) == "displaced_reachable"
    assert physical_class(np.array([0.05, 0.0, 0.70]), initial, np.array([0.0, 0.0, 1.0]), table_z=0.85, tilt_deg=0.0, grasped=False) == "dropped_or_tipped"
    assert physical_class(np.array([0.05, 0.0, 0.90]), initial, np.array([0.0, 0.0, 1.0]), table_z=0.85, tilt_deg=60.0, grasped=False) == "dropped_or_tipped"
    assert physical_class(np.array([0.0, 0.0, 1.0]), initial, np.array([0.0, 0.0, 1.0]), table_z=0.85, tilt_deg=0.0, grasped=True) == "in_hand"


def test_aperture_and_eef_from_observation():
    obs = {"robot0_gripper_qpos": np.array([0.03, -0.03]), "robot0_eef_pos": np.array([1.0, 2.0, 3.0])}
    assert aperture(obs) == pytest.approx(0.06)
    assert np.allclose(eef_position(obs), [1.0, 2.0, 3.0])


def test_proportional_retract_plans_chunks_until_guard():
    controller = ProportionalRetract(target=np.array([0.0, 0.0, 1.10]), gain=0.8, unit_metres=0.05,
                                     tolerance=0.02, settle_steps=5, max_chunks=6)
    eef = np.array([0.0, 0.0, 1.00])
    chunk = controller.next_chunk(eef)
    assert chunk.shape == (10, 7) and chunk[0, 2] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/jovyan/work/himoe-vla/moe-trap-control && python3 -m pytest test_repair_control.py -q`
Expected: `ModuleNotFoundError: No module named 'repair_controller'`

- [ ] **Step 3: 实现最小模块**

```python
# moe-trap-control/repair_controller.py
"""Train-free repair controllers that act only through env.step; no model calls."""

from __future__ import annotations

import numpy as np

OPEN, CLOSE = -1.0, 1.0
CLOSE_APERTURE = 0.05          # analysis_trap_taxonomy 的闭合阈值
DEPARTURE_M = 0.10             # phantom：末端离开闭合点
STILL_M = 0.01                 # phantom：目标位移上限
NEAR_M = 0.16                  # 闭合点到目标的距离上限
ABOVE_M = 0.10                 # 回退到目标上方的高度
LIFT_M = 0.08                  # 脚本化抓取抬升高度
TABLE_CLEARANCE_M = 0.03       # 物体中心低于桌面这么多算掉落
TILT_DEG = 45.0


def aperture(obs):
    q = np.asarray(obs["robot0_gripper_qpos"], np.float64)
    return float(q[0] - q[1])


def eef_position(obs):
    return np.asarray(obs["robot0_eef_pos"], np.float64).copy()


def retract_chunk(eef, target, gain, unit_metres, steps=10, gripper=OPEN):
    """One 10x7 LIBERO action chunk: proportional pull toward target, no rotation, gripper fixed."""
    error = np.asarray(target, np.float64) - np.asarray(eef, np.float64)
    command = np.clip(gain * error / unit_metres, -1.0, 1.0)
    chunk = np.zeros((steps, 7), np.float32)
    chunk[:, :3] = command.astype(np.float32)
    chunk[:, 6] = gripper
    return chunk


class RepairPhase:
    """Guard g2: reached when within tolerance for settle_steps consecutive env steps; else exhausted after max_chunks."""

    def __init__(self, target, tolerance, settle_steps, max_chunks):
        self.target = np.asarray(target, np.float64)
        self.tolerance, self.settle_steps, self.max_chunks = tolerance, settle_steps, max_chunks
        self.chunks, self.streak, self.reason, self.done = 0, 0, None, False
        self.history = []

    def begin_chunk(self):
        self.chunks += 1

    @property
    def exhausted(self):
        return self.chunks >= self.max_chunks and not self.done

    def observe(self, eef):
        distance = float(np.linalg.norm(np.asarray(eef, np.float64) - self.target))
        self.history.append(distance)
        self.streak = self.streak + 1 if distance < self.tolerance else 0
        if self.streak >= self.settle_steps:
            self.done, self.reason = True, "reached"
        elif self.exhausted:
            self.reason = "max_chunks"
        return self.done


class ProportionalRetract:
    def __init__(self, target, gain, unit_metres, tolerance, settle_steps, max_chunks, gripper=OPEN):
        self.gain, self.unit_metres, self.gripper = gain, unit_metres, gripper
        self.phase = RepairPhase(target, tolerance, settle_steps, max_chunks)

    def next_chunk(self, eef):
        self.phase.begin_chunk()
        return retract_chunk(eef, self.phase.target, self.gain, self.unit_metres, gripper=self.gripper)


def physical_class(position, initial, eef, table_z, tilt_deg, grasped):
    if grasped:
        return "in_hand"
    if position[2] < table_z - TABLE_CLEARANCE_M or tilt_deg > TILT_DEG or np.linalg.norm(position - eef) > 0.60:
        return "dropped_or_tipped"
    return "untouched" if np.linalg.norm(position - initial) < 0.02 else "displaced_reachable"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /home/jovyan/work/himoe-vla/moe-trap-control && python3 -m pytest test_repair_control.py -q`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
cd /home/jovyan/work/himoe-vla
git add moe-trap-control/repair_controller.py moe-trap-control/test_repair_control.py docs/superpowers/specs/2026-09-08-late-alarm-repair-control-design.md docs/superpowers/plans/2026-09-08-late-alarm-repair-control.md
git commit -m "feat: train-free repair controller primitives for the late-alarm rescue experiment"
```

---

### Task 2: 目标解析与物理类别的 env 适配层

**Files:**
- Modify: `moe-trap-control/repair_controller.py`（追加）
- Test: `moe-trap-control/test_repair_control.py`（追加）

- [ ] **Step 1: 写失败的测试（鸭子类型 env）**

```python
class _FakeSim:
    class data:
        body_xpos = np.array([[0.0, 0.0, 0.9], [0.3, 0.0, 0.9]])
        body_xquat = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])


class _FakeInner:
    parsed_problem = {"goal_state": [["On", "pot_1", "stove_1"], ["On", "pot_2", "stove_1"]],
                      "objects": {"moka_pot": ["pot_1", "pot_2"]}, "fixtures": {"stove": ["stove_1"]}}
    obj_body_id = {"pot_1": 0, "pot_2": 1}
    sim = _FakeSim()

    def _eval_predicate(self, state):
        return state[1] == "pot_2"       # pot_2 已到位，pot_1 未完成


class _FakeEnv:
    env = _FakeInner()


def test_resolve_target_returns_first_unsatisfied_movable_object():
    from repair_controller import resolve_target
    target = resolve_target(_FakeEnv())
    assert target["name"] == "pot_1"
    assert np.allclose(target["position"], [0.0, 0.0, 0.9])
    assert target["unsatisfied"] == [["On", "pot_1", "stove_1"]]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest test_repair_control.py -q -k resolve_target`
Expected: `ImportError: cannot import name 'resolve_target'`

- [ ] **Step 3: 实现**

```python
def _movable_names(inner):
    names = set()
    for group in inner.parsed_problem["objects"].values():
        names.update(group)
    return names


def resolve_target(env):
    """First unsatisfied goal predicate whose subject is a movable (free) object; None if only fixtures remain."""
    inner = env.env
    movable = _movable_names(inner)
    unsatisfied = [list(state) for state in inner.parsed_problem["goal_state"] if not inner._eval_predicate(state)]
    for state in unsatisfied:
        candidates = [name for name in state[1:] if name in movable]
        if candidates:
            name = candidates[0]
            body = inner.obj_body_id[name]
            return dict(name=name, position=np.asarray(inner.sim.data.body_xpos[body], np.float64).copy(),
                        quaternion=np.asarray(inner.sim.data.body_xquat[body], np.float64).copy(),
                        unsatisfied=unsatisfied, articulated_pending=False)
    return dict(name=None, position=None, quaternion=None, unsatisfied=unsatisfied, articulated_pending=bool(unsatisfied))


def tilt_degrees(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)     # z 分量的 z 轴旋转后
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest test_repair_control.py -q`
Expected: `8 passed`

- [ ] **Step 5: 提交** `git commit -am "feat: goal-object resolution for repair controller"`

---

### Task 3: 协议、臂注册与分叉事件

**Files:**
- Create: `moe-trap-control/repair_control.py`
- Test: `moe-trap-control/test_repair_control.py`（追加）

- [ ] **Step 1: 测试**

```python
def test_repair_events_three_fork_times():
    from repair_control import events_for, LATE_QUERY
    events = events_for("abc", knn_first=25, length=52, failed=True)
    timings = [e["timing"] for e in events]
    assert timings == ["mid", "late"]                      # early 由重放时在线决定
    assert events[0]["start_query"] == 26 and events[1]["start_query"] == LATE_QUERY == 44
    assert events_for("abc", knn_first=25, length=52, failed=False)[0]["timing"] == "mid"
    assert events_for("abc", knn_first=-1, length=52, failed=True) == [dict(events[1], event_id=events[1]["event_id"])]


def test_arm_registry_orders_by_strength():
    from repair_control import ARMS
    assert list(ARMS) == ["new_noise", "open_only", "retract_history", "retract_above_target",
                          "retract_above_target_noisy", "scripted_regrasp"]
    assert ARMS["retract_above_target_noisy"]["xy_noise_m"] == 0.02
```

- [ ] **Step 2: 跑测试确认失败** — `ModuleNotFoundError: repair_control`

- [ ] **Step 3: 实现**

```python
# moe-trap-control/repair_control.py
"""Frozen protocol for the late-alarm repair experiment: GPUs 4/5, six arms, three fork timings."""

from __future__ import annotations

import json
from pathlib import Path

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_storage import digest
from adaptive_control import RouteRisk, REFERENCE, PARAMETERS  # 冻结 kNN-20 与 v7 monitor

PROTOCOL = "moe_control.repair_recovery.v1"
GPUS = (4, 5)
RENDER_GPUS = (4, 5)
LATE_QUERY = 44
WINDOW_STEPS = 300
HORIZON_STEPS = 520
ARMS = {
    "new_noise": dict(repair="none"),
    "open_only": dict(repair="open"),
    "retract_history": dict(repair="retract", target="history"),
    "retract_above_target": dict(repair="retract", target="above_target"),
    "retract_above_target_noisy": dict(repair="retract", target="above_target", xy_noise_m=0.02),
    "scripted_regrasp": dict(repair="regrasp", target="above_target"),
}
CONTROLLER = dict(gain=0.8, unit_metres=0.05, tolerance_m=0.02, settle_steps=5, max_chunks=6,
                  open_chunks=1, above_m=0.10, lift_m=0.08, grasp_close_chunks=1)
CONTRACT = dict(
    trigger_mid="frozen global Euclidean kNN-20 first alarm + 1, deployable",
    trigger_early="physical grasp-verification rule evaluated online during C0: closed aperture, eef departed >= 0.10 m, target moved < 0.01 m since closure",
    trigger_late="fixed query 44 (80 steps left)",
    repair="proportional task-space controller through env.step only; no model call; single switch per suffix",
    handback="guard: within 0.02 m for 5 consecutive steps or 6 chunks; then native 10-step chunks with the replicate's new noise stream",
    endpoints="B: success within 300 steps from fork including repair; A: success within 520 - steps_before",
    internal_readouts="full HB probs stored for every VLA query; first post-handback query is the competence observer",
    hidden_capture=False, batch_size=1, threshold_fitting=False, training=False,
)


def event_id(main_id, timing, start):
    return stable_id(PROTOCOL, main_id, timing, int(start))


def events_for(main_id, knn_first, length, failed):
    events = []
    q = int(knn_first)
    if q >= 0 and q + 1 < length:
        events.append(dict(event_id=event_id(main_id, "mid", q + 1), timing="mid", start_query=q + 1,
                           alarm_query=q, deployable=True))
    if failed and LATE_QUERY < length:
        events.append(dict(event_id=event_id(main_id, "late", LATE_QUERY), timing="late", start_query=LATE_QUERY,
                           alarm_query=q, deployable=False))
    return events


def early_event(main_id, start):
    return dict(event_id=event_id(main_id, "early", start), timing="early", start_query=int(start),
                alarm_query=-1, deployable=True)


def seed_for(main_id, replicate, index, stream, candidate=0):
    if stream not in ("policy", "environment") or replicate < 0 or index < 0:
        raise ValueError("Invalid repair random stream")
    return int(stable_id(PROTOCOL, main_id, int(replicate), int(index), stream, int(candidate))[:8], 16)


def branch_bound(arms, replicates):
    queries = (WINDOW_STEPS // 10 + 1) * len(arms) * replicates
    return queries * 190 * 1024 + 4 * 1024**2


def scheduled_jobs(plan, inventory, output):
    jobs = []
    for task in plan["tasks"]:
        parent_id = task["main_id"]
        base = dict(inventory[task["variant_id"]], main_id=parent_id,
                    noise_seed=int(task["noise_seed"]), init_index=int(task["init_index"]))
        replay_id = parent_id + "/replay"
        jobs.append(dict(base, job_id=replay_id, depends_on=None,
                         sampling=dict(kind="replay", parent=task, max_output_bytes=task["max_output_bytes"])))
        jobs.append(dict(base, job_id=parent_id + "/branches", depends_on=replay_id,
                         sampling=dict(kind="branches", parent=task, arms=plan["arms"], replicates=plan["replicates"],
                                       replay_directory=str(Path(output) / "tasks" / replay_id),
                                       max_output_bytes=branch_bound(plan["arms"], plan["replicates"]) * 3)))
    jobs.sort(key=lambda row: (row["depends_on"] is not None, -row["sampling"]["max_output_bytes"], row["job_id"]))
    return jobs


def load_plan(path, model="long"):
    plan = json.loads(Path(path).read_text())
    verify_frozen_alarm()
    if (plan["protocol"] != PROTOCOL or plan["contract"] != CONTRACT or model != "long" or
            plan["allowed_gpus"] != list(GPUS) or plan["render_gpus"] != list(RENDER_GPUS) or
            plan["frozen_parameters_sha256"] != PARAMETERS_SHA256 or plan["arms_registry"] != ARMS or
            plan["controller"] != CONTROLLER):
        raise ValueError("Repair plan contract changed")
    for source, expected in plan["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError("Repair input changed: " + source)
    if not plan["tasks"] or len({t["main_id"] for t in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Empty/duplicated repair parents")
    for task in plan["tasks"]:
        directory = Path(task["parent_directory"])
        for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(directory / filename) != task[key]:
                raise ValueError("Parent commit changed")
        if task["events"] != events_for(task["main_id"], task["first_alarm"], task["parent_queries"], task["failed"]):
            raise ValueError("Repair trigger positions changed")
    return plan
```

分支作业按主轨迹聚合（一条作业跑该主轨迹的全部事件、臂、重复），因为 early 分叉点只有重放后才知道。

- [ ] **Step 4: 跑测试通过**；**Step 5: 提交** `git add moe-trap-control/repair_control.py && git commit -m "feat: repair experiment protocol, arms and fork events"`

---

### Task 4: 采集会话 `collect_repair_control.py`

**Files:**
- Create: `moe-trap-control/collect_repair_control.py`

从 `collect_adaptive_control.py` 复制 `RecoverySession` 骨架后做四处改动。

- [ ] **Step 1: 类骨架与 env 创建**

`__init__` 与原版相同，但 `from repair_control import ...`，`GPUS/RENDER_GPUS` 校验用新值，`self.report["protocol"] = PROTOCOL`；`new_env(extended=False)` 的 horizon 为 `self.horizon + 11 + (WINDOW_STEPS if extended else 0)`。C0 用非扩展 horizon，分支用扩展 horizon。

- [ ] **Step 2: 重放中在线评估 early 触发与物理量**

在 `replay()` 循环每个 query 开始处：

```python
target = resolve_target(self.env)
ap, eef = aperture(obs), eef_position(obs)
if target["name"] is not None:
    if closure is None and ap < CLOSE_APERTURE and previous_ap >= CLOSE_APERTURE and np.linalg.norm(eef - target["position"]) < NEAR_M:
        closure = dict(query=q, eef=eef.copy(), object=target["position"].copy(), name=target["name"])
    if closure is not None and early is None and target["name"] == closure["name"] and ap < CLOSE_APERTURE:
        if np.linalg.norm(eef - closure["eef"]) >= DEPARTURE_M and np.linalg.norm(target["position"] - closure["object"]) < STILL_M:
            early = q
            event = early_event(self.args.main_id, q)
            self.report["events"].append(event); targets[q] = event
if ap >= CLOSE_APERTURE:
    closure = None if closure is not None and q > closure["query"] + 1 and ... else closure   # 重新张开则重置闭合事件
previous_ap = ap
physics_rows.append(dict(query=q, aperture=ap, eef=eef, target_name=target["name"], target_position=target["position"],
                         unsatisfied=len(target["unsatisfied"]), knn_score=score, freeze_score=alarm["freeze_score"],
                         acceleration_score=alarm["acceleration_score"]))
```

在快照保存时同时记录 `physical_class`、`target`、`initial_target_position`（来自 `preflight_q000` 后第一次 `resolve_target`）与 `history_pose`（最近一次闭合前最后一个开爪 query 的末端位姿，若无则 None）。物理表写 `events/<id>/physics.json` 与 `c0_physics.npz`。

- [ ] **Step 3: 分支执行 REPAIR 阶段**

```python
def repair_actions(self, arm, saved_physics, obs, replicate):
    spec = ARMS[arm]
    if spec["repair"] == "none":
        return None
    eef = eef_position(obs)
    if spec["repair"] == "open":
        return [retract_chunk(eef, eef, 0.0, CONTROLLER["unit_metres"])] * CONTROLLER["open_chunks"], None
    if spec["target"] == "history":
        target = np.asarray(saved_physics["history_pose"]) if saved_physics["history_pose"] is not None else eef + [0, 0, CONTROLLER["above_m"]]
    else:
        position = np.asarray(saved_physics["target_position"])
        if spec.get("xy_noise_m"):
            rng = np.random.default_rng(seed_for(self.args.main_id, replicate, 0, "policy", 1))
            position = position + np.append(rng.normal(0, spec["xy_noise_m"], 2), 0.0)
        target = position + np.array([0.0, 0.0, CONTROLLER["above_m"]])
    return ProportionalRetract(target, CONTROLLER["gain"], CONTROLLER["unit_metres"], CONTROLLER["tolerance_m"],
                               CONTROLLER["settle_steps"], CONTROLLER["max_chunks"]), target
```

分支循环：先执行 REPAIR（每 chunk 用 `self.advance(obs, chunk, 10, steps, replicate)`，每步后 `phase.observe(eef)`；REPAIR 行写入 `branches/<arm>/repair` 分块，字段：`query`、`actions`、`sim_before/after`、`eef_before/after`、`aperture_after`、`target`、`distance_after`、`reason`），`scripted_regrasp` 在到达上方后追加：下降到 `target_z + grasp_offset`（物体 geom 包围盒半高，取 `inner.sim.model.geom_size` 的 z 分量最大值，若不可得取 0.02）、闭合 1 chunk、抬升 `lift_m`，记录 `lifted = object_z_after - object_z_before >= 0.02`。然后 VLA 循环沿用原版但 `candidate=0`、无候选池、`limit = min(10, window_remaining)`，`window_remaining = WINDOW_STEPS - branch["action_steps"]`；成功时记录 `success_step`，`success_within_original = success_step <= HORIZON_STEPS - saved["action_steps"]`。每个 VLA 行照旧写全 HB 概率。

- [ ] **Step 4: 分支作业遍历事件、臂、重复**：`branches()` 读取 replay 的 `result.json["events"]`（含 early），对 `for event in events: for replicate in range(replicates): for arm in arms:` 执行，输出目录 `branches/<event_id>/repeat<r>/<arm>/`。每个臂完成后 `atomic_json(branch.json)`；`self.save()` 每 8 个 query。

- [ ] **Step 5: 用 CPU 假环境做一次 dry-run 测试**：`test_repair_control.py::test_repair_actions_registry_covers_all_arms`，用 `_FakeEnv` 与 `saved_physics` 检查每个臂都能生成计划（`new_noise` 返回 None，其余返回控制器与目标）。

- [ ] **Step 6: 提交** `git add moe-trap-control/collect_repair_control.py && git commit -m "feat: repair collection session with online early trigger and REPAIR phase"`

---

### Task 5: 计划生成 `prepare_repair_experiment.py`

**Files:**
- Create: `moe-trap-control/prepare_repair_experiment.py`

- [ ] **Step 1: 实现**

读 `design/experiment_long_scale_audit_20260908.json` 与 `design/experiment_long_batch1_audit_20260908.json` 的 tasks，与 `design/experiment_long_screen_summary_20260908/first_alarms.csv` 连接。筛选：`suite == libero_10`；`base_task` 属于 7 个纯抓放任务（按 `task_name.split("_view_")[0]` 去掉 Pro 后缀匹配前缀）；失败者要求 `0 <= knn20 <= 44`。按基础任务分层、`stable_id(PROTOCOL, "parent_selection", main_id)` 排序取前 `--per-task`（主体 7 或 8 使总数 50；冒烟 `--stage smoke` 取 3 条不同任务）。成功者取 `knn20 >= 0` 的 15 条，同样分层。任务记录字段：`main_id, variant_id, benchmark, category, analysis_role, parent_directory, parent_queries, parent_commit_sha256, parent_manifest_sha256, noise_seed, init_index, first_alarm, failed, events, max_output_bytes`。计划字段：`protocol, model, stage, arms, arms_registry, controller, replicates, contract, source_sha256, frozen_parameters_sha256, tasks, allowed_gpus=[4,5], render_gpus=[4,5], replicas_per_gpu=8, workers_per_gpu=8, mps=True, storage_quota_gib, disk_floor_gib=8`。写前检查 `shutil.disk_usage(HERE).free - bound >= 8 GiB`；写后 `load_plan` 自检。

- [ ] **Step 2: 运行冒烟计划** `python3 prepare_repair_experiment.py --stage smoke --replicates 1 --output design/repair_plans_20260908/smoke.json`，期望打印 `parents=3`。

- [ ] **Step 3: 提交**

---

### Task 6: 编排器新增 `--repair-plan`

**Files:**
- Modify: `moe-trap-control/run_collection_preflight.py`（`run()` 内 plan 分支、worker 脚本选择、sources 列表、argparse）

- [ ] **Step 1: 改动点**

```python
repair = getattr(args, "repair_plan", None) is not None
reuse = continuation or control or repair
plan_path = args.repair_plan if repair else args.control_plan if control else args.continuation_plan if continuation else args.plan
...
elif repair:
    from repair_control import load_plan as load_repair_plan, scheduled_jobs as repair_jobs
    plan = load_repair_plan(plan_path, args.model)
    jobs = repair_jobs(plan, inventory, args.output)
...
worker_script = ("collect_repair_control.py" if repair else "collect_adaptive_control.py" if control else ...)
...
if repair:
    sources += [HERE / name for name in ("repair_control.py", "repair_controller.py", "collect_repair_control.py",
                "prepare_repair_experiment.py", "audit_repair_experiment.py")]
...
selection.add_argument("--repair-plan", type=Path)
```

`control` 分支里所有 `if control:` 的依赖调度逻辑改为 `if control or repair:`。

- [ ] **Step 2: 语法检查** `python3 -m py_compile run_collection_preflight.py` 与 `/home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python -m py_compile collect_repair_control.py repair_controller.py repair_control.py`（Python 3.8 兼容：不用 `str.removeprefix`、不用 `X | Y` 类型语法）。

- [ ] **Step 3: 提交**

---

### Task 7: P0 冒烟运行与标定

- [ ] **Step 1: 确认 GPU 4/5 空闲显存 ≥ 39,000 MiB**：`nvidia-smi -i 4,5 --query-gpu=memory.free --format=csv`
- [ ] **Step 2: 运行**

```bash
cd /home/jovyan/work/himoe-vla/moe-trap-control
nohup python3 run_collection_preflight.py --repair-plan design/repair_plans_20260908/smoke.json \
  --gpus 4 5 --render-gpus 4 5 --model long --replicas 8 --workers-per-gpu 8 --mps --paired-branches \
  --storage-quota-gib 3 --disk-floor-gib 8 --job-timeout 3600 \
  --output design/repair_smoke_20260908 > design/repair_smoke_20260908.log 2>&1 &
```

- [ ] **Step 3: 检查** `summary.json` 的 `c0_passed == 3`，每个分支 `repair/` 的 `reason` 与到达步数；若 6 个 chunk 内到不了 0.02 m，把 `gain` 调到 1.0 并重跑冒烟；若 `scripted_regrasp` 的 `lifted` 全为 False，调 `grasp_offset`。标定值写回 `CONTROLLER` 并在设计文档 amendments 记录测量值。
- [ ] **Step 4: 冻结** 设计文档状态改为 frozen，提交。

---

### Task 8: 审计与分析

**Files:**
- Create: `moe-trap-control/audit_repair_experiment.py`：核对每个主轨迹 `c0.status == passed`，事件快照哈希，每个事件的臂×重复完整，REPAIR 行的 `reason ∈ {reached, max_chunks}`，VLA 行 `executed_action_count` 之和等于 `action_steps`，配对种子由 `seed_for` 复算一致，无 hidden 字段。输出 `design/repair_<stage>_audit_20260908.json`。
- Create: `moe-trap-control/analyze_repair_experiment.py`：按 事件时刻 × 物理类别 × 臂 汇总 `B 救回/分母`、`A 救回/分母`、复陷、破坏；主轨迹与基础任务聚类 bootstrap 区间；结构读数：交回后第一个 VLA 行的 `hb_probs` 算 state token L2–L5 top-1 与熵、action token 弧的 bandedness 与到模板的 Procrustes（复用 `moe-token-geometry-0906/experiments/geometry.py::procrustes_disparity`，模板取 development 合并核）、以及交回后第 8 个 query 的 `knn_score`；逻辑回归（statsmodels 若可用，否则 numpy 迭代加权最小二乘）救回 ~ 时刻 + 类别 + 臂 + 读数。输出 `summary.json`、`metrics.csv`、`interventions.png`、`REPORT.zh.md`。

- [ ] 测试：`test_repair_control.py::test_endpoint_arithmetic`（B/A 判定函数）。
- [ ] 提交。

---

### Task 9: 主体运行、报告、迭代

- [ ] 生成主体计划：`python3 prepare_repair_experiment.py --stage main --replicates 2 --per-task 7 --output design/repair_plans_20260908/main.json`
- [ ] 运行同 Task 7 的命令，`--storage-quota-gib 9 --output design/repair_main_20260908`；每 20 分钟看 `PROGRESS` 行。
- [ ] 审计、分析、写 `moe-trap-control/REPAIR_CONTROL_EXPERIMENT.zh.md`：按设计文档 §6 的判读表逐条对号，报计数与分母。
- [ ] 迭代规则（写在报告里，不事后改主表）：若 A4 明显优于 A1 且读数能预测救回，下一轮把读数做成交回守卫；若 A2 就够，砍掉更强的臂；若全零，写否定结论并停止。
