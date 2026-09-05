#!/usr/bin/env python3
"""事件表构建 v2（PROTOCOL §8 Amendment 1）：loop/static onset 代理 → events/.../events.csv。

规则源头：himoe-vla_trap/code/analyze_trainfree_signal_matrix.py::physical_onsets（阈值不改）。
  loop  : ∃ left<right−2: ‖eef[r]−eef[l]‖≤0.045 ∧ max_obj_disp≤0.030 ∧ |grip[r]−grip[l]|≤0.012
          ∧ path(l→r)≥0.120 [∧ goal[l]−goal[r]≤0.035] → onset=最早 right
  static: 宽 2 滑窗 eef 步长≤0.020 ∧ obj 步长≤0.005 ∧ grip 步长≤0.001，连续 7 窗 → 第 7 窗+2
  trap  = min(有效 loop, 有效 static)；-1=无

proxy_grade（v2 两级）：
  full     : 仅 KITCHEN_SCENE8。物体=冻结槽位 sim[10:13]/[17:20]（sim_layout 断言），
             含 goal 项（参考=rolling-star dryrun_v10）。gripper=state[:,6:8].mean。
  objaware : 其余 43 任务。物体=sim_layout.json 全部非机器人 free joint（7 维槽）的位置；
             规则=full 去 goal 项（goal 在 obj 项下 1-Lipschitz 冗余，SCENE8 512 集实证
             逐集等价）；其余阈值、gripper 定义与 full 完全一致。
  （v1 degraded[eef+grip] 已废弃：loop specificity 0.077，验证记录保留在
   audit/proxy_validation.json 的 degraded_vs_full / task_event_summaries_v1_degraded。）

准入回归：objaware 实现先在 main16x32 SCENE8 上与 full 逐集对照（512 集三元 onset 全等 +
aligned_event_values 161/161），不通过则不覆写任何 events.csv。

语料只读；只写 events/ 与 audit/proxy_validation.json。
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
HERE = pathlib.Path(__file__).resolve().parent          # events/
AUDIT = HERE.parent / "audit"
REF_JSON = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/"
    "rolling-star-a100-long-t08-k16-20260828/analysis_dryrun_v10/"
    "physical_label_definitions.json"
)
TRAP_TABLES = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-vla_trap/results/trainfree_signal_matrix/tables"
)
SSM_CACHE = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_ssm/cache")

SUITES = ["libero_goal", "libero_long", "libero_object", "libero_spatial"]
CORPORA = {
    "grid50x8": {"root": HUB / "cache_new/HiMoE-VLA", "run_id": "right-50x8-20260903"},
    "main16x32": {"root": HUB / "cache/HiMoE-VLA", "run_id": "right-16x32"},
}
SCENE8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

# ---------------- 冻结阈值（不许改） ----------------
LOOP_EEF = 0.045
LOOP_OBJ = 0.030
LOOP_GRIP = 0.012
LOOP_PATH = 0.120
LOOP_GOAL = 0.035
STATIC_EEF = 0.020
STATIC_OBJ = 0.005
STATIC_GRIP = 0.001
STATIC_WIDTH = 2
STATIC_RUN = 7


def goal_distance(objects: np.ndarray, references: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(
        objects[:, :, None, :] - references[None, None, :, :], axis=-1
    ).min(axis=-1)
    return distance.max(axis=1)


def _static_from_windows(static_window: np.ndarray) -> int:
    run = 0
    for window_start, value in enumerate(static_window):
        run = run + 1 if value else 0
        if run >= STATIC_RUN:
            return window_start + STATIC_WIDTH
    return -1


def _onsets_obj(eef, objects, gripper, references=None) -> tuple[int, int, int]:
    """full（references 非 None，含 goal 项）与 objaware（references=None，去 goal 项）。
    左端扫描按 right 向量化；对 onset 与逐字原版等价（原版返回 right，与命中哪个 left 无关）。"""
    goal = goal_distance(objects, references) if references is not None else None
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    loop = -1
    for right in range(3, len(eef)):
        lo = slice(0, right - 2)
        ok = (
            (np.linalg.norm(eef[right] - eef[lo], axis=1) <= LOOP_EEF)
            & (np.linalg.norm(objects[right] - objects[lo], axis=2).max(axis=1) <= LOOP_OBJ)
            & (np.abs(gripper[right] - gripper[lo]) <= LOOP_GRIP)
            & (cumulative[right] - cumulative[lo] >= LOOP_PATH)
        )
        if goal is not None:
            ok &= goal[lo] - goal[right] <= LOOP_GOAL
        if ok.any():
            loop = right
            break

    object_step = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    grip_step = np.abs(np.diff(gripper))
    if len(step) >= STATIC_WIDTH:
        kernel = np.ones(STATIC_WIDTH)
        static_window = (
            (np.convolve(step, kernel, mode="valid") <= STATIC_EEF)
            & (np.convolve(object_step, kernel, mode="valid") <= STATIC_OBJ)
            & (np.convolve(grip_step, kernel, mode="valid") <= STATIC_GRIP)
        )
    else:
        static_window = np.zeros(0, bool)
    static = _static_from_windows(static_window)
    candidates = [v for v in (loop, static) if v >= 0]
    return loop, static, min(candidates, default=-1)


def physical_onsets_full(eef, objects, gripper, references):
    return _onsets_obj(eef, objects, gripper, references)


def physical_onsets_objaware(eef, objects, gripper):
    return _onsets_obj(eef, objects, gripper, None)


def load_object_slots(run: pathlib.Path) -> list[tuple[str, int]]:
    """sim_layout.json → 非机器人 free joint（7 维槽）的 (joint 名, 位置起始索引)。"""
    lay = json.load(open(run / "client" / "sim_layout.json"))
    slots = [(j["joint"], int(j["state_lo"])) for j in lay["joints"]
             if not j["is_robot"] and j["state_hi"] - j["state_lo"] == 7]
    if not slots:
        raise SystemExit(f"{run}: sim_layout 无 free-joint 物体")
    return slots


def check_scene8_layout(run: pathlib.Path) -> None:
    lay = json.load(open(run / "client" / "sim_layout.json"))
    slots = {j["joint"]: (j["state_lo"], j["state_hi"]) for j in lay["joints"]}
    assert slots.get("moka_pot_1_joint0") == (10, 17), slots
    assert slots.get("moka_pot_2_joint0") == (17, 24), slots
    assert lay["state_dim"] == 47, lay["state_dim"]


def episode_arrays(run: pathlib.Path, e_id: int, slots):
    with np.load(run / "client" / f"episode_{e_id:02d}.npz") as d:
        state = np.asarray(d["state"], np.float64)
        sim = np.asarray(d["sim_state"], np.float64)
    eef = state[:, :3]
    grip = state[:, 6:8].mean(axis=1)
    objects = np.stack([sim[:, lo:lo + 3] for _, lo in slots], axis=1)
    return eef, objects, grip, len(state)


def process_task(run: pathlib.Path, is_scene8: bool, references):
    """→ (events rows, slots)。SCENE8=full，其余=objaware（阈值同 full、去 goal 项）。"""
    summ = json.load(open(run / "client" / "summaries.json"))
    by_ep = {int(e["episode_index"]): e for e in summ}
    if is_scene8:
        check_scene8_layout(run)
        slots = [("moka_pot_1_joint0", 10), ("moka_pot_2_joint0", 17)]
        grade = "full"
    else:
        slots = load_object_slots(run)
        grade = "objaware"
    rows = []
    for e_id in sorted(by_ep):
        s = by_ep[e_id]
        eef, objects, grip, q = episode_arrays(run, e_id, slots)
        if is_scene8:
            lo, st, tr = physical_onsets_full(eef, objects, grip, references)
        else:
            lo, st, tr = physical_onsets_objaware(eef, objects, grip)
        rows.append([e_id, s.get("init_state_id", -1), s.get("repeat", -1),
                     int(bool(s["success"])), q, lo, st, tr, grade])
    return rows, slots


def write_events(out_dir: pathlib.Path, rows) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "scene", "repeat", "success", "n_queries",
                    "loop_onset_q", "static_onset_q", "trap_onset_q", "proxy_grade"])
        w.writerows(rows)


def summarize(rows) -> dict:
    arr = np.array([[r[3], r[5], r[6], r[7]] for r in rows if r[4] > 0], np.int32)
    if len(arr) == 0:
        return {"episodes": 0}
    succ = arr[:, 0] == 1
    out = {"episodes": int(len(arr)), "success": int(succ.sum()),
           "failure": int((~succ).sum())}
    for name, col in [("loop", 1), ("static", 2), ("trap", 3)]:
        ev = arr[:, col] >= 0
        out[name] = {"n": int(ev.sum()),
                     "n_fail": int((ev & ~succ).sum()),
                     "n_succ": int((ev & succ).sum())}
    return out


def admission_scene8(cname: str, run: pathlib.Path, references) -> dict:
    """准入回归：objaware（layout 槽位、无 goal）vs full（冻结槽位、含 goal）逐集对照。"""
    summ = json.load(open(run / "client" / "summaries.json"))
    check_scene8_layout(run)
    slots_layout = load_object_slots(run)          # 应恰为两只 moka pot
    mismatch = []
    trap_mine: dict[int, int] = {}
    for e in summ:
        e_id = int(e["episode_index"])
        eef, objects, grip, _ = episode_arrays(run, e_id, slots_layout)
        a = physical_onsets_full(eef, objects, grip, references)
        b = physical_onsets_objaware(eef, objects, grip)
        trap_mine[e_id] = b[2]
        if a != b:
            mismatch.append({"episode_id": e_id, "full": list(a), "objaware": list(b)})
    rec = {"corpus": cname, "episodes": len(summ),
           "layout_slots": [s[0] for s in slots_layout],
           "onset_triple_mismatch": len(mismatch),
           "mismatch_detail": mismatch[:10], "pass": len(mismatch) == 0}
    if cname == "main16x32":
        stored: dict[int, int] = {}
        with open(TRAP_TABLES / "aligned_event_values.csv") as f:
            for r in csv.DictReader(f):
                if r["corpus"] == "B" and r["relative"] == "0":
                    stored.setdefault(int(r["episode_id"]), int(r["query"]))
        common = sorted(set(stored) & set(trap_mine))
        exact = sum(stored[e] == trap_mine[e] for e in common)
        rec["vs_aligned_event_values"] = {"compared": len(common), "exact_match": exact}
        rec["pass"] = rec["pass"] and exact == len(common) == 161
    return rec


def main() -> None:
    refs = np.asarray(
        json.load(open(REF_JSON))["success_terminal_references"]["moka_pot_1_joint0"],
        np.float64,
    )
    # 保留 v1 验证记录
    pv_path = AUDIT / "proxy_validation.json"
    validation = json.loads(pv_path.read_text()) if pv_path.exists() else {}
    if "task_event_summaries" in validation:
        validation["task_event_summaries_v1_degraded"] = validation.pop("task_event_summaries")

    # ---- 准入回归（不通过则不覆写）----
    admissions = []
    for cname, cfg in CORPORA.items():
        run = cfg["root"] / "libero_long" / SCENE8 / cfg["run_id"]
        admissions.append(admission_scene8(cname, run, refs))
        print(f"[admission {cname}] mismatch={admissions[-1]['onset_triple_mismatch']} "
              f"pass={admissions[-1]['pass']} slots={admissions[-1]['layout_slots']}")
    validation["objaware_admission"] = admissions
    main_adm = next(a for a in admissions if a["corpus"] == "main16x32")
    if not main_adm["pass"]:
        pv_path.write_text(json.dumps(validation, indent=1))
        print("准入回归未通过，events.csv 未覆写。", file=sys.stderr)
        raise SystemExit(1)

    # ---- 重建（SCENE8 保持 full 不动：跳过写入）----
    summaries_v2: dict = {}
    slots_per_task: dict = {}
    for cname, cfg in CORPORA.items():
        for suite in SUITES:
            sdir = cfg["root"] / suite
            if not sdir.exists():
                continue
            for task in sorted(sdir.iterdir()):
                run = task / cfg["run_id"]
                if not (run / "client" / "summaries.json").exists():
                    continue
                is_s8 = task.name == SCENE8
                rows, slots = process_task(run, is_s8, refs if is_s8 else None)
                if not is_s8:
                    write_events(HERE / cname / suite / task.name, rows)
                slots_per_task[f"{cname}/{suite}/{task.name}"] = [s[0] for s in slots]
                s = summarize(rows)
                summaries_v2[f"{cname}/{suite}/{task.name}"] = s
                print(f"[{cname}/{suite}] {task.name}: eps={s['episodes']} "
                      f"loop={s['loop']['n']}({s['loop']['n_fail']}f) "
                      f"static={s['static']['n']}({s['static']['n_fail']}f) "
                      f"trap={s['trap']['n']}({s['trap']['n_fail']}f)"
                      + ("  [full, events.csv 未动]" if is_s8 else ""))
    validation["task_event_summaries_v2"] = summaries_v2
    validation["object_slots_v2"] = slots_per_task
    pv_path.write_text(json.dumps(validation, indent=1))
    print("validation ->", pv_path)


if __name__ == "__main__":
    main()
