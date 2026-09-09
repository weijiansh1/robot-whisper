"""Write results/manifest.json: provenance, frozen parameters, headline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capfree_common as C  # noqa: E402

DESCRIPTIONS = {
    "channel_screen_development.csv":
        "单 chunk 阈值（floor）在 development 上的全通道扫描：episode 内信息量、"
        "最优符号/阈值、FP∈[80,320] 内超出基线的 TP",
    "cusum_channel_screen_development.csv":
        "单通道 CUSUM 在 development 上的全扫描（2 归一化臂 × 2 符号 × 6 个 k）",
    "development_frontier_window.csv":
        "development 上 window 目标下三个检测器族的完整前沿",
    "development_frontier_v7budget.csv":
        "development 上 v7budget 目标下三个检测器族的完整前沿",
    "frozen_config.json":
        "冻结配置：通道、权重、逐 chunk 归一化常数、k、h、K、M、q_min，"
        "以及任务不相交二折稳定性",
    "frozen_pool.npz": "候选池通道名与相关性去重后保留的下标",
    "frozen_operating_points.csv":
        "所有臂在两个 cohort 上的冻结工作点（含 5 个提前量档、两种基线参照）",
    "frontier_all.csv":
        "描述性前沿：冻结通道/权重/k/M 不变，只扫描判决门限；含基线与零对照",
    "null_sweep.csv": "白噪声与 episode 常数零对照，走完全相同的流程",
    "suite_breakdown.csv":
        "诊断表：各臂在四个 suite 上的命中分解（suite 从不进入检测决策）",
    "tie_audit.csv": "并列组审计：把 >= 换成 > 会丢失/推迟多少个报警 episode",
    "headline.json": "头条数字：锚点、各臂、零对照、rate-matched 零对照",
    "external_first_alarms.npz": "external 上 cusum|window 的首次报警向量",
    "frontier_external.png": "图 1：external TP–FP 前沿 + 提前量代价",
    "frontier_external.svg": "图 1（矢量）",
    "screen_meta.json": "floor 扫描元信息",
    "screen_cusum_meta.json": "CUSUM 通道扫描元信息",
    "fit_log.txt": "拟合日志（贪心每一步）",
    "screen_log.txt": "floor 扫描 stdout",
    "screen_cusum_log.txt": "CUSUM 扫描 stdout",
    "fit_log_full.txt": "拟合 stdout",
    "score_external_log.txt": "external 评分 stdout",
    "_bank_names.npy": "通道 bank 的名称顺序",
}

INPUTS = [
    "moe-flow-semantics-0906/results/step_profiles/{cohort}_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/{cohort}_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/{cohort}_state_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/{cohort}_index.npz",
    "moe-unused-channels-0906/results/channels/{cohort}_quantities.npy",
    "moe-unused-channels-0906/results/channels/{cohort}_index.npz",
    "moe-hb-front-back-0905/results/layer_graphs/{cohort}.npz",
]


def sha(p: Path, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def main() -> None:
    cfg = json.loads((C.RESULTS / "frozen_config.json").read_text())
    head = json.loads((C.RESULTS / "headline.json").read_text())
    files = {}
    for p in sorted(C.RESULTS.iterdir()):
        if p.is_dir() or p.name == "manifest.json":
            continue
        files[p.name] = {
            "bytes": p.stat().st_size, "sha256_16": sha(p),
            "description": DESCRIPTIONS.get(p.name, ""),
        }
    inputs = {}
    for tpl in INPUTS:
        for cohort in ("development_main", "external_8b"):
            rel = tpl.format(cohort=cohort)
            q = C.ROOT / rel
            if q.exists():
                inputs[rel] = {"bytes": q.stat().st_size, "sha256_16": sha(q)}

    manifest = {
        "bundle": "moe-cusum-capfree-0906",
        "date": str(date.today()),
        "question": ("在冻结的无 cap 协议下，用 CUSUM 做时间积分，"
                     "能否超过 v7_guard 的 external 347/57（提前量>=4）"),
        "protocol": {
            "detector_inputs": "chunk index + routing only",
            "metric": "TP / FP / 绝对提前量 = length - alarm_chunk",
            "headline_lead": C.HEADLINE_LEAD, "leads": list(C.LEADS),
            "baseline": "still_running_q0, q0 in [2, 40)",
            "reference_statistic":
                "同等误报数下超出最优固定 chunk 基线的 TP（步进包络），"
                "另报随机化凸包参照",
            "target_fp_window": [C.FP_LO, C.FP_HI],
            "harness": "moe-capfree-0906/experiments/capfree_protocol.py "
                       "（score / fixed_chunk_baseline / baseline_frontier / "
                       "rate_matched_null / cohort_frame 全部 import，未重写）",
            "tie_rule": ">=",
        },
        "anchors_reproduced": {
            "v7_guard_external": {"tp": 439, "fp": 80, "tp_lead4": 347,
                                  "fp_lead4": 57, "tp_lead8": 261,
                                  "fp_lead8": 43, "median_lead": 13},
            "v7_guard_development": {"tp": 382, "fp": 67, "tp_lead4": 303,
                                     "fp_lead4": 38, "tp_lead8": 217,
                                     "fp_lead8": 26, "median_lead": 10},
            "capfree_baseline_lead4_fp80": {"external": ["still_running_q37", 274],
                                            "development": ["still_running_q37", 222]},
            "checked_by": "tests/test_protocol.py::test_v7_guard_anchor, "
                          "::test_capfree_baseline_anchor",
        },
        "fit": {
            "cohort": "development_main",
            "development_iterations": 6,
            "external_seen_before_fitting":
                "只看过两件事：(1) 任务书要求先复现的 v7_guard / 基线锚点数字；"
                "(2) 缓存校验时打印过 token_entropy 与 token_differentiation 在 "
                "L2/L3/L4、chunk 32 上的 external 内任务 AUC。二者都没有进入任何"
                "选择；通道、权重、k、h、K、M、q_min 全部只在 development 上拟合。",
            "channels_screened": 540,
            "banned_channels": sorted(C.BANNED),
            "normalisation_arms": ["raw", "self"],
            "selections": {m: {f: {"channels": v["channels"],
                                   "weights": v["weights"],
                                   "weight_scheme": v["weight_scheme"],
                                   "operating_point": v["operating_point"]}
                               for f, v in cfg["selections"][m]["families"].items()}
                           for m in ("window", "v7budget")},
            "stability_task_disjoint_halves":
                cfg["selections"]["_stability_task_halves"],
        },
        "headline": head,
        "controls": {
            "rate_matched_null": head["rate_matched_null"],
            "white_noise_and_episode_constant_null": head["null"],
            "length_is_negative_control": True,
            "tie_audit": "results/tie_audit.csv",
        },
        "inputs": inputs,
        "files": files,
        "reproduce": [
            "python experiments/screen_channels.py    # 单 chunk floor，development",
            "python experiments/screen_cusum.py       # 单通道 CUSUM，development",
            "python experiments/fit_composite.py      # 冻结通道/权重/k/h/K/M",
            "python experiments/score_external.py     # external 只评一次",
            "python experiments/make_figure.py",
            "python -m pytest tests/test_protocol.py -q",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "device": "cpu",
        },
    }
    (C.RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print("wrote", C.RESULTS / "manifest.json", len(files), "files")


if __name__ == "__main__":
    main()
