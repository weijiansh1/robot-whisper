#!/usr/bin/env python3
"""Reproduce every anchor before anything new is reported.

Six checks, all of which must pass before a single new number is quoted:

  1. the cap-free ``v7_guard`` anchors (TP / FP / total alarms / median lead);
  2. the four replicated within-task fixed-chunk survivors-only AUC cells;
  3. a channel that is constant inside every stratum scores *exactly* 0.5;
  4. the rate-matched null on white noise and on episode-constant channels;
  5. within-episode information for every channel, and which are bit-constant;
  6. the tie group: `>=` fires it, `>` drops it, and by how much.

Plus the physical-mode census the whole bundle is conditioned on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import modes_common as M
import common as EW
import capfree_protocol as CF

OUT = M.RESULTS

# Published cells, from the brief.
# (suite, chunk, quantity, layer, dev, ext, negatives_restricted)
#
# The fourth cell is quoted as `token_entropy|L3`, which is ambiguous about the
# denoising step and about the negative arm.  Both conventions are reported.
# It reproduces to three decimals as the *step-9* slice with the negative arm
# restricted to episodes running >= 5 more chunks - the same restriction the
# brief states for its L2 A/B cancellation figure.  As the step-0 slice with
# unrestricted negatives it is 0.476 / 0.501, i.e. not the published cell.
REFERENCE_CELLS = (
    ("libero_object", 9, "mobility_d0", "L15", 0.890, 0.908, 0),
    ("libero_spatial", 7, "flow_path", "L12", 0.783, 0.775, 0),
    ("libero_goal", 6, "hb_entropy_action", "L15", 0.248, 0.253, 0),
    ("libero_long", 14, "token_entropy_d9", "L3", 0.348, 0.352, 5),
)
ANCHORS = {
    "external_8b": {"tp_lead4": 347, "fp_lead4": 57, "median_lead_at_headline": 17.0,
                    "tp": 439, "fp": 80, "n_risk": 564},
    "development_main": {"tp_lead4": 303, "fp_lead4": 38, "median_lead_at_headline": 16.0,
                         "tp": 382, "fp": 67, "n_risk": 487},
}


def check_anchors(frames: dict) -> pd.DataFrame:
    rows = []
    for cohort, frame in frames.items():
        det = CF.load_detectors(cohort, len(frame["risk"]))
        key = next(k for k in det if "v7_guard|global" in k)
        got = M.score_modes(det[key], frame)
        want = ANCHORS[cohort]
        for field, target in want.items():
            if field == "n_risk":
                value = int(frame["risk"].sum())
            else:
                value = got[field]
            ok = abs(float(value) - float(target)) < 1e-9
            rows.append({"check": "v7_guard_capfree", "cohort": cohort,
                         "field": field, "got": value, "want": target, "ok": ok})
            assert ok, (cohort, field, value, target)
    return pd.DataFrame(rows)


def check_reference_cells(frames: dict) -> pd.DataFrame:
    rows = []
    for suite, chunk, quantity, layer, want_d, want_e, slack in REFERENCE_CELLS:
        got = {}
        for cohort, frame in frames.items():
            x = M.channel(frame, quantity, layer)
            alive = (frame["suite"] == suite) & (frame["length"] > chunk)
            take = alive & (frame["risk"] | (frame["length"] > chunk + slack))
            col = x[take, chunk][:, None]
            cell = EW.stratified_cell(col, frame["risk"][take], frame["task"][take])
            got[cohort] = float(cell["auc"][0])
        rows.append({"check": "within_task_fixed_chunk_auc", "suite": suite,
                     "chunk": chunk, "quantity": quantity, "layer": layer,
                     "negatives_need_extra_chunks": slack,
                     "auc_dev": got["development_main"], "want_dev": want_d,
                     "auc_ext": got["external_8b"], "want_ext": want_e,
                     "ok": abs(got["development_main"] - want_d) < 1e-3
                           and abs(got["external_8b"] - want_e) < 1e-3})
    frame = pd.DataFrame(rows)
    assert frame["ok"].all(), frame[~frame["ok"]]
    return frame


def check_constant_is_half(frames: dict) -> pd.DataFrame:
    """A channel constant inside every stratum must score exactly 0.5."""
    rows = []
    for cohort, frame in frames.items():
        for suite, chunk in (("libero_object", 9), ("libero_long", 14)):
            take = (frame["suite"] == suite) & (frame["length"] > chunk)
            n = int(take.sum())
            const = np.zeros((n, 1))
            elapsed = np.full((n, 1), float(chunk))  # ctrl_const_elapsed
            for name, col in (("zeros", const), ("ctrl_const_elapsed", elapsed)):
                cell = EW.stratified_cell(col, frame["risk"][take],
                                          frame["task"][take])
                auc = float(cell["auc"][0])
                rows.append({"check": "constant_is_exactly_half", "cohort": cohort,
                             "suite": suite, "chunk": chunk, "channel": name,
                             "auc": auc, "se": float(cell["se"][0]),
                             "ok": auc == 0.5})
                assert auc == 0.5, (cohort, suite, chunk, name, auc)
    return pd.DataFrame(rows)


def check_null_arm(frames: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Rate-matched null on white noise and on episode-constant channels.

    Two forms are reported.  (a) The published form: the *excess* true positives
    a null detector wins over the fixed-chunk baseline admissible at the same
    false alarm count.  Under the cap-free protocol this is 0 at every budget -
    a null cannot beat "still running at chunk q0", because it *is* a noisy
    version of it.  (b) The raw true-positive count, which is not zero: a null
    that fires 500 times on a cohort with a 3.6% risk rate hits some risks by
    chance.  Reporting only (a) as "0/564" would be a category error, so both
    are stated.
    """
    rows = []
    for cohort, frame in frames.items():
        risk, length = frame["risk"], frame["length"]
        base = CF.fixed_chunk_baseline(risk, length)
        valid = frame["valid"]
        for name in ("ctrl_white_noise", "ctrl_episode_const_rand",
                     "ctrl_flow_noise_seed"):
            x = M.channel(frame, name, "L2")
            ref = M.chunk_reference(x, valid)
            u = M.standardise(x, valid, ref, +1)
            u = np.where(np.isfinite(u), u, -np.inf)
            rm = M.running_max(M.stat_threshold(u, valid))
            grid = np.quantile(rm[np.isfinite(rm)], np.linspace(0.50, 0.999, 40))
            for h in np.unique(grid):
                first = M.first_alarm(rm, float(h))
                got = M.score_modes(first, frame)
                nul = M.score_modes(
                    CF.rate_matched_null(first, rng, length), frame)
                for arm, r in (("control_channel", got), ("rate_matched_null", nul)):
                    at = CF.baseline_frontier(base, M.HEADLINE_LEAD)(r["fp_lead4"])
                    rows.append({"check": "null_arm", "cohort": cohort,
                                 "channel": name, "arm": arm,
                                 "threshold": float(h),
                                 "alarms": r["alarms"],
                                 "tp_lead4": r["tp_lead4"],
                                 "fp_lead4": r["fp_lead4"],
                                 "baseline_tp_at_same_fp": int(at),
                                 "excess_tp_lead4": int(r["tp_lead4"] - at)})
    return pd.DataFrame(rows)


def within_episode_information(frames: dict) -> pd.DataFrame:
    rows = []
    for cohort, frame in frames.items():
        valid = frame["valid"]
        long_enough = frame["length"] >= 4
        for quantity, layer in M.channel_names(frame, include_controls=True):
            x = M.channel(frame, quantity, layer)
            xv = np.where(valid, x, np.nan)
            with np.errstate(invalid="ignore"):
                lo = np.nanmin(xv, axis=1)
                hi = np.nanmax(xv, axis=1)
            const = ~(hi > lo)
            frac_const = float(const[long_enough].mean())
            distinct = np.array([
                len(np.unique(xv[i][valid[i]])) for i in
                np.flatnonzero(long_enough)[:2000]])
            rows.append({"cohort": cohort, "quantity": quantity, "layer": layer,
                         "frac_within_episode_constant": frac_const,
                         "median_distinct_values": float(np.median(distinct)),
                         "bit_constant": bool(frac_const >= 0.999)})
    return pd.DataFrame(rows)


def check_tie_group(frames: dict) -> pd.DataFrame:
    """`>=` fires the tie group; `>` drops it.  Quantify the loss per channel."""
    rows = []
    for cohort, frame in frames.items():
        valid = frame["valid"]
        for quantity in ("set_dwell", "query_top1_churn", "query_hard_churn"):
            x = M.channel(frame, quantity, "L5")
            ref = M.chunk_reference(x, valid)
            u = M.standardise(x, valid, ref, +1)
            u = np.where(np.isfinite(u), u, -np.inf)
            rm = M.running_max(M.stat_threshold(u, valid))
            fin = rm[np.isfinite(rm)]
            h = float(np.quantile(fin, 0.98))
            # snap to an achieved value so a tie group exists at the threshold
            vals = np.unique(fin)
            h = float(vals[np.searchsorted(vals, h)])
            ge = M.first_alarm(rm, h)
            gt_mask = rm > h
            gt = np.where(gt_mask.any(axis=1), np.argmax(gt_mask, axis=1), -1)
            n_ge, n_gt = int((ge >= 0).sum()), int((gt >= 0).sum())
            rows.append({"check": "tie_group", "cohort": cohort,
                         "quantity": quantity, "threshold": h,
                         "alarms_ge": n_ge, "alarms_gt": n_gt,
                         "frac_dropped_by_strict_gt":
                             float(1 - n_gt / n_ge) if n_ge else np.nan})
            assert n_ge >= n_gt
    return pd.DataFrame(rows)


def mode_census(frames: dict) -> pd.DataFrame:
    rows = []
    for cohort, frame in frames.items():
        risk, mode, suite, task = (frame["risk"], frame["mode"],
                                   frame["suite"], frame["task"])
        for m in M.MODES + ("__all_risk__",):
            take = risk if m == "__all_risk__" else (risk & (mode == m))
            by_suite = pd.Series(suite[take]).value_counts().to_dict()
            top = pd.Series(task[take]).value_counts()
            rows.append({
                "cohort": cohort, "mode": m, "n": int(take.sum()),
                "n_risk_total": int(risk.sum()),
                "goal": by_suite.get("libero_goal", 0),
                "long": by_suite.get("libero_long", 0),
                "object": by_suite.get("libero_object", 0),
                "spatial": by_suite.get("libero_spatial", 0),
                "top_task": top.index[0] if len(top) else "",
                "top_task_n": int(top.iloc[0]) if len(top) else 0,
                "top_task_share": float(top.iloc[0] / take.sum()) if take.sum() else np.nan,
                "n_tasks_present": int(len(top)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(M.SEED)
    frames = {c: M.load(c) for c in ("development_main", "external_8b")}

    anchors = check_anchors(frames)
    M.write_csv(OUT / "check_anchors.csv", anchors)
    print("== 1. v7_guard 无 cap 锚点 ==")
    print(anchors.to_string(index=False))

    cells = check_reference_cells(frames)
    M.write_csv(OUT / "check_reference_cells.csv", cells)
    print("\n== 2. 同任务同 chunk 仅存活者 AUC 参考格 ==")
    print(cells.to_string(index=False))

    const = check_constant_is_half(frames)
    M.write_csv(OUT / "check_constant_half.csv", const)
    print("\n== 3. 层内常数通道必须恰为 0.5 ==")
    print(const.to_string(index=False))

    ties = check_tie_group(frames)
    M.write_csv(OUT / "check_tie_group.csv", ties)
    print("\n== 6. 并列组：>= 触发，> 丢弃 ==")
    print(ties.to_string(index=False))

    info = within_episode_information(frames)
    M.write_csv(OUT / "within_episode_information.csv", info)
    print("\n== 5. 单集内信息量（constant 比例最高的 12 个通道）==")
    print(info.sort_values("frac_within_episode_constant", ascending=False)
          .head(12).to_string(index=False))
    bit_const = info[info.bit_constant]
    print(f"   bit-constant 通道数 = {len(bit_const)}"
          + (": " + ", ".join(sorted(set(bit_const.quantity))) if len(bit_const) else ""))

    nul = check_null_arm(frames, rng)
    M.write_csv(OUT / "check_null_arm.csv", nul)
    print("\n== 4. 同发射率零对照 ==")
    summ = (nul.groupby(["cohort", "arm"])
            .agg(max_excess_tp=("excess_tp_lead4", "max"),
                 max_tp=("tp_lead4", "max"),
                 n_points=("tp_lead4", "size")).reset_index())
    print(summ.to_string(index=False))

    census = mode_census(frames)
    M.write_csv(OUT / "mode_census.csv", census)
    print("\n== 物理失败模式普查 ==")
    print(census.to_string(index=False))

    M.write_json(OUT / "harness_checks.json", {
        "anchors_ok": bool(anchors["ok"].all()),
        "reference_cells_ok": bool(cells["ok"].all()),
        "constant_is_half_ok": bool(const["ok"].all()),
        "n_bit_constant_channels": int(info.bit_constant.sum()),
        "max_null_excess_tp_lead4": int(
            nul[nul.arm == "rate_matched_null"].excess_tp_lead4.max()),
        "max_null_raw_tp_lead4": int(
            nul[nul.arm == "rate_matched_null"].tp_lead4.max()),
    })


if __name__ == "__main__":
    main()
