#!/usr/bin/env python3
"""Audit motion phenotypes against frozen MoE alarms on all legacy episodes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import zarr

from diagnostics import (CONFIG, PHENOTYPES, event_window_start, first_hit,
                         motion_features, route_features, valid_alarms)


HERE = Path(__file__).resolve().parent
HUB = HERE.parent
ROOT = HUB.parent
V8_RESULTS = ROOT / "moe-v8-0906/results"
INDEX_PATH = ROOT / "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv"
HISTORY_PATH = HUB / "moe-history-only/results/episode_alarms.csv"
METHODS = ("v7", "v8", "v8.2", "history_k4")
LABELS = {"still": "Measured stillness", "backtracking": "Measured backtracking",
          "periodic_return": "Measured periodic return", "command_jitter": "Predicted-command jitter",
          "route_periodic_front": "Front-route periodic return",
          "route_periodic_back": "Back-route periodic return"}
ZH = {"still": "末端停滞", "backtracking": "末端方向反复", "periodic_return": "末端周期性回返",
      "command_jitter": "预测动作抖动代理", "route_periodic_front": "前层路由周期性回返",
      "route_periodic_back": "后层路由周期性回返"}
TASK_LABELS = {"open_the_middle_drawer_of_the_cabinet": "Middle drawer",
               "open_the_top_drawer_and_put_the_bowl_inside": "Drawer + bowl",
               "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "Scene8 moka pots",
               "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "Bowl on ramekin",
               "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "Bowl on stove"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_index() -> pd.DataFrame:
    frame = pd.read_csv(INDEX_PATH)
    if not np.array_equal(frame.row, np.arange(len(frame))):
        raise ValueError("legacy index order changed")
    if frame.duplicated(["task", "episode"]).any():
        raise ValueError("duplicate episode identity")
    frame["source_run"] = "cache/HiMoE-VLA/" + frame.task + "/right-16x32"
    frame["suite"] = frame.task.str.split("/").str[0]
    frame["task_name"] = frame.task.str.split("/").str[1]
    if len(frame) != 2560 or int(frame.failure.sum()) != 307:
        raise ValueError("unexpected legacy corpus; revise the explicit protocol first")
    keep = ["row", "source_run", "suite", "task", "task_name", "episode", "init_state_id",
            "flow_noise_seed", "length", "success", "failure"]
    return frame[keep].copy()


def join_history(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    columns = ["source_run", "episode", "init_state", "seed", "length", "success",
               "freeze_back_half_k4", "primary_failure_reason"]
    merged = frame.merge(history[columns], on=["source_run", "episode"], how="left",
                         validate="one_to_one", suffixes=("", "_history"), sort=False)
    for left, right in (("length", "length_history"), ("success", "success_history"),
                        ("init_state_id", "init_state"), ("flow_noise_seed", "seed")):
        if not np.array_equal(merged[left], merged[right]):
            raise ValueError(f"history identity mismatch: {left}")
    if not np.array_equal(merged.row, frame.row):
        raise ValueError("history merge reordered episodes")
    return merged


def existing_methods(frame: pd.DataFrame) -> tuple[dict, dict, dict]:
    sys.path.insert(0, str(ROOT / "moe-v8-0906/experiments"))
    import evaluate_full_corpus as v8
    import freeze_v82 as v82

    lengths = frame.length.to_numpy(int)
    with np.load(V8_RESULTS / "v8_full_corpus_alarms.npz", allow_pickle=False) as z:
        cached = {method: z[f"legacy_main16x32|{method}"] for method in ("v7", "v8")}
    with np.load(V8_RESULTS / "v82_alarms.npz", allow_pickle=False) as z:
        cached["v8.2"] = z["legacy_main16x32|v8.2"]
    history = join_history(frame, pd.read_csv(HISTORY_PATH))
    cached["history_k4"] = history.freeze_back_half_k4.to_numpy(int)
    alarms, audit = {}, {}
    for method, first in cached.items():
        alarms[method], invalid = valid_alarms(first, lengths)
        audit[method] = {"cached_alarms": int((first >= 0).sum()),
                         "outside_observed_episode": int(invalid.sum()),
                         "at_length_boundary": int((first == lengths).sum()),
                         "valid_alarms": int((alarms[method] >= 0).sum())}
    with np.load(V8_RESULTS / "legacy_main16x32_flow_speed.npz", allow_pickle=False) as z:
        speed = z["flow_speed"]
    config = json.loads((V8_RESULTS / "v82_summary.json").read_text())
    thresholds = config["thresholds"]
    cfg = config["config"]
    scores = {key: np.full(speed.shape[:2], np.nan, dtype=np.float32)
              for key in ("v8_frontback_score", "v8_curvature_score")}
    rebuilt = {key: np.full(len(frame), -1, dtype=int) for key in ("v8", "v8.2")}
    for i, length in enumerate(lengths):
        sample = speed[i:i + 1, :length]
        if not np.isfinite(sample).all():
            raise ValueError(f"missing flow speed inside episode {i}")
        if length <= v8.WIDTH:
            rebuilt["v8"][i] = rebuilt["v8.2"][i] = alarms["v7"][i]
            continue
        heads = v8.heads_from_flow_speed(sample)
        scores["v8_frontback_score"][i, v8.WIDTH:length] = heads["frontback_flowpath"][0, v8.WIDTH:]
        scores["v8_curvature_score"][i, v8.WIDTH:length] = heads["curvature_3step"][0, v8.WIDTH:]
        firsts = [v8.confirmed_first(series, thresholds[name], v8.DIRECTION[name], v8.CONFIRM)
                  for name, series in heads.items()]
        rebuilt["v8"][i] = int(v8.union(np.array([alarms["v7"][i]]), *firsts)[0])
        heads82 = v82.build_heads(sample, cfg["baseline"], cfg["width"])
        firsts82 = [v82.moving_first(series, thresholds[name], v8.DIRECTION[name], cfg["slope"],
                                    cfg["confirm"], config["earliest_chunk"])
                   for name, series in heads82.items()]
        rebuilt["v8.2"][i] = int(v8.union(np.array([alarms["v7"][i]]), *firsts82)[0])
    for method, values in rebuilt.items():
        mismatch = int((values != alarms[method]).sum())
        audit[method]["unpadded_rebuild_mismatches"] = mismatch
        if mismatch:
            raise ValueError(f"unpadded {method} disagrees with valid cached alarms: {mismatch}")
    frame["primary_failure_reason"] = history.primary_failure_reason.fillna("").to_numpy()
    return alarms, scores, {"alarm_cache_audit": audit, "v82_profile": config}


def extract(frame: pd.DataFrame, output: Path) -> tuple[dict, dict]:
    history_monitor = read_module("motion_history_monitor", HUB / "moe-history-only/monitor.py")
    speed_extractor = read_module("motion_flow_extractor", ROOT / "moe-v8-0906/experiments/extract_flow_speed.py")
    with np.load(V8_RESULTS / "legacy_main16x32_flow_speed.npz", allow_pickle=False) as z:
        speed_cache = z["flow_speed"]
    arrays = {}
    audits, checksums = [], []
    n, qmax = len(frame), int(frame.length.max())
    history = join_history(frame, pd.read_csv(HISTORY_PATH))
    for task, task_rows in frame.groupby("task", sort=False):
        run = HUB / task_rows.source_run.iloc[0]
        summaries_path = run / "client/summaries.json"
        summaries = sorted(json.loads(summaries_path.read_text()), key=lambda x: x["episode_index"])
        if len(summaries) != len(task_rows):
            raise ValueError(f"summary count mismatch: {task}")
        lengths = np.array([s["inference_calls"] for s in summaries], dtype=int)
        episode_ids = np.array([s["episode_index"] for s in summaries], dtype=int)
        group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        ids, steps = group["episode_id"][:], group["control_step"][:]
        if not np.array_equal(ids, np.repeat(episode_ids, lengths)) or not np.all(np.diff(steps) == 1):
            raise ValueError(f"server episode/time mismatch: {task}")
        if not np.array_equal(lengths, task_rows.length) or len(steps) != sum(lengths):
            raise ValueError(f"episode length mismatch: {task}")
        raw = np.asarray(group["hb_router_probs"][:, :, 9, 1:, :])
        audit = {"task": task, "episodes": len(task_rows), "queries": int(sum(lengths)),
                 "summary_sha256": digest(summaries_path), "raw_final_routes_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                 "server_order_verified": True, "history_k4_raw_mismatches": 0}
        first_row = int(task_rows.index[0])
        sample = np.asarray(group["hb_router_probs"][:lengths[0]])
        err = float(np.max(np.abs(speed_extractor.flow_speed(sample) - speed_cache[first_row, :lengths[0]])))
        audit["first_episode_flow_cache_max_error"] = err
        if err > 1e-6:
            raise ValueError(f"flow cache/raw mismatch: {task}, {err}")
        cursor = 0
        for (i, row), summary in zip(task_rows.iterrows(), summaries):
            for name, key in (("episode", "episode_index"), ("init_state_id", "init_state_id"),
                              ("flow_noise_seed", "flow_noise_seed"), ("success", "success")):
                if row[name] != summary[key]:
                    raise ValueError(f"identity mismatch: {task}/{row.episode}, {name}")
            length = int(row.length)
            action_steps = int(summary["action_steps"])
            if not 10 * (length - 1) < action_steps <= 10 * length:
                raise ValueError("unexpected execution length/replan protocol")
            frame.loc[i, "action_steps"] = action_steps
            frame.loc[i, "last_chunk_executed"] = action_steps - 10 * (length - 1)
            path = run / "client" / f"episode_{int(row.episode):02d}.npz"
            with np.load(path, allow_pickle=False) as z:
                state, actions = z["state"], z["actions"]
            if len(state) != length:
                raise ValueError(f"state length mismatch: {path}")
            checksums.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
            routes = raw[cursor:cursor + length]
            values = motion_features(state, actions) | route_features(routes)
            h = history_monitor.features_from_roots(history_monitor.root_action_routes(routes))
            rule = history_monitor.Rule("freeze_back_half_k4", "freeze", threshold=0.5, confirmations=4)
            score = history_monitor.score_stream(h[None], rule)[0]
            actual = first_hit(np.isfinite(score) & (score <= rule.threshold))
            expected = int(history.freeze_back_half_k4.iloc[i])
            if actual != expected:
                raise ValueError(f"raw history alarm mismatch: row {i}, {actual} != {expected}")
            values["history_k4_score"] = score
            for key, value in values.items():
                if key not in arrays:
                    arrays[key] = np.full((n, qmax) + value.shape[1:], np.nan, dtype=np.float32)
                arrays[key][i, :length] = value
            cursor += length
        audits.append(audit)
        print(f"extracted {task}: {len(task_rows)} episodes, {sum(lengths)} queries", flush=True)
    np.savez_compressed(output / "features.npz", **arrays)
    write_json(output / "input_checksums.json", checksums)
    audit = {"config": CONFIG.as_dict(), "sources": audits,
             "episodes": len(frame), "queries": int(frame.length.sum()),
             "partial_final_chunks": int((frame.last_chunk_executed < 10).sum()),
             "diagnostics_sha256": digest(HERE / "diagnostics.py"),
             "features_sha256": digest(output / "features.npz"),
             "input_checksums_sha256": digest(output / "input_checksums.json")}
    write_json(output / "extraction_audit.json", audit)
    return arrays, audit


def method_metrics(frame: pd.DataFrame, alarms: dict) -> pd.DataFrame:
    rows = []
    groups = [("all", frame)] + list(frame.groupby("task", sort=False))
    for group, subset in groups:
        ix = subset.index.to_numpy()
        risk = subset.failure.to_numpy(bool)
        lengths = subset.length.to_numpy(int)
        for method, values in alarms.items():
            first = values[ix]
            fired = first >= 0
            timely = fired & ((lengths - first) >= 4)
            tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
            rows.append(dict(group=group, method=method, failures=int(risk.sum()), successes=int((~risk).sum()),
                             tp_any=tp, fp_any=fp, tp_lead4=int((timely & risk).sum()),
                             historical_fp_lead4=int((timely & ~risk).sum()),
                             precision_any=tp / (tp + fp) if tp + fp else np.nan,
                             recall_any=tp / risk.sum() if risk.sum() else np.nan))
    return pd.DataFrame(rows)


def phenotype_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, subset in [("all", frame)] + list(frame.groupby("task", sort=False)):
        for event in PHENOTYPES:
            earliest = ((0 if event == "command_jitter" else CONFIG.recurrence_window
                         if "periodic" in event else CONFIG.movement_window) + CONFIG.confirmations - 1)
            for outcome, s in subset.groupby("success"):
                eligible = s.length.to_numpy() > earliest
                positive = s[f"first_{event}"].to_numpy() >= 0
                rows.append(dict(group=group, event=event, outcome="success" if outcome else "failure",
                                 episodes=len(s), window_eligible=int(eligible.sum()),
                                 positive=int(positive.sum()),
                                 fraction_all=float(positive.mean()),
                                 fraction_eligible=float(positive.sum() / eligible.sum()) if eligible.any() else np.nan))
    return pd.DataFrame(rows)


def event_alignment(frame: pd.DataFrame, alarms: dict) -> pd.DataFrame:
    rows = []
    for event in PHENOTYPES:
        for outcome in ("all", "failure", "success"):
            indices = frame.index[frame[f"first_{event}"] >= 0].to_numpy()
            if outcome != "all":
                indices = indices[frame.success.to_numpy()[indices] == (outcome == "success")]
            end = frame[f"first_{event}"].to_numpy(int)[indices]
            start = np.array([event_window_start(event, q) for q in end])
            for method, first in alarms.items():
                a = first[indices]
                fired = a >= 0
                lead = end[fired] - a[fired]
                rows.append(dict(event=event, outcome=outcome, method=method, event_episodes=len(indices),
                                 detected_ever=int(fired.sum()),
                                 strictly_before_confirmation=int((fired & (a < end)).sum()),
                                 strictly_before_evidence_window=int((fired & (a < start)).sum()),
                                 median_confirmation_minus_alarm=float(np.median(lead)) if len(lead) else np.nan))
    return pd.DataFrame(rows)


def matched_auc(frame: pd.DataFrame, score: np.ndarray) -> dict:
    score = np.asarray(score)
    numerator, pairs, groups = 0.0, 0, 0
    for _, subset in frame.groupby(["task", "init_state_id"], sort=False):
        x = score[subset.index]
        y = subset.failure.to_numpy(bool)
        ok = np.isfinite(x)
        x, y = x[ok], y[ok]
        nf, ns = int(y.sum()), int((~y).sum())
        if nf and ns:
            numerator += rankdata(x)[y].sum() - nf * (nf + 1) / 2
            pairs += nf * ns
            groups += 1
    return dict(auc=numerator / pairs if pairs else np.nan, pairs=pairs, groups=groups)


def fixed_query_metrics(frame: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    rows = []
    features = {"pose_speed": -1, "pose_reversal": 1, "pose_return_ratio": -1,
                "command_high_fraction": 1, "route_back_speed": -1,
                "route_back_return_ratio": -1, "v8_frontback_score": -1, "v8_curvature_score": 1}
    for q in (7, 13, 19, 31):
        live = frame[frame.length > q]
        for name, direction in features.items():
            score = direction * arrays[name][:, q]
            valid = live[np.isfinite(score[live.index])]
            result = matched_auc(valid, score)
            rows.append(dict(query=q, feature=name, failure_direction=direction,
                             observed_episodes=len(live), finite_episodes=len(valid),
                             failures=int(valid.failure.sum()), successes=int(valid.success.sum()), **result))
    return pd.DataFrame(rows)


def matched_correlation(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> dict:
    left, right, groups = [], [], 0
    for _, subset in frame.groupby(["task", "init_state_id"], sort=False):
        a, b = x[subset.index], y[subset.index]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        a, b = rankdata(a), rankdata(b)
        a, b = a - a.mean(), b - b.mean()
        if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
            continue
        left.extend(a)
        right.extend(b)
        groups += 1
    a, b = np.asarray(left), np.asarray(right)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(rank_correlation=float(a @ b / denominator) if denominator > 0 else np.nan,
                contributing_episodes=len(a), groups=groups)


def coupling_metrics(frame: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    rows = []
    pairs = (("pose_step", "route_back_step"), ("pose_step", "route_front_step"),
             ("command_high_fraction", "v8_curvature_score"))
    for q in (7, 13, 19, 31):
        live = frame[frame.length > q]
        for x, y in pairs:
            result = matched_correlation(live, arrays[x][:, q], arrays[y][:, q])
            rows.append(dict(query=q, signal_x=x, signal_y=y, **result))
    return pd.DataFrame(rows)


def strongest_candidates(frame: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    rows = []
    for event in PHENOTYPES[2:]:
        if event == "command_jitter":
            score = np.minimum.reduce([arrays["command_rms"] / CONFIG.command_rms_floor,
                                       arrays["command_high_fraction"] / CONFIG.command_high_fraction,
                                       arrays["command_reversal"] / CONFIG.reversal_fraction])
        else:
            prefix = "pose" if event == "periodic_return" else "route_" + event.rsplit("_", 1)[1]
            score = np.minimum(CONFIG.recurrence_ratio / np.maximum(arrays[f"{prefix}_return_ratio"], 1e-12),
                               CONFIG.recurrence_ratio / np.maximum(arrays[f"{prefix}_return_valley"], 1e-12))
            if prefix == "pose":
                score = np.minimum.reduce([score, arrays["pose_speed"] / CONFIG.movement_floor_m,
                                           arrays["pose_span"] / CONFIG.span_floor_m])
            else:
                score = np.minimum(score, arrays[f"{prefix}_relative_speed"] / CONFIG.relative_still)
                score = np.where(arrays[f"{prefix}_speed"] > 1e-9, score, np.nan)
        sustained = np.full_like(score, np.nan)
        sustained[:, 1:] = np.minimum(score[:, :-1], score[:, 1:])
        for i, row in frame.iterrows():
            values = sustained[i, :int(row.length)]
            finite = np.flatnonzero(np.isfinite(values))
            if not len(finite):
                continue
            q = int(finite[np.argmax(values[finite])])
            rows.append(dict(row=i, event=event, query=q, success=bool(row.success),
                             joint_threshold_fraction=float(values[q])))
    return pd.DataFrame(rows)


def select_examples(frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    selected, used = [], set()
    for event in PHENOTYPES:
        for success in (False, True):
            subset = frame[(frame[f"first_{event}"] >= 0) & (frame.success == success)]
            if len(subset):
                subset = subset.sort_values([f"first_{event}", "row"])
                row = subset.iloc[(len(subset) - 1) // 2]
                role, focus = "median_confirmation", int(row[f"first_{event}"])
            else:
                strongest = candidates[(candidates.event == event) & (candidates.success == success)]
                if not len(strongest):
                    continue
                candidate = strongest.sort_values(["joint_threshold_fraction", "row"], ascending=[False, True]).iloc[0]
                row = frame.iloc[int(candidate.row)]
                role, focus = "strongest_nondetection", int(candidate["query"])
            key = (int(row.row), event)
            if key in used:
                continue
            used.add(key)
            selected.append(dict(row=int(row.row), event=event, role=role, paired_to=-1, focus_query=focus))
            control = frame[(frame.task == row.task) & (frame.init_state_id == row.init_state_id)
                            & (frame.success != row.success)]
            if len(control):
                control = control.assign(seed_gap=np.abs(control.flow_noise_seed - row.flow_noise_seed))
                other = control.sort_values(["seed_gap", "row"]).iloc[0]
                selected.append(dict(row=int(other.row), event=event, role="matched_opposite_outcome",
                                     paired_to=int(row.row), focus_query=min(focus, int(other.length) - 1)))
    return pd.DataFrame(selected)


def plot_examples(frame: pd.DataFrame, arrays: dict, examples: pd.DataFrame, output: Path) -> None:
    (output / "examples").mkdir(exist_ok=True)
    colors = {"v7": "#737373", "v8": "#3478b0", "v8.2": "#b34145", "history_k4": "#38806c"}
    for index, example in examples.iterrows():
        row = frame.iloc[int(example.row)]
        i, length, event = int(row.row), int(row.length), example.event
        q = np.arange(length)
        with np.load(HUB / row.source_run / "client" / f"episode_{int(row.episode):02d}.npz", allow_pickle=False) as z:
            commands = z["actions"][:, :, :3]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        xyz = arrays["position"][i, :length]
        axes[0, 0].plot(q, 1000 * (xyz - xyz[0]), label=["x", "y", "z"])
        axes[0, 0].set(ylabel="Position relative to start (mm)", title="Measured end-effector position")
        axes[0, 0].legend(ncol=3, fontsize=8)
        axes[0, 1].plot(q, arrays["pose_speed"][i, :length] * 1000, color="#38806c", label="mean step")
        axes[0, 1].axhline(1, color="#737373", linestyle=":", linewidth=1)
        axes[0, 1].set(ylabel="mm / query", title="Six-transition movement")
        axes[1, 0].plot(q, arrays["route_front_speed"][i, :length], label="front", color="#b87d20")
        axes[1, 0].plot(q, arrays["route_back_speed"][i, :length], label="back", color="#3478b0")
        axes[1, 0].set(ylabel="RMS Hellinger distance", title="Final-flow route mobility")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(q, arrays["command_high_fraction"][i, :length], color="#b34145", label="high-frequency power")
        axes[1, 1].axhline(0.5, color="#737373", linestyle=":", linewidth=1)
        axes[1, 1].set(ylabel="Fraction", ylim=(-0.05, 1.05), title="Predicted-command spectrum")
        confirmation = int(row[f"first_{event}"])
        focus = int(example.focus_query)
        for prefix, label, color in (("pose", "pose", "#38806c"), ("route_front", "front route", "#b87d20"),
                                     ("route_back", "back route", "#3478b0")):
            lag = np.array([arrays[f"{prefix}_lag{k}"][i, focus] for k in range(1, 7)])
            if np.isfinite(lag).all() and lag[0] > 0:
                axes[2, 0].plot(np.arange(1, 7), lag / lag[0], "o-", label=label, color=color)
        axes[2, 0].axhline(0.5, color="#737373", linestyle=":", linewidth=1)
        axes[2, 0].set(xlabel="Lag (queries)", ylabel="D(lag) / D(1)", title=f"Return profile at q{focus}")
        if axes[2, 0].get_legend_handles_labels()[0]:
            axes[2, 0].legend(fontsize=8)
        else:
            axes[2, 0].text(0.5, 0.5, "Fewer than 13 observations at this query", ha="center", va="center",
                            transform=axes[2, 0].transAxes, fontsize=9, color="#737373")
        axes[2, 1].plot(np.arange(10), commands[focus], marker=".", label=["x", "y", "z"])
        axes[2, 1].set(xlabel="Predicted action index", ylabel="Command units", title=f"Ten predicted commands at q{focus}")
        if focus == length - 1 and row.last_chunk_executed < 10:
            axes[2, 1].axvspan(row.last_chunk_executed - 0.5, 9.5, color="#dddddd", alpha=0.7)
        for ax in axes[:2].flat:
            ax.set_xlabel("Query q (zero-based)")
            ax.set_xlim(-0.5, length - 0.5)
            if confirmation >= 0:
                ax.axvspan(event_window_start(event, confirmation), confirmation, color="#999999", alpha=0.12)
                ax.axvline(confirmation, color="#222222", linestyle="--", linewidth=1)
            for method in METHODS:
                alarm = int(row[f"alarm_{method}"])
                if alarm >= 0:
                    ax.axvline(alarm, color=colors[method], linewidth=0.9, alpha=0.85)
        for ax in axes.flat:
            ax.grid(alpha=0.15)
        method_text = ", ".join(f"{m}: q{int(row[f'alarm_{m}'])}" if row[f"alarm_{m}"] >= 0 else f"{m}: none" for m in METHODS)
        fig.suptitle(f"{TASK_LABELS[row.task_name]} | ep {int(row.episode)} | init {int(row.init_state_id)} | "
                     f"seed {int(row.flow_noise_seed)} | {'success' if row.success else 'failure'}\n"
                     f"{LABELS[event]} | {example.role}\n{method_text}", fontsize=11)
        filename = f"{index:02d}_{event}_row{i}.png"
        fig.savefig(output / "examples" / filename, dpi=140)
        plt.close(fig)
        examples.loc[index, "plot"] = f"examples/{filename}"
    examples.to_csv(output / "examples.csv", index=False)


def plot_overview(phenotypes: pd.DataFrame, metrics: pd.DataFrame, alignment: pd.DataFrame,
                  fixed: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    short = ["Still", "Backtrack", "Periodic", "Cmd jitter", "Route front", "Route back"]
    positions = np.arange(len(PHENOTYPES))
    for offset, outcome, color in ((-0.18, "failure", "#b34145"), (0.18, "success", "#38806c")):
        s = phenotypes[(phenotypes.group == "all") & (phenotypes.outcome == outcome)].set_index("event").loc[list(PHENOTYPES)]
        axes[0, 0].bar(positions + offset, 100 * s.fraction_all, width=0.36, label=outcome, color=color)
    axes[0, 0].set(xticks=positions, xticklabels=short, ylabel="Episodes (%)", title="Any observed phenotype; overlapping flags")
    axes[0, 0].tick_params(axis="x", labelrotation=20)
    axes[0, 0].legend()
    s = metrics[(metrics.group == "all") & metrics.method.isin(METHODS)].set_index("method").loc[list(METHODS)]
    axes[0, 1].scatter(s.fp_any, 100 * s.tp_lead4 / s.failures, c=["#737373", "#3478b0", "#b34145", "#38806c"], s=55)
    offsets = {"v7": (-25, -20), "v8": (-45, -3), "v8.2": (12, 12), "history_k4": (8, 5)}
    for method, r in s.iterrows():
        axes[0, 1].annotate(method, (r.fp_any, 100 * r.tp_lead4 / r.failures), xytext=offsets[method],
                            textcoords="offset points", fontsize=9,
                            arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6))
    axes[0, 1].margins(x=0.13, y=0.16)
    axes[0, 1].set(xlabel="All valid false alarms on successes (count)", ylabel="Failure recall at lead >=4 (%)",
                   title="Frozen methods; no time filtering of false alarms")
    subset = alignment[(alignment.method == "v8.2") & (alignment.outcome == "failure")].set_index("event").loc[list(PHENOTYPES)]
    denominator = subset.event_episodes.replace(0, np.nan)
    axes[1, 0].bar(positions - 0.18, 100 * subset.strictly_before_confirmation / denominator, 0.36, label="before confirmation", color="#3478b0")
    axes[1, 0].bar(positions + 0.18, 100 * subset.strictly_before_evidence_window / denominator, 0.36, label="before evidence window", color="#b87d20")
    tick_labels = [f"{label}\n(n={int(n)})" for label, n in zip(short, subset.event_episodes)]
    axes[1, 0].set(xticks=positions, xticklabels=tick_labels, ylabel="All phenotype-positive failures (%)", ylim=(0, 118),
                   xlim=(-0.6, len(PHENOTYPES) - 0.4),
                   title="v8.2 timing: recognition is not physical onset")
    axes[1, 0].tick_params(axis="x", labelrotation=20)
    axes[1, 0].legend(fontsize=9, loc="upper right")
    for x, n in enumerate(subset.event_episodes):
        if n == 0:
            axes[1, 0].text(x, 4, "n/a", ha="center", color="#737373", fontsize=9)
    for name, label, color in (("pose_speed", "Pose immobility", "#38806c"),
                               ("route_back_speed", "Route immobility", "#3478b0"),
                               ("command_high_fraction", "Command high-frequency power", "#b34145")):
        s = fixed[fixed.feature == name]
        axes[1, 1].plot(s["query"], s.auc, "o-", label=label, color=color)
    axes[1, 1].axhline(0.5, linestyle=":", color="#737373")
    axes[1, 1].set(xlabel="Fixed observed query", ylabel="Within-task/init AUC", ylim=(0, 1),
                   title="Failure discrimination at matched observation times")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("VLA hub motion / MoE audit: 2,560 episodes, 307 failures", fontsize=15)
    fig.savefig(output / "overview.png", dpi=160)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)


def write_report(frame: pd.DataFrame, phenotypes: pd.DataFrame, metrics: pd.DataFrame,
                 alignment: pd.DataFrame, audit: dict, examples: pd.DataFrame,
                 candidates: pd.DataFrame, coupling: pd.DataFrame, output: Path) -> None:
    lines = ["# 运动与 MoE 联合观察", "", "## 本轮范围", "",
             "已分析 `right-16x32` 的全部 5 个任务、2,560 条轨迹、51,308 次推理：2,253 成功、307 失败。",
             "这是历史数据上的描述性分析；新运动判据没有用这些失败标签调参，但这些数据已经被此前研究查看过。",
             "末端 xyz 每 10 个策略动作观察一次。动作频谱来自生成的 10 步指令，不能替代真实高频运动记录。",
             f"其中 {int(frame.last_chunk_executed.lt(10).sum())} 条轨迹最后一个 chunk 只执行部分动作；频谱仍明确表示完整预测指令。", "",
             "## 观察到的模式", "", "各模式可以重叠，不是物理失败原因标签。分母包括短轨迹，窗口可用率另见 `phenotype_metrics.csv`。", "",
             "| 固定描述性判据 | 失败轨迹 / 307 | 成功轨迹 / 2253 |", "|---|---:|---:|"]
    p = phenotypes[phenotypes.group == "all"].set_index(["event", "outcome"])
    for event in PHENOTYPES:
        lines.append(f"| {ZH[event]} | {int(p.loc[(event, 'failure'), 'positive'])} | {int(p.loc[(event, 'success'), 'positive'])} |")
    largest = p.xs("failure", level="outcome").loc[list(PHENOTYPES[:4]), "positive"].idxmax()
    no_hits = [ZH[event] for event in PHENOTYPES if int(p.loc[event, "positive"].sum()) == 0]
    lines += ["", f"在本轮四类运动判据中，失败轨迹最多出现的是{ZH[largest]}。",
              f"严格判据下未检出的模式：{'、'.join(no_hits) if no_hits else '无'}。未检出不能证明所有振动均不存在。",
              f"周期判据在成功轨迹中有 {int(p.loc[('periodic_return', 'success'), 'window_eligible'])}/2253 条具备足够窗口，"
              f"失败轨迹为 {int(p.loc[('periodic_return', 'failure'), 'window_eligible'])}/307；全程发生率不是等观察时长的因果比较。"]
    lines += ["", "周期性回返要求在 13 个观察点中实际返回先前状态，并且相对相邻 lag 出现明显距离低谷；",
              "不把距离曲线的弯曲、路由熵变大或方向改变一次直接算成周期。只检查 2–5 query 周期，未检出不表示所有频率均不存在。",
              "动作抖动代理有幅度、频谱和反向差分三个门槛；发生在最终成功的轨迹中也不能自动称为误动作。", "",
              "下面显示最接近阈值的程度：对所有必需条件取最弱的一项相对阈值比例，连续两 query 取较小值、全轨迹取最大值。",
              "只有比例达到 1 才满足完整判据。极值案例及同期相关性是在看到首轮计数后补充的描述性检查，未改变任何判据或阈值。", "",
              "| 模式 | 失败中的最高比例 | 成功中的最高比例 |", "|---|---:|---:|"]
    for event, group in candidates.groupby("event", sort=False):
        failure_max = group.loc[~group.success, "joint_threshold_fraction"].max()
        success_max = group.loc[group.success, "joint_threshold_fraction"].max()
        lines.append(f"| {ZH[event]} | {failure_max:.3f} | {success_max:.3f} |")
    lines += ["",
              "## 冻结方法的失败检测", "", "所有告警必须满足 `0 <= q < length`。提前 4 chunk 沿用旧协议 `length - q >= 4`，包含即将执行的 chunk。",
              "主表保留成功轨迹上的全部有效误报，避免用事后时长过滤掉误报。", "",
              "| 方法 | 任意时刻检出失败 | 提前至少4 chunk检出 | 全部有效成功误报 | 旧口径提前4 chunk误报 |",
              "|---|---:|---:|---:|---:|"]
    for r in metrics[(metrics.group == "all") & metrics.method.isin(METHODS)].itertuples():
        lines.append(f"| {r.method} | {r.tp_any}/307 | {r.tp_lead4}/307 | {r.fp_any}/2253 | {r.historical_fp_lead4} |")
    lines += ["", "**缓存问题已经在本分析中处理：**旧 v8/v8.2 在变长轨迹的 NaN 补齐区间上求和，可能产生假的前后层反转。",
              "本次逐条用未补齐的 flow-speed 重算新增检测头，确认与裁去越界值后的缓存一致；原缓存保留。", "",
              "| 方法 | 缓存中 q>=length 的告警 | 其中 q==length |", "|---|---:|---:|"]
    for method in METHODS:
        row = audit["alarm_cache_audit"][method]
        lines.append(f"| {method} | {row['outside_observed_episode']} | {row['at_length_boundary']} |")
    lines += ["", "## 是提前还是事后", "",
              "下面分母是具有该模式的全部失败轨迹，包括从未告警者。q 从 0 开始。",
              "“早于确认”仅表示比积累完整证据更早，不能写成早于物理异常发生；“早于观察窗”要求告警早于用于该事件的最早位置样本。", "",
              "| 模式 | 有该模式的失败 | v8.2早于确认 | v8.2早于整个观察窗 |", "|---|---:|---:|---:|"]
    for r in alignment[(alignment.method == "v8.2") & (alignment.outcome == "failure")].itertuples():
        lines.append(f"| {ZH[r.event]} | {r.event_episodes} | {r.strictly_before_confirmation} | {r.strictly_before_evidence_window} |")
    lines += ["", "## 怎么看图", "", "![总体观察](overview.png)", "",
              "固定 query 的曲线只比较同任务、同初态、仍有该次观察的成功/失败配对。AUC 按配对数加权，方向预先固定。",
              "样本、可配对初态和配对数见 `fixed_query_metrics.csv`。这些是多项描述性比较，不给独立显著性或跨任务泛化保证。", "",
              "同期关系另外按同任务、同初态、固定 query 比较：将两个信号各自在组内转成秩并去均值，再计算相关性。",
              "结果见 `coupling_metrics.csv`；这不说明 MoE 导致了物理运动。后层路由更新与已观察到的末端位移的结果如下：", "",
              "| query | 组内秩相关 | 有效轨迹 | 初态组数 |", "|---|---:|---:|---:|"]
    for r in coupling[coupling.signal_y == "route_back_step"].itertuples():
        lines.append(f"| {r.query} | {r.rank_correlation:.3f} | {r.contributing_episodes} | {r.groups} |")
    lines += ["",
              "案例按各模式首次确认时刻的中位数选取，分别保留成功和失败；能匹配时加入同任务、同初态、最接近噪声 seed 的反结局对照。",
              "没有通过判据的模式展示最接近阈值的案例，明确标为 `strongest_nondetection`；这些极值只用于检查现象，不作为检测效果证据。",
              "图中灰色区域是证据窗，黑色虚线是确认时刻，彩色竖线为各方法告警；最后一个动作图的灰色后缀表示未执行预测。", ""]
    for r in examples.itertuples():
        row = frame.iloc[int(r.row)]
        lines.append(f"- [{ZH[r.event]}，row {r.row}，{'成功' if row.success else '失败'}，{r.role}]({r.plot})")
    lines += ["", "## 对方法设计的含义", "",
              "1. 保留 v8/v8.2 作为一般路由异常基线；不能把其 flow 曲率或 turbulence 名称直接解释为机器人的物理振荡。",
              "2. 将末端停滞、非周期方向反复、明确周期回返、预测动作抖动分别记录。物理状态只进入本轮事后评价，冻结 MoE 方法不读取它。",
              "3. 按任务和固定 query 检查成功对照后，才能判断运动信号是否具有失败特异性。整体发生率受轨迹长短与任务构成影响。",
              "4. 下一轮需恢复模拟器状态、执行真实动作前缀，并对齐下个保存状态后，记录每步 xyz/姿态/关节速度/接触/目标进展来验证真实振动。",
              "5. 只有独立数据确认有效后，再设计同一触发状态下的原策略、平滑/滞回、v8触发重规划对照。本轮没有执行控制干预，不能声称抖动已减少或成功率已提高。", "",
              "## 复现", "", "```bash", "python VLA_MUI_HUB/moe-motion-diagnostics/analyze.py",
              "python -m pytest VLA_MUI_HUB/moe-motion-diagnostics -q", "```", "",
              "`episode_metrics.csv` 可回查每条轨迹及所有首次确认/告警；`features.npz` 保留逐 query 分数。",
              "`extraction_audit.json`、`input_checksums.json` 和 `summary.json` 记录原数据对齐、逐条物理文件摘要、缓存核验与配置。",
              "`--reuse-features` 仅复用这次已提取的特征，不代表重新校验所有原始 Zarr 字节。", ""]
    (output / "report.zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_index()
    alarms, scores, audit = existing_methods(frame)
    if args.reuse_features:
        extracted = json.loads((output / "extraction_audit.json").read_text())
        if (extracted["config"] != CONFIG.as_dict()
                or extracted["diagnostics_sha256"] != digest(HERE / "diagnostics.py")
                or extracted["features_sha256"] != digest(output / "features.npz")):
            raise ValueError("feature cache/config changed; rerun extraction")
        old = pd.read_csv(output / "episode_metrics.csv")
        for key in ("row", "source_run", "episode", "length", "success"):
            if not np.array_equal(old[key], frame[key]):
                raise ValueError(f"cached episode identity mismatch: {key}")
        frame["action_steps"] = old.action_steps
        frame["last_chunk_executed"] = old.last_chunk_executed
        with np.load(output / "features.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        arrays, extracted = extract(frame, output)
    arrays.update(scores)
    for method, first in alarms.items():
        frame[f"alarm_{method}"] = first
    for event in PHENOTYPES:
        frame[f"first_{event}"] = [first_hit(arrays[event][i, :int(length)] == 1)
                                    for i, length in enumerate(frame.length)]
    for event in PHENOTYPES:
        alarms[f"descriptor_{event}"] = frame[f"first_{event}"].to_numpy(int)
    frame.to_csv(output / "episode_metrics.csv", index=False)
    metrics = method_metrics(frame, alarms)
    phenotypes = phenotype_metrics(frame)
    alignment = event_alignment(frame, {k: alarms[k] for k in METHODS})
    fixed = fixed_query_metrics(frame, arrays)
    coupling = coupling_metrics(frame, arrays)
    candidates = strongest_candidates(frame, arrays)
    for name, table in (("method_metrics", metrics), ("phenotype_metrics", phenotypes),
                         ("event_alignment", alignment), ("fixed_query_metrics", fixed),
                         ("coupling_metrics", coupling), ("candidate_extremes", candidates)):
        table.to_csv(output / f"{name}.csv", index=False)
    examples = select_examples(frame, candidates)
    plot_examples(frame, arrays, examples, output)
    plot_overview(phenotypes, metrics, alignment, fixed, output)
    audit.update(config=CONFIG.as_dict(), extraction=extracted, reused_features=args.reuse_features,
                 index_sha256=digest(INDEX_PATH), source_code_sha256={p.name: digest(p) for p in (HERE / "diagnostics.py", HERE / "analyze.py", HERE / "PROTOCOL.md")},
                 limitations=["retrospective five-task cohort", "no dense measured vibration", "phenotypes are not failure labels",
                              "no new holdout", "no controller intervention", "no new v7 calibration"])
    write_json(output / "summary.json", audit)
    write_report(frame, phenotypes, metrics, alignment, audit, examples, candidates, coupling, output)
    print(metrics[metrics.group == "all"].to_string(index=False), flush=True)
    print(phenotypes[phenotypes.group == "all"].to_string(index=False), flush=True)
    print(f"report: {output / 'report.zh.md'}", flush=True)


if __name__ == "__main__":
    main()
