#!/usr/bin/env python3
"""Select WATCH and ACT on development. External is never opened here.

WATCH is selected three unsupervised ways (a1 solo-best, a2 solo-best under an
equal budget share, a3 frame-diverse greedy marginal gain) and one supervised
way (b minimax physical-mode coverage, plus b4 at matched size), so the
question "does an unsupervised criterion approximate the supervised one" has an
answer rather than an assumption. Only (b)/(b4) read physical-mode labels.

ACT is selected independently of WATCH, because AND viability is extremely
head-dependent: whether a pair yields 0 true positives or 259 is not predictable
from either head's solo performance. Two ACT objectives are scored: A (maximise
precision) as first declared, and B (maximise early true positives at high
precision) from the addendum.

Objectives, constraints and tie-breaks are those written in results/PREREG.md.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import twotier_lib as lib


# ---- predeclared, see PREREG.md -----------------------------------------
MAX_TIMELY_FPR = 0.005
WATCH_SIZES = (2, 3, 4, 5)
WATCH_MATCHED_SIZE = 4
GREEDY_MAX_HEADS = 5
ACT_MIN_TP = 50
ACT_MIN_TP_RELAXED = 25
ACT_B_MIN_PRECISION = 0.95
CONTROLLER_TRIO = (
    "mobility|global",
    "flow_path|global",
    "expert_load_effective_rank|global",
)
SKIP = {"schema", "risk", "suite", "length", "physical_mode"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lib.RESULTS)
    return parser.parse_args()


def head_pool(alarms: dict[str, np.ndarray], mode: str, secondary: bool) -> list[str]:
    keys = [k for k in alarms if k.rsplit("|", 1)[1] == mode]
    if not secondary:
        keys = [k for k in keys if not k.startswith("v7_guard")]
    return sorted(keys)


def duplicate_groups(alarms: dict[str, np.ndarray], keys: Sequence[str]) -> list[list[str]]:
    seen: dict[bytes, list[str]] = {}
    for key in keys:
        seen.setdefault(np.asarray(alarms[key], np.int16).tobytes(), []).append(key)
    return [sorted(group) for group in seen.values() if len(group) > 1]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohort = lib.load_cohort("development_main")
    data = np.load(args.output / "development_first_alarms.npz", allow_pickle=True)
    if not np.array_equal(data["risk"].astype(bool), cohort["risk"]):
        raise ValueError("development alarm cache is not aligned with the labels")
    modes = data["physical_mode"].astype(str)
    counts = lib.mode_counts(modes, cohort["risk"])
    scored = lib.scored_modes(counts)
    fp_cap = MAX_TIMELY_FPR * int((~cohort["risk"]).sum())
    alarms = {k: np.asarray(data[k], np.int16) for k in data.files if k not in SKIP}

    def score(keys: Sequence[str], k: int = 1) -> dict[str, Any]:
        keys = list(keys)
        record = lib.evaluate(
            lib.combine([alarms[key] for key in keys], k), cohort, modes, scored, counts
        )
        record.update(
            {
                "heads": "+".join(keys),
                "n_heads": len(keys),
                "k": k,
                "frames": "+".join(sorted(lib.frames_of(keys))),
                "n_frames": len(lib.frames_of(keys)),
                "has_duplicate_pair": any(
                    np.array_equal(alarms[a], alarms[b]) for a, b in combinations(keys, 2)
                ),
            }
        )
        # The a1 drop rule refuses to go below two heads, so a1 can terminate
        # over budget. Flag it rather than silently comparing an infeasible rule
        # against feasible ones.
        record["fits_dev_fp_cap"] = bool(record["fp"] <= fp_cap)
        return record

    all_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    duplicates: dict[str, Any] = {}

    def emit(role: str, record: dict[str, Any] | None, pool: str, mode: str) -> None:
        if record is not None:
            all_rows.append({**record, "pool": pool, "mode": mode, "role": role})

    for secondary in (False, True):
        pool_name = "secondary_with_v7" if secondary else "primary"
        for mode in lib.MODES:
            if secondary and mode == "per_task":
                continue  # v7_guard is global-only; the pool would be identical
            keys = head_pool(alarms, mode, secondary)
            tag = f"{pool_name}|{mode}"
            duplicates[tag] = duplicate_groups(alarms, keys)

            singles = {key: score([key]) for key in keys}
            for key, row in singles.items():
                emit("single", row, pool_name, mode)

            # ---------------- WATCH (a1): solo-best per frame, then drop ---
            eligible = {k: r for k, r in singles.items() if r["fp"] <= fp_cap}
            per_frame_a1: dict[str, str] = {}
            for key in sorted(eligible):
                frame = lib.head_frame(key)
                best = per_frame_a1.get(frame)
                if best is None or (eligible[key]["tp"], eligible[key]["precision"]) > (
                    eligible[best]["tp"],
                    eligible[best]["precision"],
                ):
                    per_frame_a1[frame] = key
            chosen = [per_frame_a1[f] for f in sorted(per_frame_a1)]
            trail = [score(chosen)]
            while trail[-1]["fp"] > fp_cap and len(chosen) > 2:
                drop = min(chosen, key=lambda key: (singles[key]["tp"], key))
                chosen = [key for key in chosen if key != drop]
                trail.append(score(chosen))
            watch_a1, watch_a1_uncapped = trail[-1], trail[0]
            emit("watch_a1", watch_a1, pool_name, mode)
            if len(trail) > 1:
                emit("watch_a1_uncapped", watch_a1_uncapped, pool_name, mode)

            # ---------------- WATCH (a2): solo-best under a budget share ---
            share = fp_cap / 4.0
            per_frame_a2: dict[str, str] = {}
            for key in sorted(keys):
                if singles[key]["fp"] > share:
                    continue
                frame = lib.head_frame(key)
                best = per_frame_a2.get(frame)
                if best is None or (singles[key]["tp"], singles[key]["precision"]) > (
                    singles[best]["tp"],
                    singles[best]["precision"],
                ):
                    per_frame_a2[frame] = key
            watch_a2 = score([per_frame_a2[f] for f in sorted(per_frame_a2)])
            emit("watch_a2", watch_a2, pool_name, mode)

            # ---------------- WATCH (a3): greedy marginal TP gain ----------
            def greedy(one_per_frame: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                picked: list[str] = []
                used: set[str] = set()
                steps: list[dict[str, Any]] = []
                current = 0
                while len(picked) < GREEDY_MAX_HEADS:
                    best: tuple[tuple[int, int], str, dict[str, Any]] | None = None
                    for key in sorted(keys):
                        if key in picked:
                            continue
                        if one_per_frame and lib.head_frame(key) in used:
                            continue
                        row = score(picked + [key])
                        if row["fp"] > fp_cap:
                            continue
                        rank = (row["tp"] - current, -row["fp"])
                        if best is None or rank > best[0]:
                            best = (rank, key, row)
                    if best is None or best[0][0] <= 0:
                        break
                    picked.append(best[1])
                    used.add(lib.head_frame(best[1]))
                    current = best[2]["tp"]
                    steps.append(best[2])
                if not picked:
                    raise RuntimeError(f"{tag}: greedy found no admissible head")
                return steps[-1], steps

            watch_a3, a3_steps = greedy(True)
            watch_a3_free, _ = greedy(False)
            emit("watch_a3", watch_a3, pool_name, mode)
            emit("watch_a3_free", watch_a3_free, pool_name, mode)
            for step in a3_steps[:-1]:
                emit("watch_a3_prefix", step, pool_name, mode)

            # ---------------- WATCH (b): minimax mode coverage -------------
            cross: list[dict[str, Any]] = []
            same: list[dict[str, Any]] = []
            for size in WATCH_SIZES:
                for subset in combinations(keys, size):
                    row = score(list(subset))
                    if row["fp"] > fp_cap:
                        continue
                    (cross if row["n_frames"] >= 2 else same).append(row)
            if not cross:
                raise RuntimeError(f"{tag}: no cross-frame OR fits the FP cap")
            cross_frame = pd.DataFrame(cross).sort_values(
                ["worst_mode_coverage", "mean_mode_coverage", "n_heads", "fp"],
                ascending=[False, False, True, True],
                kind="stable",
            )
            watch_b = cross_frame.iloc[0].to_dict()
            emit("watch_b", watch_b, pool_name, mode)
            matched = cross_frame[cross_frame["n_heads"] == WATCH_MATCHED_SIZE]
            watch_b4 = matched.iloc[0].to_dict() if len(matched) else None
            emit("watch_b4", watch_b4, pool_name, mode)
            cross_frame.head(30).assign(pool=pool_name, mode=mode).to_csv(
                args.output / f"development_watch_shortlist_{pool_name}_{mode}.csv",
                index=False,
            )

            # ---------------- same-frame controls, matched sizes -----------
            same_frame = pd.DataFrame(same)
            controls: dict[str, Any] = {}
            for size in sorted(
                {
                    int(watch_a1["n_heads"]),
                    int(watch_a2["n_heads"]),
                    int(watch_a3["n_heads"]),
                    int(watch_b["n_heads"]),
                }
            ):
                block = same_frame[same_frame["n_heads"] == size] if len(same_frame) else same_frame
                if not len(block):
                    controls[f"size_{size}"] = None
                    continue
                pick = block.sort_values(
                    ["worst_mode_coverage", "mean_mode_coverage", "fp"],
                    ascending=[False, False, True],
                    kind="stable",
                ).iloc[0].to_dict()
                controls[f"size_{size}"] = pick
                emit(f"same_frame_control_size{size}", pick, pool_name, mode)
                # and the same-frame maximum-TP control at that size
                pick_tp = block.sort_values(
                    ["tp", "precision"], ascending=False, kind="stable"
                ).iloc[0].to_dict()
                controls[f"size_{size}_max_tp"] = pick_tp
                emit(f"same_frame_maxtp_size{size}", pick_tp, pool_name, mode)

            # ---------------- fixed references -----------------------------
            # The controller's trio is a global-mode rule; emit it only in the
            # global blocks so the `mode` column never mislabels it.
            reference: dict[str, Any] = {}
            if mode == "global" and all(key in alarms for key in CONTROLLER_TRIO):
                reference["controller_cross_frame_trio"] = score(list(CONTROLLER_TRIO))
                emit(
                    "reference_controller_trio",
                    reference["controller_cross_frame_trio"],
                    pool_name,
                    mode,
                )

            # ---------------- ACT: independent of WATCH --------------------
            act_rows: list[dict[str, Any]] = []
            for size, ks in ((2, (2,)), (3, (2, 3))):
                for subset in combinations(keys, size):
                    if len(lib.frames_of(subset)) < 2:
                        continue
                    for k in ks:
                        act_rows.append(score(list(subset), k))
            act_frame = pd.DataFrame(act_rows)
            act_frame.sort_values(
                ["precision", "tp"], ascending=False, kind="stable"
            ).head(40).assign(pool=pool_name, mode=mode, objective="A").to_csv(
                args.output / f"development_act_shortlist_{pool_name}_{mode}.csv",
                index=False,
            )

            def pick_act(
                minimum: int, filt: Callable[[pd.DataFrame], pd.DataFrame], order, asc
            ) -> dict[str, Any] | None:
                block = filt(act_frame[act_frame["tp"] >= minimum])
                if not len(block):
                    return None
                return block.sort_values(list(order), ascending=list(asc), kind="stable").iloc[0].to_dict()

            act_min, relaxed = ACT_MIN_TP, False
            act_a = pick_act(
                act_min,
                lambda f: f,
                ["precision", "tp", "n_heads", "median_alarm_chunk"],
                [False, False, True, True],
            )
            if act_a is None:
                act_min, relaxed = ACT_MIN_TP_RELAXED, True
                act_a = pick_act(
                    act_min,
                    lambda f: f,
                    ["precision", "tp", "n_heads", "median_alarm_chunk"],
                    [False, False, True, True],
                )
            act_b = pick_act(
                act_min,
                lambda f: f[f["precision"] >= ACT_B_MIN_PRECISION],
                ["low_prior_tp", "lift", "tp", "n_heads"],
                [False, False, False, True],
            )
            emit("act_a", act_a, pool_name, mode)
            emit("act_b", act_b, pool_name, mode)

            def overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
                a, b = set(left["heads"].split("+")), set(right["heads"].split("+"))
                return {
                    "shared": sorted(a & b),
                    "jaccard": len(a & b) / len(a | b),
                    "identical_set": a == b,
                    "worst_mode_gap": float(
                        left["worst_mode_coverage"] - right["worst_mode_coverage"]
                    ),
                    "mode_cv_gap": float(left["mode_cv"] - right["mode_cv"]),
                }

            selections[tag] = lib.plain(
                {
                    "pool": pool_name,
                    "mode": mode,
                    "pool_heads": keys,
                    "watch_a1": watch_a1,
                    "watch_a1_uncapped": watch_a1_uncapped,
                    "watch_a1_per_frame_choice": per_frame_a1,
                    "watch_a2": watch_a2,
                    "watch_a2_per_frame_choice": per_frame_a2,
                    "watch_a3": watch_a3,
                    "watch_a3_free": watch_a3_free,
                    "watch_a3_order": [step["heads"].split("+")[-1] for step in a3_steps],
                    "watch_b": watch_b,
                    "watch_b4": watch_b4,
                    "same_frame_controls": controls,
                    "reference": reference,
                    "act_a": act_a,
                    "act_b": act_b,
                    "act_min_tp_used": act_min,
                    "act_constraint_relaxed_post_hoc": relaxed,
                    "act_candidates": int(len(act_frame)),
                    "watch_cross_frame_candidates": int(len(cross)),
                    "watch_same_frame_candidates": int(len(same)),
                    "development_agreement": {
                        f"{name}_vs_b": overlap(rule, watch_b)
                        for name, rule in (
                            ("a1", watch_a1),
                            ("a2", watch_a2),
                            ("a3", watch_a3),
                        )
                    }
                    | (
                        {
                            f"{name}_vs_b4": overlap(rule, watch_b4)
                            for name, rule in (
                                ("a1", watch_a1),
                                ("a2", watch_a2),
                                ("a3", watch_a3),
                            )
                        }
                        if watch_b4 is not None
                        else {}
                    ),
                }
            )
            print(f"=== {tag}")
            for name, rule in (
                ("a1", watch_a1),
                ("a2", watch_a2),
                ("a3", watch_a3),
                ("b ", watch_b),
                ("b4", watch_b4),
                ("A ", act_a),
                ("B ", act_b),
            ):
                if rule is None:
                    print(f"  {name}: INFEASIBLE", flush=True)
                    continue
                print(
                    f"  {name} k={rule['k']} tp={rule['tp']:>3d} fp={rule['fp']:>3d} "
                    f"prec={rule['precision']:.3f} lift={rule['lift']:.3f} "
                    f"early={rule['low_prior_tp']}/{rule['low_prior_fp']} "
                    f"worst={rule['worst_mode_coverage']:.3f} cv={rule['mode_cv']:.3f} "
                    f"| {rule['heads']}",
                    flush=True,
                )

    pd.DataFrame(all_rows).to_csv(args.output / "development_rules.csv", index=False)
    (args.output / "development_selection.json").write_text(
        json.dumps(
            lib.plain(
                {
                    "schema": "himoe.two_tier.development_selection.v2",
                    "external_opened": False,
                    "constraints": {
                        "max_timely_fpr": MAX_TIMELY_FPR,
                        "development_fp_cap": fp_cap,
                        "watch_sizes": list(WATCH_SIZES),
                        "watch_matched_size": WATCH_MATCHED_SIZE,
                        "greedy_max_heads": GREEDY_MAX_HEADS,
                        "act_min_tp": ACT_MIN_TP,
                        "act_b_min_precision": ACT_B_MIN_PRECISION,
                    },
                    "development_mode_counts": counts,
                    "modes_scored": scored,
                    "mode_floor": lib.MODE_FLOOR,
                    "duplicates": duplicates,
                    "selections": selections,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nwrote development_selection.json and development_rules.csv", flush=True)


if __name__ == "__main__":
    main()
