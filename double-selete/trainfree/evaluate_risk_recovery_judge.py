#!/usr/bin/env python3
"""Evaluate a train-free online risk alarm and deadline recovery allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import online_multihead_alarm as v1
import online_precision_cascade_alarm as precision
from risk_recovery_monitor import (
    COMPONENT_NAMES,
    RecoveryProfile,
    allocation_state,
    quantile_higher,
    terminal_components,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
HUB = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
PROTOCOL = HERE / "RISK_RECOVERY_JUDGE_PROTOCOL.md"
DEFAULT_OUTPUT = HERE / "results/online_risk_recovery_judge"
MAIN_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
EXTRA_REFERENCE_CACHE = (
    HERE / "results/online_multihead_hub_external/unlabeled_query_features.npz"
)
EXTERNAL_CACHE = (
    HERE / "results/online_precision_cascade_external/unlabeled_query_features.npz"
)
FIRST_ALARMS = HERE / "results/online_precision_cascade_shared/first_alarms.npz"
EXTENSION = HERE / "results/timeout_extension_plus10"
MAIN_RUN = "right-50x8-20260903"
EXTERNAL_RUN = "right-50x8b-20260903"
HORIZONS = {
    "libero_spatial": 22,
    "libero_object": 28,
    "libero_goal": 30,
    "libero_long": 52,
}
ALLOCATION_QUANTILES = (0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-components", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def source_path(task: str, run_id: str, episode: int) -> Path:
    return HUB / task / run_id / "client" / f"episode_{episode:02d}.npz"


def validate_component_identity(
    component_cache: dict[str, np.ndarray], route_cache: dict[str, np.ndarray]
) -> None:
    if str(component_cache["schema"]) != "himoe.risk_recovery.components.v1":
        raise ValueError("unexpected recovery-component schema")
    if tuple(component_cache["component_names"].astype(str)) != COMPONENT_NAMES:
        raise ValueError("recovery-component order mismatch")
    for name in (
        "task_names",
        "task_index",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "length",
    ):
        if not np.array_equal(component_cache[name], route_cache[name]):
            raise ValueError(f"component/route identity mismatch: {name}")


def build_component_cache(
    route_cache: dict[str, np.ndarray],
    run_id: str,
    path: Path,
    reuse: bool,
) -> dict[str, np.ndarray]:
    if reuse and path.is_file():
        cached = load_npz(path)
        validate_component_identity(cached, route_cache)
        if str(cached["run_id"]) != run_id:
            raise ValueError("component-cache run ID mismatch")
        return cached

    feature_names = route_cache["feature_names"].astype(str).tolist()
    feature_index = {name: feature_names.index(name) for name in v1.FEATURES}
    task_names = route_cache["task_names"].astype(str)
    task_index = route_cache["task_index"].astype(int)
    episodes = route_cache["episode"].astype(int)
    lengths = route_cache["length"].astype(int)
    components = np.full(
        (len(episodes), len(COMPONENT_NAMES)), np.nan, dtype=np.float32
    )
    logical_digest = hashlib.sha256()
    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"expected 400 trajectories for {task}")
        for row in take:
            episode = int(episodes[row])
            query = int(lengths[row]) - 1
            client_path = source_path(task, run_id, episode)
            with np.load(client_path, allow_pickle=False) as archive:
                state = np.asarray(archive["state"], dtype=np.float32)
                actions = np.asarray(archive["actions"], dtype=np.float32)
            if len(state) != int(lengths[row]) or len(actions) != len(state):
                raise ValueError(f"client/route length mismatch in {client_path}")
            route = {
                name: float(route_cache["features"][row, query, feature_index[name]])
                for name in (
                    "route_mobility",
                    "front_feedback_split",
                    "lag_recurrence",
                )
            }
            components[row] = terminal_components(
                state, actions, route, query=query
            )
            logical_digest.update(task.encode("utf-8"))
            logical_digest.update(np.asarray(episode, dtype="<i4").tobytes())
            logical_digest.update(np.asarray(state, dtype="<f4").tobytes())
            logical_digest.update(np.asarray(actions, dtype="<f4").tobytes())
        print(
            f"[components {run_id} {task_position + 1}/{len(task_names)}] {task}",
            flush=True,
        )

    output = {
        "schema": np.asarray("himoe.risk_recovery.components.v1"),
        "component_names": np.asarray(COMPONENT_NAMES),
        "run_id": np.asarray(run_id),
        "task_names": route_cache["task_names"],
        "task_index": route_cache["task_index"],
        "episode": route_cache["episode"],
        "init_state_id": route_cache["init_state_id"],
        "flow_noise_seed": route_cache["flow_noise_seed"],
        "length": route_cache["length"],
        "components": components,
        "logical_source_sha256": np.asarray(logical_digest.hexdigest()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return output


def task_horizons(cache: dict[str, np.ndarray]) -> np.ndarray:
    tasks = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    return np.asarray([HORIZONS[task.split("/", 1)[0]] for task in tasks])


def crossfit_scores(
    cache: dict[str, np.ndarray], components: np.ndarray
) -> np.ndarray:
    task_index = cache["task_index"].astype(int)
    init_state = cache["init_state_id"].astype(int)
    completed = cache["length"].astype(int) < task_horizons(cache)
    output = np.full(len(components), np.nan, dtype=np.float32)
    for task_position in range(len(cache["task_names"])):
        task = task_index == task_position
        for held_out in np.unique(init_state[task]):
            reference = task & completed & (init_state != held_out)
            test = task & (init_state == held_out)
            profile = RecoveryProfile(components[reference])
            output[test] = profile.scores(components[test])
    if not np.isfinite(output).all():
        raise ValueError("cross-fitted recovery score is non-finite")
    return output


def external_scores(
    reference_cache: dict[str, np.ndarray],
    reference_components: np.ndarray,
    test_cache: dict[str, np.ndarray],
    test_components: np.ndarray,
) -> np.ndarray:
    reference_tasks = reference_cache["task_names"].astype(str)
    test_tasks = test_cache["task_names"].astype(str)
    if not np.array_equal(reference_tasks, test_tasks):
        raise ValueError("reference and external task orders differ")
    reference_index = reference_cache["task_index"].astype(int)
    test_index = test_cache["task_index"].astype(int)
    completed = (
        reference_cache["length"].astype(int) < task_horizons(reference_cache)
    )
    output = np.full(len(test_components), np.nan, dtype=np.float32)
    for task_position in range(len(test_tasks)):
        reference = (reference_index == task_position) & completed
        test = test_index == task_position
        profile = RecoveryProfile(reference_components[reference])
        output[test] = profile.scores(test_components[test])
    if not np.isfinite(output).all():
        raise ValueError("external recovery score is non-finite")
    return output


def main_thresholds(
    cache: dict[str, np.ndarray], scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    init_state = cache["init_state_id"].astype(int)
    censored = cache["length"].astype(int) == task_horizons(cache)
    q75 = np.full(len(scores), np.nan, dtype=np.float32)
    q90 = np.full(len(scores), np.nan, dtype=np.float32)
    for held_out in np.unique(init_state):
        calibration = scores[censored & (init_state != held_out)]
        test = init_state == held_out
        q75[test] = quantile_higher(calibration, ALLOCATION_QUANTILES[0])
        q90[test] = quantile_higher(calibration, ALLOCATION_QUANTILES[1])
    return q75, q90


def external_thresholds(
    reference_cache: dict[str, np.ndarray], reference_scores: np.ndarray
) -> tuple[float, float]:
    censored = (
        reference_cache["length"].astype(int) == task_horizons(reference_cache)
    )
    values = reference_scores[censored]
    return tuple(
        quantile_higher(values, quantile) for quantile in ALLOCATION_QUANTILES
    )


def label_table(
    cohort: str, cache: dict[str, np.ndarray], path: Path
) -> pd.DataFrame:
    task_names = cache["task_names"].astype(str)
    index = pd.DataFrame(
        {
            "sealed_row": np.arange(len(cache["episode"])),
            "task": task_names[cache["task_index"].astype(int)],
            "episode": cache["episode"].astype(int),
            "init_state_id": cache["init_state_id"].astype(int),
            "flow_noise_seed": cache["flow_noise_seed"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(path)[
        [
            "task",
            "episode",
            "original_failure",
            "failure",
            "late_success_plus10_queries",
            "late_success_first_extra_query",
        ]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("sealed_row")
    if merged["failure"].isna().any():
        raise ValueError(f"{cohort} labels do not align")
    merged["cohort"] = cohort
    merged["category"] = np.where(
        ~merged["original_failure"],
        "timely",
        np.where(merged["late_success_plus10_queries"], "late", "persistent"),
    )
    return merged.reset_index(drop=True)


def first_alarm_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = load_npz(FIRST_ALARMS)
    levels = archive["levels"].astype(str).tolist()
    quantiles = archive["quantiles"].astype(float)
    q95 = int(np.flatnonzero(np.isclose(quantiles, 0.95))[0])
    k2 = levels.index("mobility_w4_k2")
    k4 = levels.index("mobility_w4_k4")
    return (
        archive["main_first"][:, k2, q95].astype(int),
        archive["main_first"][:, k4, q95].astype(int),
        archive["external_first"][:, k2, q95].astype(int),
        archive["external_first"][:, k4, q95].astype(int),
    )


def episode_table(
    labels: pd.DataFrame,
    first_warning: np.ndarray,
    first_hard: np.ndarray,
    recovery_score: np.ndarray,
    q75: np.ndarray,
    q90: np.ndarray,
) -> pd.DataFrame:
    if not all(
        len(values) == len(labels)
        for values in (first_warning, first_hard, recovery_score, q75, q90)
    ):
        raise ValueError("decision arrays do not align")
    output = labels.copy()
    output["first_warning_query"] = first_warning
    output["first_hard_alarm_query"] = first_hard
    output["warning"] = first_warning >= 0
    output["hard_alarm"] = first_hard >= 0
    output["recovery_score_at_last_planned_query"] = recovery_score
    output["extend_q75_threshold"] = q75
    output["extend_q90_threshold"] = q90
    decisions = []
    for original_failure, score, lower, upper in zip(
        output["original_failure"], recovery_score, q75, q90
    ):
        decisions.append(
            allocation_state(float(score), float(lower), float(upper))
            if original_failure
            else "not_invoked"
        )
    output["allocation"] = decisions
    return output


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def risk_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, block in table.groupby("cohort", sort=False):
        risk = block["original_failure"].to_numpy(dtype=bool)
        timely = ~risk
        late = block["late_success_plus10_queries"].to_numpy(dtype=bool)
        persistent = block["failure"].to_numpy(dtype=bool)
        for name, first_column in (
            ("warning_k2_q95", "first_warning_query"),
            ("hard_alarm_k4_q95", "first_hard_alarm_query"),
        ):
            first = block[first_column].to_numpy(dtype=int)
            alarm = first >= 0
            tp = int((alarm & risk).sum())
            fp = int((alarm & timely).sum())
            rows.append(
                {
                    "cohort": cohort,
                    "detector": name,
                    "risk_n": int(risk.sum()),
                    "timely_n": int(timely.sum()),
                    "tp": tp,
                    "fp": fp,
                    "risk_recall": ratio(tp, risk.sum()),
                    "timely_fpr": ratio(fp, timely.sum()),
                    "risk_precision": ratio(tp, tp + fp),
                    "late_recall": ratio((alarm & late).sum(), late.sum()),
                    "persistent_recall": ratio(
                        (alarm & persistent).sum(), persistent.sum()
                    ),
                    "late_alarm_n": int((alarm & late).sum()),
                    "persistent_alarm_n": int((alarm & persistent).sum()),
                }
            )
    return pd.DataFrame(rows)


def recovery_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, all_rows in table.groupby("cohort", sort=False):
        block = all_rows[all_rows["original_failure"]].copy()
        late = block["late_success_plus10_queries"].to_numpy(dtype=bool)
        persistent = block["failure"].to_numpy(dtype=bool)
        score = block["recovery_score_at_last_planned_query"].to_numpy(float)
        base_rate = float(late.mean())
        auc = float(roc_auc_score(late, score))
        average_precision = float(average_precision_score(late, score))
        candidates = (
            ("extend_all", np.ones(len(block), dtype=bool)),
            (
                "extend_q75",
                block["allocation"].isin(["extend_watch", "strong_extend"]).to_numpy(),
            ),
            ("strong_extend_q90", block["allocation"].eq("strong_extend").to_numpy()),
        )
        first_success = block["late_success_first_extra_query"].fillna(10).to_numpy(float)
        for decision, selected in candidates:
            recovered = int((selected & late).sum())
            wasted = int((selected & persistent).sum())
            query_cost = float(
                np.where(late[selected], first_success[selected], 10.0).sum()
            )
            precision_value = ratio(recovered, selected.sum())
            rows.append(
                {
                    "cohort": cohort,
                    "decision": decision,
                    "risk_n": len(block),
                    "late_n": int(late.sum()),
                    "persistent_n": int(persistent.sum()),
                    "base_late_rate": base_rate,
                    "recovery_roc_auc": auc,
                    "recovery_average_precision": average_precision,
                    "selected_n": int(selected.sum()),
                    "late_recovered": recovered,
                    "persistent_extended": wasted,
                    "extension_precision": precision_value,
                    "precision_lift": ratio(precision_value, base_rate),
                    "late_recall": ratio(recovered, late.sum()),
                    "extra_query_cost": query_cost,
                    "queries_per_recovered": ratio(query_cost, recovered),
                }
            )
    return pd.DataFrame(rows)


def recovery_by_task(table: pd.DataFrame) -> pd.DataFrame:
    risk = table[table["original_failure"]].copy()
    risk["extend_q75"] = risk["allocation"].isin(
        ["extend_watch", "strong_extend"]
    )
    rows = []
    for (cohort, task), block in risk.groupby(["cohort", "task"], sort=True):
        selected = block["extend_q75"]
        late = block["late_success_plus10_queries"]
        rows.append(
            {
                "cohort": cohort,
                "task": task,
                "risk_n": len(block),
                "late_n": int(late.sum()),
                "persistent_n": int(block["failure"].sum()),
                "mean_recovery_score": float(
                    block["recovery_score_at_last_planned_query"].mean()
                ),
                "extend_q75_n": int(selected.sum()),
                "late_recovered_q75": int((selected & late).sum()),
                "extension_precision_q75": ratio(
                    (selected & late).sum(), selected.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def joint_policy_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, all_rows in table.groupby("cohort", sort=False):
        block = all_rows[all_rows["original_failure"]].copy()
        late = block["late_success_plus10_queries"].to_numpy(dtype=bool)
        persistent = block["failure"].to_numpy(dtype=bool)
        active = block["allocation"].isin(
            ["extend_watch", "strong_extend"]
        ).to_numpy()
        hard = block["hard_alarm"].to_numpy(dtype=bool)
        first_success = block["late_success_first_extra_query"].fillna(10).to_numpy(float)
        policies = (
            ("economy_active_q75", active),
            ("safety_hard_or_active_q75", hard | active),
            ("diagnostic_hard_and_active_q75", hard & active),
            ("diagnostic_hard_only", hard),
        )
        for policy, extend in policies:
            intervene = ~extend
            recovered = int((extend & late).sum())
            query_cost = float(
                np.where(late[extend], first_success[extend], 10.0).sum()
            )
            rows.append(
                {
                    "cohort": cohort,
                    "policy": policy,
                    "risk_n": len(block),
                    "extend_n": int(extend.sum()),
                    "late_recovered": recovered,
                    "persistent_extended": int((extend & persistent).sum()),
                    "late_recall": ratio(recovered, late.sum()),
                    "extension_precision": ratio(recovered, extend.sum()),
                    "extra_query_cost": query_cost,
                    "queries_per_recovered": ratio(query_cost, recovered),
                    "intervene_n": int(intervene.sum()),
                    "persistent_intervened": int((intervene & persistent).sum()),
                    "late_intervened": int((intervene & late).sum()),
                    "intervention_precision": ratio(
                        (intervene & persistent).sum(), intervene.sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def recovery_by_suite(table: pd.DataFrame) -> pd.DataFrame:
    risk = table[table["original_failure"]].copy()
    risk["suite"] = risk["task"].str.split("/", n=1).str[0]
    risk["extend_q75"] = risk["allocation"].isin(
        ["extend_watch", "strong_extend"]
    )
    rows: list[dict[str, Any]] = []
    for (cohort, suite), block in risk.groupby(["cohort", "suite"], sort=True):
        late = block["late_success_plus10_queries"].to_numpy(dtype=bool)
        score = block["recovery_score_at_last_planned_query"].to_numpy(float)
        selected = block["extend_q75"].to_numpy(dtype=bool)
        auc = (
            float(roc_auc_score(late, score))
            if late.any() and (~late).any()
            else float("nan")
        )
        rows.append(
            {
                "cohort": cohort,
                "suite": suite,
                "risk_n": len(block),
                "late_n": int(late.sum()),
                "recovery_roc_auc": auc,
                "extend_q75_n": int(selected.sum()),
                "late_recovered_q75": int((selected & late).sum()),
                "extension_precision_q75": ratio(
                    (selected & late).sum(), selected.sum()
                ),
                "late_recall_q75": ratio((selected & late).sum(), late.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_cache = v1.load_feature_cache(MAIN_CACHE)
    external_cache = v1.load_feature_cache(EXTERNAL_CACHE)
    extra_reference = v1.load_feature_cache(EXTRA_REFERENCE_CACHE)
    wanted_tasks = set(external_cache["task_names"].astype(str))
    external_reference = precision.combine_feature_caches(
        [main_cache, extra_reference], wanted_tasks
    )

    main_components_cache = build_component_cache(
        main_cache,
        MAIN_RUN,
        output / "main_terminal_components.npz",
        args.reuse_components,
    )
    reference_components_cache = build_component_cache(
        external_reference,
        MAIN_RUN,
        output / "external_reference_terminal_components.npz",
        args.reuse_components,
    )
    external_components_cache = build_component_cache(
        external_cache,
        EXTERNAL_RUN,
        output / "external_terminal_components.npz",
        args.reuse_components,
    )

    main_score = crossfit_scores(
        main_cache, main_components_cache["components"]
    )
    reference_score = crossfit_scores(
        external_reference, reference_components_cache["components"]
    )
    external_score = external_scores(
        external_reference,
        reference_components_cache["components"],
        external_cache,
        external_components_cache["components"],
    )
    main_q75, main_q90 = main_thresholds(main_cache, main_score)
    external_q75_value, external_q90_value = external_thresholds(
        external_reference, reference_score
    )
    external_q75 = np.full(len(external_score), external_q75_value, dtype=np.float32)
    external_q90 = np.full(len(external_score), external_q90_value, dtype=np.float32)

    # Outcomes are loaded only after every train-free score and threshold is fixed.
    main_labels = label_table(
        "development_main",
        main_cache,
        EXTENSION / "development_main_clean_labels.csv",
    )
    external_labels = label_table(
        "external_8b",
        external_cache,
        EXTENSION / "external_8b_clean_labels.csv",
    )
    main_warning, main_hard, external_warning, external_hard = first_alarm_arrays()
    decisions = pd.concat(
        (
            episode_table(
                main_labels,
                main_warning,
                main_hard,
                main_score,
                main_q75,
                main_q90,
            ),
            episode_table(
                external_labels,
                external_warning,
                external_hard,
                external_score,
                external_q75,
                external_q90,
            ),
        ),
        ignore_index=True,
    )
    risk = risk_metrics(decisions)
    recovery = recovery_metrics(decisions)
    joint = joint_policy_metrics(decisions)
    by_suite = recovery_by_suite(decisions)
    by_task = recovery_by_task(decisions)
    decisions.to_csv(output / "episode_decisions.csv", index=False)
    risk.to_csv(output / "risk_metrics.csv", index=False)
    recovery.to_csv(output / "recovery_metrics.csv", index=False)
    joint.to_csv(output / "joint_policy_metrics.csv", index=False)
    by_suite.to_csv(output / "recovery_by_suite.csv", index=False)
    by_task.to_csv(output / "recovery_by_task.csv", index=False)

    summary = {
        "schema": "himoe.risk_recovery_judge.summary.v1",
        "status": "complete_posthoc_exploration",
        "train_free_runtime": True,
        "query_causal": True,
        "future_or_sim_state_used": False,
        "outcomes_loaded_after_scores_and_thresholds": True,
        "risk": risk.to_dict(orient="records"),
        "recovery": recovery.to_dict(orient="records"),
        "joint_policies": joint.to_dict(orient="records"),
        "recovery_by_suite": by_suite.to_dict(orient="records"),
        "external_allocation_thresholds": {
            "q75": external_q75_value,
            "q90": external_q90_value,
        },
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "monitor_sha256": sha256(HERE / "risk_recovery_monitor.py"),
            "evaluator_sha256": sha256(Path(__file__)),
            "episode_decisions_sha256": sha256(output / "episode_decisions.csv"),
            "risk_metrics_sha256": sha256(output / "risk_metrics.csv"),
            "recovery_metrics_sha256": sha256(output / "recovery_metrics.csv"),
            "joint_policy_metrics_sha256": sha256(
                output / "joint_policy_metrics.csv"
            ),
            "recovery_by_suite_sha256": sha256(output / "recovery_by_suite.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nRisk head")
    print(risk.to_string(index=False))
    print("\nRecovery allocator")
    print(recovery.to_string(index=False))
    print("\nJoint policies")
    print(joint.to_string(index=False))
    print("\nRecovery by suite")
    print(by_suite.to_string(index=False))


if __name__ == "__main__":
    main()
