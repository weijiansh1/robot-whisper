#!/usr/bin/env python3
"""Test conditional front-to-back transfer on same-snapshot noise forks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy.stats import binomtest, rankdata, wilcoxon


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
OUTPUT = BUNDLE / "results/conditional_forks"
PILOT_RECORDS = PROJECT / "himoe-route-capture/runs/fork-pilot-n32-client/fork_records.json"
PILOT_ROUTES = PROJECT / "himoe-route-capture/runs/fork-pilot-n32/routes.zarr"
ROLLING_ROOT = PROJECT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
ROLLING_FORMAL = ROLLING_ROOT / "formal"
ROLLING_ROUTES = ROLLING_FORMAL / "server/routes.zarr"
ROLLING_LABELS = ROLLING_ROOT / "analysis/candidate_physical_labels.csv"
sys.path.insert(0, str(BUNDLE / "transfer"))

from field import (  # noqa: E402
    WITHIN_FEATURE_NAMES,
    compute_within_transfer,
    encode_transfer_views,
)


FORK_VIEWS = ("action_relation", "action_shape", "action_conditional", "action_partial")
SIDES = ("front", "back")
LOCAL_METRICS = ("d0", "d9", "log_ratio", "path", "terminal_z")
FIXED_AXES = {
    "loop": (
        ("fork|action_conditional|front|log_ratio", 1),
        ("fork|action_conditional|conversion|log_ratio", 1),
        ("fork|action_partial|front|path", 1),
        ("single|flow|action_conditional|gap_curvature", 1),
    ),
    "static": (
        ("fork|action_conditional|back|d9", -1),
        ("fork|action_partial|back|d9", -1),
        ("fork|action_conditional|conversion|log_ratio", -1),
        ("single|flow|action_conditional|terminal_alignment", 1),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_pilot() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    records = json.loads(PILOT_RECORDS.read_text(encoding="utf-8"))

    def keep(record: dict[str, Any]) -> bool:
        spread = float(record.get("candidate_spread", 0.0))
        return spread <= 0.0 or float(record.get("rerun_drift", 0.0)) <= 0.05 * spread

    records = [record for record in records if keep(record)]
    rows = np.asarray([record["trace_row"] for record in records], dtype=np.int64)
    store = zarr.open_group(PILOT_ROUTES, mode="r")
    probability = np.asarray(store["hb_router_probs"].oindex[rows, :, :, :, :])
    metadata = [
        {
            "snapshot": f"episode{record['episode']}_q{record['fork_step']}",
            "candidate": int(record["candidate"]),
            "success": bool(record["success_in_window"]),
            "loop": False,
            "static": False,
        }
        for record in records
    ]
    return probability, metadata, {
        "design": "same simulator snapshot, 32 independent Gaussian flow-noise candidates",
        "candidates": len(records),
        "snapshots": len({row["snapshot"] for row in metadata}),
        "source_sha256": sha256(PILOT_RECORDS),
    }


def load_rolling() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    labels = {}
    with ROLLING_LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[int(row["episode_id"])] = row
    metadata = []
    manifests = sorted(ROLLING_FORMAL.glob("worker*/snapshot_*/manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or len(manifest.get("candidate_summaries", ())) != 16:
            raise ValueError(f"incomplete rolling snapshot: {manifest_path}")
        snapshot_key = f"{manifest_path.parent.parent.name}_snapshot{int(manifest['snapshot_index']):03d}"
        for candidate in manifest["candidate_summaries"]:
            episode_id = int(candidate["episode_id"])
            label = labels[episode_id]
            metadata.append(
                {
                    "snapshot": snapshot_key,
                    "candidate": int(candidate["candidate"]),
                    "episode_id": episode_id,
                    "success": bool(candidate["success"]),
                    "loop": as_bool(label["label_loop_or_cycling"]),
                    "static": as_bool(label["label_stagnation"]),
                }
            )
    store = zarr.open_group(ROLLING_ROUTES, mode="r")
    episode = np.asarray(store["episode_id"], dtype=np.int64)
    control = np.asarray(store["control_step"], dtype=np.int64)
    first_row: dict[int, int] = {}
    for row, episode_id in enumerate(episode):
        current = first_row.get(int(episode_id))
        if current is None or control[row] < control[current]:
            first_row[int(episode_id)] = row
    rows = np.asarray([first_row[row["episode_id"]] for row in metadata], dtype=np.int64)
    probability = np.asarray(store["hb_router_probs"].oindex[rows, :, :, :, :])
    return probability, metadata, {
        "design": "same simulator snapshot, 16 independent full-rollout noise streams; q0 routing",
        "candidates": len(metadata),
        "snapshots": len({row["snapshot"] for row in metadata}),
        "source_sha256": sha256(ROLLING_LABELS),
    }


def pairwise_median(value: torch.Tensor) -> float:
    return float(torch.pdist(value.float()).median().item())


def leave_one_out_distance(value: torch.Tensor) -> torch.Tensor:
    count = value.shape[0]
    center = (value.sum(dim=0, keepdim=True) - value) / max(count - 1, 1)
    return torch.linalg.vector_norm(value - center, dim=-1)


def local_metrics(value: torch.Tensor) -> torch.Tensor:
    """Candidate distance to the leave-one-out same-snapshot routing center."""

    distance = torch.stack(
        [leave_one_out_distance(value[:, flow]) for flow in range(value.shape[1])], dim=1
    )
    d0 = distance[:, 0]
    d9 = distance[:, -1]
    path = torch.abs(distance[:, 1:] - distance[:, :-1]).sum(dim=1)
    terminal_scale = d9.median().clamp_min(1e-8)
    return torch.stack(
        (
            d0,
            d9,
            torch.log((d9 + 1e-8) / (d0 + 1e-8)),
            path,
            d9 / terminal_scale,
        ),
        dim=-1,
    )


def candidate_feature_names() -> tuple[str, ...]:
    names = [
        f"fork|{view}|{side}|{metric}"
        for view in FORK_VIEWS
        for side in SIDES
        for metric in LOCAL_METRICS
    ]
    names.extend(f"fork|{view}|conversion|log_ratio" for view in FORK_VIEWS)
    names.extend(f"single|{name}" for name in WITHIN_FEATURE_NAMES)
    return tuple(names)


CANDIDATE_FEATURE_NAMES = candidate_feature_names()


def analyze_dataset(dataset: str, device_index: int) -> dict[str, Any]:
    probability, metadata, audit = load_pilot() if dataset == "fork_pilot_n32" else load_rolling()
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tensor = torch.as_tensor(probability, dtype=torch.float32, device=device)
    paired, _ = encode_transfer_views(tensor)
    within = torch.as_tensor(compute_within_transfer(tensor)["features"], device=device)
    del tensor, probability

    candidate_matrix = torch.empty(
        (len(metadata), len(CANDIDATE_FEATURE_NAMES)), dtype=torch.float32, device=device
    )
    ensemble_rows = []
    snapshot_ids = sorted({row["snapshot"] for row in metadata})
    local_width = len(FORK_VIEWS) * len(SIDES) * len(LOCAL_METRICS)
    for snapshot in snapshot_ids:
        selected_np = np.asarray(
            [index for index, row in enumerate(metadata) if row["snapshot"] == snapshot],
            dtype=np.int64,
        )
        selected = torch.as_tensor(selected_np, device=device)
        cursor = 0
        ratios: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for view in FORK_VIEWS:
            front, back = paired[view]
            front = front.index_select(0, selected)
            back = back.index_select(0, selected)
            view_ratios = []
            for side, value in (("front", front), ("back", back)):
                metrics = local_metrics(value)
                width = metrics.shape[1]
                candidate_matrix[selected, cursor : cursor + width] = metrics
                cursor += width
                dispersion = [pairwise_median(value[:, flow]) for flow in range(10)]
                ratio = dispersion[-1] / max(dispersion[0], 1e-12)
                view_ratios.append(metrics[:, LOCAL_METRICS.index("log_ratio")])
                ensemble_rows.append(
                    {
                        "dataset": dataset,
                        "snapshot": snapshot,
                        "view": view,
                        "side": side,
                        "d0": dispersion[0],
                        "d9": dispersion[-1],
                        "ratio_d9_d0": ratio,
                        "log_slope": float(
                            np.polyfit(np.arange(10), np.log(np.maximum(dispersion, 1e-12)), 1)[0]
                        ),
                        "candidates": len(selected_np),
                    }
                )
            ratios[view] = (view_ratios[0], view_ratios[1])
        if cursor != local_width:
            raise RuntimeError("local fork schema mismatch")
        for view in FORK_VIEWS:
            front_ratio, back_ratio = ratios[view]
            candidate_matrix[selected, cursor] = back_ratio - front_ratio
            cursor += 1
        candidate_matrix[selected, cursor:] = within.index_select(0, selected)

    matrix = candidate_matrix.detach().cpu().numpy()
    outcome_rows = []
    for outcome in ("success", "loop", "static"):
        for axis_index, axis in enumerate(CANDIDATE_FEATURE_NAMES):
            snapshot_aucs = []
            for snapshot in snapshot_ids:
                selected = np.asarray(
                    [index for index, row in enumerate(metadata) if row["snapshot"] == snapshot],
                    dtype=np.int64,
                )
                label = np.asarray([metadata[index][outcome] for index in selected], dtype=bool)
                if not label.any() or label.all():
                    continue
                value = matrix[selected, axis_index]
                positive = value[label][:, None]
                negative = value[~label][None, :]
                snapshot_aucs.append(
                    float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative))
                )
            outcome_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "axis": axis,
                    "mixed_snapshots": len(snapshot_aucs),
                    "snapshot_aucs": snapshot_aucs,
                }
            )
    feature_indices = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
    for outcome, definition in FIXED_AXES.items():
        snapshot_aucs = []
        for snapshot in snapshot_ids:
            selected = np.asarray(
                [index for index, row in enumerate(metadata) if row["snapshot"] == snapshot],
                dtype=np.int64,
            )
            label = np.asarray([metadata[index][outcome] for index in selected], dtype=bool)
            if not label.any() or label.all():
                continue
            columns = []
            for axis, direction in definition:
                value = matrix[selected, feature_indices[axis]]
                percentile = rankdata(value, method="average") / (len(value) + 1.0)
                columns.append(percentile if direction > 0 else 1.0 - percentile)
            score = np.column_stack(columns).mean(axis=1)
            positive = score[label][:, None]
            negative = score[~label][None, :]
            snapshot_aucs.append(
                float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative))
            )
        outcome_rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "axis": f"fixed_composite|{outcome}",
                "mixed_snapshots": len(snapshot_aucs),
                "snapshot_aucs": snapshot_aucs,
            }
        )
    audit.update(
        {
            "device": device_index,
            "device_name": torch.cuda.get_device_name(device),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "candidate_axes": len(CANDIDATE_FEATURE_NAMES),
            "mixed_snapshots": {
                outcome: sum(
                    0
                    < sum(metadata[index][outcome] for index, row in enumerate(metadata) if row["snapshot"] == snapshot)
                    < sum(row["snapshot"] == snapshot for row in metadata)
                    for snapshot in snapshot_ids
                )
                for outcome in ("success", "loop", "static")
            },
        }
    )
    return {
        "dataset": dataset,
        "audit": audit,
        "ensemble_rows": ensemble_rows,
        "outcome_rows": outcome_rows,
    }


def worker(dataset: str, device_index: int, queue: mp.Queue) -> None:
    try:
        queue.put(analyze_dataset(dataset, device_index))
    except Exception as error:  # pragma: no cover
        queue.put({"dataset": dataset, "error": repr(error)})


def bootstrap_ci(
    values: np.ndarray, draws: int, seed: int, statistic: str = "mean"
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    estimate = np.median(sampled, axis=1) if statistic == "median" else sampled.mean(axis=1)
    return float(np.quantile(estimate, 0.025)), float(np.quantile(estimate, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_ensemble(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["dataset"], row["view"], row["side"]) for row in rows})
    for ordinal, (dataset, view, side) in enumerate(keys):
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset and row["view"] == view and row["side"] == side
        ]
        ratio = np.asarray([row["ratio_d9_d0"] for row in selected])
        low, high = bootstrap_ci(ratio, draws, 7103 + ordinal, statistic="median")
        contracting = int(np.count_nonzero(ratio < 1.0))
        summaries.append(
            {
                "dataset": dataset,
                "view": view,
                "side": side,
                "snapshots": len(selected),
                "median_ratio_d9_d0": float(np.median(ratio)),
                "bootstrap_median_95_low": low,
                "bootstrap_median_95_high": high,
                "contracting_snapshots": contracting,
                "sign_test_p_contraction": float(
                    binomtest(contracting, len(selected), 0.5, alternative="greater").pvalue
                ),
                "median_d0": float(np.median([row["d0"] for row in selected])),
                "median_d9": float(np.median([row["d9"] for row in selected])),
            }
        )
    return summaries


def summarize_outcomes(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    summaries = []
    for ordinal, row in enumerate(rows):
        auc = np.asarray(row.pop("snapshot_aucs"), dtype=np.float64)
        low, high = bootstrap_ci(auc, draws, 13001 + ordinal)
        if len(auc) < 2 or np.allclose(auc, 0.5):
            p_value = float("nan")
        else:
            p_value = float(wilcoxon(auc - 0.5).pvalue)
        row.update(
            {
                "mean_within_snapshot_auc_high": float(auc.mean()) if len(auc) else float("nan"),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wilcoxon_p_vs_half": p_value,
            }
        )
        summaries.append(row)
    return summaries


def summarize_handoff(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    summaries = []
    for ordinal, (dataset, view) in enumerate(
        sorted({(row["dataset"], row["view"]) for row in rows})
    ):
        selected = [row for row in rows if row["dataset"] == dataset and row["view"] == view]
        by_snapshot: dict[str, dict[str, float]] = {}
        for row in selected:
            by_snapshot.setdefault(row["snapshot"], {})[row["side"]] = row["ratio_d9_d0"]
        contrast = np.asarray(
            [
                np.log(value["back"] + 1e-12) - np.log(value["front"] + 1e-12)
                for value in by_snapshot.values()
            ],
            dtype=np.float64,
        )
        low, high = bootstrap_ci(contrast, draws, 17011 + ordinal, statistic="median")
        positive = int(np.count_nonzero(contrast > 0.0))
        summaries.append(
            {
                "dataset": dataset,
                "view": view,
                "snapshots": len(contrast),
                "median_log_back_front_ratio": float(np.median(contrast)),
                "equivalent_back_front_ratio": float(np.exp(np.median(contrast))),
                "bootstrap_median_95_low": low,
                "bootstrap_median_95_high": high,
                "positive_snapshots": positive,
                "sign_test_p_back_greater": float(
                    binomtest(positive, len(contrast), 0.5, alternative="greater").pvalue
                ),
                "wilcoxon_p_back_greater": float(
                    wilcoxon(contrast, alternative="greater").pvalue
                ),
            }
        )
    return summaries


def plot_contraction(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["view"] in {"action_conditional", "action_partial"}]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
    colors = {"front": "#2d6f8e", "back": "#c65342"}
    for panel, dataset in zip(axes, ("fork_pilot_n32", "rolling_star_k16")):
        subset = [row for row in selected if row["dataset"] == dataset]
        x = np.arange(2)
        for offset, side in ((-0.18, "front"), (0.18, "back")):
            value = [
                next(
                    row["median_ratio_d9_d0"]
                    for row in subset
                    if row["view"] == view and row["side"] == side
                )
                for view in ("action_conditional", "action_partial")
            ]
            panel.bar(x + offset, value, width=0.34, color=colors[side], label=side)
        panel.axhline(1.0, color="#555555", lw=0.8, ls="--")
        panel.set_xticks(x, ("conditional Gram", "partial affinity"))
        panel.set_title(dataset)
        panel.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("median same-snapshot D9 / D0")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "conditional_fork_contraction.png", dpi=180)
    fig.savefig(output / "conditional_fork_contraction.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(dataset, device, queue))
        for device, dataset in enumerate(("fork_pilot_n32", "rolling_star_k16"))
    ]
    for process in processes:
        process.start()
    payloads = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    failures = [payload for payload in payloads if "error" in payload]
    if failures:
        raise RuntimeError(str(failures))

    args.output.mkdir(parents=True, exist_ok=True)
    ensemble_rows = [row for payload in payloads for row in payload["ensemble_rows"]]
    outcome_rows = [row for payload in payloads for row in payload["outcome_rows"]]
    ensemble_summary = summarize_ensemble(ensemble_rows, args.bootstrap)
    handoff_summary = summarize_handoff(ensemble_rows, args.bootstrap)
    outcome_summary = summarize_outcomes(outcome_rows, args.bootstrap)
    write_csv(args.output / "ensemble_snapshot_rows.csv", ensemble_rows)
    write_csv(args.output / "ensemble_contraction.csv", ensemble_summary)
    write_csv(args.output / "handoff_contrast.csv", handoff_summary)
    write_csv(args.output / "candidate_axis_auc.csv", outcome_summary)
    plot_contraction(ensemble_summary, args.output)

    fixed = {}
    for outcome, definitions in FIXED_AXES.items():
        fixed[outcome] = []
        for axis, direction in definitions:
            row = next(
                item
                for item in outcome_summary
                if item["dataset"] == "rolling_star_k16"
                and item["outcome"] == outcome
                and item["axis"] == axis
            )
            fixed[outcome].append(
                {
                    **row,
                    "direction": direction,
                    "direction_locked_auc": (
                        row["mean_within_snapshot_auc_high"]
                        if direction > 0
                        else 1.0 - row["mean_within_snapshot_auc_high"]
                    ),
                }
            )
        composite = next(
            item
            for item in outcome_summary
            if item["dataset"] == "rolling_star_k16"
            and item["outcome"] == outcome
            and item["axis"] == f"fixed_composite|{outcome}"
        )
        fixed[outcome].append({**composite, "direction": 1, "direction_locked_auc": composite["mean_within_snapshot_auc_high"]})
    post_hoc = {}
    for outcome in ("success", "loop", "static"):
        candidates = [
            row
            for row in outcome_summary
            if row["dataset"] == "rolling_star_k16"
            and row["outcome"] == outcome
            and row["mixed_snapshots"] >= 3
            and np.isfinite(row["mean_within_snapshot_auc_high"])
        ]
        post_hoc[outcome] = sorted(
            (
                {
                    **row,
                    "two_sided_auc": max(
                        row["mean_within_snapshot_auc_high"],
                        1.0 - row["mean_within_snapshot_auc_high"],
                    ),
                }
                for row in candidates
            ),
            key=lambda row: row["two_sided_auc"],
            reverse=True,
        )[:20]
    summary = {
        "schema": "himoe.conditional_fork_transfer.v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "datasets": [payload["audit"] for payload in payloads],
        "candidate_axes": len(CANDIDATE_FEATURE_NAMES),
        "ensemble_contraction": ensemble_summary,
        "handoff_contrast": handoff_summary,
        "fixed_outcome_axes": fixed,
        "post_hoc_axis_audit": post_hoc,
        "caveat": (
            "Independent-noise q0 forks measure ensemble sensitivity, not infinitesimal stability. "
            "The post-hoc scan is descriptive; only fixed axes have a predeclared direction."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"datasets": summary["datasets"], "fixed": fixed, "top": {key: value[:5] for key, value in post_hoc.items()}}, indent=2))


if __name__ == "__main__":
    main()
