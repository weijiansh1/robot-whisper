#!/usr/bin/env python3
"""Measure same-snapshot ensemble contraction of weighted MoE graph views."""

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

import numpy as np
import torch
import zarr
from scipy.stats import binomtest, wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
OUTPUT = BUNDLE / "results/fork_contraction"
PILOT_RECORDS = PROJECT / "himoe-route-capture/runs/fork-pilot-n32-client/fork_records.json"
PILOT_ROUTES = PROJECT / "himoe-route-capture/runs/fork-pilot-n32/routes.zarr"
ROLLING_ROOT = PROJECT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
ROLLING_FORMAL = ROLLING_ROOT / "formal"
ROLLING_ROUTES = ROLLING_FORMAL / "server/routes.zarr"
ROLLING_LABELS = ROLLING_ROOT / "analysis/candidate_physical_labels.csv"
sys.path.insert(0, str(BUNDLE / "graph"))

from structure import (  # noqa: E402
    LAYER_GROUPS,
    VIEW_NAMES,
    WITHIN_FEATURE_NAMES,
    compute_within_structure,
    encode_graph_views,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
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
            "row": int(record["trace_row"]),
        }
        for record in records
    ]
    audit = {
        "design": "same simulator snapshot, 32 independent Gaussian flow-noise candidates",
        "candidates": len(records),
        "snapshots": len({row["snapshot"] for row in metadata}),
        "outcome": "success_in_window",
        "source_hashes": {"fork_records.json": sha256(PILOT_RECORDS)},
    }
    return probability, metadata, audit


def load_rolling() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    label_by_episode: dict[int, dict[str, str]] = {}
    with ROLLING_LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_by_episode[int(row["episode_id"])] = row

    metadata = []
    manifests = sorted(ROLLING_FORMAL.glob("worker*/snapshot_*/manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or len(manifest.get("candidate_summaries", ())) != 16:
            raise ValueError(f"incomplete rolling snapshot: {manifest_path}")
        worker = manifest_path.parent.parent.name
        snapshot = int(manifest["snapshot_index"])
        snapshot_key = f"{worker}_snapshot{snapshot:03d}"
        for candidate in manifest["candidate_summaries"]:
            episode_id = int(candidate["episode_id"])
            label = label_by_episode[episode_id]
            success = bool(candidate["success"])
            if success != as_bool(label["success"]):
                raise ValueError(f"outcome mismatch for episode {episode_id}")
            metadata.append(
                {
                    "snapshot": snapshot_key,
                    "candidate": int(candidate["candidate"]),
                    "episode_id": episode_id,
                    "success": success,
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
    if len(np.unique(rows)) != len(rows):
        raise ValueError("rolling candidate query-zero rows are not one-to-one")
    for record, row in zip(metadata, rows):
        record["row"] = int(row)
    probability = np.asarray(store["hb_router_probs"].oindex[rows, :, :, :, :])
    audit = {
        "design": "same simulator snapshot, 16 independent full rollout noise streams; query zero only",
        "candidates": len(metadata),
        "snapshots": len({row["snapshot"] for row in metadata}),
        "outcome": "eventual rollout success and physical failure taxonomy",
        "manifests": len(manifests),
        "source_hashes": {
            "candidate_physical_labels.csv": sha256(ROLLING_LABELS),
            "manifest_digest": hashlib.sha256(
                "".join(sha256(path) for path in manifests).encode("ascii")
            ).hexdigest(),
        },
    }
    return probability, metadata, audit


def pairwise_summary(embedding: torch.Tensor) -> tuple[float, float]:
    distance = torch.pdist(embedding.float())
    return float(distance.median().item()), float(distance.mean().item())


def analyze_dataset(dataset: str, device_index: int) -> dict[str, Any]:
    probability, metadata, audit = load_pilot() if dataset == "fork_pilot_n32" else load_rolling()
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tensor = torch.as_tensor(probability, dtype=torch.float32, device=device)
    views, _ = encode_graph_views(tensor)
    within = compute_within_structure(tensor)["features"]
    del tensor, probability

    snapshot_ids = sorted({row["snapshot"] for row in metadata})
    flow_rows: list[dict[str, Any]] = []
    contraction_rows: list[dict[str, Any]] = []
    for snapshot in snapshot_ids:
        selected_np = np.asarray(
            [index for index, row in enumerate(metadata) if row["snapshot"] == snapshot],
            dtype=np.int64,
        )
        selected = torch.as_tensor(selected_np, device=device)
        for group_name, layer_indices in LAYER_GROUPS.items():
            layer = torch.as_tensor(layer_indices, device=device)
            for view_name in VIEW_NAMES:
                value = views[view_name].index_select(0, selected).index_select(1, layer)
                medians = []
                means = []
                for flow in range(value.shape[2]):
                    embedding = value[:, :, flow].flatten(1) / np.sqrt(len(layer_indices))
                    median, mean = pairwise_summary(embedding)
                    medians.append(median)
                    means.append(mean)
                    flow_rows.append(
                        {
                            "dataset": dataset,
                            "snapshot": snapshot,
                            "layer_group": group_name,
                            "view": view_name,
                            "flow": flow,
                            "median_dispersion": median,
                            "mean_dispersion": mean,
                            "candidates": len(selected_np),
                        }
                    )
                log_distance = np.log(np.maximum(medians, 1e-12))
                slope = float(np.polyfit(np.arange(10, dtype=np.float64), log_distance, 1)[0])
                contraction_rows.append(
                    {
                        "dataset": dataset,
                        "snapshot": snapshot,
                        "layer_group": group_name,
                        "view": view_name,
                        "d0": medians[0],
                        "d9": medians[-1],
                        "ratio_d9_d0": medians[-1] / max(medians[0], 1e-12),
                        "log_slope": slope,
                        "declining_transition_fraction": float(
                            np.mean(np.diff(np.asarray(medians)) < 0.0)
                        ),
                        "candidates": len(selected_np),
                    }
                )

    state_indices = [
        index for index, name in enumerate(WITHIN_FEATURE_NAMES) if name.startswith("state|")
    ]
    outcome_rows = []
    for outcome_name in ("success", "loop", "static"):
        for state_index in state_indices:
            differences = []
            snapshot_aucs = []
            for snapshot in snapshot_ids:
                selected = np.asarray(
                    [index for index, row in enumerate(metadata) if row["snapshot"] == snapshot],
                    dtype=np.int64,
                )
                label = np.asarray([metadata[index][outcome_name] for index in selected], dtype=bool)
                value = within[selected, state_index].astype(np.float64)
                if label.any() and (~label).any():
                    differences.append(float(value[label].mean() - value[~label].mean()))
                    positive = value[label][:, None]
                    negative = value[~label][None, :]
                    snapshot_aucs.append(
                        float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative))
                    )
            outcome_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome_name,
                    "axis": WITHIN_FEATURE_NAMES[state_index],
                    "mixed_snapshots": len(differences),
                    "mean_within_snapshot_difference": (
                        float(np.mean(differences)) if differences else float("nan")
                    ),
                    "snapshot_differences": differences,
                    "snapshot_aucs": snapshot_aucs,
                }
            )
    audit["mixed_success_snapshots"] = sum(
        0 < sum(metadata[index]["success"] for index, row in enumerate(metadata) if row["snapshot"] == snapshot)
        < sum(row["snapshot"] == snapshot for row in metadata)
        for snapshot in snapshot_ids
    )
    audit["device"] = device_index
    audit["device_name"] = torch.cuda.get_device_name(device)
    audit["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return {
        "dataset": dataset,
        "audit": audit,
        "flow_rows": flow_rows,
        "contraction_rows": contraction_rows,
        "outcome_rows": outcome_rows,
    }


def worker(dataset: str, device_index: int, queue: mp.Queue) -> None:
    try:
        queue.put(analyze_dataset(dataset, device_index))
    except Exception as error:  # pragma: no cover
        queue.put({"dataset": dataset, "error": repr(error)})


def bootstrap_ci(
    values: np.ndarray,
    draws: int,
    seed: int,
    statistic: str = "median",
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    estimate = sampled.mean(axis=1) if statistic == "mean" else np.median(sampled, axis=1)
    return float(np.quantile(estimate, 0.025)), float(np.quantile(estimate, 0.975))


def summarize_contraction(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["dataset"], row["layer_group"], row["view"]) for row in rows})
    for ordinal, (dataset, group, view) in enumerate(keys):
        selected = [
            row for row in rows
            if row["dataset"] == dataset and row["layer_group"] == group and row["view"] == view
        ]
        ratio = np.asarray([row["ratio_d9_d0"] for row in selected], dtype=np.float64)
        difference = np.asarray([row["d9"] - row["d0"] for row in selected], dtype=np.float64)
        contracting = int(np.count_nonzero(ratio < 1.0))
        low, high = bootstrap_ci(ratio, draws, 1729 + ordinal)
        if np.allclose(difference, 0.0):
            signed_p = 1.0
        else:
            signed_p = float(wilcoxon(difference, alternative="less").pvalue)
        summaries.append(
            {
                "dataset": dataset,
                "layer_group": group,
                "view": view,
                "snapshots": len(selected),
                "median_ratio_d9_d0": float(np.median(ratio)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "contracting_snapshots": contracting,
                "fraction_contracting": contracting / len(selected),
                "sign_test_p_contraction": float(
                    binomtest(contracting, len(selected), 0.5, alternative="greater").pvalue
                ),
                "wilcoxon_p_d9_less_d0": signed_p,
                "median_d0": float(np.median([row["d0"] for row in selected])),
                "median_d9": float(np.median([row["d9"] for row in selected])),
                "median_log_slope": float(np.median([row["log_slope"] for row in selected])),
            }
        )
    return summaries


def summarize_outcomes(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    summaries = []
    for ordinal, row in enumerate(rows):
        difference = np.asarray(row.pop("snapshot_differences"), dtype=np.float64)
        auc = np.asarray(row.pop("snapshot_aucs"), dtype=np.float64)
        low, high = bootstrap_ci(difference, draws, 8111 + ordinal, statistic="mean")
        auc_low, auc_high = bootstrap_ci(auc, draws, 12143 + ordinal, statistic="mean")
        if len(difference) < 2 or np.allclose(difference, 0.0):
            signed_p = float("nan")
        else:
            signed_p = float(wilcoxon(difference).pvalue)
        if len(auc) < 2 or np.allclose(auc, 0.5):
            auc_p = float("nan")
        else:
            auc_p = float(wilcoxon(auc - 0.5).pvalue)
        row.update(
            {
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wilcoxon_p_two_sided": signed_p,
                "mean_within_snapshot_auc": float(auc.mean()) if len(auc) else float("nan"),
                "auc_bootstrap_95_low": auc_low,
                "auc_bootstrap_95_high": auc_high,
                "auc_wilcoxon_p_vs_half": auc_p,
            }
        )
        summaries.append(row)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_contraction(flow_rows: list[dict[str, Any]], output: Path) -> None:
    curves = (
        ("front", "edge_action", "front action edges", "#287271", "-"),
        ("front", "token_gram", "front token affinity", "#d98947", "-"),
        ("back", "edge_action", "back action edges", "#287271", "--"),
        ("back", "token_gram", "back token affinity", "#d98947", "--"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for panel, dataset in zip(axes, ("fork_pilot_n32", "rolling_star_k16")):
        for group, view, label, color, linestyle in curves:
            selected = [
                row for row in flow_rows
                if row["dataset"] == dataset and row["layer_group"] == group and row["view"] == view
            ]
            snapshots = sorted({row["snapshot"] for row in selected})
            matrix = np.asarray(
                [
                    [
                        next(
                            row["median_dispersion"] for row in selected
                            if row["snapshot"] == snapshot and row["flow"] == flow
                        )
                        for flow in range(10)
                    ]
                    for snapshot in snapshots
                ],
                dtype=np.float64,
            )
            normalized = matrix / np.maximum(matrix[:, :1], 1e-12)
            center = np.median(normalized, axis=0)
            low, high = np.quantile(normalized, (0.25, 0.75), axis=0)
            panel.plot(range(10), center, color=color, ls=linestyle, lw=2.0, label=label)
            panel.fill_between(range(10), low, high, color=color, alpha=0.10)
        panel.axhline(1.0, color="#777777", lw=0.8, ls=":")
        panel.set_yscale("log")
        panel.set_title(dataset)
        panel.set_xlabel("flow step")
        panel.grid(alpha=0.15)
    axes[0].set_ylabel("snapshot dispersion / dispersion at flow 0")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Same-snapshot MoE graph ensemble contraction", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "fork_contraction.png", dpi=180)
    fig.savefig(output / "fork_contraction.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    jobs = (("fork_pilot_n32", 0), ("rolling_star_k16", 1))
    processes = [context.Process(target=worker, args=(*job, queue)) for job in jobs]
    for process in processes:
        process.start()
    payloads = [queue.get() for _ in processes]
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"fork worker failed with exit code {process.exitcode}")
    failures = [payload for payload in payloads if "error" in payload]
    if failures:
        raise RuntimeError(str(failures))

    flow_rows = [row for payload in payloads for row in payload["flow_rows"]]
    contraction_rows = [row for payload in payloads for row in payload["contraction_rows"]]
    outcome_rows = [row for payload in payloads for row in payload["outcome_rows"]]
    contraction_summary = summarize_contraction(contraction_rows, args.bootstrap)
    outcome_summary = summarize_outcomes(outcome_rows, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "flow_dispersion.csv", flow_rows)
    write_csv(args.output / "snapshot_contraction.csv", contraction_rows)
    write_csv(args.output / "contraction_summary.csv", contraction_summary)
    write_csv(args.output / "outcome_contrast.csv", outcome_summary)
    plot_contraction(flow_rows, args.output)

    common_contractions = []
    replicated_contractions = []
    for group in LAYER_GROUPS:
        for view in VIEW_NAMES:
            pair = [
                row for row in contraction_summary
                if row["layer_group"] == group and row["view"] == view
            ]
            if len(pair) == 2 and all(row["fraction_contracting"] > 0.5 for row in pair):
                common_contractions.append(
                    {
                        "layer_group": group,
                        "view": view,
                        "ratios": {row["dataset"]: row["median_ratio_d9_d0"] for row in pair},
                    }
                )
            if len(pair) == 2 and all(row["bootstrap_95_high"] < 1.0 for row in pair):
                replicated_contractions.append(
                    {
                        "layer_group": group,
                        "view": view,
                        "ratios": {row["dataset"]: row["median_ratio_d9_d0"] for row in pair},
                        "bootstrap_95_high": {
                            row["dataset"]: row["bootstrap_95_high"] for row in pair
                        },
                    }
                )
    summary = {
        "schema": "himoe.graph_fork_contraction.v1",
        "interpretation": "global independent-noise ensemble contraction within one forward flow",
        "not_measured": "infinitesimal local Lyapunov stability or calibrated task-success probability",
        "graph_views": list(VIEW_NAMES),
        "layer_groups": list(LAYER_GROUPS),
        "gpu_accelerated": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "datasets": {payload["dataset"]: payload["audit"] for payload in payloads},
        "common_contractions": common_contractions,
        "replicated_contractions": replicated_contractions,
        "contraction_summary": contraction_summary,
        "outcome_summary": outcome_summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "datasets": summary["datasets"],
                "common_contractions": common_contractions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
