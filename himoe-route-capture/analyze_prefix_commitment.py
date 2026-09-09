"""Analyze nested early-prefix counterfactual rollout grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import zarr

from analyze_commitment_grid import _commitment_statistics


@dataclass
class PrefixCell:
    summary: dict
    as_prefix: np.ndarray
    hb_prefix: np.ndarray
    flow_noises: np.ndarray
    states: np.ndarray
    action_chunks: np.ndarray


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _load_cells(server_dir: pathlib.Path, client_dir: pathlib.Path) -> tuple[list[PrefixCell], dict]:
    summaries = json.loads((client_dir / "summaries.json").read_text())
    group = zarr.open(str(server_dir / "routes.zarr"), mode="r")
    required = ("as_probs", "hb_router_probs", "episode_id", "control_step")
    missing = [name for name in required if name not in group]
    if missing:
        raise RuntimeError("route store is missing arrays: %s" % missing)

    total = int(sum(int(summary["inference_calls"]) for summary in summaries))
    available = int(group["hb_router_probs"].shape[0])
    if total != available:
        raise RuntimeError(
            "client has %d inference calls but route store has %d; flush the server first"
            % (total, available)
        )

    control_steps = np.asarray(group["control_step"][:])
    if len(control_steps) > 1 and not np.all(np.diff(control_steps) == 1):
        raise RuntimeError("server control_step axis is not contiguous")

    cells = []
    cursor = 0
    for summary in summaries:
        count = int(summary["inference_calls"])
        branch = int(summary["branch_controls"])
        if count <= 0:
            raise RuntimeError("episode %s contains no inference calls" % summary["episode_index"])
        episode_ids = np.asarray(group["episode_id"][cursor : cursor + count])
        if not np.all(episode_ids == int(summary["episode_index"])):
            raise RuntimeError("server/client episode ids disagree at trace offset %d" % cursor)

        archive = np.load(client_dir / ("episode_%04d.npz" % int(summary["episode_index"])))
        flow_noises = np.asarray(archive["flow_noises"], dtype=np.float32)
        states = np.asarray(archive["states"], dtype=np.float32)
        action_chunks = np.asarray(archive["action_chunks"], dtype=np.float32)
        if not (len(flow_noises) == len(states) == len(action_chunks) == count):
            raise RuntimeError("client archive length mismatch for episode %d" % summary["episode_index"])

        prefix_len = min(branch, count)
        cells.append(
            PrefixCell(
                summary=summary,
                as_prefix=np.asarray(group["as_probs"][cursor : cursor + prefix_len], dtype=np.float32),
                hb_prefix=np.asarray(
                    group["hb_router_probs"][cursor : cursor + prefix_len], dtype=np.float32
                ),
                flow_noises=flow_noises,
                states=states,
                action_chunks=action_chunks,
            )
        )
        cursor += count
    return cells, dict(group.attrs)


def _integrity(cells: list[PrefixCell], branches: list[int]) -> dict:
    observation_hashes = {cell.summary["initial_observation_sha256"] for cell in cells}
    max_as = 0.0
    max_hb = 0.0
    max_state = 0.0
    max_action = 0.0
    prefix_noise_mismatches = 0
    prefix_pairs = 0
    truncated_prefix_cells = 0

    rows = sorted({int(cell.summary["grid_row"]) for cell in cells})
    cols = sorted({int(cell.summary["grid_col"]) for cell in cells})

    for branch in branches:
        for row in rows:
            same_prefix = [
                cell
                for cell in cells
                if int(cell.summary["branch_controls"]) == branch
                and int(cell.summary["grid_row"]) == row
            ]
            if not same_prefix:
                continue
            reference = same_prefix[0]
            truncated_prefix_cells += sum(len(cell.as_prefix) < branch for cell in same_prefix)
            for cell in same_prefix[1:]:
                common = min(branch, len(reference.as_prefix), len(cell.as_prefix))
                max_as = max(max_as, _max_abs(reference.as_prefix[:common], cell.as_prefix[:common]))
                max_hb = max(max_hb, _max_abs(reference.hb_prefix[:common], cell.hb_prefix[:common]))
                max_state = max(max_state, _max_abs(reference.states[:common], cell.states[:common]))
                max_action = max(
                    max_action, _max_abs(reference.action_chunks[:common], cell.action_chunks[:common])
                )
                for offset in range(common):
                    prefix_noise_mismatches += int(
                        _array_sha256(reference.flow_noises[offset])
                        != _array_sha256(cell.flow_noises[offset])
                    )
                prefix_pairs += 1

    future_noise_mismatches = 0
    future_tensors_compared = 0
    for branch in branches:
        for col in cols:
            same_future = [
                cell
                for cell in cells
                if int(cell.summary["branch_controls"]) == branch
                and int(cell.summary["grid_col"]) == col
                and len(cell.flow_noises) > branch
            ]
            if not same_future:
                continue
            reference = same_future[0]
            for cell in same_future[1:]:
                common = min(len(reference.flow_noises), len(cell.flow_noises)) - branch
                for offset in range(max(0, common)):
                    future_tensors_compared += 1
                    future_noise_mismatches += int(
                        _array_sha256(reference.flow_noises[branch + offset])
                        != _array_sha256(cell.flow_noises[branch + offset])
                    )

    nested_noise_mismatches = 0
    nested_pairs = 0
    nested_max_as = 0.0
    nested_max_hb = 0.0
    nested_max_state = 0.0
    nested_max_action = 0.0
    for row in rows:
        references = {}
        for branch in branches:
            candidates = [
                cell
                for cell in cells
                if int(cell.summary["branch_controls"]) == branch
                and int(cell.summary["grid_row"]) == row
                and int(cell.summary["grid_col"]) == cols[0]
            ]
            if candidates:
                references[branch] = candidates[0]
        ordered = sorted(references)
        for left_index, left_branch in enumerate(ordered):
            for right_branch in ordered[left_index + 1 :]:
                left = references[left_branch]
                right = references[right_branch]
                common = min(left_branch, len(left.as_prefix), len(right.as_prefix))
                nested_max_as = max(
                    nested_max_as, _max_abs(left.as_prefix[:common], right.as_prefix[:common])
                )
                nested_max_hb = max(
                    nested_max_hb, _max_abs(left.hb_prefix[:common], right.hb_prefix[:common])
                )
                nested_max_state = max(
                    nested_max_state, _max_abs(left.states[:common], right.states[:common])
                )
                nested_max_action = max(
                    nested_max_action,
                    _max_abs(left.action_chunks[:common], right.action_chunks[:common]),
                )
                for offset in range(common):
                    nested_noise_mismatches += int(
                        _array_sha256(left.flow_noises[offset])
                        != _array_sha256(right.flow_noises[offset])
                    )
                nested_pairs += 1

    prebranch_success_cells = int(
        sum(bool(cell.summary.get("success_before_branch", False)) for cell in cells)
    )
    valid = (
        len(observation_hashes) == 1
        and prefix_noise_mismatches == 0
        and future_noise_mismatches == 0
        and nested_noise_mismatches == 0
        and max_as == 0.0
        and max_hb == 0.0
        and max_state == 0.0
        and max_action == 0.0
        and nested_max_as == 0.0
        and nested_max_hb == 0.0
        and nested_max_state == 0.0
        and nested_max_action == 0.0
    )
    return {
        "n_cells": len(cells),
        "n_initial_observation_hashes": len(observation_hashes),
        "initial_observation_sha256": next(iter(observation_hashes))
        if len(observation_hashes) == 1
        else None,
        "same_prefix_pairs_compared": prefix_pairs,
        "same_prefix_noise_mismatches": prefix_noise_mismatches,
        "same_prefix_max_abs_as_route_difference": max_as,
        "same_prefix_max_abs_hb_route_difference": max_hb,
        "same_prefix_max_abs_state_difference": max_state,
        "same_prefix_max_abs_action_difference": max_action,
        "same_future_tensors_compared": future_tensors_compared,
        "same_future_noise_mismatches": future_noise_mismatches,
        "nested_branch_pairs_compared": nested_pairs,
        "nested_prefix_noise_mismatches": nested_noise_mismatches,
        "nested_max_abs_as_route_difference": nested_max_as,
        "nested_max_abs_hb_route_difference": nested_max_hb,
        "nested_max_abs_state_difference": nested_max_state,
        "nested_max_abs_action_difference": nested_max_action,
        "truncated_prefix_cells": truncated_prefix_cells,
        "prebranch_success_cells": prebranch_success_cells,
        "valid": valid,
    }


def _outcome_grid(
    cells: list[PrefixCell], branch: int, n_prefix: int, n_future: int
) -> np.ndarray | None:
    selected = {
        (int(cell.summary["grid_row"]), int(cell.summary["grid_col"])): cell
        for cell in cells
        if int(cell.summary["branch_controls"]) == branch
    }
    if len(selected) != n_prefix * n_future:
        return None
    outcomes = np.empty((n_prefix, n_future), dtype=np.float64)
    for row in range(n_prefix):
        for col in range(n_future):
            outcomes[row, col] = float(selected[(row, col)].summary["success"])
    return outcomes


def _load_k1_baseline(
    client_dir: pathlib.Path, n_prefix: int, n_future: int
) -> tuple[np.ndarray, dict]:
    config = json.loads((client_dir / "experiment_config.json").read_text())
    if int(config["n_first"]) < n_prefix or int(config["n_future"]) < n_future:
        raise RuntimeError("k=1 baseline grid is smaller than the prefix grid")
    summaries = json.loads((client_dir / "summaries.json").read_text())
    by_key = {
        (int(summary["grid_row"]), int(summary["grid_col"])): summary
        for summary in summaries
        if int(summary["grid_row"]) < n_prefix and int(summary["grid_col"]) < n_future
    }
    if len(by_key) != n_prefix * n_future:
        raise RuntimeError("k=1 baseline is incomplete")
    outcomes = np.empty((n_prefix, n_future), dtype=np.float64)
    for row in range(n_prefix):
        for col in range(n_future):
            outcomes[row, col] = float(by_key[(row, col)]["success"])
    return outcomes, config


def _write_report(path: pathlib.Path, analysis: dict) -> None:
    integrity = analysis["integrity"]
    lines = [
        "# Early-prefix commitment curve",
        "",
        "The first k complete control steps are replayed exactly; only policy noise from control step k onward changes.",
        "",
        "## Integrity",
        "",
        "- Cells: `%d`" % integrity["n_cells"],
        "- Initial observation hashes: `%d`" % integrity["n_initial_observation_hashes"],
        "- Same-prefix AS route max difference: `%.9g`"
        % integrity["same_prefix_max_abs_as_route_difference"],
        "- Same-prefix HB route max difference: `%.9g`"
        % integrity["same_prefix_max_abs_hb_route_difference"],
        "- Same-prefix state max difference: `%.9g`"
        % integrity["same_prefix_max_abs_state_difference"],
        "- Same-prefix action max difference: `%.9g`"
        % integrity["same_prefix_max_abs_action_difference"],
        "- Future-noise mismatches: `%d / %d`"
        % (integrity["same_future_noise_mismatches"], integrity["same_future_tensors_compared"]),
        "- Nested-prefix route/state/action differences: `%.9g / %.9g / %.9g / %.9g`"
        % (
            integrity["nested_max_abs_as_route_difference"],
            integrity["nested_max_abs_hb_route_difference"],
            integrity["nested_max_abs_state_difference"],
            integrity["nested_max_abs_action_difference"],
        ),
        "- Successes before branch: `%d`" % integrity["prebranch_success_cells"],
        "- Valid: `%s`" % integrity["valid"],
        "",
    ]

    if not analysis["complete"]:
        lines.extend(["The grid is incomplete; commitment statistics were not computed.", ""])
        path.write_text("\n".join(lines))
        return

    lines.extend(
        [
            "## Curve",
            "",
            "| branch k | fixed action steps | mixed rows | stable rows | H(Y|prefix) | route-row p | success |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in analysis["curve"]:
        stats = entry["statistics"]
        lines.append(
            "| %d | %d | %d/%d | %d/%d | %.4f | %.4f | %.3f |"
            % (
                entry["branch_controls"],
                entry["branch_controls"] * analysis["experiment_config"]["replan_steps"],
                stats["mixed_rows"],
                stats["n_first_routes"],
                stats["stable_rows"],
                stats["n_first_routes"],
                stats["conditional_entropy_bits"],
                stats["permutation"]["mutual_information_upper_p"],
                stats["overall_success_rate"],
            )
        )

    lines.extend(["", "## Outcome matrices", ""])
    for entry in analysis["curve"]:
        branch = entry["branch_controls"]
        outcomes = np.asarray(entry["outcomes"], dtype=int)
        lines.extend(["### k=%d" % branch, "", "```text"])
        for row, values in enumerate(outcomes):
            pattern = " ".join("S" if value else "F" for value in values)
            lines.append("prefix %02d: %s  p=%.3f" % (row, pattern, values.mean()))
        lines.extend(["```", ""])
    path.write_text("\n".join(lines))


def _plot_matrices(path: pathlib.Path, curve: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    cmap = ListedColormap(["#d75a4a", "#2f8f62"])
    for axis, entry in zip(axes.flat, curve):
        outcomes = np.asarray(entry["outcomes"])
        axis.imshow(outcomes, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        axis.set_title("branch k=%d" % entry["branch_controls"])
        axis.set_xlabel("future-noise stream")
        axis.set_ylabel("fixed prefix row")
        axis.set_xticks(np.arange(outcomes.shape[1]))
        axis.set_yticks(np.arange(outcomes.shape[0]))
    for axis in axes.flat[len(curve) :]:
        axis.set_axis_off()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_curve(path: pathlib.Path, curve: list[dict]) -> None:
    import matplotlib.pyplot as plt

    branches = np.asarray([entry["branch_controls"] for entry in curve])
    entropy = np.asarray([entry["statistics"]["conditional_entropy_bits"] for entry in curve])
    stable = np.asarray(
        [
            entry["statistics"]["stable_rows"] / entry["statistics"]["n_first_routes"]
            for entry in curve
        ]
    )
    row_rates = np.asarray([entry["statistics"]["row_success_rates"] for entry in curve])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    axes[0].plot(branches, entropy, marker="o", color="#4b70a7")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("branch control k")
    axes[0].set_ylabel("H(outcome | fixed prefix), bits")
    axes[0].set_title("Residual outcome uncertainty")

    axes[1].plot(branches, stable, marker="o", color="#2f8f62")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("branch control k")
    axes[1].set_ylabel("sample-stable prefix fraction")
    axes[1].set_title("Empirical commitment")

    for row in range(row_rates.shape[1]):
        axes[2].plot(branches, row_rates[:, row], marker="o", alpha=0.72, lw=1)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("branch control k")
    axes[2].set_ylabel("P(success | fixed prefix)")
    axes[2].set_title("Per-trunk continuation success")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--k1-client-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    server_dir = pathlib.Path(args.server_dir)
    client_dir = pathlib.Path(args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((client_dir / "experiment_config.json").read_text())
    branches = [int(value) for value in config["branch_controls"]]
    n_prefix = int(config["n_prefix"])
    n_future = int(config["n_future"])
    cells, route_attrs = _load_cells(server_dir, client_dir)
    integrity = _integrity(cells, branches)
    grids = {branch: _outcome_grid(cells, branch, n_prefix, n_future) for branch in branches}
    complete = all(grid is not None for grid in grids.values())

    if not integrity["valid"]:
        raise RuntimeError("prefix-grid integrity check failed: %s" % integrity)
    if not complete and not args.allow_partial:
        raise RuntimeError("prefix grid is incomplete; pass --allow-partial for integrity-only analysis")

    analysis = {
        "protocol": "early-prefix-commitment-analysis-v1",
        "experiment_config": config,
        "route_store_attrs": route_attrs,
        "integrity": integrity,
        "complete": complete,
    }

    if complete:
        entries = []
        if args.k1_client_dir:
            baseline, baseline_config = _load_k1_baseline(
                pathlib.Path(args.k1_client_dir), n_prefix, n_future
            )
            if int(baseline_config["first_seed_base"]) != int(config["prefix_seed_base"]):
                raise RuntimeError("k=1 baseline prefix seeds do not match")
            if int(baseline_config["future_seed_base"]) != int(config["future_seed_base"]):
                raise RuntimeError("k=1 baseline future seeds do not match")
            entries.append(
                {
                    "branch_controls": 1,
                    "source": str(args.k1_client_dir),
                    "outcomes": baseline.astype(int).tolist(),
                    "statistics": _commitment_statistics(
                        baseline, args.permutations, args.seed + 1
                    ),
                }
            )

        for branch in branches:
            grid = grids[branch]
            entries.append(
                {
                    "branch_controls": branch,
                    "source": str(client_dir),
                    "outcomes": grid.astype(int).tolist(),
                    "statistics": _commitment_statistics(
                        grid, args.permutations, args.seed + branch
                    ),
                }
            )
        entries.sort(key=lambda entry: entry["branch_controls"])
        analysis["curve"] = entries

    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    _write_report(out / "REPORT.md", analysis)
    if complete:
        _plot_matrices(out / "prefix_commitment_matrices.png", analysis["curve"])
        _plot_curve(out / "prefix_commitment_curve.png", analysis["curve"])

    print(
        json.dumps(
            {
                "complete": complete,
                "integrity": integrity,
                "curve": [
                    {
                        "branch_controls": entry["branch_controls"],
                        "mixed_rows": entry["statistics"]["mixed_rows"],
                        "stable_rows": entry["statistics"]["stable_rows"],
                        "conditional_entropy_bits": entry["statistics"][
                            "conditional_entropy_bits"
                        ],
                        "route_effect_p": entry["statistics"]["permutation"][
                            "mutual_information_upper_p"
                        ],
                    }
                    for entry in analysis.get("curve", [])
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
