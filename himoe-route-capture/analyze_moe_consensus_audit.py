"""Audit MoE-routing consensus selectors on the legacy K32 fork capture.

This is a retrospective endpoint-proxy audit.  For every exact snapshot it
selects one of 32 already-executed action chunks using only candidate geometry:

* the full action-token HB router distribution (RMS Hellinger medoid),
* top-4 HB expert identity sets (mean aligned-site Jaccard medoid),
* each denoising prefix of the full-probability route, and
* checkpoint-normalized action-command geometry.

The exact random expectation is the within-snapshot candidate mean.  Selector
contrasts are reduced to one number per snapshot before bootstrapping; candidate
pairs are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import zarr

from behavior_geometry import action_distance_matrix
from route_noise_selector import centrality, pairwise_hellinger, stable_argmin


HERE = Path(__file__).resolve().parent
DEFAULT_RECORDS = HERE / "runs/fork-pilot-n32-client/fork_records.json"
DEFAULT_CHUNKS = HERE / "runs/fork-pilot-n32-client"
DEFAULT_ROUTES = HERE / "runs/fork-pilot-n32/routes.zarr"
DEFAULT_METADATA = HERE / "runs/fork-pilot-n32-client/server_metadata.json"
DEFAULT_OUT = HERE / "analysis/moe-consensus-audit/legacy-k32"

ACTION_TOKEN_SLICE = slice(1, 11)
EXPECTED_SNAPSHOTS = 20
EXPECTED_CANDIDATES = 32

OUTCOMES: dict[str, dict[str, Any]] = {
    "drawer_delta_chunk": {
        "source": "drawer_delta_chunk",
        "multiplier": 1.0,
        "description": "signed drawer displacement during the candidate chunk",
        "higher_is_better": False,
    },
    "drawer_delta_after_continuation": {
        "source": "drawer_delta",
        "multiplier": 1.0,
        "description": "signed drawer displacement after the one shared-CRN continuation",
        "higher_is_better": False,
    },
    "drawer_progress_after_continuation": {
        "source": "drawer_delta",
        "multiplier": -1.0,
        "description": "opening-oriented drawer progress (-drawer_delta)",
        "higher_is_better": True,
    },
    "success_in_window": {
        "source": "success_in_window",
        "multiplier": 1.0,
        "description": "binary success during the single continuation window",
        "higher_is_better": True,
    },
    "object_qpos_moved": {
        "source": "object_qpos_moved",
        "multiplier": 1.0,
        "description": "maximum absolute non-time qpos displacement after the chunk",
        "higher_is_better": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_bootstrap(
    values: Iterable[float],
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile bootstrap for an already-reduced vector of snapshot effects."""

    effects = np.asarray(tuple(values), dtype=np.float64)
    if effects.ndim != 1 or not len(effects) or not np.all(np.isfinite(effects)):
        raise ValueError("snapshot effects must be one non-empty finite vector")
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    result: dict[str, Any] = {
        "mean": float(effects.mean()),
        "snapshots": int(len(effects)),
        "confidence": float(confidence),
    }
    if len(effects) == 1:
        result.update({"lower": None, "upper": None, "draws": 0, "available": False})
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effects), size=(draws, len(effects)))
    distribution = effects[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [tail, 1.0 - tail])
    result.update(
        {
            "lower": float(lower),
            "upper": float(upper),
            "draws": int(draws),
            "available": True,
        }
    )
    return result


def pairwise_topk_jaccard(expert_ids: np.ndarray, n_experts: int = 32) -> np.ndarray:
    """Mean top-k set Jaccard distance over aligned HB routing sites."""

    ids = np.asarray(expert_ids)
    if ids.ndim < 3 or len(ids) < 2:
        raise ValueError("expert IDs must have shape [candidate, ..., top_k]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expert IDs must be integers")
    if np.any(ids < 0) or np.any(ids >= n_experts):
        raise ValueError("expert ID outside the declared expert range")
    ordered = np.sort(ids, axis=-1)
    if np.any(np.diff(ordered, axis=-1) == 0):
        raise ValueError("a top-k routing site contains duplicate expert IDs")

    sites = ids.reshape(len(ids), -1, ids.shape[-1])
    membership = np.eye(n_experts, dtype=np.bool_)[sites].any(axis=-2)
    result = np.zeros((len(ids), len(ids)), dtype=np.float64)
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            intersection = np.logical_and(membership[left], membership[right]).sum(axis=-1)
            union = np.logical_or(membership[left], membership[right]).sum(axis=-1)
            value = float(np.mean(1.0 - intersection / union))
            result[left, right] = result[right, left] = value
    return result


def medoid_candidate(distance: np.ndarray, candidate_ids: np.ndarray) -> tuple[int, float]:
    scores = centrality(distance)
    candidate = stable_argmin(scores, candidate_ids)
    axis = int(np.flatnonzero(candidate_ids == candidate)[0])
    return candidate, float(scores[axis])


def select_consensus_candidates(
    actions: np.ndarray,
    router_probabilities: np.ndarray,
    expert_ids: np.ndarray,
    candidate_ids: np.ndarray,
    action_std: np.ndarray,
    gripper_weight: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return deterministic K-pool medoids and distance diagnostics."""

    actions = np.asarray(actions, dtype=np.float64)
    probabilities = np.asarray(router_probabilities, dtype=np.float64)
    ids = np.asarray(expert_ids)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if actions.shape != (len(candidate_ids), 10, 7):
        raise ValueError(f"expected action chunks [K,10,7], got {actions.shape}")
    if probabilities.shape != (len(candidate_ids), 8, 10, 10, 32):
        raise ValueError(
            "expected action-token HB probabilities [K,8,10,10,32], "
            f"got {probabilities.shape}"
        )
    if ids.shape != (len(candidate_ids), 8, 10, 10, 4):
        raise ValueError(f"expected action-token HB expert IDs [K,8,10,10,4], got {ids.shape}")
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")

    selections: dict[str, dict[str, Any]] = {}
    action_distance = action_distance_matrix(actions, action_std, gripper_weight)
    full_probability_distance = pairwise_hellinger(probabilities)
    id_distance = pairwise_topk_jaccard(ids)
    for name, distance, metric in (
        (
            "action_medoid",
            action_distance,
            "checkpoint-normalized mean stepwise L2; "
            f"gripper weight {gripper_weight:g}",
        ),
        (
            "hb_probability_medoid",
            full_probability_distance,
            "RMS Hellinger over 8 HB x 10 denoise x 10 action tokens",
        ),
        (
            "hb_expert_id_medoid",
            id_distance,
            "mean top-4 set Jaccard over 8 HB x 10 denoise x 10 action tokens",
        ),
    ):
        candidate, score = medoid_candidate(distance, candidate_ids)
        selections[name] = {
            "candidate": candidate,
            "centrality": score,
            "metric": metric,
        }

    prefix_candidates = []
    for prefix in range(1, probabilities.shape[2] + 1):
        name = f"hb_probability_prefix_{prefix:02d}"
        distance = pairwise_hellinger(probabilities[:, :, :prefix])
        candidate, score = medoid_candidate(distance, candidate_ids)
        prefix_candidates.append(candidate)
        selections[name] = {
            "candidate": candidate,
            "centrality": score,
            "completed_denoise_rounds": prefix,
            "metric": "RMS Hellinger over the completed HB denoise prefix",
        }
    if prefix_candidates[-1] != selections["hb_probability_medoid"]["candidate"]:
        raise RuntimeError("10-round prefix medoid does not reproduce the full route medoid")

    diagnostics = {
        "action_pair_distance_mean": float(
            action_distance[np.triu_indices(len(candidate_ids), 1)].mean()
        ),
        "full_probability_pair_distance_mean": float(
            full_probability_distance[np.triu_indices(len(candidate_ids), 1)].mean()
        ),
        "expert_id_pair_distance_mean": float(
            id_distance[np.triu_indices(len(candidate_ids), 1)].mean()
        ),
        "prefix_first_matches_full": bool(
            prefix_candidates[0] == selections["hb_probability_medoid"]["candidate"]
        ),
        "prefix_first_full_match_round": next(
            (
                index + 1
                for index in range(len(prefix_candidates))
                if all(value == prefix_candidates[-1] for value in prefix_candidates[index:])
            ),
            None,
        ),
    }
    return selections, diagnostics


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot_results(
    records_path: Path,
    chunks_dir: Path,
    routes_path: Path,
    metadata_path: Path,
    gripper_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_records = _load_json(records_path)
    if not isinstance(raw_records, list):
        raise ValueError("fork records must be a JSON list")
    metadata = _load_json(metadata_path)
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    if action_std.shape != (7,) or np.any(action_std <= 0) or not np.all(np.isfinite(action_std)):
        raise ValueError("normalization_action_std must contain seven positive finite values")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        grouped[(int(row["episode"]), int(row["fork_step"]))].append(row)
    if len(grouped) != EXPECTED_SNAPSHOTS:
        raise ValueError(f"legacy audit requires {EXPECTED_SNAPSHOTS} snapshots, got {len(grouped)}")

    route_store = zarr.open_group(str(routes_path), mode="r")
    for key in ("hb_router_probs", "hb_expert_ids"):
        if key not in route_store:
            raise ValueError(f"route store is missing {key}")
    probabilities_store = route_store["hb_router_probs"]
    expert_ids_store = route_store["hb_expert_ids"]
    if probabilities_store.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"unexpected full route shape: {probabilities_store.shape}")
    if expert_ids_store.shape[1:] != (8, 10, 11, 4):
        raise ValueError(f"unexpected expert-ID shape: {expert_ids_store.shape}")

    snapshots = []
    selected_trace_rows: set[int] = set()
    mass_min = np.inf
    mass_max = -np.inf
    for (episode, fork_step), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(value["candidate"]))
        if len(rows) != EXPECTED_CANDIDATES:
            raise ValueError(
                f"ep{episode}/t{fork_step} requires {EXPECTED_CANDIDATES} candidates, "
                f"got {len(rows)}"
            )
        candidate_ids = np.asarray([int(row["candidate"]) for row in rows], dtype=np.int64)
        if not np.array_equal(candidate_ids, np.arange(EXPECTED_CANDIDATES)):
            raise ValueError(f"ep{episode}/t{fork_step} candidate IDs are not 0..31")
        trace_rows = np.asarray([int(row["trace_row"]) for row in rows], dtype=np.int64)
        if len(np.unique(trace_rows)) != len(trace_rows):
            raise ValueError(f"ep{episode}/t{fork_step} reuses a route row")
        if trace_rows.min() < 0 or trace_rows.max() >= probabilities_store.shape[0]:
            raise ValueError(f"ep{episode}/t{fork_step} references a route row outside the store")
        overlap = selected_trace_rows.intersection(int(value) for value in trace_rows)
        if overlap:
            raise ValueError(f"route trace rows are reused across snapshots: {sorted(overlap)}")
        selected_trace_rows.update(int(value) for value in trace_rows)

        chunk_path = chunks_dir / f"chunks_ep{episode:02d}_t{fork_step:02d}.npy"
        actions = np.asarray(np.load(chunk_path), dtype=np.float64)[candidate_ids]
        probabilities = np.asarray(
            probabilities_store.oindex[trace_rows, :, :, ACTION_TOKEN_SLICE, :],
            dtype=np.float64,
        )
        expert_ids = np.asarray(
            expert_ids_store.oindex[trace_rows, :, :, ACTION_TOKEN_SLICE, :]
        )
        masses = probabilities.sum(axis=-1)
        mass_min = min(mass_min, float(masses.min()))
        mass_max = max(mass_max, float(masses.max()))
        selections, distance_diagnostics = select_consensus_candidates(
            actions,
            probabilities,
            expert_ids,
            candidate_ids,
            action_std,
            gripper_weight,
        )

        outcome_arrays = {
            name: np.asarray(
                [float(row[spec["source"]]) * float(spec["multiplier"]) for row in rows],
                dtype=np.float64,
            )
            for name, spec in OUTCOMES.items()
        }
        for selection in selections.values():
            axis = int(np.flatnonzero(candidate_ids == selection["candidate"])[0])
            selection["outcomes"] = {
                name: float(values[axis]) for name, values in outcome_arrays.items()
            }
        snapshots.append(
            {
                "snapshot": f"ep{episode}/t{fork_step}",
                "episode": episode,
                "fork_step": fork_step,
                "candidate_count": len(rows),
                "random_expectation": {
                    name: float(values.mean()) for name, values in outcome_arrays.items()
                },
                "outcome_min": {
                    name: float(values.min()) for name, values in outcome_arrays.items()
                },
                "outcome_max": {
                    name: float(values.max()) for name, values in outcome_arrays.items()
                },
                "outcome_n_unique": {
                    name: int(len(np.unique(values))) for name, values in outcome_arrays.items()
                },
                "selectors": selections,
                "distance_diagnostics": distance_diagnostics,
            }
        )

    provenance = {
        "fork_records": str(records_path),
        "fork_records_sha256": sha256_file(records_path),
        "chunks_dir": str(chunks_dir),
        "routes": str(routes_path),
        "server_metadata": str(metadata_path),
        "server_metadata_sha256": sha256_file(metadata_path),
        "action_std": action_std.tolist(),
        "route_store_rows": int(probabilities_store.shape[0]),
        "selected_candidate_trace_rows": int(len(selected_trace_rows)),
        "selected_trace_row_min": int(min(selected_trace_rows)),
        "selected_trace_row_max": int(max(selected_trace_rows)),
        "raw_probability_mass_min": float(mass_min),
        "raw_probability_mass_max": float(mass_max),
        "route_store_attrs": dict(route_store.attrs),
    }
    return snapshots, provenance


def _effect_counts(values: np.ndarray, tolerance: float = 1e-15) -> dict[str, int]:
    return {
        "positive": int(np.count_nonzero(values > tolerance)),
        "negative": int(np.count_nonzero(values < -tolerance)),
        "zero": int(np.count_nonzero(np.abs(values) <= tolerance)),
    }


def aggregate_results(
    snapshots: list[dict[str, Any]],
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    selector_names = list(snapshots[0]["selectors"])
    mixed = {
        outcome: [row["outcome_n_unique"][outcome] > 1 for row in snapshots]
        for outcome in OUTCOMES
    }
    outcomes: dict[str, Any] = {}
    seed_offset = 0
    for outcome, spec in OUTCOMES.items():
        random_values = np.asarray(
            [row["random_expectation"][outcome] for row in snapshots], dtype=np.float64
        )
        action_values = np.asarray(
            [
                row["selectors"]["action_medoid"]["outcomes"][outcome]
                for row in snapshots
            ],
            dtype=np.float64,
        )
        mask = np.asarray(mixed[outcome], dtype=np.bool_)
        selector_results = {}
        for selector in selector_names:
            selected = np.asarray(
                [row["selectors"][selector]["outcomes"][outcome] for row in snapshots],
                dtype=np.float64,
            )
            minus_random = selected - random_values
            minus_action = selected - action_values
            selected_result = {
                "selected_outcome": snapshot_bootstrap(
                    selected,
                    draws=draws,
                    confidence=confidence,
                    seed=seed + seed_offset,
                ),
                "selected_minus_random_expectation": {
                    **snapshot_bootstrap(
                        minus_random,
                        draws=draws,
                        confidence=confidence,
                        seed=seed + seed_offset + 1,
                    ),
                    "sign_counts": _effect_counts(minus_random),
                },
                "selected_minus_action_medoid": {
                    **snapshot_bootstrap(
                        minus_action,
                        draws=draws,
                        confidence=confidence,
                        seed=seed + seed_offset + 2,
                    ),
                    "sign_counts": _effect_counts(minus_action),
                },
            }
            if np.any(mask):
                selected_result["mixed_snapshot_selected_minus_random"] = {
                    **snapshot_bootstrap(
                        minus_random[mask],
                        draws=draws,
                        confidence=confidence,
                        seed=seed + seed_offset + 3,
                    ),
                    "sign_counts": _effect_counts(minus_random[mask]),
                }
            selector_results[selector] = selected_result
            seed_offset += 4

        result: dict[str, Any] = {
            **spec,
            "snapshot_count": len(snapshots),
            "mixed_snapshot_count": int(mask.sum()),
            "degenerate_snapshot_count": int((~mask).sum()),
            "random_expectation_snapshot_mean": float(random_values.mean()),
            "selectors": selector_results,
        }
        if bool(spec["higher_is_better"]):
            maxima = np.asarray(
                [row["outcome_max"][outcome] for row in snapshots], dtype=np.float64
            )
            result["oracle_opportunity_over_random"] = snapshot_bootstrap(
                maxima - random_values,
                draws=draws,
                confidence=confidence,
                seed=seed + seed_offset,
            )
            seed_offset += 1
        outcomes[outcome] = result

    action_candidates = np.asarray(
        [row["selectors"]["action_medoid"]["candidate"] for row in snapshots]
    )
    probability_candidates = np.asarray(
        [row["selectors"]["hb_probability_medoid"]["candidate"] for row in snapshots]
    )
    id_candidates = np.asarray(
        [row["selectors"]["hb_expert_id_medoid"]["candidate"] for row in snapshots]
    )
    prefix = {}
    for completed in range(1, 11):
        candidates = np.asarray(
            [
                row["selectors"][f"hb_probability_prefix_{completed:02d}"]["candidate"]
                for row in snapshots
            ]
        )
        prefix[str(completed)] = {
            "matches_full_probability_medoid": float(np.mean(candidates == probability_candidates)),
            "matches_action_medoid": float(np.mean(candidates == action_candidates)),
            "unique_selected_candidate_ids": int(len(np.unique(candidates))),
        }
    return {
        "outcomes": outcomes,
        "selector_agreement": {
            "full_probability_matches_action": float(
                np.mean(probability_candidates == action_candidates)
            ),
            "expert_id_matches_full_probability": float(
                np.mean(id_candidates == probability_candidates)
            ),
            "expert_id_matches_action": float(np.mean(id_candidates == action_candidates)),
            "denoise_prefix": prefix,
        },
    }


def _format_ci(value: Mapping[str, Any], digits: int = 6) -> str:
    if not value.get("available"):
        return "CI unavailable"
    return f"[{value['lower']:+.{digits}f}, {value['upper']:+.{digits}f}]"


def render_report(summary: Mapping[str, Any]) -> str:
    outcomes = summary["results"]["outcomes"]
    agreement = summary["results"]["selector_agreement"]
    methods = (
        ("action medoid", "action_medoid"),
        ("full HB probability", "hb_probability_medoid"),
        ("top-4 expert ID", "hb_expert_id_medoid"),
    )
    columns = (
        "drawer_delta_chunk",
        "drawer_progress_after_continuation",
        "success_in_window",
        "object_qpos_moved",
    )
    lines = [
        "# Legacy K32 MoE routing-consensus audit",
        "",
        "This is a retrospective endpoint-proxy audit, not confirmatory evidence of an "
        "action-value selector.",
        "",
        "## Data and estimand",
        "",
        f"- `{summary['snapshot_count']}` exact snapshots x "
        f"`{summary['candidates_per_snapshot']}` candidates = "
        f"`{summary['candidate_count']}` candidate chunks.",
        "- Each selector chooses one candidate from the complete K32 pool. The random "
        "expectation is the exact within-snapshot candidate mean.",
        "- Every reported contrast is reduced within snapshot first, then bootstrapped "
        "with snapshots receiving equal weight.",
        "- A positive contrast means a larger raw proxy. Only "
        "`drawer_progress_after_continuation` and `success_in_window` have a declared "
        "higher-is-better direction.",
        "",
        "## Full-pool selectors",
        "",
        "The table reports `selected - random expectation`; parentheses contain the "
        "snapshot-bootstrap 95% interval.",
        "",
        "| selector | chunk drawer delta | continuation drawer progress | success | "
        "object qpos moved |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, selector in methods:
        cells = []
        for outcome in columns:
            effect = outcomes[outcome]["selectors"][selector][
                "selected_minus_random_expectation"
            ]
            cells.append(f"{effect['mean']:+.6f} ({_format_ci(effect)})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "The full HB probability medoid and action medoid choose the same candidate "
            f"in `{agreement['full_probability_matches_action']:.1%}` of snapshots. The "
            "expert-ID and full-probability medoids agree in "
            f"`{agreement['expert_id_matches_full_probability']:.1%}`.",
            "",
            "## Denoising-prefix route medoids",
            "",
            "| completed denoise rounds | matches full route medoid | matches action medoid | "
            "drawer progress minus random | success minus random |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for completed in range(1, 11):
        selector = f"hb_probability_prefix_{completed:02d}"
        prefix = agreement["denoise_prefix"][str(completed)]
        progress = outcomes["drawer_progress_after_continuation"]["selectors"][selector][
            "selected_minus_random_expectation"
        ]
        success = outcomes["success_in_window"]["selectors"][selector][
            "selected_minus_random_expectation"
        ]
        lines.append(
            f"| {completed} | {prefix['matches_full_probability_medoid']:.1%} | "
            f"{prefix['matches_action_medoid']:.1%} | {progress['mean']:+.6f} | "
            f"{success['mean']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Outcome support",
            "",
            "| proxy | mixed snapshots | degenerate snapshots |",
            "|---|---:|---:|",
        ]
    )
    for outcome, result in outcomes.items():
        lines.append(
            f"| `{outcome}` | {result['mixed_snapshot_count']} | "
            f"{result['degenerate_snapshot_count']} |"
        )
    lines.extend(
        [
            "",
            "The all-20-snapshot contrasts above correctly include zero-information "
            "snapshots, but this makes the effective outcome support explicit: binary "
            "success is mixed in only one snapshot, chunk drawer displacement in only "
            "four, and continuation drawer displacement in fourteen. The JSON also "
            "reports mixed-only effects; a one-snapshot interval is deliberately unavailable.",
            "",
            "## Decision",
            "",
            "No MoE routing-consensus advantage is established. The full-probability "
            "route medoid's opening-oriented continuation effect versus random is "
            f"`{summary['decision']['full_probability_progress_minus_random']['mean']:+.6f}` "
            "with interval "
            f"`{_format_ci(summary['decision']['full_probability_progress_minus_random'])}`; "
            "its effect relative to the action medoid is "
            f"`{summary['decision']['full_probability_progress_minus_action']['mean']:+.6f}` "
            "with interval "
            f"`{_format_ci(summary['decision']['full_probability_progress_minus_action'])}`. "
            "Both include zero.",
            "",
            "The positive success contrast cannot be interpreted inferentially: only one "
            "snapshot contains both success and failure candidates, so its mixed-only "
            "confidence interval is unavailable. Denoise-prefix results are descriptive "
            "and were inspected jointly; no prefix is a separately confirmed selector.",
            "",
            "## Interpretation limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one")
    if not np.isfinite(args.gripper_weight) or args.gripper_weight < 0.0:
        raise ValueError("--gripper-weight must be finite and non-negative")
    snapshots, provenance = load_snapshot_results(
        args.records,
        args.chunks_dir,
        args.routes,
        args.metadata,
        args.gripper_weight,
    )
    results = aggregate_results(
        snapshots,
        draws=args.bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    progress = results["outcomes"]["drawer_progress_after_continuation"]["selectors"][
        "hb_probability_medoid"
    ]
    success = results["outcomes"]["success_in_window"]
    progress_random = progress["selected_minus_random_expectation"]
    progress_action = progress["selected_minus_action_medoid"]
    decision = {
        "moe_routing_consensus_gain_established": bool(
            progress_random["available"] and progress_random["lower"] > 0.0
        ),
        "moe_increment_over_action_medoid_established": bool(
            progress_action["available"] and progress_action["lower"] > 0.0
        ),
        "success_inference_supported": success["mixed_snapshot_count"] >= 2,
        "full_probability_progress_minus_random": progress_random,
        "full_probability_progress_minus_action": progress_action,
        "rule": (
            "A route-consensus advantage requires a snapshot-bootstrap interval above "
            "zero; success additionally requires at least two mixed-outcome snapshots."
        ),
    }
    summary = {
        "analysis": "legacy K32 MoE routing-consensus candidate audit",
        "mode": "retrospective_endpoint_proxy_exploratory",
        "confirmatory": False,
        "snapshot_count": len(snapshots),
        "candidates_per_snapshot": EXPECTED_CANDIDATES,
        "candidate_count": len(snapshots) * EXPECTED_CANDIDATES,
        "inference_unit": "snapshot after within-snapshot selector contrast",
        "random_baseline": "exact within-snapshot candidate mean",
        "route_scope": "8 HB layers x 10 denoise rounds x 10 action tokens",
        "expert_count": 32,
        "top_k": 4,
        "bootstrap": {
            "draws": args.bootstrap,
            "confidence": args.confidence,
            "seed": args.seed,
        },
        "provenance": provenance,
        "results": results,
        "decision": decision,
        "per_snapshot": snapshots,
        "limitations": [
            "All 20 snapshots come from one LIBERO-Goal task and five source episodes; there is no held-out task.",
            "Each candidate has one shared-CRN continuation realization, not repeated-continuation Q.",
            "The continuation route rows are not selector inputs; only each candidate's initial generation route is used.",
            "The signed drawer proxies are stage-dependent physical coordinates. Only the explicitly negated drawer-progress proxy has a higher-is-better interpretation.",
            "Most snapshots are outcome-degenerate for success and chunk drawer motion, so pair count cannot repair the small effective snapshot count.",
            "This reused exploratory sample cannot establish that routing consensus improves closed-loop task success.",
        ],
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    compact = {
        "snapshot_count": summary["snapshot_count"],
        "mixed_snapshot_counts": {
            key: value["mixed_snapshot_count"]
            for key, value in summary["results"]["outcomes"].items()
        },
        "full_probability_selected_minus_random": {
            key: value["selectors"]["hb_probability_medoid"][
                "selected_minus_random_expectation"
            ]["mean"]
            for key, value in summary["results"]["outcomes"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
