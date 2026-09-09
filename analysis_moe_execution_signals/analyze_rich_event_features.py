#!/usr/bin/env python3
"""Run the frozen matched analysis for rich HB-MoE event snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_moe_execution_signals.execution_features import (
    sparse_hellinger_distance,
    support_jaccard_distance,
)


SCHEMA = "himoe.rich_event_functional_analysis.v1"
FEATURES = (
    "routed_authority",
    "expert_cancellation",
    "expert_disagreement_ratio",
    "routed_shared_cosine",
    "functional_flow_delta",
)
SECONDARY_SCALARS = (
    "top4_top5_logit_margin",
    "tail_mass",
    "exec_entropy",
)
LEADS = (-4, -2)
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_flow_delta(sketch: np.ndarray) -> np.ndarray:
    value = np.asarray(sketch, np.float64)
    current = value[:, :, 1:, 1:, :]
    previous = value[:, :, :-1, 1:, :]
    numerator = np.linalg.norm(current - previous, axis=-1)
    denominator = np.linalg.norm(current, axis=-1) + np.linalg.norm(
        previous, axis=-1
    )
    return numerator / np.maximum(denominator, EPS)


def extract_features(arrays: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Extract preregistered scalars and descriptive layer-token maps."""
    required = set(FEATURES[:-1]) | set(SECONDARY_SCALARS) | {
        "routed_output_sketch",
        "topk_idx",
        "topk_exec_weight",
        "hb_layers",
        "n_denoise",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise RuntimeError("snapshot is missing arrays: %s" % missing)
    scalar_shape = (1, 8, 10, 11)
    for key in set(FEATURES[:-1]) | set(SECONDARY_SCALARS):
        if arrays[key].shape != scalar_shape:
            raise RuntimeError("unexpected %s shape: %s" % (key, arrays[key].shape))
    if arrays["routed_output_sketch"].shape != (1, 8, 10, 11, 32):
        raise RuntimeError(
            "unexpected routed_output_sketch shape: %s"
            % (arrays["routed_output_sketch"].shape,)
        )
    if arrays["topk_idx"].shape != (1, 8, 10, 11, 4):
        raise RuntimeError("unexpected topk_idx shape")
    if arrays["topk_exec_weight"].shape != (1, 8, 10, 11, 4):
        raise RuntimeError("unexpected topk_exec_weight shape")
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fc":
            if not np.isfinite(value).all():
                raise RuntimeError("non-finite values in %s" % key)

    maps: dict[str, np.ndarray] = {}
    features: dict[str, float] = {}
    for key in FEATURES[:-1] + SECONDARY_SCALARS:
        # B,L,D,T -> L,T after excluding the state token.
        layer_token = np.asarray(arrays[key], np.float64)[:, :, :, 1:].mean(
            axis=(0, 2)
        )
        maps[key] = layer_token
        features[key] = float(layer_token[-4:].mean())
    functional_sites = _relative_flow_delta(arrays["routed_output_sketch"])
    functional_map = functional_sites.mean(axis=(0, 2))
    maps["functional_flow_delta"] = functional_map
    features["functional_flow_delta"] = float(functional_map[-4:].mean())

    ids = arrays["topk_idx"][:, :, :, 1:]
    weights = np.asarray(arrays["topk_exec_weight"], np.float32)[:, :, :, 1:]
    maps["support_flow_delta_sites"] = support_jaccard_distance(
        ids[:, :, 1:], ids[:, :, :-1]
    )
    maps["execution_flow_delta_sites"] = sparse_hellinger_distance(
        ids[:, :, 1:],
        weights[:, :, 1:],
        ids[:, :, :-1],
        weights[:, :, :-1],
    )
    maps["functional_flow_delta_sites"] = functional_sites
    return features, maps


def _load_snapshot(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: int(row["row_id"]))
    row_ids = [int(row["row_id"]) for row in records]
    if row_ids != list(range(len(records))):
        raise RuntimeError("capture records must have contiguous unique row IDs")
    return records


def _pooled_auc(event: np.ndarray, control: np.ndarray) -> float:
    values = np.concatenate((control, event))
    ranks = rankdata(values, method="average")
    n_control = len(control)
    n_event = len(event)
    rank_sum_event = ranks[n_control:].sum()
    return float(
        (rank_sum_event - n_event * (n_event + 1) / 2.0)
        / (n_event * n_control)
    )


def _cell_summary(
    event: np.ndarray,
    control: np.ndarray,
    *,
    bootstrap_rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int]:
    event = np.asarray(event, np.float64)
    control = np.asarray(control, np.float64)
    difference = event - control
    n = len(difference)
    standard_deviation = float(difference.std(ddof=1))
    mean = float(difference.mean())
    if standard_deviation <= EPS:
        statistic = float(np.sign(mean) * np.inf) if abs(mean) > EPS else 0.0
        effect = statistic
    else:
        statistic = mean / (standard_deviation / np.sqrt(n))
        effect = mean / standard_deviation
    indices = bootstrap_rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = difference[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "n_pairs": n,
        "event_mean": float(event.mean()),
        "control_mean": float(control.mean()),
        "paired_mean_difference": mean,
        "paired_median_difference": float(np.median(difference)),
        "paired_standardized_effect": effect,
        "paired_t_statistic": statistic,
        "event_greater_fraction": float((difference > 0).mean()),
        "pooled_event_high_auc": _pooled_auc(event, control),
        "bootstrap_mean_ci95_low": float(lower),
        "bootstrap_mean_ci95_high": float(upper),
    }


def sign_flip_max_t(
    differences: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint paired sign-flip inference with missing cells represented by NaN."""
    values = np.asarray(differences, np.float64)
    if values.ndim != 2:
        raise ValueError("differences must be pair x cell")
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.asarray((-1.0, 1.0), np.float64),
        size=(n_permutations, values.shape[0]),
    )
    observed = np.empty(values.shape[1], np.float64)
    permuted = np.empty((n_permutations, values.shape[1]), np.float64)
    for cell in range(values.shape[1]):
        finite = np.isfinite(values[:, cell])
        sample = values[finite, cell]
        if len(sample) < 2:
            raise RuntimeError("each primary cell requires at least two pairs")
        observed_sd = sample.std(ddof=1)
        observed[cell] = (
            sample.mean() / (observed_sd / np.sqrt(len(sample)))
            if observed_sd > EPS
            else 0.0
        )
        signed = signs[:, finite] * sample[None, :]
        means = signed.mean(axis=1)
        sum_squares = np.square(sample).sum()
        variances = (sum_squares - len(sample) * np.square(means)) / (
            len(sample) - 1
        )
        standard_errors = np.sqrt(np.maximum(variances, 0.0) / len(sample))
        permuted[:, cell] = np.divide(
            means,
            standard_errors,
            out=np.zeros_like(means),
            where=standard_errors > EPS,
        )
    absolute = np.abs(permuted)
    maximum = absolute.max(axis=1)
    raw = np.asarray(
        [
            (1 + np.count_nonzero(absolute[:, cell] >= abs(observed[cell])))
            / (n_permutations + 1)
            for cell in range(values.shape[1])
        ]
    )
    max_t = np.asarray(
        [
            (1 + np.count_nonzero(maximum >= abs(observed[cell])))
            / (n_permutations + 1)
            for cell in range(values.shape[1])
        ]
    )
    return raw, max_t


def _finite_spearman(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    statistic = spearmanr(
        np.concatenate([np.asarray(value).reshape(-1) for value in left]),
        np.concatenate([np.asarray(value).reshape(-1) for value in right]),
    ).statistic
    return float(statistic) if np.isfinite(statistic) else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute != 0 and (absolute < 1e-3 or absolute >= 1e3):
        return "%.3g" % value
    return "%.4f" % value


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Rich HB-MoE matched event pilot",
        "",
        "## Validity gates",
        "",
        "- Functional capture: `%s`; recorder-transparent rows: %d/%d."
        % (
            str(result["capture"]["passed"]).lower(),
            result["capture"]["capture_transparent_rows"],
            result["capture"]["rows"],
        ),
        "- GPU power limits: all selected cards were at their hardware maximum; "
        "OOM kills during capture: %d -> %d."
        % (
            result["capture"]["preflight_memory"]["oom_kill"],
            result["capture"]["postflight_memory"]["oom_kill"],
        ),
        "- Restoration-qualified rows: %d/%d; complete pairs: %s."
        % (
            result["capture"]["restore_pass_rows"],
            result["capture"]["rows"],
            ", ".join(
                "%+d=%d" % (int(lead), count)
                for lead, count in result["complete_pairs_by_lead"].items()
            ),
        ),
        "- Historical action reproduction is diagnostic only: exact %d/%d, "
        "maximum absolute difference %.4f."
        % (
            result["capture"]["original_action_bitwise_rows"],
            result["capture"]["rows"],
            result["capture"]["original_action_max_abs_error"],
        ),
        "",
        "## Preregistered results",
        "",
        "| Lead | Feature | N | Event-control | Paired d | AUC | raw p | maxT p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["primary"]:
        lines.append(
            "| %+d | `%s` | %d | %s | %s | %.3f | %.4g | %.4g |"
            % (
                row["lead"],
                row["feature"],
                row["n_pairs"],
                _format_number(row["paired_mean_difference"]),
                _format_number(row["paired_standardized_effect"]),
                row["pooled_event_high_auc"],
                row["permutation_p_raw"],
                row["permutation_p_max_t"],
            )
        )
    significant = result["max_t_significant_cells"]
    lines.extend(
        (
            "",
            "## Conclusion",
            "",
            (
                "- %d/%d primary cells pass family-wise maxT at 0.05: %s."
                % (
                    len(significant),
                    len(result["primary"]),
                    ", ".join(
                        "%s@%+d" % (row["feature"], row["lead"])
                        for row in significant
                    )
                    or "none",
                )
            ),
            "- This is a frozen-state observational result, not evidence that "
            "rerouting improves control or that a token prefix is safe.",
            "- Layer/token maps and route-function correlations are saved as "
            "descriptive secondary outputs only.",
            "",
        )
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol_sha256 = _sha256(args.protocol)
    capture = json.loads(args.capture_summary.read_text(encoding="utf-8"))
    if not capture.get("passed"):
        raise RuntimeError("capture summary did not pass")
    records = _load_records(args.records)
    if len(records) != int(capture["rows"]):
        raise RuntimeError("capture summary and record counts disagree")
    if not all(row["capture_transparent_bitwise"] for row in records):
        raise RuntimeError("non-transparent capture row found")

    row_features: dict[int, dict[str, float]] = {}
    row_maps: dict[int, dict[str, np.ndarray]] = {}
    support_sites: list[np.ndarray] = []
    execution_sites: list[np.ndarray] = []
    functional_sites: list[np.ndarray] = []
    expected_hb_layers: np.ndarray | None = None
    for record in records:
        row_id = int(record["row_id"])
        snapshot = Path(record["snapshot"])
        if _sha256(snapshot) != record["snapshot_sha256"]:
            raise RuntimeError("snapshot digest mismatch at row %d" % row_id)
        arrays = _load_snapshot(snapshot)
        if int(arrays["episode_id"]) != int(record["global_episode"]):
            raise RuntimeError("snapshot episode metadata mismatch")
        if int(arrays["control_step"]) != int(record["query_index"]):
            raise RuntimeError("snapshot control-step metadata mismatch")
        if int(arrays["n_denoise"]) != 10:
            raise RuntimeError("unexpected denoise count")
        if not np.allclose(
            np.asarray(arrays["topk_exec_weight"], np.float32).sum(axis=-1),
            1.0,
            atol=2e-3,
        ):
            raise RuntimeError("Top-4 execution weights are not normalized")
        for bounded_key in (
            "tail_mass",
            "routed_authority",
            "expert_cancellation",
            "expert_disagreement_ratio",
        ):
            bounded = np.asarray(arrays[bounded_key], np.float32)
            if np.any(bounded < -2e-3) or np.any(bounded > 1.0 + 2e-3):
                raise RuntimeError("out-of-range values in %s" % bounded_key)
        hb_layers = np.asarray(arrays["hb_layers"], np.int16)
        if hb_layers.shape != (8,) or len(np.unique(hb_layers)) != 8:
            raise RuntimeError("expected eight unique HB layer identities")
        if expected_hb_layers is None:
            expected_hb_layers = hb_layers.copy()
        elif not np.array_equal(hb_layers, expected_hb_layers):
            raise RuntimeError("HB layer identities differ between snapshots")
        features, maps = extract_features(arrays)
        row_features[row_id] = features
        row_maps[row_id] = maps
        if record["restore_pass"]:
            support_sites.append(maps["support_flow_delta_sites"])
            execution_sites.append(maps["execution_flow_delta_sites"])
            functional_sites.append(maps["functional_flow_delta_sites"])

    pair_ids = sorted({int(row["pair_id"]) for row in records})
    cells = [(lead, feature) for lead in LEADS for feature in FEATURES]
    differences = np.full((len(pair_ids), len(cells)), np.nan, np.float64)
    primary: list[dict[str, Any]] = []
    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    complete_pairs_by_lead: dict[str, int] = {}
    secondary_payload: dict[str, np.ndarray] = {}
    by_key = {
        (int(row["pair_id"]), int(row["relative_query"]), bool(row["event"])): row
        for row in records
    }
    matched_comparisons_same_gpu = all(
        int(by_key[(pair_id, lead, True)]["physical_gpu"])
        == int(by_key[(pair_id, lead, False)]["physical_gpu"])
        for pair_id in pair_ids
        for lead in LEADS
    )
    if not matched_comparisons_same_gpu:
        raise RuntimeError("an event/control comparison crosses physical GPUs")
    for lead in LEADS:
        valid_pairs = []
        for pair_id in pair_ids:
            event = by_key[(pair_id, lead, True)]
            control = by_key[(pair_id, lead, False)]
            if event["restore_pass"] and control["restore_pass"]:
                valid_pairs.append(pair_id)
        complete_pairs_by_lead[str(lead)] = len(valid_pairs)
        for feature in FEATURES:
            events = np.asarray(
                [row_features[int(by_key[(pair, lead, True)]["row_id"])][feature] for pair in valid_pairs]
            )
            controls = np.asarray(
                [row_features[int(by_key[(pair, lead, False)]["row_id"])][feature] for pair in valid_pairs]
            )
            cell_index = cells.index((lead, feature))
            for pair, difference in zip(valid_pairs, events - controls, strict=True):
                differences[pair_ids.index(pair), cell_index] = difference
            summary = _cell_summary(
                events,
                controls,
                bootstrap_rng=bootstrap_rng,
                n_bootstrap=args.bootstrap,
            )
            primary.append({"lead": lead, "feature": feature, **summary})

            event_maps = np.stack(
                [row_maps[int(by_key[(pair, lead, True)]["row_id"])][feature] for pair in valid_pairs]
            )
            control_maps = np.stack(
                [row_maps[int(by_key[(pair, lead, False)]["row_id"])][feature] for pair in valid_pairs]
            )
            prefix = "%s_lead_%s" % (feature, str(lead).replace("-", "m"))
            secondary_payload[prefix + "_event_mean"] = event_maps.mean(axis=0)
            secondary_payload[prefix + "_control_mean"] = control_maps.mean(axis=0)
            secondary_payload[prefix + "_paired_difference"] = (
                event_maps - control_maps
            ).mean(axis=0)

    raw_p, max_t_p = sign_flip_max_t(
        differences,
        n_permutations=args.permutations,
        seed=args.permutation_seed,
    )
    for index, row in enumerate(primary):
        row["permutation_p_raw"] = float(raw_p[index])
        row["permutation_p_max_t"] = float(max_t_p[index])
        row["max_t_significant_0_05"] = bool(max_t_p[index] <= 0.05)

    row_payload: dict[str, np.ndarray] = {
        "row_id": np.asarray([row["row_id"] for row in records], np.int32),
        "pair_id": np.asarray([row["pair_id"] for row in records], np.int16),
        "event": np.asarray([row["event"] for row in records], bool),
        "lead": np.asarray([row["relative_query"] for row in records], np.int8),
        "restore_pass": np.asarray([row["restore_pass"] for row in records], bool),
    }
    for feature in FEATURES + SECONDARY_SCALARS:
        row_payload[feature] = np.asarray(
            [row_features[int(row["row_id"])][feature] for row in records],
            np.float64,
        )
    np.savez_compressed(args.row_features, **row_payload)
    np.savez_compressed(args.secondary_maps, **secondary_payload)

    correlations = {
        "support_churn_vs_functional_delta": _finite_spearman(
            support_sites, functional_sites
        ),
        "execution_churn_vs_functional_delta": _finite_spearman(
            execution_sites, functional_sites
        ),
    }
    significant = [
        {"lead": row["lead"], "feature": row["feature"]}
        for row in primary
        if row["max_t_significant_0_05"]
    ]
    result = {
        "schema": SCHEMA,
        "passed": bool(
            len(primary) == len(cells)
            and all(row["n_pairs"] == 23 for row in primary)
            and np.all(max_t_p >= raw_p)
        ),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "capture_summary": str(args.capture_summary.resolve()),
        "capture_summary_sha256": _sha256(args.capture_summary),
        "capture": capture,
        "features": list(FEATURES),
        "hb_layers": expected_hb_layers.astype(int).tolist(),
        "leads": list(LEADS),
        "complete_pairs_by_lead": complete_pairs_by_lead,
        "matched_comparisons_same_gpu": matched_comparisons_same_gpu,
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "primary": primary,
        "max_t_significant_cells": significant,
        "secondary_correlations": correlations,
        "row_features": str(args.row_features.resolve()),
        "secondary_maps": str(args.secondary_maps.resolve()),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.primary_csv, primary)
    args.report.write_text(_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=directory / "RICH_EVENT_PROTOCOL.md"
    )
    parser.add_argument(
        "--capture-summary",
        type=Path,
        default=directory / "rich_event_functional_32d" / "summary.json",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=directory / "rich_event_functional_32d" / "records.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=directory / "rich_event_analysis_summary.json"
    )
    parser.add_argument(
        "--primary-csv", type=Path, default=directory / "rich_event_primary.csv"
    )
    parser.add_argument(
        "--row-features",
        type=Path,
        default=directory / "rich_event_row_features.npz",
    )
    parser.add_argument(
        "--secondary-maps",
        type=Path,
        default=directory / "rich_event_secondary_maps.npz",
    )
    parser.add_argument(
        "--report", type=Path, default=directory / "rich_event_analysis_report.md"
    )
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--permutation-seed", type=int, default=20260905)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
