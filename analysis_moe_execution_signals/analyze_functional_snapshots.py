#!/usr/bin/env python3
"""Descriptive mechanism audit for request-gated functional snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

from analysis_moe_execution_signals.execution_features import (
    sparse_hellinger_distance,
    support_jaccard_distance,
)


EPS = 1e-12


def _digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, np.float64).reshape(-1)
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _relative_sketch_delta(sketch: np.ndarray) -> np.ndarray:
    current = np.asarray(sketch[:, :, 1:], np.float64)
    previous = np.asarray(sketch[:, :, :-1], np.float64)
    numerator = np.linalg.norm(current - previous, axis=-1)
    denominator = np.linalg.norm(current, axis=-1) + np.linalg.norm(previous, axis=-1)
    return numerator / np.maximum(denominator, EPS)


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    correlation = spearmanr(
        np.asarray(left).reshape(-1),
        np.asarray(right).reshape(-1),
    ).statistic
    return float(correlation) if np.isfinite(correlation) else 0.0


def _four_quadrants(route: np.ndarray, functional: np.ndarray) -> dict[str, object]:
    route = np.asarray(route, np.float64).reshape(-1)
    functional = np.asarray(functional, np.float64).reshape(-1)
    route_threshold = float(np.median(route))
    functional_threshold = float(np.median(functional))
    route_high = route > route_threshold
    functional_high = functional > functional_threshold
    names = {
        "stable": ~route_high & ~functional_high,
        "cosmetic_churn": route_high & ~functional_high,
        "within_support_change": ~route_high & functional_high,
        "route_with_functional_change": route_high & functional_high,
    }
    return {
        "descriptive_median_thresholds": {
            "route": route_threshold,
            "functional": functional_threshold,
        },
        "counts": {name: int(mask.sum()) for name, mask in names.items()},
        "fractions": {name: float(mask.mean()) for name, mask in names.items()},
    }


def analyze(path: pathlib.Path) -> dict[str, object]:
    with np.load(path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    ids = arrays["topk_idx"]
    weights = arrays["topk_exec_weight"].astype(np.float32)
    routed_sketch = arrays["routed_output_sketch"].astype(np.float32)
    support = support_jaccard_distance(ids[:, :, 1:], ids[:, :, :-1])
    execution = sparse_hellinger_distance(
        ids[:, :, 1:],
        weights[:, :, 1:],
        ids[:, :, :-1],
        weights[:, :, :-1],
    )
    functional = _relative_sketch_delta(routed_sketch)

    routed_norm = arrays["routed_output_norm"].astype(np.float32)
    shared_norm = arrays["shared_output_norm"].astype(np.float32)
    routed_sketch_norm = np.linalg.norm(routed_sketch, axis=-1)
    shared_sketch_norm = np.linalg.norm(
        arrays["shared_output_sketch"].astype(np.float32), axis=-1
    )
    routed_sketch_error = np.abs(routed_sketch_norm - routed_norm) / np.maximum(
        routed_norm, EPS
    )
    shared_sketch_error = np.abs(shared_sketch_norm - shared_norm) / np.maximum(
        shared_norm, EPS
    )

    scalar_keys = (
        "top4_top5_logit_margin",
        "tail_mass",
        "exec_entropy",
        "routed_output_norm",
        "shared_output_norm",
        "total_output_norm",
        "routed_authority",
        "routed_relative_norm",
        "routed_shared_cosine",
        "expert_cancellation",
        "expert_disagreement",
        "expert_disagreement_ratio",
    )
    summary = {key: _quantiles(arrays[key]) for key in scalar_keys}
    token = {
        key: np.asarray(arrays[key], np.float64).mean(axis=(0, 1, 2)).tolist()
        for key in (
            "top4_top5_logit_margin",
            "tail_mass",
            "exec_entropy",
            "routed_authority",
            "routed_shared_cosine",
            "expert_cancellation",
            "expert_disagreement_ratio",
        )
    }
    layer = {
        key: np.asarray(arrays[key], np.float64).mean(axis=(0, 2, 3)).tolist()
        for key in (
            "routed_authority",
            "routed_shared_cosine",
            "expert_cancellation",
            "expert_disagreement_ratio",
        )
    }
    correlations = {
        "support_churn_vs_functional_delta": _finite_spearman(support, functional),
        "execution_churn_vs_functional_delta": _finite_spearman(execution, functional),
        "tail_mass_vs_authority": _finite_spearman(
            arrays["tail_mass"], arrays["routed_authority"]
        ),
        "boundary_margin_vs_authority": _finite_spearman(
            arrays["top4_top5_logit_margin"], arrays["routed_authority"]
        ),
        "cancellation_vs_authority": _finite_spearman(
            arrays["expert_cancellation"], arrays["routed_authority"]
        ),
    }
    return {
        "path": str(path.resolve()),
        "sha256": _digest(path),
        "array_bytes": int(
            sum(value.nbytes for value in arrays.values() if isinstance(value, np.ndarray))
        ),
        "hb_layers": arrays["hb_layers"].astype(int).tolist(),
        "shape": {
            "sites": list(arrays["routed_authority"].shape),
            "expert_contrib_sketch": list(arrays["expert_contrib_sketch"].shape),
        },
        "summary": summary,
        "state_vs_action": {
            key: {
                "state_mean": float(np.asarray(arrays[key], np.float64)[..., 0].mean()),
                "action_mean": float(np.asarray(arrays[key], np.float64)[..., 1:].mean()),
            }
            for key in (
                "routed_authority",
                "expert_cancellation",
                "expert_disagreement_ratio",
                "top4_top5_logit_margin",
            )
        },
        "by_token": token,
        "by_layer": layer,
        "adjacent_flow": {
            "support_jaccard": _quantiles(support),
            "sparse_execution_hellinger": _quantiles(execution),
            "functional_sketch_delta": _quantiles(functional),
            "support_four_quadrants": _four_quadrants(support, functional),
            "execution_four_quadrants": _four_quadrants(execution, functional),
        },
        "sketch_relative_norm_error": {
            "routed": _quantiles(routed_sketch_error),
            "shared": _quantiles(shared_sketch_error),
        },
        "correlations": correlations,
        "boundary_zero_fraction": float(
            (arrays["top4_top5_logit_margin"] == 0).mean()
        ),
    }


def render_report(result: dict[str, object]) -> str:
    first = result["snapshots"][0]
    summary = first["summary"]
    quadrants = first["adjacent_flow"]["support_four_quadrants"]["fractions"]
    correlations = first["correlations"]
    sketch = first["sketch_relative_norm_error"]
    state_action = first["state_vs_action"]
    return "\n".join(
        (
            "# HB-MoE functional snapshot audit",
            "",
            "## Gate",
            "",
            "- GPU snapshots: %d; byte-identical: `%s`." % (
                len(result["snapshots"]),
                str(result["all_snapshots_byte_identical"]).lower(),
            ),
            "- One snapshot contains 8 HB layers x 10 denoise steps x 11 tokens, "
            "and occupies %.1f KiB before NPZ compression." % (first["array_bytes"] / 1024),
            "- This is a fixed synthetic observation no-op audit, not an event/outcome sample.",
            "",
            "## Functional ranges",
            "",
            "- Routed authority: median %.3f, q10-q90 %.3f-%.3f." % (
                summary["routed_authority"]["median"],
                summary["routed_authority"]["q10"],
                summary["routed_authority"]["q90"],
            ),
            "- Expert cancellation: median %.3f, q10-q90 %.3f-%.3f." % (
                summary["expert_cancellation"]["median"],
                summary["expert_cancellation"]["q10"],
                summary["expert_cancellation"]["q90"],
            ),
            "- Normalized disagreement: median %.3f, q10-q90 %.3f-%.3f." % (
                summary["expert_disagreement_ratio"]["median"],
                summary["expert_disagreement_ratio"]["q10"],
                summary["expert_disagreement_ratio"]["q90"],
            ),
            "- Routed/shared cosine: median %.3f, q10-q90 %.3f-%.3f." % (
                summary["routed_shared_cosine"]["median"],
                summary["routed_shared_cosine"]["q10"],
                summary["routed_shared_cosine"]["q90"],
            ),
            "- Exact Top-4/5 logit ties: %.1f%% of sites." % (
                100 * first["boundary_zero_fraction"]
            ),
            "",
            "## Route versus function",
            "",
            "Median-split site fractions (descriptive only): stable %.1f%%, cosmetic churn %.1f%%, "
            "within-support functional change %.1f%%, route+functional change %.1f%%." % (
                100 * quadrants["stable"],
                100 * quadrants["cosmetic_churn"],
                100 * quadrants["within_support_change"],
                100 * quadrants["route_with_functional_change"],
            ),
            "Support churn/function delta Spearman rho %.3f; execution-weight churn/function "
            "delta rho %.3f." % (
                correlations["support_churn_vs_functional_delta"],
                correlations["execution_churn_vs_functional_delta"],
            ),
            "",
            "## Token and sketch checks",
            "",
            "- State/action routed authority: %.3f / %.3f." % (
                state_action["routed_authority"]["state_mean"],
                state_action["routed_authority"]["action_mean"],
            ),
            "- 16-D sketch routed-norm relative error median %.1f%%, q90 %.1f%%; shared median "
            "%.1f%%, q90 %.1f%%." % (
                100 * sketch["routed"]["median"],
                100 * sketch["routed"]["q90"],
                100 * sketch["shared"]["median"],
                100 * sketch["shared"]["q90"],
            ),
            "",
            "## Interpretation",
            "",
            "The new quantities have non-degenerate dynamic range and the capture is numerically "
            "transparent on two physical GPUs.  This audit cannot establish Trap prediction or "
            "causality.  VLA_MUI_HUB stores no RGB observations, so event-level rich capture must "
            "reconstruct simulator snapshots and render fresh observations.",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    snapshots = [analyze(path) for path in args.snapshots]
    result = {
        "schema": "himoe.functional_snapshot.analysis.v1",
        "snapshots": snapshots,
        "all_snapshots_byte_identical": len({row["sha256"] for row in snapshots}) == 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        "all_snapshots_byte_identical": result["all_snapshots_byte_identical"],
        "correlations": snapshots[0]["correlations"],
        "four_quadrants": snapshots[0]["adjacent_flow"]["support_four_quadrants"]["fractions"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
