#!/usr/bin/env python3
"""Build a complete, non-inferential inventory of MoE analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    family: str
    pattern: str


RULES = (
    Rule(
        "meta_audit",
        r"^(AUDIT-|_repro_logs$|moe-current-data-audit$|routing-go-no-go|"
        r"routing-organization-synthesis$|failure-type-(method-review|synthesis)$)",
    ),
    Rule(
        "rolling_or_early_outcome",
        r"^(initial-state-early|t34-closure$|full-geometry-routing-ladder$|"
        r"single-chunk-early-signal$|moe-rollout-trend$|commitment-grid-s24$|"
        r"prefix-commitment-k6$)",
    ),
    Rule(
        "capture_and_fidelity",
        r"^(fork-|libero30-|slice-control|objstate-|preaction64|prefix-commitment|"
        r"commitment-grid-s24-pilot$|within64-|routing-cloud-|flow-lead(-cpu)?$|"
        r"k32-route-trajectories$|branch-replay$)",
    ),
    Rule(
        "router_structure_and_numeric",
        r"^(router-audit$|near-tie$|router-quantization|graph-phase$|"
        r"graph-separability$|moe-connectivity-graph$|routing-graph|"
        r"moe-action-decision-formation$)",
    ),
    Rule(
        "routing_dynamics_and_state_readout",
        r"^(control-route-speed$|markov-routing|intraquery-|unsupervised-moe-dynamics$|"
        r"weighted-noise-evolution$|cumulative-moe-prefix$|state-route-layer-denoise|"
        r"cross-task-layer-denoise|state-routing-concept-matrix$|moe-state-impact$)",
    ),
    Rule(
        "expert_mechanism_and_output",
        r"^(expert-contribution$|expert-activation|raw-expert-amplitude$|"
        r"unweighted-expert-norm$|absolute-expert-tension$|moe-svd$|"
        r"fixed-expert-flow-displacement$|raw-size-hidden-matched$|"
        r"offline-vector-screen$)",
    ),
    Rule(
        "intervention_and_pruning",
        r"^(branch-ablation$|expert-swap$|raw-pruning|raw-internal-activation|"
        r"internal-mlp-activation|runtime-scalar|runtime-vector|route-prune-vote$|"
        r"prune-curve$)",
    ),
    Rule(
        "candidate_selection_and_budget",
        r"^(action-majority-vote$|adaptive-k$|budget-gain$|oracle-gate$|"
        r"route-noise-selector$|self-supervised-noise-methods$|early-action-head$|"
        r"moe-value-signal$|moe-consensus-audit$|candidate-level-soft-routing$|"
        r"behavior-geometry|action-commitment-pilot$)",
    ),
    Rule(
        "denoise_stopping",
        r"^(adaptive-denoise-stop|denoise-stop-sweep|unsupervised-stop-signals|"
        r"unsupervised-denoise-validation$|value-location$)",
    ),
    Rule(
        "physical_failure_labels",
        r"^(failure-behavior-taxonomy$|failure-event-audit$|"
        r"residual-failure-physical-audit|residual-failure-subtypes)",
    ),
    Rule(
        "failure_routing_and_recovery",
        r"^(failure-routing-clusters|all-outcome-routing-clusters|aligned-route-kernel|"
        r"alternative-routing-organizations|routing-state-grammar|route-change-events|"
        r"residual-failure-dynamics|residual-failure-moe-atlas|failure-moe-signatures|"
        r"failure-rollout-periodicity$|unsupervised-replanning-loops|"
        r"within-task-unsupervised-loops|replanning-reset-trap$|post-error-|"
        r"failure-type-)",
    ),
)


def family_for(name: str) -> str:
    for rule in RULES:
        if re.search(rule.pattern, name):
            return rule.family
    return "unclassified"


def canonical_name(name: str) -> str:
    name = re.sub(r"-repro-seed\d+$", "", name)
    name = re.sub(r"-repro(?:-chained)?$", "", name)
    name = re.sub(r"-pca32-20260828$", "-20260828", name)
    return name


def file_count_and_size(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def reports_under(path: Path, root: Path) -> list[str]:
    reports = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        lower = item.name.lower()
        if lower in {"report.md", "report_zh.md"} or lower.endswith("report.md"):
            reports.append(str(item.relative_to(root)))
    return sorted(reports)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    analysis = root / "analysis"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for directory in sorted(
        path
        for path in analysis.iterdir()
        if path.is_dir() and path.resolve() != out_dir
    ):
        files, size = file_count_and_size(directory)
        reports = reports_under(directory, root)
        summaries = sorted(
            str(item.relative_to(root)) for item in directory.rglob("summary.json")
        )
        rows.append(
            {
                "directory": directory.name,
                "canonical_name": canonical_name(directory.name),
                "audit_family": family_for(directory.name),
                "is_reproduction_or_sensitivity": bool(
                    re.search(r"repro|pca32", directory.name)
                ),
                "file_count": files,
                "bytes": size,
                "report_count": len(reports),
                "summary_count": len(summaries),
                "reports": ";".join(reports),
                "summaries": ";".join(summaries),
            }
        )

    csv_path = out_dir / "directory_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    family_counts: dict[str, int] = {}
    canonical_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["audit_family"]] = family_counts.get(
            row["audit_family"], 0
        ) + 1
        canonical_counts[row["canonical_name"]] = canonical_counts.get(
            row["canonical_name"], 0
        ) + 1

    legacy_reports = sorted(
        str(path.relative_to(root.parent))
        for path in root.parent.glob("himoe-vla-*.md")
    )
    payload = {
        "schema": "himoe-moe-experiment-inventory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_directory_count": len(rows),
        "canonical_directory_count": len(canonical_counts),
        "reproduction_or_sensitivity_directory_count": sum(
            bool(row["is_reproduction_or_sensitivity"]) for row in rows
        ),
        "directories_with_report": sum(row["report_count"] > 0 for row in rows),
        "directories_with_summary": sum(row["summary_count"] > 0 for row in rows),
        "family_counts": dict(sorted(family_counts.items())),
        "unclassified_directories": [
            row["directory"] for row in rows if row["audit_family"] == "unclassified"
        ],
        "legacy_parent_reports": legacy_reports,
        "directory_inventory_sha256": sha256(csv_path),
    }
    (out_dir / "inventory_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["unclassified_directories"]:
        raise SystemExit("unclassified analysis directories remain")


if __name__ == "__main__":
    main()
