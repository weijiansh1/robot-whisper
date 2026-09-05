#!/usr/bin/env python3
"""Validate train-free isolation, budget matching, and reported conclusions."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(frame: pd.DataFrame, **query) -> pd.Series:
    take = np.ones(len(frame), bool)
    for key, value in query.items():
        take &= frame[key] == value
    selected = frame[take]
    if len(selected) != 1:
        raise AssertionError(f"expected one row for {query}, got {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    required = (
        "unlabeled_scores.npz",
        "unlabeled_snapshot_scores.csv",
        "unlabeled_assignments.csv",
        "assignment_audit_labeled.csv",
        "score_manifest.json",
        "evaluation_manifest.json",
        "snapshot_auc.csv",
        "multilabel_auc_deltas.csv",
        "selection_by_group.csv",
        "selection_metrics.csv",
        "bootstrap_deltas.csv",
        "component_auc.csv",
        "onset_alignment.csv",
        "onset_bootstrap_deltas.csv",
    )
    for name in required:
        if not (RESULTS / name).exists():
            raise AssertionError(f"missing result: {name}")

    score_manifest = json.loads((RESULTS / "score_manifest.json").read_text())
    evaluation_manifest = json.loads((RESULTS / "evaluation_manifest.json").read_text())
    assert score_manifest["train_free"] is True
    assert score_manifest["fit_calls"] == 0
    assert score_manifest["calibrated_parameters"] == 0
    assert score_manifest["label_columns_read"] == []
    assert score_manifest["protocol_sha256"] == sha256(HERE / "PROTOCOL.md")
    assert evaluation_manifest["protocol_sha256"] == sha256(HERE / "PROTOCOL.md")
    assert evaluation_manifest["score_manifest_sha256"] == sha256(
        RESULTS / "score_manifest.json"
    )
    assert evaluation_manifest["label_counts"] == {
        "loop": 57,
        "static": 146,
        "trap": 193,
    }

    source = (HERE / "score_heads.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "sklearn" not in imports
    for forbidden in ("loop_onset", "static_onset", "physical_onsets", "load_corpus"):
        assert forbidden not in source

    with np.load(RESULTS / "unlabeled_scores.npz") as data:
        assert data["episode"].shape == (512,)
        for name in (
            "loop_soft",
            "static_soft",
            "double_soft_max",
            "single_soft_mean",
            "single_mobility",
        ):
            assert data[name].shape == (512, 52)
            assert np.isfinite(data[name][:, 34]).all()

    assignments = pd.read_csv(RESULTS / "unlabeled_assignments.csv")
    assert len(assignments) == 512 * 2 * 3
    assignment_count = assignments.groupby(["query", "budget_per_head", "group"])[
        ["loop_selected", "static_selected"]
    ].sum()
    budgets = assignment_count.index.get_level_values("budget_per_head").to_numpy()
    assert np.array_equal(assignment_count.loop_selected.to_numpy(), budgets)
    assert np.array_equal(assignment_count.static_selected.to_numpy(), budgets)

    by_group = pd.read_csv(RESULTS / "selection_by_group.csv")
    for family in ("soft", "hard_diagnostic"):
        pivot = by_group[by_group.family == family].pivot_table(
            index=["query", "budget_per_head", "group"],
            columns="selector",
            values="n_selected",
        )
        union = pivot[f"double_{family}_union"]
        assert (union >= pivot.index.get_level_values("budget_per_head")).all()
        assert (union <= 2 * pivot.index.get_level_values("budget_per_head")).all()
        assert np.array_equal(union, pivot[f"single_{family}_mean_matched"])
        assert np.array_equal(union, pivot[f"double_{family}_max_matched"])

    multi = pd.read_csv(RESULTS / "multilabel_auc_deltas.csv")
    typed = one(
        multi,
        query=34,
        left="typed_soft_heads",
        right="single_soft_mean",
    )
    assert np.isclose(typed.left_macro_subtype_auc, 0.8045, atol=5e-5)
    assert np.isclose(typed.right_macro_subtype_auc, 0.6303, atol=5e-5)
    assert typed.delta > 0 and typed.ci_low > 0

    auc = pd.read_csv(RESULTS / "snapshot_auc.csv")
    max_trap = one(auc, query=34, score="double_soft_max", target="trap")
    mean_trap = one(auc, query=34, score="single_soft_mean", target="trap")
    mobility_trap = one(auc, query=34, score="single_mobility", target="trap")
    assert max_trap.group_macro_auc < mean_trap.group_macro_auc
    assert max_trap.group_macro_auc < mobility_trap.group_macro_auc

    boot = pd.read_csv(RESULTS / "bootstrap_deltas.csv")
    loop = one(
        boot,
        query=34,
        budget_per_head=8,
        family="soft",
        left="loop_soft_topk",
        right="single_soft_mean_topk",
        metric="loop_recall",
    )
    static = one(
        boot,
        query=34,
        budget_per_head=8,
        family="soft",
        left="static_soft_topk",
        right="single_soft_mean_topk",
        metric="static_recall",
    )
    assert loop.delta > 0 and loop.ci_low > 0
    assert static.delta > 0 and static.ci_low > 0
    assert (boot.bootstrap_draws == 20_000).all()

    onset = pd.read_csv(RESULTS / "onset_bootstrap_deltas.csv")
    assert (onset.bootstrap_draws == 20_000).all()
    assert (
        one(onset, event="loop", left="loop_soft", right="single_soft_mean").ci_low > 0
    )
    assert (
        one(onset, event="static", left="static_soft", right="single_soft_mean").ci_low
        > 0
    )
    print("TRAINFREE_VALIDATION_OK")


if __name__ == "__main__":
    main()
