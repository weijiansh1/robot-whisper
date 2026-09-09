"""A2 of the token-matching pre-registration: does the structure deform before failure?

Scoring is conditioned on the episode still running at chunk q, within a task. That
condition exactly cancels the survival base rate, which is the confound that dominates
every uncontrolled number on this corpus: 1,442 of 1,442 annotated failures run to their
run horizon while 129 of 34,656 successes do.

Heads are only those fixed in `results-token-matching/PREREG.zh.md`: three per-layer
metrics over eight layers, three cross-layer metrics, each summarised at flow 0, flow 9
and by the least-squares slope across the ten steps. That is 81 heads, so the
`right-50x8` design selects one head per family and the `right-50x8b` design is read
once. The full development profile is reported so the selection is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from moe_grammar.extract_token_matching import N_FLOW, N_LAYERS, PER_FLOW, PER_LAYER, feature_names

CHUNKS = (4, 6, 8, 10, 12, 16, 20, 24)
TIME = np.arange(N_FLOW, dtype=np.float64) - (N_FLOW - 1) / 2.0
TIME_SS = float((TIME**2).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-dir", type=Path, default=Path("artifacts/token-matching"))
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-token-matching"))
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def head_names() -> list[str]:
    heads = [
        f"{metric}|layer_{layer}|{summary}"
        for metric in PER_LAYER
        for layer in range(N_LAYERS)
        for summary in ("f0", "f9", "slope")
    ]
    heads += [f"{metric}|{summary}" for metric in PER_FLOW for summary in ("f0", "f9", "slope")]
    return heads


def summarise(values: np.ndarray) -> np.ndarray:
    """Collapse the ten flow steps to the three pre-registered summaries."""
    centred = values - values.mean(axis=-1, keepdims=True)
    slope = (centred * TIME).sum(axis=-1) / TIME_SS
    return np.stack([values[..., 0], values[..., -1], slope], axis=-1)


def load_design(matching_dir: Path, features_dir: Path, marker: str) -> dict[str, Any]:
    names = feature_names()
    index = {name: position for position, name in enumerate(names)}
    per_task: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(str(matching_dir / "*.npz"))):
        if marker not in os.path.basename(path):
            continue
        matching = np.load(path, allow_pickle=False)
        outcome = np.load(features_dir / os.path.basename(path), allow_pickle=False)
        raw = matching["features"]
        blocks = []
        for metric in PER_LAYER:
            for layer in range(N_LAYERS):
                columns = [index[f"{metric}|layer_{layer}|flow_{f}"] for f in range(N_FLOW)]
                blocks.append(summarise(raw[:, columns]))
        for metric in PER_FLOW:
            columns = [index[f"{metric}|flow_{f}"] for f in range(N_FLOW)]
            blocks.append(summarise(raw[:, columns]))
        task = json.loads(str(matching["metadata_json"].item()))["task_key"]
        record = {
            "values": np.concatenate(blocks, axis=1).astype(np.float32),
            "episode": matching["episode_id"].astype(np.int64),
            "step": matching["episode_step"].astype(np.int64),
            "success": outcome["success"].astype(bool),
        }
        if task in per_task:
            previous = per_task[task]
            offset = previous["episode"].max() + 1
            record["episode"] = record["episode"] + offset
            for key in ("values", "episode", "step", "success"):
                record[key] = np.concatenate([previous[key], record[key]])
        per_task[task] = record
    return per_task


def survival_auc(values: np.ndarray, failed: np.ndarray) -> float | None:
    """Mann-Whitney AUC of failure versus success among episodes still present."""
    positive = values[failed]
    negative = values[~failed]
    if len(positive) < 5 or len(negative) < 5:
        return None
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    combined = np.concatenate([positive, negative])
    # Average ranks within ties so a constant head reads exactly 0.5.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    total = ranks[: len(positive)].sum()
    return float(
        (total - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))
    )


def evaluate(per_task: dict[str, Any], heads: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for position, head in enumerate(heads):
        by_chunk: dict[int, dict[str, Any]] = {}
        for chunk in CHUNKS:
            scores = []
            for record in per_task.values():
                rows = np.flatnonzero(record["step"] == chunk)
                if len(rows) < 20:
                    continue
                episodes = record["episode"][rows]
                failed = ~np.asarray(
                    [record["success"][record["episode"] == e][0] for e in episodes]
                )
                value = survival_auc(record["values"][rows, position], failed)
                if value is not None:
                    scores.append(value)
            if len(scores) >= 10:
                array = np.asarray(scores)
                by_chunk[chunk] = {
                    "tasks": len(array),
                    "macro_auc": float(array.mean()),
                    "tasks_above_half": int(np.sum(array > 0.5)),
                }
        if by_chunk:
            deviations = [abs(v["macro_auc"] - 0.5) for v in by_chunk.values()]
            output[head] = {
                "by_chunk": by_chunk,
                "mean_abs_deviation": float(np.mean(deviations)),
                "signed_mean": float(np.mean([v["macro_auc"] for v in by_chunk.values()]) - 0.5),
            }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    heads = head_names()
    print(f"{len(heads)} pre-registered heads", flush=True)

    development = load_design(args.matching_dir, args.features_dir, "right-50x8-")
    print(f"development: {len(development)} tasks", flush=True)
    dev = evaluate(development, heads)
    del development

    families = {"sa_coupling": [], "aa_coupling": [], "sa_gap": [], **{m: [] for m in PER_FLOW}}
    for head, record in dev.items():
        families[head.split("|")[0]].append((record["mean_abs_deviation"], head))
    selected = {
        family: max(items)[1] for family, items in families.items() if items
    }
    print("\nselected on development:")
    for family, head in selected.items():
        print(f"  {family:14s} -> {head:34s} |AUC-0.5| = {dev[head]['mean_abs_deviation']:.4f}")

    external = load_design(args.matching_dir, args.features_dir, "right-50x8b-")
    print(f"\nexternal: {len(external)} tasks", flush=True)
    ext = evaluate(external, list(selected.values()))
    del external

    summary = {
        "schema_version": 1,
        "heads": len(heads),
        "chunks": list(CHUNKS),
        "selected_on_development": selected,
        "development": {head: dev[head] for head in selected.values()},
        "development_all_heads": {
            head: {
                "mean_abs_deviation": record["mean_abs_deviation"],
                "signed_mean": record["signed_mean"],
            }
            for head, record in dev.items()
        },
        "external": ext,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "a2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n{'head':34s} {'dev |AUC-.5|':>12s} {'ext |AUC-.5|':>12s} {'符号一致':>9s}")
    for family, head in selected.items():
        if head not in ext:
            continue
        d, e = dev[head], ext[head]
        agree = sum(
            1
            for chunk in CHUNKS
            if chunk in d["by_chunk"]
            and chunk in e["by_chunk"]
            and np.sign(d["by_chunk"][chunk]["macro_auc"] - 0.5)
            == np.sign(e["by_chunk"][chunk]["macro_auc"] - 0.5)
        )
        total = sum(1 for c in CHUNKS if c in d["by_chunk"] and c in e["by_chunk"])
        print(
            f"{head:34s} {d['mean_abs_deviation']:12.4f} {e['mean_abs_deviation']:12.4f} "
            f"{agree:5d}/{total:<3d}"
        )


if __name__ == "__main__":
    main()
