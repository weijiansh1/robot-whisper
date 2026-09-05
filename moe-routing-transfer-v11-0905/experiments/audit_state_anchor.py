#!/usr/bin/env python3
"""Audit how nearly flow-invariant the state-token router distribution is."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


BUNDLE = Path(__file__).resolve().parent.parent
BUILD_SUMMARY = BUNDLE / "results/conditional_profiles/build_summary.json"
OUTPUT = BUNDLE / "results/state_anchor_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-summary", type=Path, default=BUILD_SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(values)),
        "fraction_exact": float(np.mean(values == 0.0)),
        "median_row_max_range": float(np.quantile(values, 0.5)),
        "p95_row_max_range": float(np.quantile(values, 0.95)),
        "p99_row_max_range": float(np.quantile(values, 0.99)),
        "max_row_max_range": float(values.max(initial=0.0)),
    }


def main() -> None:
    args = parse_args()
    build = json.loads(args.build_summary.read_text(encoding="utf-8"))
    by_corpus: dict[str, list[np.ndarray]] = {}
    tasks = []
    for record in build["records"]:
        store = zarr.open_group(record["raw_routes"], mode="r")
        probability = store["hb_router_probs"]
        row_maxima = []
        for start in range(0, probability.shape[0], args.batch_size):
            state = np.asarray(
                probability[start : start + args.batch_size, :, :, 0, :],
                dtype=np.float32,
            )
            # Maximum probability range over flow, layers, and experts for each query.
            row_maxima.append((state.max(axis=2) - state.min(axis=2)).max(axis=(1, 2)))
        values = np.concatenate(row_maxima)
        by_corpus.setdefault(record["corpus"], []).append(values)
        tasks.append(
            {
                "corpus": record["corpus"],
                "suite": record["suite"],
                "task": record["task"],
                **summarize(values),
            }
        )

    corpus = {
        name: summarize(np.concatenate(parts)) for name, parts in sorted(by_corpus.items())
    }
    all_values = np.concatenate([part for parts in by_corpus.values() for part in parts])
    result = {
        "schema": "himoe.state_anchor_audit.v1",
        "definition": (
            "Per query, max_e,l(max_f p[l,f,state,e] - min_f p[l,f,state,e]); "
            "state token is token index 0."
        ),
        "source": "original hb_router_probs, not derived profiles",
        "corpora": corpus,
        "all": summarize(all_values),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"corpora": corpus, "all": result["all"]}, indent=2))


if __name__ == "__main__":
    main()
