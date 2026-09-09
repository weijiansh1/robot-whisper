#!/usr/bin/env python3
"""Freeze matched event/control rows before collecting functional features."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_moe_execution_signals.confirm_cache_new import (
    ROOT,
    RUNS,
    load_confirmation_corpus,
)


SCHEMA = "himoe.rich_event_capture_plan.v1"


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pick(rng: np.random.Generator, values: np.ndarray) -> int:
    if not len(values):
        raise ValueError("cannot choose from an empty array")
    return int(values[int(rng.integers(0, len(values)))])


def build_plan(n_pairs: int, seed: int, leads: tuple[int, ...]) -> dict:
    corpus, inventory = load_confirmation_corpus()
    rng = np.random.default_rng(seed)
    onset = corpus.static_onset
    meta = corpus.meta.reset_index(drop=True)
    success = meta.success.to_numpy(bool)
    lengths = meta["T"].to_numpy(int)
    groups = corpus.match_group.astype(str)
    event_mask = onset >= max(-min(leads), 0)
    control_mask = success & (corpus.trap_onset < 0) & (corpus.static_onset < 0)

    run_by_name = {path.name: path for path in RUNS}
    candidate_groups = []
    for group in sorted(np.unique(groups)):
        events = np.flatnonzero(event_mask & (groups == group))
        usable = []
        for event_index in events:
            queries = [int(onset[event_index]) + lead for lead in leads]
            controls = np.flatnonzero(control_mask & (groups == group))
            if all(
                query >= 0
                and query < lengths[event_index]
                and np.any(lengths[controls] > query)
                for query in queries
            ):
                usable.append(event_index)
        if usable:
            candidate_groups.append((group, np.asarray(usable, np.int32)))
    if len(candidate_groups) < n_pairs:
        raise RuntimeError(
            "requested %d independent groups, only %d are eligible"
            % (n_pairs, len(candidate_groups))
        )

    group_order = rng.permutation(len(candidate_groups))[:n_pairs]
    pairs = []
    rows = []
    for pair_id, group_position in enumerate(group_order):
        group, events = candidate_groups[int(group_position)]
        event_index = _pick(rng, events)
        query_max = max(int(onset[event_index]) + lead for lead in leads)
        controls = np.flatnonzero(
            control_mask & (groups == group) & (lengths > query_max)
        )
        same_run = controls[
            meta.iloc[controls].source_run.to_numpy()
            == str(meta.iloc[event_index].source_run)
        ]
        control_index = _pick(rng, same_run if len(same_run) else controls)

        def episode_record(index: int) -> dict:
            item = meta.iloc[index]
            run = run_by_name[str(item.source_run)]
            episode = int(item.source_episode_id)
            episode_path = run / "client" / ("episode_%02d.npz" % episode)
            if not episode_path.is_file():
                raise FileNotFoundError(episode_path)
            return {
                "global_episode": int(index),
                "source_run": str(item.source_run),
                "source_episode": episode,
                "episode_path": str(episode_path.relative_to(ROOT)),
                "flow_noise_seed": int(item.flow_noise_seed),
                "success": bool(item.success),
                "length": int(item["T"]),
            }

        event_record = episode_record(event_index)
        control_record = episode_record(control_index)
        pair = {
            "pair_id": pair_id,
            "init_state_id": int(group),
            "static_onset_query": int(onset[event_index]),
            "event": event_record,
            "control": control_record,
        }
        pairs.append(pair)
        for lead in leads:
            query = int(onset[event_index]) + int(lead)
            for role, record in (("event", event_record), ("control", control_record)):
                rows.append(
                    {
                        "row_id": len(rows),
                        "pair_id": pair_id,
                        "role": role,
                        "event_kind": "static",
                        "relative_query": int(lead),
                        "query_index": query,
                        "init_state_id": int(group),
                        **record,
                    }
                )

    return {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection_code_sha256": _code_sha256(),
        "frozen_before_functional_capture": True,
        "selection_seed": int(seed),
        "event_kind": "static_query_proxy",
        "leads": list(leads),
        "control_definition": "success_and_no_aggregate_trap_and_no_static",
        "matching": "same_init_state_and_same_absolute_query",
        "one_event_pair_per_init_state": True,
        "inventory": inventory,
        "n_pairs": len(pairs),
        "n_rows": len(rows),
        "pairs": pairs,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--leads", default="-4,-2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    leads = tuple(int(value) for value in args.leads.split(","))
    if not leads or any(value > 0 for value in leads):
        raise ValueError("leads must be non-empty and non-positive")
    plan = build_plan(args.pairs, args.seed, leads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote %s: %d pairs / %d rows / groups=%s"
        % (
            args.out,
            plan["n_pairs"],
            plan["n_rows"],
            ",".join(str(pair["init_state_id"]) for pair in plan["pairs"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
