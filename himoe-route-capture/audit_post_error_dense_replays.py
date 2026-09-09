#!/usr/bin/env python3
"""Contact-audit selected query-level transport-loss proxies.

The dense replay artifact records every simulator substep in each saved action
chunk.  This audit requires sustained bilateral target-object contact before
the query-level loss and complete finger-contact loss at the post-loss query.
It is a validation audit of selected cases, not a population estimator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
ANALYSIS_DIR = HERE / "analysis/post-error-recovery-hub"
DENSE_ROOT = ANALYSIS_DIR / "dense-replays"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-csv", type=pathlib.Path, default=ANALYSIS_DIR / "event_inventory.csv"
    )
    parser.add_argument(
        "--recurrence-csv",
        type=pathlib.Path,
        default=ANALYSIS_DIR / "recurrence_inventory.csv",
    )
    parser.add_argument("--replay-root", type=pathlib.Path, default=DENSE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ANALYSIS_DIR)
    parser.add_argument("--minimum-bilateral-run", type=int, default=3)
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"unexpected boolean: {value}")
    return value == "True"


def max_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def first_true_run(mask: np.ndarray, minimum: int) -> int | None:
    current = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        current = current + 1 if value else 0
        if current >= minimum:
            return index - minimum + 1
    return None


def load_events(path: pathlib.Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        key = (row["task"].split("/", 1)[1], int(row["episode"]))
        if key in output:
            raise ValueError(f"duplicate event key: {key}")
        output[key] = row
    return output


def load_recurrence(path: pathlib.Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["anchor_kind"] == "failed_chunk"
        ]
    output = {}
    for row in rows:
        key = (row["task"].split("/", 1)[1], int(row["episode"]))
        if key in output:
            raise ValueError(f"duplicate failed-chunk recurrence key: {key}")
        output[key] = row
    return output


def verify_artifact(root: pathlib.Path, descriptor: dict[str, Any]) -> None:
    if descriptor.get("status") != "complete":
        raise ValueError(f"incomplete replay artifact: {root}")
    if not descriptor.get("replay_fidelity_passed"):
        raise ValueError(f"replay fidelity failed: {root}")
    for key in ("events", "layout"):
        item = descriptor["files"][key]
        actual = sha256_file(root / item["file"])
        if actual != item["sha256"]:
            raise ValueError(f"artifact hash mismatch: {root / item['file']}")


def target_finger_tapes(
    active: np.ndarray,
    pair_names: np.ndarray,
    layout: dict[str, Any],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    roles = {row["geom"]: row for row in layout["geom_roles"]}
    left = np.zeros(active.shape[:2], dtype=bool)
    right = np.zeros(active.shape[:2], dtype=bool)
    for pair_index, name in enumerate(pair_names.tolist()):
        geoms = str(name).split(" <-> ")
        target_contact = any(
            geom == target or geom.startswith(target + "_") for geom in geoms
        )
        if not target_contact:
            continue
        sides = {roles.get(geom, {}).get("finger_side") for geom in geoms}
        if "left" in sides:
            left |= active[:, :, pair_index]
        if "right" in sides:
            right |= active[:, :, pair_index]
    return left, right


def contact_pattern(
    left: np.ndarray, right: np.ndarray, valid: np.ndarray, lo: int, hi: int
) -> str:
    chunks = []
    for query in range(max(0, lo), min(len(valid), hi)):
        symbols = []
        for step in range(valid.shape[1]):
            if not valid[query, step]:
                continue
            if left[query, step] and right[query, step]:
                symbols.append("B")
            elif left[query, step]:
                symbols.append("L")
            elif right[query, step]:
                symbols.append("R")
            else:
                symbols.append(".")
        chunks.append(f"q{query}:{''.join(symbols)}")
    return ";".join(chunks)


def audit_replay(
    root: pathlib.Path,
    event: dict[str, str],
    recurrence: dict[str, str],
    minimum_run: int,
) -> dict[str, Any]:
    descriptor = json.loads((root / "artifact.json").read_text())
    layout = json.loads((root / "layout.json").read_text())
    verify_artifact(root, descriptor)
    with np.load(root / "events.npz", allow_pickle=False) as archive:
        active = np.asarray(archive["contact_active"], dtype=bool)
        pair_names = np.asarray(archive["contact_pair_names"])
        valid = np.asarray(archive["valid_steps"], dtype=bool)
        replay_drift = np.asarray(archive["source_replay_drift"], dtype=np.float64)

    target = event["target"].removesuffix("_joint0")
    left, right = target_finger_tapes(active, pair_names, layout, target)
    left &= valid
    right &= valid
    bilateral = left & right
    any_finger = left | right
    bout_start = int(event["bout_start"])
    drop_query = int(event["drop_query"])
    if not 0 <= bout_start < drop_query < len(valid):
        raise ValueError(f"event/replay query mismatch: {root}")

    pre = bilateral[bout_start:drop_query].reshape(-1)
    future = bilateral[drop_query:].reshape(-1)
    pre_run = max_true_run(pre)
    future_run = max_true_run(future)
    future_start = first_true_run(future, minimum_run)
    post_has_finger_contact = bool(any_finger[drop_query, 0])
    confirmed = pre_run >= minimum_run and not post_has_finger_contact
    regrasp = future_run >= minimum_run
    future_regrasp_query = (
        drop_query + future_start // valid.shape[1]
        if future_start is not None
        else None
    )
    terminal_success = parse_bool(event["terminal_success"])
    if not confirmed:
        verdict = "rejected_no_stable_bilateral_precontact"
    elif regrasp and terminal_success:
        verdict = "confirmed_loss_regrasp_terminal_success"
    elif regrasp:
        verdict = "confirmed_loss_regrasp_terminal_failure"
    elif terminal_success:
        verdict = "confirmed_loss_no_regrasp_terminal_success"
    else:
        verdict = "confirmed_loss_no_regrasp_terminal_failure"

    return {
        "task": event["task"],
        "episode": int(event["episode"]),
        "init_state_id": int(event["init_state_id"]),
        "flow_noise_seed": int(event["flow_noise_seed"]),
        "target": event["target"],
        "bout_start": bout_start,
        "drop_query": drop_query,
        "terminal_success": terminal_success,
        "minimum_bilateral_run": minimum_run,
        "maximum_pre_loss_bilateral_run": pre_run,
        "post_query_has_target_finger_contact": post_has_finger_contact,
        "contact_validated_transport_loss": confirmed,
        "maximum_future_bilateral_run": future_run,
        "contact_validated_regrasp": regrasp,
        "first_contact_validated_regrasp_query": future_regrasp_query,
        "verdict": verdict,
        "route_action_hellinger_similarity": float(
            recurrence["route_action_hellinger_similarity"]
        ),
        "route_action_top4_jaccard": float(
            recurrence["route_action_top4_jaccard"]
        ),
        "action_cosine_similarity": float(
            recurrence["action_cosine_similarity"]
        ),
        "hidden_action_cosine_similarity": float(
            recurrence["hidden_action_cosine_similarity"]
        ),
        "local_contact_pattern": contact_pattern(
            left, right, valid, bout_start - 1, drop_query + 5
        ),
        "replay_fidelity_max_abs": float(replay_drift.max()),
        "replay_fidelity_tolerance": float(
            descriptor["replay_fidelity_tolerance"]
        ),
        "source_episode_sha256": descriptor["source_episode_sha256"],
        "events_sha256": descriptor["files"]["events"]["sha256"],
        "artifact": str(root / "artifact.json"),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Dense-contact audit of selected post-loss proxies",
        "",
        "This purposive audit covers every terminal-success candidate in the main",
        "query-level cohort plus five failure candidates spanning all four event-bearing",
        "tasks. It is not a random sample and must not be used to estimate prevalence.",
        "",
        "A proxy is contact-confirmed only when the target has bilateral finger contact",
        "for at least three consecutive simulator substeps before `drop_query`, followed",
        "by no target-finger contact at that query's first physical point. Regrasp uses",
        "the same three-substep bilateral criterion after the landmark.",
        "",
        "| task | ep | outcome | pre bilateral run | confirmed loss | future run | regrasp | verdict |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| %s | %d | %s | %d | %s | %d | %s | %s |"
            % (
                row["task"],
                row["episode"],
                "success" if row["terminal_success"] else "failure",
                row["maximum_pre_loss_bilateral_run"],
                str(row["contact_validated_transport_loss"]).lower(),
                row["maximum_future_bilateral_run"],
                str(row["contact_validated_regrasp"]).lower(),
                row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "- Audited: %d; contact-confirmed loss: %d; rejected proxy: %d."
            % (
                summary["audited_cases"],
                summary["contact_validated_transport_losses"],
                summary["rejected_query_proxies"],
            ),
            "- Among confirmed losses: %d terminal success, %d terminal failure."
            % (
                summary["confirmed_terminal_success"],
                summary["confirmed_terminal_failure"],
            ),
            "- Contact-confirmed regrasp after a confirmed loss: %d."
            % summary["confirmed_loss_then_regrasp"],
            "- Maximum query-state replay drift: %.3e (tolerance %.1e)."
            % (
                summary["maximum_replay_fidelity_abs"],
                summary["replay_fidelity_tolerance"],
            ),
            "",
            "The ramekin success candidate later performs a real bilateral grasp, but",
            "its query-level landmark was not preceded by stable bilateral contact; it",
            "therefore cannot validate an initial slip. The stove episode 3 candidate is",
            "the only audited contact-confirmed loss followed by regrasp and task success.",
            "",
        ]
    )
    if summary["contact_validated_mixed_pairs"]:
        lines.extend(
            [
                "## Contact-validated matched micro-case",
                "",
                "Failure-minus-success recurrence differences are shown only for a",
                "contact-confirmed pair sharing task and initial state.",
                "",
                "| task | init | pair | route Hellinger | top-4 Jaccard | action cosine | hidden cosine |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for pair in summary["contact_validated_mixed_pairs"]:
            lines.append(
                "| %s | %d | %d vs %d | %+.4f | %+.4f | %+.4f | %+.4f |"
                % (
                    pair["task"],
                    pair["init_state_id"],
                    pair["failure_episode"],
                    pair["success_episode"],
                    pair["failure_minus_success"][
                        "route_action_hellinger_similarity"
                    ],
                    pair["failure_minus_success"]["route_action_top4_jaccard"],
                    pair["failure_minus_success"]["action_cosine_similarity"],
                    pair["failure_minus_success"][
                        "hidden_action_cosine_similarity"
                    ],
                )
            )
        lines.extend(
            [
                "",
                "There is only one such pair. It is a case study, not an effect estimate.",
                "The post-loss physical states are not matched, so it does not isolate routing.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.minimum_bilateral_run < 1:
        raise ValueError("minimum bilateral run must be positive")
    events = load_events(args.event_csv)
    recurrence = load_recurrence(args.recurrence_csv)
    roots = sorted(args.replay_root.glob("task_*/episode_*"))
    if not roots:
        raise RuntimeError(f"no dense replay artifacts under {args.replay_root}")
    rows = []
    for root in roots:
        descriptor = json.loads((root / "artifact.json").read_text())
        key = (str(descriptor["task_name"]), int(descriptor["episode"]))
        if key not in events:
            raise ValueError(f"dense replay is not in the main event inventory: {key}")
        if key not in recurrence:
            raise ValueError(f"dense replay has no failed-chunk recurrence row: {key}")
        rows.append(
            audit_replay(
                root, events[key], recurrence[key], args.minimum_bilateral_run
            )
        )

    confirmed = [row for row in rows if row["contact_validated_transport_loss"]]
    metrics = (
        "route_action_hellinger_similarity",
        "route_action_top4_jaccard",
        "action_cosine_similarity",
        "hidden_action_cosine_similarity",
    )
    by_cluster: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in confirmed:
        by_cluster.setdefault((row["task"], row["init_state_id"]), []).append(row)
    matched_pairs = []
    for (task, init_state), group in sorted(by_cluster.items()):
        successes = [row for row in group if row["terminal_success"]]
        failures = [row for row in group if not row["terminal_success"]]
        for success in successes:
            for failure in failures:
                matched_pairs.append(
                    {
                        "task": task,
                        "init_state_id": init_state,
                        "success_episode": success["episode"],
                        "failure_episode": failure["episode"],
                        "failure_minus_success": {
                            metric: failure[metric] - success[metric]
                            for metric in metrics
                        },
                    }
                )
    summary = {
        "schema": "himoe.post_error_dense_contact_audit.v1",
        "selection_contract": (
            "all two main-cohort terminal-success candidates plus five purposively "
            "selected terminal failures spanning four event-bearing tasks"
        ),
        "population_estimate_allowed": False,
        "minimum_consecutive_bilateral_substeps": args.minimum_bilateral_run,
        "audited_cases": len(rows),
        "contact_validated_transport_losses": len(confirmed),
        "rejected_query_proxies": len(rows) - len(confirmed),
        "confirmed_terminal_success": int(
            sum(row["terminal_success"] for row in confirmed)
        ),
        "confirmed_terminal_failure": int(
            sum(not row["terminal_success"] for row in confirmed)
        ),
        "confirmed_loss_then_regrasp": int(
            sum(row["contact_validated_regrasp"] for row in confirmed)
        ),
        "contact_validated_mixed_pairs": matched_pairs,
        "maximum_replay_fidelity_abs": max(
            row["replay_fidelity_max_abs"] for row in rows
        ),
        "replay_fidelity_tolerance": min(
            row["replay_fidelity_tolerance"] for row in rows
        ),
        "cases": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "dense_replay_audit.csv", rows)
    (args.out_dir / "dense_replay_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    (args.out_dir / "dense_replay_audit.md").write_text(
        render_report(summary, rows)
    )
    print(f"wrote {args.out_dir / 'dense_replay_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
