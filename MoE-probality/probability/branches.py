"""Outcome-blind checkpoint selection and strict future branch result validation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .evaluation import wilson


def branch_candidates(episodes, per_task=2, continuations=64):
    rows = []
    selected = episodes[(episodes.split == "test_unseen_init") & (episodes.first_alarm_q >= 0)].copy()
    selected["selection_key"] = [hashlib.sha256(f"{r.source_run}:{r.episode}:20260907".encode()).hexdigest()
                                 for r in selected.itertuples()]
    for _, group in selected.groupby("task", sort=True):
        used = set()
        for alarm in group.sort_values("selection_key").head(per_task).itertuples():
            q = int(alarm.first_alarm_q)
            controls = episodes[(episodes.split == "test_unseen_init") & (episodes.task == alarm.task)
                                & (episodes.run_id == alarm.run_id) & (episodes.length > q)
                                & ((episodes.first_alarm_q < 0) | (episodes.first_alarm_q > q))
                                & ~episodes.episode_row.isin(used)].copy()
            controls["selection_key"] = [hashlib.sha256(f"control:{r.source_run}:{r.episode}:{q}".encode()).hexdigest()
                                         for r in controls.itertuples()]
            matched = controls.sort_values("selection_key").head(1)
            pair = f"pair_{alarm.episode_row}_q{q}"
            for role, row in [("alarm", alarm), *[("control", r) for r in matched.itertuples()]]:
                used.add(row.episode_row)
                rows.append(dict(pair_id=pair, checkpoint_id=f"e{row.episode_row}_q{q}", role=role,
                                 source_run=row.source_run, episode=int(row.episode), query=q,
                                 suite=row.suite, task=row.task, init_state=int(row.init_state),
                                 remaining_action_steps=int(row.max_steps-q*row.replan_steps),
                                 requested_continuations=continuations,
                                 preserve_current_chunk=True,
                                 checkpoint_status="requires_full_closed_loop_reconstruction",
                                 physical_trap_verified="unknown",
                                 control_available=bool(len(matched))))
    return pd.DataFrame(rows)


def summarize_branches(records):
    """Only verified, aligned, completed natural continuations yield soft labels.

    A CSV row alone cannot establish that a simulator was restored correctly.
    Provenance digests must refer to the collector's saved validation artifacts.
    """
    required = {"checkpoint_id", "branch_id", "future_seed", "snapshot_sha256",
                "policy_sha256", "current_chunk_sha256", "restore_audit_sha256",
                "remaining_action_steps", "success", "complete", "policy_unchanged",
                "current_chunk_preserved", "full_restore_verified", "independent_future_rng"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"missing branch provenance columns: {sorted(missing)}")
    for column in ("complete", "policy_unchanged", "current_chunk_preserved",
                   "full_restore_verified", "independent_future_rng"):
        if not records[column].eq(True).all():
            raise ValueError(f"invalid natural-continuation evidence: {column}")
    if not records.success.isin([True, False, 0, 1]).all():
        raise ValueError("branch success must be a binary observed budget outcome")
    if records.duplicated(["checkpoint_id", "branch_id"]).any() or records.duplicated(["checkpoint_id", "future_seed"]).any():
        raise ValueError("duplicate branch or future-noise stream")
    for column in ("snapshot_sha256", "policy_sha256", "current_chunk_sha256", "restore_audit_sha256"):
        if not records[column].astype(str).str.fullmatch("[a-f0-9]{64}").all():
            raise ValueError(f"invalid SHA-256: {column}")
    if not (pd.to_numeric(records.remaining_action_steps) > 0).all():
        raise ValueError("branch must start with positive execution budget")
    rows = []
    for checkpoint, group in records.groupby("checkpoint_id", sort=True):
        for column in ("snapshot_sha256", "policy_sha256", "current_chunk_sha256", "remaining_action_steps"):
            if group[column].nunique() != 1:
                raise ValueError(f"branches do not share the same starting condition: {checkpoint}/{column}")
        k, n = int(group.success.sum()), len(group)
        lo, hi = wilson(k, n)
        rows.append(dict(checkpoint_id=checkpoint, branches=n, successes=k,
                         success_probability=k/n, wilson_low=lo, wilson_high=hi,
                         standard_error=float(np.sqrt((k/n)*(1-k/n)/n)),
                         soft_label_trials=n, natural_escape_probability=None))
    return pd.DataFrame(rows)
