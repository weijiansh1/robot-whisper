"""First-pass characterisation of MoE routing across the 30-task corpus.

The question this can answer: do different tasks route to different experts, and
is any such difference bigger than the corpus's own sampling noise?

Two guards are built in, both from things this project already got wrong once:

* **A null.**  "Task A routes differently from task B" is meaningless without
  knowing how far apart two halves of the *same* task land.  Every between-task
  distance here is computed between half-samples, and compared against the
  within-task split-half distance at the identical sample size.  A between-task
  distance that does not clear the within-task baseline is not evidence.
* **No token pooling.**  The state token and the action tokens have very
  different router sharpness (0.80-0.89 vs 0.998-0.999 normalised entropy on the
  earlier batches); pooling them produced a misleading "routing is almost
  uniform" reading before.  They are kept separate throughout.

Reads the corpus through INDEX.csv, so it only ever sees verified tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np
import zarr

import corpus_layout as cl

STATE_TOKEN = 0  # suffix index 0 is the state token; 1..10 are the action tokens


def load_index(root):
    with (pathlib.Path(root) / "INDEX.csv").open() as stream:
        return list(csv.DictReader(stream))


def task_expert_counts(root, suite, task_dir, rows, n_experts=32):
    """Per-CONTROL-STEP top-4 expert counts, split by token kind.

    Returns action/state arrays shaped [n_steps, n_hb_layers, n_experts], the AS
    choices [n_steps, n_as_layers, n_as_experts], and the episode index of each
    step.  Counting per step rather than per episode is what lets the caller
    equalise sample size across tasks: episodes differ in length by 2.5x, and a
    task estimated from more control steps gets a lower split-half floor for free.

    The directory comes from INDEX.csv's ``path`` column rather than being
    rebuilt from root+benchmark+task: the corpus may live under VLA_MUI_HUB,
    where the layout differs.
    """
    path = pathlib.Path(rows[0]["path"])
    group = zarr.open_group(str(cl.server_dir(path) / "routes.zarr"), mode="r")
    ids = np.asarray(group["hb_expert_ids"][:])          # [N, L, D, S, K]
    as_ids = np.asarray(group["as_expert_ids"][:])       # [N, A]
    n_as_experts = int(group.attrs.get("n_as_experts", 3))
    n_layers = ids.shape[1]

    action, state, as_counts, episode_of = [], [], [], []
    for row in rows:
        start = int(row["control_step_offset"])
        stop = start + int(row["inference_calls"])
        for step in range(start, stop):
            frame = ids[step]
            action.append(np.stack([
                np.bincount(frame[layer, :, STATE_TOKEN + 1:, :].ravel(), minlength=n_experts)
                for layer in range(n_layers)]))
            state.append(np.stack([
                np.bincount(frame[layer, :, STATE_TOKEN, :].ravel(), minlength=n_experts)
                for layer in range(n_layers)]))
            as_counts.append(np.stack([
                np.bincount(as_ids[step, layer:layer + 1], minlength=n_as_experts)
                for layer in range(as_ids.shape[1])]))
            episode_of.append(int(row["episode_index"]))
    return (np.asarray(action, np.int64), np.asarray(state, np.int64),
            np.asarray(as_counts, np.int64), np.asarray(episode_of))


def _distribution(counts):
    """counts [..., L, E] -> per-layer probability distribution."""
    total = counts.sum(-1, keepdims=True)
    return np.divide(counts, np.maximum(total, 1), dtype=np.float64)


def js_distance(p, q):
    """Jensen-Shannon distance per layer, averaged over layers (bits, sqrt scale)."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(np.where(mask, a * np.log2(np.where(mask, a / np.maximum(b, 1e-300), 1)), 0.0), -1)

    divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.mean(np.sqrt(np.maximum(divergence, 0.0))))


def split_halves(counts, episode_of, equal_steps=None, seed=0):
    """Even/odd episodes, so a drift over the episode order cannot fake a split.

    ``equal_steps`` subsamples each half to exactly that many control steps, which
    removes the sample-size advantage a task with longer episodes would otherwise
    have when its split-half floor is used as the noise baseline.
    """
    halves = []
    rng = np.random.default_rng(seed)
    for parity in (0, 1):
        take = np.flatnonzero(episode_of % 2 == parity)
        if equal_steps is not None:
            if len(take) < equal_steps:
                raise ValueError("only %d steps in half, need %d" % (len(take), equal_steps))
            take = rng.choice(take, equal_steps, replace=False)
        halves.append(counts[take].sum(0))
    return halves[0], halves[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--equal-steps", type=int,
                    help="subsample every half to this many control steps")
    args = ap.parse_args()

    rows_by_task = defaultdict(list)
    for row in load_index(args.root):
        rows_by_task[(row["suite"], row["task_dir"])].append(row)
    if not rows_by_task:
        raise SystemExit("INDEX.csv is empty; run validate/build_index first")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    halves, totals, within, as_profile = {}, {}, {}, {}
    for (suite, task_dir), rows in sorted(rows_by_task.items()):
        action, state, as_counts, episode_of = task_expert_counts(
            args.root, suite, task_dir, rows)
        first, second = split_halves(action, episode_of, args.equal_steps)
        halves[(suite, task_dir)] = (_distribution(first), _distribution(second))
        totals[(suite, task_dir)] = {
            "action": _distribution(action.sum(0)),
            "state": _distribution(state.sum(0)),
        }
        within[(suite, task_dir)] = js_distance(*halves[(suite, task_dir)])
        as_profile[(suite, task_dir)] = _distribution(as_counts.sum(0)).tolist()
        print("%-8s %-56s episodes=%3d  within-task split-half JS=%.4f"
              % (suite, task_dir[:56], action.shape[0], within[(suite, task_dir)]), flush=True)

    keys = sorted(halves)
    # between-task distances use half-samples on both sides, so they carry the
    # same sampling noise as the within-task baseline they are compared against
    between = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                between[i, j] = between[j, i] = js_distance(halves[a][0], halves[b][0])

    within_values = np.array([within[k] for k in keys])
    same_suite, cross_suite = [], []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                (same_suite if a[0] == b[0] else cross_suite).append(between[i, j])

    def stats(values):
        """Partial corpora legitimately have empty groups (one suite, one task)."""
        if not values:
            return {"mean": None, "min": None, "n_pairs": 0}
        return {"mean": float(np.mean(values)), "min": float(np.min(values)),
                "n_pairs": len(values)}

    summary = {
        "n_tasks": len(keys),
        "within_task_split_half_JS": {
            "mean": float(within_values.mean()), "max": float(within_values.max()),
            "min": float(within_values.min())},
        "between_task_JS_same_suite": stats(same_suite),
        "between_task_JS_cross_suite": stats(cross_suite),
    }
    ratio = (summary["between_task_JS_same_suite"]["mean"] / max(within_values.mean(), 1e-12)
             if same_suite else None)
    summary["separability_ratio_same_suite_over_within"] = ratio

    print("\n=== task separability (action tokens, HB layers) ===")
    print("within-task split-half JS   mean %.4f  (this is the noise floor)"
          % summary["within_task_split_half_JS"]["mean"])
    for label, group in (("same suite ", "between_task_JS_same_suite"),
                         ("cross suite", "between_task_JS_cross_suite")):
        item = summary[group]
        if item["n_pairs"]:
            print("between-task JS %s mean %.4f  min %.4f  (%d pairs)"
                  % (label, item["mean"], item["min"], item["n_pairs"]))
        else:
            print("between-task JS %s no pairs in this corpus subset" % label)
    if ratio is not None:
        print("ratio between/within = %.2f  %s"
              % (ratio, "tasks separate" if ratio > 2 else "NOT above the noise floor"))

    np.savez_compressed(
        out / "routing_profiles.npz",
        tasks=np.array(["%s/%s" % k for k in keys]),
        between=between,
        within=within_values,
        action=np.stack([totals[k]["action"] for k in keys]),
        state=np.stack([totals[k]["state"] for k in keys]),
    )
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary,
         "within_task": {"%s/%s" % k: within[k] for k in keys},
         "as_routing": {"%s/%s" % k: as_profile[k] for k in keys}}, indent=1))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
