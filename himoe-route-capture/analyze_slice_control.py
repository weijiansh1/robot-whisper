"""Is the corpus's task separability confounded with the MIG slice it ran on?

`libero30-right-v1` split every suite the same way: task_id 0-4 on the 2g.35gb
slice, 5-9 on 1g.35gb.  Slice is therefore *perfectly* confounded with the low/high
half of the task id, and the headline "between-task JS = 3.4x the within-task
floor" cannot distinguish "different tasks route differently" from "different SM
counts route differently" for the 0-4 vs 5-9 pairs.

`slice-control-v1` re-ran two tasks (goal/t00, object/t00) on the *other* slice
under an otherwise identical protocol, which isolates the slice effect at fixed
task.  This script measures it and puts it on the same axis as the two numbers it
has to be compared against:

    within-task split-half JS   the sampling noise floor
    cross-slice JS at same task the quantity of interest
    between-task JS same suite  the effect the corpus claims

A cross-slice distance near the noise floor clears the corpus.  One near the
between-task distance means the separability result is uninterpretable as it
stands.

Aggregated routing profiles also move when the *mix* of control steps moves, so
episode length and success are reported alongside: a slice that fails more often
runs longer episodes and shifts the profile without any per-state change.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

STATE_TOKEN = 0
N_EXPERTS = 32


def _episode_bounds(summaries):
    """Start offset of each episode in the flat zarr, by cumulative inference_calls."""
    offset, bounds = 0, []
    for row in summaries:
        calls = int(row["inference_calls"])
        bounds.append((offset, offset + calls))
        offset += calls
    return bounds, offset


def load_task(task_dir: pathlib.Path):
    summaries = json.loads((task_dir / "client" / "summaries.json").read_text())
    group = zarr.open_group(str(task_dir / "server" / "routes.zarr"), mode="r")
    ids = np.asarray(group["hb_expert_ids"][:])
    bounds, total = _episode_bounds(summaries)
    if total != ids.shape[0]:
        raise RuntimeError("%s: sum(inference_calls)=%d != zarr steps=%d"
                           % (task_dir, total, ids.shape[0]))
    # the corpus writes episode_id since 2026-08-13; use it to check the offsets
    # rather than trusting the cumulative sum, which is what older batches had to do
    if "episode_id" in group:
        stored = np.asarray(group["episode_id"][:])
        for row, (start, stop) in zip(summaries, bounds):
            want = int(row["episode_index"])
            if not np.all(stored[start:stop] == want):
                raise RuntimeError("%s: episode_id disagrees with the offsets at %d"
                                   % (task_dir, start))

    n_layers = ids.shape[1]
    action, state, episode_of = [], [], []
    for row, (start, stop) in zip(summaries, bounds):
        for step in range(start, stop):
            frame = ids[step]
            action.append(np.stack([
                np.bincount(frame[layer, :, STATE_TOKEN + 1:, :].ravel(), minlength=N_EXPERTS)
                for layer in range(n_layers)]))
            state.append(np.stack([
                np.bincount(frame[layer, :, STATE_TOKEN, :].ravel(), minlength=N_EXPERTS)
                for layer in range(n_layers)]))
            episode_of.append(int(row["episode_index"]))
    return {
        "action": np.asarray(action, np.int64),
        "state": np.asarray(state, np.int64),
        "episode_of": np.asarray(episode_of),
        "summaries": summaries,
        "meta": json.loads((task_dir / "meta.json").read_text()),
    }


def _distribution(counts):
    total = counts.sum(-1, keepdims=True)
    return np.divide(counts, np.maximum(total, 1), dtype=np.float64)


def js_distance(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(np.where(mask, a * np.log2(np.where(mask, a / np.maximum(b, 1e-300), 1)), 0.0), -1)

    return float(np.mean(np.sqrt(np.maximum(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))))


def half(counts, episode_of, parity, n_steps, rng):
    take = np.flatnonzero(episode_of % 2 == parity)
    if len(take) < n_steps:
        raise ValueError("half has %d steps, need %d" % (len(take), n_steps))
    return _distribution(counts[rng.choice(take, n_steps, replace=False)].sum(0))


def compare(a, b, key, seed=0):
    """Split-half floors on both sides and the cross-slice distance, all at one size."""
    rng = np.random.default_rng(seed)
    sizes = []
    for run in (a, b):
        sizes += [int(np.sum(run["episode_of"] % 2 == parity)) for parity in (0, 1)]
    n = min(sizes)
    halves = {}
    for name, run in (("a", a), ("b", b)):
        for parity in (0, 1):
            halves[(name, parity)] = half(run[key], run["episode_of"], parity, n, rng)
    return {
        "equal_steps": n,
        "within_a": js_distance(halves[("a", 0)], halves[("a", 1)]),
        "within_b": js_distance(halves[("b", 0)], halves[("b", 1)]),
        # both cross terms use the same parity on each side, so an even/odd drift
        # cannot masquerade as a slice effect
        "cross": float(np.mean([js_distance(halves[("a", p)], halves[("b", p)])
                                for p in (0, 1)])),
        "cross_all_pairs": float(np.mean([js_distance(halves[("a", p)], halves[("b", q)])
                                          for p in (0, 1) for q in (0, 1)])),
    }


def behaviour(run):
    rows = run["summaries"]
    return {
        "episodes": len(rows),
        "success": sum(1 for r in rows if r["success"]),
        "mean_action_steps": float(np.mean([r["action_steps"] for r in rows])),
        "control_steps": int(run["action"].shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/libero30-right-v1")
    ap.add_argument("--control", default="corpus/slice-control-v1")
    ap.add_argument("--reference-summary", default="analysis/libero30-right-v1/summary.json",
                    help="for the between-task distance this has to be compared against")
    ap.add_argument("--out", default="analysis/slice-control-v1")
    args = ap.parse_args()

    corpus, control = pathlib.Path(args.corpus), pathlib.Path(args.control)
    reference = None
    ref_path = pathlib.Path(args.reference_summary)
    if ref_path.exists():
        reference = json.loads(ref_path.read_text())["summary"]

    pairs = []
    for suite_dir in sorted(control.glob("libero_*")):
        for task_dir in sorted(suite_dir.glob("t*")):
            meta = json.loads((task_dir / "meta.json").read_text())
            if meta.get("status") != "complete":
                print("skip %s/%s (status=%s)"
                      % (suite_dir.name, task_dir.name, meta.get("status")))
                continue
            original = corpus / suite_dir.name / task_dir.name
            if not original.exists():
                print("skip %s/%s (not in the main corpus)" % (suite_dir.name, task_dir.name))
                continue
            pairs.append((suite_dir.name, task_dir.name, original, task_dir))

    if not pairs:
        raise SystemExit("no task ran on both slices; the control has nothing to say")

    results = []
    for suite, task, path_a, path_b in pairs:
        a, b = load_task(path_a), load_task(path_b)
        entry = {
            "suite": suite,
            "task": task,
            "slice_a": a["meta"]["mig_profile"],
            "slice_b": b["meta"]["mig_profile"],
            "behaviour_a": behaviour(a),
            "behaviour_b": behaviour(b),
            "action": compare(a, b, "action"),
            "state": compare(a, b, "state"),
        }
        results.append(entry)
        print("\n=== %s/%s   %s vs %s ==="
              % (suite, task, entry["slice_a"], entry["slice_b"]))
        for side, label in (("behaviour_a", entry["slice_a"]), ("behaviour_b", entry["slice_b"])):
            item = entry[side]
            print("  %-9s %2d/%d success  %.1f action steps  %d control steps"
                  % (label, item["success"], item["episodes"],
                     item["mean_action_steps"], item["control_steps"]))
        for key in ("action", "state"):
            item = entry[key]
            print("  %-6s tokens  within %s %.4f | within %s %.4f | CROSS-SLICE %.4f  (n=%d steps)"
                  % (key, entry["slice_a"], item["within_a"], entry["slice_b"],
                     item["within_b"], item["cross"], item["equal_steps"]))

    print("\n=== verdict ===")
    if reference:
        print("corpus within-task floor      %.4f" % reference["within_task_split_half_JS"]["mean"])
        print("corpus between-task same suite %.4f" % reference["between_task_JS_same_suite"]["mean"])
    cross = [r["action"]["cross"] for r in results]
    floor = [x for r in results for x in (r["action"]["within_a"], r["action"]["within_b"])]
    print("this control: cross-slice %.4f (mean of %d tasks), own floor %.4f"
          % (float(np.mean(cross)), len(cross), float(np.mean(floor))))
    if reference:
        between = reference["between_task_JS_same_suite"]["mean"]
        share = float(np.mean(cross)) / between if between else float("nan")
        print("cross-slice / between-task = %.2f" % share)
        print("interpretation:", "slice effect is at the noise floor; the separability "
              "result stands" if float(np.mean(cross)) < 1.5 * float(np.mean(floor)) else
              "slice moves routing beyond the noise floor; the 0-4 vs 5-9 half of the "
              "between-task matrix is confounded")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "slice_control.json").write_text(json.dumps(
        {"pairs": results, "reference": reference}, indent=1))
    print("wrote %s" % (out / "slice_control.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
