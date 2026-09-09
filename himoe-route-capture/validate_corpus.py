"""Integrity pass over a captured routing corpus.

Structural checks live in ``corpus_layout.verify_task``; this adds everything
that needs the zarr open and the frozen task table cross-referenced:

* ``episode_id`` in the store matches the client's ``episode_index`` for every
  slice -- the check that the old accumulate-``inference_calls`` scheme could not
  do, because a dropped chunk just shifts every later episode silently;
* ``control_step`` is globally contiguous (same dropped-chunk detector
  ``within64_lib`` relies on);
* the prompt the client actually sent matches the frozen task language;
* the server's checkpoint sha256 is the one ``suites.py`` pins for that suite;
* per-suite success rate against the published Section 4.1 numbers.

Needs zarr.  Exit code is non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import zarr

import corpus_layout as cl

# Reference success rates.  goal/spatial/object are OUR Section 4.1 reproduction
# (checkpoint-right, 50 episodes/task); long has never been reproduced here, so the
# only number available is the paper's own claim -- flagged as such, never as a repro.
SECTION_41 = {"goal": 97.8, "spatial": 94.2, "object": 96.6, "long": 95.8}
REFERENCE_SOURCE = {"goal": "repro", "spatial": "repro", "object": "repro", "long": "paper-claim"}
EXPECTED_SHA = {
    "long": "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256",
    "goal": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
    "spatial": "1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137",
    "object": "f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa",
}


def check_task(root, task, expect_episodes, resolve=None):
    path = resolve(task) if resolve else cl.task_dir(
        root, task["benchmark"], task["task_id"], task["name"])
    # canonical key, not path.name: in the hub layout the leaf directory is the
    # run_id and is identical for every task
    result = {"task_dir": cl.task_dir_name(task["task_id"], task["name"]),
              "suite": task["suite"], "path": str(path), "problems": [],
              "episodes": 0, "successes": 0}
    if not path.is_dir():
        result["problems"].append("missing")
        return result

    structural = cl.verify_task(path, expect_episodes=expect_episodes)
    result["problems"].extend(structural["problems"])
    result["episodes"] = structural["episodes"]
    result["successes"] = structural["successes"]
    if structural["episodes"] == 0:
        return result

    summaries = cl.load_summaries(path)
    meta = cl.read_task_meta(path) or {}

    # prompt actually sent vs frozen table
    language = task.get("language", "")
    bad_prompt = {item.get("prompt") for item in summaries} - {language}
    if language and bad_prompt:
        result["problems"].append("prompt mismatch: %r vs frozen %r"
                                  % (sorted(bad_prompt)[:1], language))

    # checkpoint identity
    server_meta_path = cl.client_dir(path) / "server_metadata.json"
    if server_meta_path.is_file():
        server_meta = json.loads(server_meta_path.read_text())
        actual = server_meta.get("checkpoint_sha256", "")
        if actual != EXPECTED_SHA[task["suite"]]:
            result["problems"].append("checkpoint sha256 %s not the pinned one" % actual[:12])
        if server_meta.get("libero_wrist_layout") != meta.get("wrist_layout"):
            result["problems"].append("wrist layout %s != meta %s"
                                      % (server_meta.get("libero_wrist_layout"),
                                         meta.get("wrist_layout")))

    sampling = meta.get("sampling")
    if sampling and not sampling.get("complete", True):
        result["problems"].append(
            "sampling incomplete: %s/%s episodes" % (sampling.get("actual_episodes"),
                                                     sampling.get("designed_episodes")))

    # zarr: episode labelling and contiguity
    group = zarr.open_group(str(cl.server_dir(path) / "routes.zarr"), mode="r")
    episode_id = np.asarray(group["episode_id"][:])
    control_step = np.asarray(group["control_step"][:])
    if control_step.size and not (np.diff(control_step) == 1).all():
        result["problems"].append("control_step not contiguous; a chunk was dropped")

    cursor = 0
    for item in summaries:
        count = int(item.get("inference_calls", 0))
        chunk = episode_id[cursor:cursor + count]
        expected = int(item["episode_index"])
        if chunk.size != count:
            result["problems"].append("zarr short at episode %d" % expected)
            break
        if not np.all(chunk == expected):
            result["problems"].append(
                "episode_id mismatch at episode %d: saw %s" % (expected, np.unique(chunk)[:3]))
            break
        cursor += count
    result["control_steps"] = int(episode_id.size)
    result["ok"] = not result["problems"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="capture-layout root (omit when using --run-id)")
    ap.add_argument("--run-id", help="validate a run inside the hub")
    ap.add_argument("--hub-root", default=str(cl.HUB_ROOT))
    ap.add_argument("--model", default=cl.HUB_MODEL)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", help="write REPORT.json here")
    args = ap.parse_args()

    if args.run_id:
        tasks = list(cl.iter_hub_tasks(args.hub_root))
        resolve = cl.hub_resolver(args.run_id, args.hub_root, args.model)
    elif args.root:
        tasks, resolve = list(cl.iter_planned_tasks(args.root)), None
    else:
        raise SystemExit("give --run-id or --root")
    results = [check_task(args.root, task, args.episodes, resolve) for task in tasks]
    present = [r for r in results if "missing" not in r["problems"]]

    print("%-4s %-58s %-8s %s" % ("", "task", "success", "problems"))
    for item in present:
        print("%-4s %-58s %2d/%-5d %s"
              % ("ok " if item.get("ok") else "FAIL", item["task_dir"][:58],
                 item["successes"], item["episodes"],
                 "; ".join(item["problems"]) or "-"))

    print("\n=== per-suite success rate vs published Section 4.1 ===")
    suite_stats = {}
    for suite in ("goal", "spatial", "object", "long"):
        rows = [r for r in present if r["suite"] == suite]
        episodes = sum(r["episodes"] for r in rows)
        successes = sum(r["successes"] for r in rows)
        if not episodes:
            continue
        rate = 100.0 * successes / episodes
        suite_stats[suite] = {"tasks": len(rows), "episodes": episodes,
                              "successes": successes, "rate": round(rate, 1),
                              "published": SECTION_41[suite]}
        print("%-8s %3d tasks  %4d ep  %5.1f%%   ref %.1f%% (%s)   delta %+.1f pp"
              % (suite, len(rows), episodes, rate, SECTION_41[suite],
                 REFERENCE_SOURCE[suite], rate - SECTION_41[suite]))

    failed = [r["task_dir"] for r in present if not r.get("ok")]
    missing = [r["task_dir"] for r in results if "missing" in r["problems"]]
    print("\n%d/%d planned tasks present, %d failed, %d missing"
          % (len(present), len(results), len(failed), len(missing)))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"tasks": results, "suites": suite_stats,
             "failed": failed, "missing": missing}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
