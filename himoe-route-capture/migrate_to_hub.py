"""Move a captured routing corpus into the VLA_MUI_HUB tree.

The hub already ships the destination: 74 empty task directories under
``cache/HiMoE-VLA/<benchmark>/<task>/`` (each holding a ``.gitkeep``).  This
fills 30 of them, one ``<run_id>`` subdirectory per task, and adds nothing to
the hub's layout.

What does NOT move: INDEX.csv, VALIDATION.json, analysis/, _logs/, _preflight/,
_tasks.json.  Those describe the run as a whole rather than any one task, they
are derived (regenerable from the moved data), and the hub -- like the MUI_HUB
it mirrors -- has no cross-run level to put them in.  They stay in the capture
repo and are rebuilt against the hub paths after the move.

Same filesystem, so the move is a rename: instant, no copy, no extra space.
Verifies before and after; refuses to clobber an existing run_id.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import corpus_layout as cl


def plan_moves(root, run_id, hub_root, model):
    """(planned task, source, destination) for every complete task."""
    moves = []
    for planned in cl.iter_planned_tasks(root):
        source = cl.task_dir(root, planned["benchmark"], planned["task_id"], planned["name"])
        if not source.is_dir():
            continue
        destination = cl.hub_task_dir(planned["name"], planned["benchmark"],
                                      run_id, hub_root, model)
        moves.append((planned, source, destination))
    return moves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="captured corpus root")
    ap.add_argument("--run-id", required=True, help="<tag>_<YYYYMMDD_HHMMSS>")
    ap.add_argument("--hub-root", default=str(cl.HUB_ROOT))
    ap.add_argument("--model", default=cl.HUB_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    hub_root = pathlib.Path(args.hub_root)
    moves = plan_moves(root, args.run_id, hub_root, args.model)
    if not moves:
        raise SystemExit("nothing to move under %s" % root)

    # pre-flight: the hub task directory must already exist (the hub owns the
    # task list; creating one here would mean the name does not match the hub's)
    problems = []
    for planned, source, destination in moves:
        if not destination.parent.is_dir():
            problems.append("hub has no task dir for %s/%s"
                            % (planned["benchmark"], planned["name"]))
        if destination.exists():
            problems.append("run_id already present: %s" % destination)
        if not cl.is_complete(source):
            problems.append("not complete, refusing to move: %s" % source.name)
    if problems:
        for problem in problems:
            print("BLOCKED:", problem)
        return 1

    print("%d tasks -> %s/cache/%s/<benchmark>/<task>/%s/"
          % (len(moves), hub_root, args.model, args.run_id))
    if args.dry_run:
        for planned, source, destination in moves[:3]:
            print("  %s\n    -> %s" % (source, destination))
        print("  ... (%d more)" % max(0, len(moves) - 3))
        return 0

    moved = []
    for planned, source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        cl.write_task_meta(destination, status=cl.STATUS_COMPLETE, run_id=args.run_id,
                           hub_benchmark=planned["benchmark"], hub_model=args.model)
        moved.append(destination)
    print("moved %d task directories" % len(moved))

    # the source suite dirs are now empty shells; the run-level files stay put
    for benchmark in cl.BENCHMARKS:
        suite = cl.suite_dir(root, benchmark)
        leftovers = [p for p in suite.iterdir()] if suite.is_dir() else []
        if leftovers and all(p.name == "_suite.json" for p in leftovers):
            shutil.move(str(suite / "_suite.json"),
                        str(hub_root / "cache" / args.model / benchmark / "_suite.json"))
            suite.rmdir()
        elif not leftovers and suite.is_dir():
            suite.rmdir()
        elif leftovers:
            print("left in place (unexpected contents): %s -> %s"
                  % (suite, [p.name for p in leftovers]))

    cl.update_manifest(root, hub={"root": str(hub_root), "model": args.model,
                                  "run_id": args.run_id,
                                  "layout": "cache/<model>/<benchmark>/<task>/<run_id>/"})
    print("recorded hub location in %s/MANIFEST.json" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
