#!/usr/bin/env python3
"""Emit rescue.json — one rollout, one alarm, one intervention, told on a single timeline.

What happened
-------------
A LIBERO rollout (task 8, init state 7) was run with the routing detector scoring every
control step online.  At q32 the detector fired: the chunk-to-chunk change in the router's
top-8 sets had spent three consecutive steps below 0.95x the rollout's own opening rate.

At that step the simulator state was saved and eight candidate action chunks were drawn
from it, each on its own flow-noise stream.  Executing one of them and running on to the
horizon recovers the task in eight steps.  Leaving the original chunk in place does not:
the trunk runs the full 52-step horizon and fails.

    q0 ............ q32 ............ q40 ...... q52
                     |                 |          |
                     alarm             rescued    trunk gives up
                     fork here         succeeds   (horizon)

So the page is one timeline, not two views.  Before q32 both video cells show the same
trajectory, because it *is* the same trajectory.  After q32 they diverge.

What this does and does not show
--------------------------------
It shows that the detector's firing point is actionable: intervening there converted a
rollout that failed into one that succeeded.  It does not show that the *moment* is
special -- forking at a random earlier step rescued equally often (3/8 vs 3/8) -- nor that
routing can pick *which* candidate to keep: on this rollout the largest-routing-change
candidate failed and the smallest succeeded, on a sample of one.

Nor is every state rescuable.  The same procedure on init state 0 rescued 0 of 8.

Usage:  python3 demo/build_rescue_data.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "analysis_triggered_fork"

FPS, FRAMES_PER_STEP = 20, 10

# The trunk and the rescued continuation come from the same saved state at the alarm.
# `rescue_i7` re-ran that trunk and wrote its video; `min_c01_success` is the continuation
# that recovered.  The two forked-and-failed clips are kept so the page can show that most
# candidates did not recover.
TRUNK = RUNS / "rescue_i7" / "trunk_failure.mp4"
RESCUED = RUNS / "rescue_i7" / "min_c01_success.mp4"
NOT_RESCUED = RUNS / "rescue_i7" / "max_c05_failure.mp4"
MANIFEST = RUNS / "rescue_i7" / "manifest.json"
FORK_MANIFEST = RUNS / "i7_s20260830" / "manifest.json"


def remux(src: Path, name: str) -> str | None:
    """Copy with the moov atom moved to the front, so the browser can seek it."""
    if not src.is_file():
        print(f"  ! missing {src}")
        return None
    out = HERE / "video"
    out.mkdir(exist_ok=True)
    dst = out / name
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-c", "copy", "-movflags", "+faststart", str(dst)],
                       capture_output=True)
    if r.returncode:
        shutil.copy(src, dst)
        print(f"  ! remux failed for {src.name}, copied verbatim")
    print(f"  {name:22s} <- {src.name}")
    return f"video/{name}"


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return round(float(out.stdout.strip()), 3)
    except ValueError:
        return 0.0


def main() -> None:
    m = json.loads(MANIFEST.read_text())
    fork = json.loads(FORK_MANIFEST.read_text()) if FORK_MANIFEST.is_file() else {}

    alarm = int(m["alarm_query"])
    trunk = m["trunk"]
    arms = m["arms"]
    rescued = arms["min"]           # the continuation that recovered
    failed_arm = arms["max"]

    print("videos")
    videos = dict(
        trunk=remux(TRUNK, "rescue_trunk.mp4"),
        rescued=remux(RESCUED, "rescue_ok.mp4"),
        not_rescued=remux(NOT_RESCUED, "rescue_bad.mp4"),
    )

    doc = dict(
        config=dict(theta=m["detector"]["theta"], W=m["detector"]["W"],
                    W0=m["detector"]["W0"], K=m["detector"]["K"],
                    layers=m["detector"]["layers"], topk=m["detector"]["topk"],
                    fps=FPS, frames_per_step=FRAMES_PER_STEP),
        task=m["task"],
        alarm=alarm,
        trunk=dict(queries=trunk["queries"], success=trunk["success"],
                   ratios=trunk["ratios"],
                   # raw chunk-to-chunk pattern distances drive the film's S1/S2 beats
                   distances=trunk.get("distances"),
                   duration=duration(TRUNK)),
        rescued=dict(queries=rescued["queries"], ends_at=alarm + rescued["queries"],
                     candidate=rescued["candidate"],
                     routing_change=rescued["routing_change"],
                     duration=duration(RESCUED)),
        not_rescued=dict(queries=failed_arm["queries"],
                         ends_at=alarm + failed_arm["queries"],
                         candidate=failed_arm["candidate"],
                         routing_change=failed_arm["routing_change"],
                         duration=duration(NOT_RESCUED)),
        candidate_scores=m["candidate_scores"],
        trunk_change_at_alarm=m["trunk_change_at_alarm"],
        # Honest scope, rendered on the page rather than left to the reader.
        caveats=dict(
            fork_successes=fork.get("triggered", {}).get("successes"),
            fork_n=fork.get("triggered", {}).get("n"),
            control_successes=None,
            note="forking at a random earlier step rescued equally often (3/8 vs 3/8); "
                 "init state 0 rescued 0/8",
        ),
        videos=videos,
    )
    (HERE / "rescue.json").write_text(json.dumps(doc, separators=(",", ":")))

    print(f"\nalarm at q{alarm} · trunk {trunk['queries']} steps, "
          f"{'success' if trunk['success'] else 'FAILURE'}")
    print(f"rescued continuation: {rescued['queries']} steps, ends at q{alarm + rescued['queries']}")
    print(f"rescue.json  {(HERE / 'rescue.json').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
