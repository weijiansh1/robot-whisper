#!/usr/bin/env python3
"""Build a hash-verified catalog for raw and paired rolling-star videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


RAW_PATTERN = re.compile(
    r"formal/worker(?P<worker>\d+)/videos/s(?P<snapshot>\d+)_c"
    r"(?P<candidate>\d+)_(?P<outcome>success|failure)\.mp4$"
)
COMPARISON_PAIRS = {
    "pair_01_success_c07_vs_failure_c06.mp4": {
        "left": (0, 0, 7),
        "right": (0, 0, 6),
    },
    "pair_02_success_c03_vs_failure_c02.mp4": {
        "left": (0, 0, 3),
        "right": (0, 0, 2),
    },
    "pair_03_worker3_success_c07_vs_only_failure_c02.mp4": {
        "left": (3, 0, 7),
        "right": (3, 0, 2),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload["format"]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
        "duration_s": float(fmt["duration"]),
        "bytes": int(fmt["size"]),
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    analysis_dir = (args.analysis_dir or (run_root / "analysis")).resolve()
    labels = pd.read_csv(analysis_dir / "candidate_physical_labels.csv")
    label_index = labels.set_index(["worker", "snapshot", "candidate"])
    rows: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[int, int, int], str] = {}
    for path in sorted(run_root.glob("formal/worker*/videos/*.mp4")):
        relative = path.relative_to(run_root).as_posix()
        match = RAW_PATTERN.fullmatch(relative)
        if match is None:
            raise ValueError(f"unrecognized raw video path: {relative}")
        worker = int(match.group("worker"))
        snapshot = int(match.group("snapshot"))
        candidate = int(match.group("candidate"))
        key = (worker, snapshot, candidate)
        label = label_index.loc[key]
        outcome = match.group("outcome")
        if bool(label["success"]) != (outcome == "success"):
            raise ValueError(f"video/label outcome mismatch: {relative}")
        raw_by_key[key] = relative
        rows.append(
            {
                "kind": "raw",
                "path": relative,
                "worker": worker,
                "snapshot": snapshot,
                "candidate": candidate,
                "outcome": outcome,
                "primary_failure_type": (
                    "" if outcome == "success" else label["primary_failure_type"]
                ),
                "left_source": "",
                "right_source": "",
                "sha256": sha256_file(path),
                **probe(path),
            }
        )
    for path in sorted((run_root / "comparison_videos").glob("*.mp4")):
        pair = COMPARISON_PAIRS.get(path.name)
        if pair is None:
            raise ValueError(f"comparison video has no source mapping: {path.name}")
        left_key = pair["left"]
        right_key = pair["right"]
        if left_key not in raw_by_key or right_key not in raw_by_key:
            raise ValueError(f"comparison source is missing for {path.name}")
        right_label = label_index.loc[right_key]
        rows.append(
            {
                "kind": "comparison",
                "path": path.relative_to(run_root).as_posix(),
                "worker": right_key[0],
                "snapshot": right_key[1],
                "candidate": right_key[2],
                "outcome": "success_left_failure_right",
                "primary_failure_type": right_label["primary_failure_type"],
                "left_source": raw_by_key[left_key],
                "right_source": raw_by_key[right_key],
                "sha256": sha256_file(path),
                **probe(path),
            }
        )
    columns = [
        "kind",
        "path",
        "worker",
        "snapshot",
        "candidate",
        "outcome",
        "primary_failure_type",
        "left_source",
        "right_source",
        "width",
        "height",
        "frame_rate",
        "duration_s",
        "bytes",
        "sha256",
    ]
    with (run_root / "video_catalog.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    comparisons = [row for row in rows if row["kind"] == "comparison"]
    raw = [row for row in rows if row["kind"] == "raw"]
    markdown = [
        "# Video catalog",
        "",
        "## Matched comparisons",
        "",
        "Success is on the left and the matched failure is on the right.",
        "",
        "| video | failure type | duration (s) | resolution | sha256 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in comparisons:
        markdown.append(
            f"| [{Path(row['path']).name}]({row['path']}) | "
            f"`{row['primary_failure_type']}` | {row['duration_s']:.2f} | "
            f"{row['width']}x{row['height']} | `{row['sha256']}` |"
        )
    markdown.extend(
        [
            "",
            "## Raw videos",
            "",
            "| video | outcome | failure type | duration (s) | sha256 |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in raw:
        markdown.append(
            f"| [{Path(row['path']).name}]({row['path']}) | {row['outcome']} | "
            f"`{row['primary_failure_type']}` | {row['duration_s']:.2f} | "
            f"`{row['sha256']}` |"
        )
    (run_root / "VIDEO_CATALOG.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"cataloged {len(comparisons)} comparisons and {len(raw)} raw videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
