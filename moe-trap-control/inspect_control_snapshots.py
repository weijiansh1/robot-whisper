#!/usr/bin/env python3
"""Render the exact model-input images at paired recovery snapshots, on CPU."""

import argparse
import json
from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

from collect_preflight_worker import input_digest
from collection_protocol import HERE
from collection_storage import atomic_json, digest, load_snapshot, records

sys.path.insert(0, str(HERE.parent / "himoe-libero-bridge/src"))
from himoe_libero_bridge.preprocess import build_policy_observation


def run(args):
    directory = args.run / "tasks" / args.main_id / "replay"
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "completed" or result["c0"]["status"] != "passed":
        raise ValueError("Only fully restored snapshots may be inspected")
    main = {int(row["query"]): row for row in records(Path(result["parent_directory"]) / "main")}
    events = sorted(result["events"], key=lambda e: e["start_query"])
    image = Image.new("RGB", (488, 110 + 266 * len(events)), "#fafafa")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=14)
    title = result["variant"]["base_task"].replace("_", " ")
    for i, line in enumerate(textwrap.wrap(title, 55)):
        draw.text((12, 8 + 17 * i), line, fill="#202020", font=font)
    draw.text((12, 77), "Native outcome: %s | %s" % ("success" if result["success"] else "failure", args.main_id[:12]), fill="#404040", font=font)
    evidence = []
    for i, event in enumerate(events):
        path = directory / "events" / event["event_id"] / "snapshot"
        snapshot = load_snapshot(path)
        request = build_policy_observation(snapshot["observation"], snapshot["prompt"])
        expected = main[event["start_query"]]["input_sha256"].item().decode()
        if input_digest(request) != expected:
            raise ValueError("Rendered inputs differ from the recorded model input")
        y = 110 + i * 266
        label = "After alarm" if event["deployable"] else "Before alarm (retrospective)"
        draw.text((12, y), "%s: q%d, step %d" % (label, event["start_query"], event["action_steps_before"]), fill="#202020", font=font)
        draw.text((12, y + 19), "Scene", fill="#555555", font=font)
        draw.text((252, y + 19), "Wrist", fill="#555555", font=font)
        image.paste(Image.fromarray(request["observation/image"]), (12, y + 38))
        image.paste(Image.fromarray(request["observation/wrist_image"]), (252, y + 38))
        evidence.append(dict(event_id=event["event_id"], input_sha256=expected,
            snapshot_manifest_sha256=digest(path / "manifest.json")))
    if args.output.exists():
        raise ValueError("Refusing to replace an inspection image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    atomic_json(args.output.with_suffix(".json"), dict(main_id=args.main_id, run=str(args.run), events=evidence,
        image_sha256=digest(args.output), renderer_sha256=digest(__file__), model_queries=0, gpu_compute=False))
    print(str(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--main-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
