#!/usr/bin/env python3
"""Build a dependency-free, file://-ready MoE trajectory viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent.parent / "viz" / "himoe-moe-terminal-trajectory.html"


def escape_script_body(text: str) -> str:
    return text.replace("</script", r"<\/script").replace("</SCRIPT", r"<\/SCRIPT")


def build(output: Path) -> None:
    html = (HERE / "index.html").read_text()
    css = (HERE / "styles.css").read_text()
    app = (HERE / "app.js").read_text()
    data_text = (HERE / "data.json").read_text().strip()
    payload = json.loads(data_text)

    stylesheet = '<link rel="stylesheet" href="styles.css">'
    application = '<script src="app.js" defer></script>'
    if html.count(stylesheet) != 1 or html.count(application) != 1:
        raise RuntimeError("index.html asset markers changed")

    html = html.replace(stylesheet, f"<style>\n{css}\n</style>")
    inline = (
        '<script id="trajectory-data" type="application/json">\n'
        f"{escape_script_body(data_text)}\n"
        "</script>\n"
        "<script>\n"
        f"{escape_script_body(app)}\n"
        "</script>"
    )
    html = html.replace(application, inline)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)

    built = output.read_text()
    if '<script src="' in built or 'rel="stylesheet"' in built:
        raise RuntimeError("generated viewer still references external assets")
    if built.count('id="trajectory-data"') != 1:
        raise RuntimeError("generated viewer is missing embedded data")
    schema = payload.get("schema", {})
    if schema.get("name") != "himoe_moe_control_trajectory_v1" or schema.get("version") != 2:
        raise RuntimeError("unexpected trajectory data schema")
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
