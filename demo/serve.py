#!/usr/bin/env python3
"""Static server for the demo, with byte-range support.

`python3 -m http.server` answers a Range request with a full 200, so a browser
cannot seek inside the rollout clips and `video.currentTime` silently does
nothing — the panels stay black.  This handler answers 206 properly.

    python3 demo/serve.py [port]     # default 8899, serves demo/
"""

from __future__ import annotations

import functools
import http.server
import os
import re
import socketserver
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if os.path.isdir(path) or not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        a, b = m.group(1), m.group(2)
        start = int(a) if a else max(0, size - int(b or 0))
        end = int(b) if (b and a) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_remaining", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(65536, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(
            ("0.0.0.0", port), functools.partial(Handler, directory=ROOT)) as srv:
        print(f"serving {ROOT} at http://0.0.0.0:{port}/index.html")
        srv.serve_forever()


if __name__ == "__main__":
    main()
