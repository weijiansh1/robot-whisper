"""Synthetic load generator for the policy servers: replays one saved observation, no simulator.

Usage: bench_server_load.py --ports 9510,9530 --connections 4 --seconds 60
Prints aggregate requests/second and per-request latency percentiles; used to pick
how many server processes per GPU saturate the GPU.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import os
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)   # websockets>=14 would otherwise route 127.0.0.1 through the container proxy
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))
from himoe_libero_bridge.client import PolicyClient  # noqa: E402


def load_observation(trace):
    with np.load(trace) as z:
        img, wrist, state = z["images"][0], z["wrist_images"][0], z["states"][0]
        noise = z["flow_noises"][0]
    import json
    summary = json.loads((Path(trace).parent / "summary.json").read_text())
    return {"observation/image": img.astype(np.uint8), "observation/wrist_image": wrist.astype(np.uint8),
            "observation/state": state.astype(np.float32), "prompt": summary["selected_task_prompt"],
            "flow/noise": noise.astype(np.float32)}


def worker(port, obs, seconds, out, errors):
    lat = []
    try:
        for attempt in range(5):
            try:
                client = PolicyClient("127.0.0.1", port, connect_timeout=30, inference_timeout=120)
                break
            except Exception as error:  # noqa: BLE001
                time.sleep(1.0)
                if attempt == 4:
                    raise
        with client:
            end = time.time() + seconds
            while time.time() < end:
                t0 = time.perf_counter()
                client.infer(dict(obs))
                lat.append(time.perf_counter() - t0)
    except Exception as error:  # noqa: BLE001
        errors.append("%s: %r" % (port, error))
    out.append((port, lat))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ports", required=True)
    p.add_argument("--connections", type=int, default=4, help="concurrent connections per port")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--trace", default=None)
    a = p.parse_args()
    trace = a.trace or sorted((ROOT / "simulations").glob("pro-local/episode-*/episode-trace.npz"))[0]
    obs = load_observation(trace)
    ports = [int(x) for x in a.ports.split(",")]
    out, threads, errors = [], [], []
    t0 = time.time()
    for port in ports:
        for _ in range(a.connections):
            t = threading.Thread(target=worker, args=(port, obs, a.seconds, out, errors))
            t.start()
            threads.append(t)
            time.sleep(0.3)
    for t in threads:
        t.join()
    wall = time.time() - t0
    total = sum(len(l) for _, l in out)
    lat = np.concatenate([l for _, l in out if l]) if total else np.array([np.nan])
    if errors:
        print("errors: %d, e.g. %s" % (len(errors), errors[0][:200]), flush=True)
    print("ports=%s connections/port=%d wall=%.0fs requests=%d  -> %.2f req/s total, %.2f req/s per port; latency p50 %.0f ms p90 %.0f ms"
          % (ports, a.connections, wall, total, total / wall, total / wall / len(ports), 1000 * np.median(lat), 1000 * np.percentile(lat, 90)), flush=True)


if __name__ == "__main__":
    main()
