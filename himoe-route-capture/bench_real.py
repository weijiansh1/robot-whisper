"""Storage benchmark on the REAL routing traces (180 LIBERO episodes).

Replaces the synthetic bracket in bench_bounds.py.  Note the real LIBERO trace
shape is [T, flow=10, layer=8, action=10, topk=4] -- n_action_steps is 10 for
the LIBERO checkpoints, not the 50 of the upstream default config, so the
per-control-step footprint is much smaller than the earlier estimate.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

RUNS = ["runs/right50", "runs/left50", "runs/within"]
ROOT = pathlib.Path("/home/jovyan/work/himoe-route-capture")


def du(p) -> int:
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True,
                              text=True).stdout.split()[0])


def collect():
    ids, wts, ep_index = [], [], []
    for run in RUNS:
        d = ROOT / run
        sm = json.loads((d / "summaries.json").read_text())
        for s in sm:
            idx = s.get("episode_index", s["init_state_id"])
            z = np.load(d / ("episode_%02d.npz" % idx))
            a = z["expert_ids"].astype(np.uint8)
            b = z["expert_weights"].astype(np.float16)
            ids.append(a)
            wts.append(b)
            ep_index.append(np.full(len(a), len(ep_index), np.int32))
    return np.concatenate(ids), np.concatenate(wts), np.concatenate(ep_index)


def write_zarr(path, arrays, chunk, codec_for):
    if pathlib.Path(path).exists():
        shutil.rmtree(path)
    root = zarr.create_group(store=path, overwrite=True)
    for name, a in arrays.items():
        root.create_array(name=name, shape=a.shape, chunks=(chunk, *a.shape[1:]),
                          dtype=a.dtype, compressors=codec_for(a.dtype))[:] = a
    return du(path)


def main():
    ids, wts, ep = collect()
    n = len(ids)
    arrays = {"expert_ids": ids, "expert_weights": wts, "episode": ep}
    raw = sum(a.nbytes for a in arrays.values())
    print("真实 trace: %d 个 control step, 来自 180 集" % n)
    print("单个 control step 形状 %s (flow, layer, action, topk)" % (ids.shape[1:],))
    for k, a in arrays.items():
        print("  %-15s %-8s %-24s %10s B" % (k, a.dtype, str(a.shape), f"{a.nbytes:,}"))
    print("\n未压缩合计 %s B = %.2f KiB / control step\n" % (f"{raw:,}", raw / n / 1024))

    print("%-40s %12s %8s %11s" % ("config", "bytes", "ratio", "B/step"))
    print("-" * 74)

    def row(lab, size):
        print("%-40s %12s %7.2fx %10.0f" % (lab, f"{size:,}", raw / size, size / n))

    np.savez_compressed("/tmp/real.npz", **arrays)
    row("npz_compressed (ZIP deflate)", du("/tmp/real.npz"))

    for lvl in (1, 3, 5, 9, 19):
        row("zarr chunk=256  zstd-%-2d (no shuffle)" % lvl,
            write_zarr("/tmp/zr_z%d" % lvl, arrays, 256, lambda dt: [ZstdCodec(level=lvl)]))

    def mixed(lvl):
        def f(dt):
            if dt.itemsize == 1:
                return [ZstdCodec(level=lvl)]
            return [BloscCodec(cname="zstd", clevel=lvl, shuffle="bitshuffle",
                               typesize=dt.itemsize)]
        return f

    for lvl in (3, 5, 9):
        row("zarr chunk=256  zstd-%-2d + fp16 bitshuffle" % lvl,
            write_zarr("/tmp/zr_m%d" % lvl, arrays, 256, mixed(lvl)))

    for chunk in (32, 128, 256, 1024, 4096):
        row("zarr chunk=%-5d zstd-5 + fp16 bitshuffle" % chunk,
            write_zarr("/tmp/zr_c%d" % chunk, arrays, chunk, mixed(5)))

    print("\n=== 每个数组单独看 ===")
    for k in ("expert_ids", "expert_weights"):
        a = {k: arrays[k]}
        z = write_zarr("/tmp/one_z", a, 256, lambda dt: [ZstdCodec(level=5)])
        b = write_zarr("/tmp/one_b", a, 256, mixed(5))
        print("  %-15s raw %10s  zstd-5 %10s (%.2fx)  +bitshuffle %10s (%.2fx)"
              % (k, f"{arrays[k].nbytes:,}", f"{z:,}", arrays[k].nbytes / z,
                 f"{b:,}", arrays[k].nbytes / b))

    print("\n=== 与之前合成数据估计的对照 ===")
    best = write_zarr("/tmp/zr_best", arrays, 256, mixed(5))
    print("  合成「均匀随机」下界估计: 1.3x")
    print("  合成「偏态+粘滞」估计:   1.5x")
    print("  真实 trace 实测:          %.2fx" % (raw / best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
