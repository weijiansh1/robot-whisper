"""Bracket the achievable compression, and measure recorder overhead.

The end-to-end test's trace is near-constant (random-init experts, slowly
varying input), so its ratio is an optimistic bound.  Uniform-random routing
is the pessimistic bound.  A real trace lands in between; rerun this on one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

sys.path.insert(0, str(Path(__file__).parent))
from himoe_route_store import ZarrRouteReader  # noqa: E402

N, L, D, S, K, E = 128, 8, 10, 51, 4, 32


def du(p) -> int:
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0])


def size_of(name, arrays, shuffle=None, level=3, chunk=64) -> int:
    path = f"/tmp/bnd_{name}"
    if Path(path).exists():
        shutil.rmtree(path)
    root = zarr.create_group(store=path, overwrite=True)
    for k, a in arrays.items():
        comp = (
            [ZstdCodec(level=level)]
            if shuffle is None or a.dtype.itemsize == 1
            else [BloscCodec(cname="zstd", clevel=level, shuffle=shuffle, typesize=a.dtype.itemsize)]
        )
        root.create_array(name=k, shape=a.shape, chunks=(chunk, *a.shape[1:]),
                          dtype=a.dtype, compressors=comp)[:] = a
    return du(path)


def report(label, arrays):
    raw = sum(a.nbytes for a in arrays.values())
    z = size_of(label + "_z", arrays, None)
    b = size_of(label + "_b", arrays, "bitshuffle")
    print(f"{label:24s} raw {raw:>10,}  zstd-3 {z:>9,} ({raw/z:6.1f}x)  "
          f"blosc-bitshuf {b:>9,} ({raw/b:6.1f}x)   {min(z,b)/N/1024:6.2f} KiB/step")


def main() -> int:
    rng = np.random.default_rng(0)

    print("=== compression bracket ===")
    # pessimistic: uniform-random routing, no structure at all
    rand = {
        "hb_expert_ids": rng.integers(0, E, (N, L, D, S, K), dtype=np.uint8),
        "hb_selected_prob": rng.random((N, L, D, S, K)).astype(np.float16),
        "hb_entropy": rng.random((N, L, D, S)).astype(np.float16),
    }
    report("uniform random (floor)", rand)

    # plausible: each layer has a few dominant experts, routing drifts slowly
    ids = np.empty((N, L, D, S, K), np.uint8)
    for l in range(L):
        p = rng.dirichlet(np.full(E, 0.35))
        ids[:, l] = rng.choice(E, size=(N, D, S, K), p=p).astype(np.uint8)
    keep = rng.random((N, L, D, S, K)) < 0.7  # 70% of slots persist across steps
    ids = np.where(keep, ids[:1], ids)
    probs = (0.25 + 0.05 * rng.standard_normal((N, L, D, S, K))).astype(np.float16)
    skewed = {
        "hb_expert_ids": ids,
        "hb_selected_prob": probs,
        "hb_entropy": (2.5 + 0.3 * rng.standard_normal((N, L, D, S))).astype(np.float16),
    }
    report("skewed+sticky (likely)", skewed)

    # observed in the end-to-end run
    r = ZarrRouteReader("/tmp/routes.zarr")
    obs = {k: np.asarray(r[k][:]) for k in ("hb_expert_ids", "hb_selected_prob", "hb_entropy")}
    report("test trace (ceiling)", obs)
    uniq = len(np.unique(obs["hb_expert_ids"].reshape(N, -1), axis=0))
    print(f"  ^ degenerate: only {uniq}/{N} distinct routing patterns across control steps")

    print("\n=== recorder overhead ===")
    sys.path.insert(0, "/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/"
                       "cache/upstream/HiMoE-VLA/src")
    from moevla.models.himoe import ASMoEConfig, HBMoEConfig
    from moevla.models.modeling_moe import ASMoE, HBMoE
    from test_end_to_end import FakeExpert
    from himoe_router_recorder import HiMoERouteRecorder

    torch.manual_seed(0)
    core = FakeExpert().cuda().float().eval()
    x0 = torch.randn(1, S, 1024, device="cuda")
    dm = (torch.rand(1, 24, device="cuda") > 0.5).float()

    def run(steps, recorder=None):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(steps):
                if recorder:
                    recorder.begin_control_step(0, i)
                x = x0
                for _ in range(D):
                    x = core(x, dm)
                if recorder:
                    recorder.end_control_step()

    run(3)
    torch.cuda.synchronize(); t = time.time(); run(20); torch.cuda.synchronize()
    base = (time.time() - t) / 20

    rec = HiMoERouteRecorder(core, store_full_probs=False).attach()
    run(3, rec)
    torch.cuda.synchronize(); t = time.time(); run(20, rec); torch.cuda.synchronize()
    hooked = (time.time() - t) / 20
    rec.close()

    rec2 = HiMoERouteRecorder(core, store_full_probs=True).attach()
    run(3, rec2)
    torch.cuda.synchronize(); t = time.time(); run(20, rec2); torch.cuda.synchronize()
    full = (time.time() - t) / 20
    rec2.close()

    print(f"  no hooks            {base*1000:7.1f} ms/control step")
    print(f"  recorder (lean)     {hooked*1000:7.1f} ms  (+{(hooked/base-1)*100:.1f}%)")
    print(f"  recorder (full probs){full*1000:6.1f} ms  (+{(full/base-1)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
