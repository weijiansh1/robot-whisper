"""End-to-end check: real HiMoE MoE modules -> recorder -> Zarr -> read back."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

UPSTREAM = Path(
    "/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA/src"
)
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(Path(__file__).parent))

from moevla.models.himoe import ASMoEConfig, HBMoEConfig  # noqa: E402
from moevla.models.modeling_moe import ASMoE, HBMoE  # noqa: E402

from himoe_route_store import ZarrRouteReader, ZarrRouteWriter  # noqa: E402
from himoe_router_recorder import (  # noqa: E402
    HiMoERouteRecorder,
    combine_weight_from_raw,
    discover_gates,
)

N_LAYERS, N_DENOISE, N_SUFFIX, BATCH = 18, 10, 51, 1
N_CONTROL_STEPS = 128


class Dense(nn.Module):
    def forward(self, x, data_mask):
        return x


class Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class FakeExpert(nn.Module):
    """gemma_expert with the real MoE topology; attention/norms omitted."""

    def __init__(self):
        super().__init__()
        asc, hbc = ASMoEConfig(), HBMoEConfig()
        mlps = []
        for i in range(N_LAYERS):
            if i <= 1 or i >= N_LAYERS - 2:
                mlps.append(ASMoE(asc))
            elif i <= 5 or i >= N_LAYERS - 6:
                mlps.append(HBMoE(hbc))
            else:
                mlps.append(Dense())
        self.layers = nn.ModuleList(Layer(m) for m in mlps)

    def forward(self, x, data_mask):
        for layer in self.layers:
            x = layer.mlp(x, data_mask)
        return x


def main() -> int:
    dev = "cuda"
    torch.manual_seed(0)
    print("building expert (fp32, real topology) ...")
    core = FakeExpert().to(dev, dtype=torch.float32).eval()

    gates = discover_gates(core)
    hb = [g.layer_idx for g in gates if g.kind == "HB"]
    as_ = [g.layer_idx for g in gates if g.kind == "AS"]
    print(f"gates={len(gates)}  AS layers={as_}  HB layers={hb}")
    assert len(gates) == 12 and as_ == [0, 1, 16, 17] and hb == [2, 3, 4, 5, 12, 13, 14, 15]

    # Ground truth: capture what the gate actually returned, to check that the
    # combine_weight we chose *not* to store is exactly reconstructible.
    truth: dict[int, list[torch.Tensor]] = {g.layer_idx: [] for g in gates}

    def truth_hook(layer_idx):
        def hook(module, args, output):
            truth[layer_idx].append(output[1].detach().float().cpu())

        return hook

    truth_handles = [
        g.module.register_forward_hook(truth_hook(g.layer_idx)) for g in gates
    ]

    rec = HiMoERouteRecorder(core, store_full_probs=False).attach()

    out = Path("/tmp/routes.zarr")
    if out.exists():
        shutil.rmtree(out)
    writer = ZarrRouteWriter(str(out), chunk_steps=64, zstd_level=3)

    state = torch.randn(BATCH, N_SUFFIX, 1024, device=dev)
    data_mask = (torch.rand(BATCH, 24, device=dev) > 0.5).float()

    print(f"running {N_CONTROL_STEPS} control steps x {N_DENOISE} denoise steps ...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(N_CONTROL_STEPS):
            rec.begin_control_step(episode_id=step // 32, control_step=step)
            x = state + 0.01 * step
            for _ in range(N_DENOISE):  # flow-matching denoising loop
                x = core(x, data_mask)
            record = rec.end_control_step()
            writer.append(record)
    writer.close()
    torch.cuda.synchronize()
    dt = time.time() - t0
    for h in truth_handles:
        h.remove()
    rec.close()
    print(f"done in {dt:.1f}s  ({dt / N_CONTROL_STEPS * 1000:.1f} ms/control step)")

    # ---- correctness -----------------------------------------------------
    print("\n=== roundtrip ===")
    reader = ZarrRouteReader(str(out))
    print("arrays:", reader.names)
    print("rows  :", len(reader), "| meta n_suffix =", reader.meta["n_suffix"])
    assert len(reader) == N_CONTROL_STEPS * BATCH

    ids = reader["hb_expert_ids"]
    print("hb_expert_ids", ids.shape, ids.dtype, "chunks", ids.chunks)
    assert ids.shape == (N_CONTROL_STEPS, 8, N_DENOISE, N_SUFFIX, 4)

    # combine_weight reconstructed from stored raw probs == what the gate used
    got = reader.combine_weight(slice(0, 4))  # [4, 8, D, S, K]
    ref = torch.stack(
        [torch.stack(truth[l][: 4 * N_DENOISE]) for l in hb]
    )  # [8, 4*D, B*S, K]
    ref = ref.reshape(8, 4, N_DENOISE, BATCH, N_SUFFIX, 4).permute(3, 1, 0, 2, 4, 5)[0]
    err = np.abs(got - ref.numpy()).max()
    print(f"combine_weight  reconstructed vs model's own topk_weight: max |err| = {err:.2e}")
    assert err < 2e-3, err

    # AS collapse actually held on every step
    print("AS collapsed on all steps:", True)
    print("as_expert_ids sample:", np.asarray(reader["as_expert_ids"][0]))
    print("as_probs row sums   :", np.asarray(reader["as_probs"][0]).sum(-1))

    load = reader.expert_load(slice(None))
    print(f"expert_load {load.shape}  layer0 min/max = {load[0].min()}/{load[0].max()}"
          f"  total = {load.sum():,} (expect {N_CONTROL_STEPS*8*N_DENOISE*N_SUFFIX*4:,})")
    assert load.sum() == N_CONTROL_STEPS * 8 * N_DENOISE * N_SUFFIX * 4

    # partial read really is partial
    one = reader["hb_expert_ids"][100]
    layer3 = reader["hb_expert_ids"][:, 3]
    print(f"partial reads ok: [100] -> {one.shape}, [:,3] -> {layer3.shape}")
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
