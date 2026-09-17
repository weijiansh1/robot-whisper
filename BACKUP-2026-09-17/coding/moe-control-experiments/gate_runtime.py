"""Request-scoped upstream gate hooks in a memory-bounded local process."""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
UPSTREAM = Path("/data/coding/robot-whisper-0909/moe-trap-control")
CHECKPOINT = Path("/data/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10")
CHECKPOINT_SHA = "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256"
MEMORY_LIMIT_MIB = 16384

for path in ("/data/srv/src", "/data/srv/packages/openpi-client/src", "/data/srv", str(UPSTREAM)):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_isolated():
    import torch
    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_libero_bridge.protocol import server_metadata
    from gate_capture import GateProbePolicy

    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected the single local GPU")
    free, total = torch.cuda.mem_get_info()
    if free < (MEMORY_LIMIT_MIB + 768) * 1024 ** 2:
        raise RuntimeError("Insufficient free memory for bounded isolated load")
    torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_MIB * 1024 ** 2 / total, 0)
    start = time.monotonic()
    policy = HiMoEPolicy(str(CHECKPOINT), suite="long", upstream_root="/data/srv",
                        require_cuda=True, libero_wrist_layout="released-left")
    policy._policy.model.eval().requires_grad_(False)
    if policy.metadata["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint")
    dtypes = {}
    for value in policy._policy.model.parameters():
        key = str(value.dtype)
        dtypes[key] = dtypes.get(key, 0) + value.numel() * value.element_size()
    wrapped = GateProbePolicy(policy)
    metadata = server_metadata(wrapped.backend_name)
    metadata.update(wrapped.metadata)
    return wrapped, {
        "load_seconds": time.monotonic() - start, "metadata": metadata,
        "parameter_bytes_by_dtype": dtypes, "allocator_limit_mib": MEMORY_LIMIT_MIB,
        "autocast_cache_enabled": False, "requires_grad": False,
        "allocated_mib": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024 ** 2,
    }


def infer_isolated(wrapped, request):
    import torch

    started = time.monotonic()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
        response = wrapped.infer(request)
    torch.cuda.synchronize()
    if response.get("flow/noise_sha256") != hashlib.sha256(request["flow/noise"].tobytes()).hexdigest():
        raise RuntimeError("Noise acknowledgement mismatch")
    if not np.isfinite(response["actions"]).all():
        raise RuntimeError("Non-finite policy actions")
    layers = wrapped.policy._routing_layers + wrapped.as_layers
    if any(gate._forward_hooks for _, gate in layers):
        raise RuntimeError("Request leaked a routing hook")
    return response, {
        "seconds": time.monotonic() - started,
        "allocated_mib": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024 ** 2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
    }
