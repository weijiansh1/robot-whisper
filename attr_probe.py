#!/usr/bin/env python
"""Step 0 of the HB-vs-flow gradient attribution: load the checkpoint on a MIG
slice through the validated bridge path and introspect what a forward needs.

Run with:
  CUDA_VISIBLE_DEVICES=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
  ~/.cache/himoe-libero-bridge/envs/model/bin/python attr_probe.py
"""
import dataclasses
import pathlib
import sys

import torch

CKPT = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
                    "cache/checkpoints/HiMoE-VLA-Libero-Spatial")
UPSTREAM = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
                        "cache/upstream/HiMoE-VLA")
sys.path.insert(0, str(UPSTREAM / "src"))
sys.path.insert(0, str(UPSTREAM / "packages/openpi-client/src"))

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"n={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  device0 = {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

from moevla.policies import policy_config           # noqa: E402
from moevla.training import config as mcfg          # noqa: E402

train_config = mcfg.get_training_config("libero_spatial_no_noops_lerobot_finetune")
dataset_config = dataclasses.replace(
    mcfg.get_dataset_config("libero_spatial_no_noops_lerobot"),
    assets_base_dir=str(CKPT),
)
print("\nloading trained policy ...")
policy = policy_config.create_trained_policy(
    train_config, dataset_config, CKPT, default_prompt=None)
model = policy.model
print(f"  model {type(model).__name__}  dtype={next(model.parameters()).dtype} "
      f"device={next(model.parameters()).device}")
mc = model.config
for k in ("n_action_steps", "max_action_dim", "max_state_dim", "num_steps",
          "tokenizer_max_length", "proj_width"):
    print(f"  config.{k} = {getattr(mc, k, '<absent>')}")

print("\npolicy attributes:")
for a in sorted(vars(policy)):
    v = getattr(policy, a)
    print(f"  {a:26} {type(v).__name__}")

print("\nMoE modules found:")
from moevla.models.modeling_moe import HBMoE, ASMoE, MoEGate_load_bal  # noqa: E402
hb, as_, gates = [], [], []
for name, m in model.named_modules():
    if isinstance(m, HBMoE):
        hb.append(name)
    elif isinstance(m, ASMoE):
        as_.append(name)
    elif isinstance(m, MoEGate_load_bal):
        gates.append((name, tuple(m.weight.shape), m.alpha, m.seq_aux, m.top_k))
print(f"  HBMoE  x{len(hb)}: {hb}")
print(f"  ASMoE  x{len(as_)}")
for n, s, a, sq, k in gates:
    print(f"  gate {n}  W{s} alpha={a} seq_aux={sq} top_k={k}")

print("\nn_shared_experts on HB blocks:")
for name in hb[:2]:
    m = dict(model.named_modules())[name]
    print(f"  {name}: n_shared_experts={m.config.n_shared_experts} "
          f"has shared_experts={hasattr(m, 'shared_experts')} "
          f"norm_topk_prob={m.config.norm_topk_prob}")

print("\nforward signature:")
import inspect                                       # noqa: E402
print("  " + str(inspect.signature(model.forward)))
print("\nOK -- checkpoint loads and the MoE graph is reachable.")
