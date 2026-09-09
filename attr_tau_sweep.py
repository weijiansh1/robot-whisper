#!/usr/bin/env python
"""HB-Reg vs flow-loss gradient attribution at the released HiMoE-VLA checkpoint.

Answers the one question the corpus cannot: is lambda_HB * ||g_HB|| large enough
relative to ||g_flow|| to be the thing that shaped the router, or is it negligible?

Design notes (the three traps):
 1. The gate returns aux_loss ALREADY multiplied by alpha, and AddAuxiliaryLoss
    injects grad 1.0 at the block output -- the aux term never enters a scalar loss.
    So: replace AddAuxiliaryLoss with a passthrough (stop auto-injection) and stash
    each gate's aux tensor, then backward it explicitly, divided by alpha.
 2. `self.training` gates BOTH the aux loss and the dispatch path; the eval branch
    moe_infer is @torch.no_grad(), so g_flow w.r.t. the gate does not exist in eval.
    So: model.eval() for determinism, then .train() only on the MoE modules.
 3. forward() takes explicit noise=/time=, so both backwards share one exact graph.

Run:
  CUDA_VISIBLE_DEVICES=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a \
  MOEVLA_DATA_HOME=/home/jovyan/.cache/himoe-libero-bridge/moevla-data \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
  ~/.cache/himoe-libero-bridge/envs/model/bin/python attr_hb_vs_flow.py
"""
import dataclasses
import io
import time
import json
import pathlib
import sys

import os

MIG = "MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a"          # 2g.35gb
# The MIG slices report only ~10.6 GiB free of 32.5 (another tenant), and the fp32
# model needs ~16 GiB -- model.to(cuda) dies inside CUDACachingAllocator.  This is a
# precision-sensitive magnitude comparison, so use CPU/fp32 rather than bf16.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")   # as server.py:103 does

import numpy as np
import torch

CKPT = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
                    "cache/checkpoints/HiMoE-VLA-Libero-Goal")
UP = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
                  "cache/upstream/HiMoE-VLA")
LEROBOT = pathlib.Path("/home/jovyan/work/home-cache/libero-lerobot/"
                       "physical-intelligence-libero")
sys.path.insert(0, str(UP / "src"))
sys.path.insert(0, str(UP / "packages/openpi-client/src"))

TASK = "turn on the stove"      # libero_goal; only 11/92 parquet files intact
CHUNK, N_BATCH, BATCH = 10, 4, 2
TAUS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
SEED = 0

from moevla.models import model as mmodel                       # noqa: E402
from moevla.models.modeling_moe import (HBMoE, ASMoE, MoEGate_load_bal,  # noqa: E402
                                        MoEGate_mutual_info)
from moevla.policies import policy_config                       # noqa: E402
from moevla.training import config as mcfg                      # noqa: E402
from moevla.transforms import Normalize                         # noqa: E402
import moevla.models.modeling_moe as MM                         # noqa: E402


# ----------------------------------------------------------------- data
def build_items():
    import pyarrow.parquet as pq
    from PIL import Image

    tasks = {json.loads(l)["task"]: json.loads(l)["task_index"]
             for l in open(LEROBOT / "meta/tasks.jsonl")}
    eps = [json.loads(l) for l in open(LEROBOT / "meta/episodes.jsonl")]
    mine = [e for e in eps if TASK in e.get("tasks", [])]
    print(f"  task_index={tasks[TASK]}  {len(mine)} demo episodes")

    items, used = [], []
    for e in mine:
        if len(items) >= N_BATCH * BATCH:
            break
        ei = e["episode_index"]
        f = LEROBOT / f"data/chunk-000/episode_{ei:06d}.parquet"
        if not f.exists():
            continue
        try:
            t = pq.read_table(f).to_pydict()
        except Exception as exc:                 # 81/92 files are truncated
            print(f"    skip ep{ei}: {type(exc).__name__}")
            continue
        n = len(t["state"])
        if n < CHUNK + 2:
            continue
        used.append(ei)
        for fi in (n // 4, n // 2):                     # two frames per episode
            if fi + CHUNK > n:
                continue
            items.append({
                "image": np.asarray(Image.open(io.BytesIO(t["image"][fi]["bytes"]))
                                    .convert("RGB")),
                "wrist_image": np.asarray(
                    Image.open(io.BytesIO(t["wrist_image"][fi]["bytes"])).convert("RGB")),
                "state": np.asarray(t["state"][fi], np.float32),
                "actions": np.stack([np.asarray(t["actions"][fi + k], np.float32)
                                     for k in range(CHUNK)]),
                "prompt": TASK,
            })
    print(f"  {len(items)} items from episodes {used[:6]}...")
    return items[:N_BATCH * BATCH]


def make_pipeline():
    dc = mcfg.get_dataset_config("libero_goal_no_noops_lerobot")
    dc = dataclasses.replace(dc, assets_base_dir=str(CKPT))
    tc = mcfg.get_training_config("libero_goal_no_noops_lerobot_finetune")
    d = dc.data.create(dc.assets_dirs, tc.model)
    print(f"  norm_stats keys: {list(d.norm_stats)}  data_mask sum="
          f"{sum(d.data_mask)}/{len(d.data_mask)}")
    stages = ([*d.repack_transforms.inputs, *d.data_transforms.inputs,
               Normalize(norm_stats=d.norm_stats, data_mask=d.data_mask),
               *d.model_transforms.inputs])
    print("  stages: " + " -> ".join(type(s).__name__ for s in stages))
    return stages


def collate(items, stages):
    out = []
    for it in items:
        x = it
        for s in stages:
            x = s(x)
        out.append(x)
    batch = {}
    for k in out[0]:
        if isinstance(out[0][k], dict):
            batch[k] = {kk: np.stack([o[k][kk] for o in out]) for kk in out[0][k]}
        else:
            batch[k] = np.stack([np.asarray(o[k]) for o in out])
    return batch


# ----------------------------------------------------------------- harness
class Passthrough(torch.autograd.Function):
    """AddAuxiliaryLoss with auto-injection disabled."""
    @staticmethod
    def forward(ctx, x, loss):
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {dev}")
    print("building data pipeline ...")
    stages = make_pipeline()
    items = build_items()

    print("\nloading policy ...")
    tc = mcfg.get_training_config("libero_goal_no_noops_lerobot_finetune")
    dc = dataclasses.replace(
        mcfg.get_dataset_config("libero_goal_no_noops_lerobot"),
        assets_base_dir=str(CKPT))
    policy = policy_config.create_trained_policy(tc, dc, CKPT, default_prompt=None)
    model = policy.model
    print(f"  loaded, dtype={next(model.parameters()).dtype}")

    # trap 2: eval everywhere for determinism, train ONLY on the MoE modules
    model.eval()
    n_moe = 0
    for m in model.modules():
        if isinstance(m, (HBMoE, ASMoE)):
            m.train()
            n_moe += 1
    print(f"  {n_moe} MoE modules set to train() (rest stay eval)")

    # MoEGate_mutual_info.forward hardcodes a .to(bfloat16) on its input (it relied on
    # CUDA autocast).  Cast only those 4 AS gate weights to bf16 so the AS path matches
    # training numerics; everything else, including all 8 HB gates, stays fp32.
    n_as = 0
    for m in model.modules():
        if isinstance(m, MoEGate_mutual_info):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            n_as += 1
    print(f"  {n_as} AS gate weights cast to bf16 (HB gates remain fp32)")

    # trap 1: stop auto-injection, stash each gate's aux
    MM.AddAuxiliaryLoss = Passthrough
    gates, stash = {}, {}
    for name, m in model.named_modules():
        if isinstance(m, MoEGate_load_bal):
            gates[name] = m
    print(f"  {len(gates)} HB gates; alpha={next(iter(gates.values())).alpha}")

    orig_fwd = MoEGate_load_bal.forward

    def patched(self, hidden_states):
        idx, w, aux = orig_fwd(self, hidden_states)
        stash[id(self)] = aux
        return idx, w, aux
    MoEGate_load_bal.forward = patched

    # only the 8 HB gate weights need grads
    for p in model.parameters():
        p.requires_grad_(False)
    targets = {n: g.weight for n, g in gates.items()}
    for p in targets.values():
        p.requires_grad_(True)

    ALPHA = next(iter(gates.values())).alpha
    acc = {n: {"gf": 0.0, "gh": 0.0} for n in targets}
    G = {n: [] for n in targets}
    flow_vals, hb_vals = [], []

    t0 = time.time()
    # pre-collate once so every tau sees identical batches
    batches = []
    for b in range(N_BATCH):
        sub = items[b * BATCH:(b + 1) * BATCH]
        if len(sub) < BATCH:
            break
        batches.append(collate(sub, stages))
    print(f"  {len(batches)} batches pre-collated\n")

    res = {t: {n: {"gf": [], "gh": []} for n in targets} for t in TAUS}
    for tau in TAUS:
        for b, batch in enumerate(batches):
            obs = mmodel.from_dict({k: v for k, v in batch.items() if k != "actions"})
            obs = mmodel.preprocess_observation(
                {k: (torch.as_tensor(v).to(dev) if not isinstance(v, dict)
                     else {kk: torch.as_tensor(vv).to(dev) for kk, vv in v.items()})
                 for k, v in obs.items()}, train=True)
            actions = torch.as_tensor(batch["actions"]).to(dev).float()
            dm = torch.as_tensor(batch["data_mask"]).to(dev)
            state = obs["state"].float()
            # paired: same noise for a given batch index across all tau
            noise = torch.randn(actions.shape, generator=torch.Generator(
                device=dev).manual_seed(1000 + b), device=dev, dtype=torch.float32)
            time_ = torch.full((actions.shape[0],), float(tau), device=dev,
                               dtype=torch.float32)
            stash.clear()
            loss = model(obs["images"], obs["image_masks"], obs["tokenized_prompt"],
                         obs["tokenized_prompt_mask"], state, actions, dm,
                         noise=noise, time=time_)
            hb_aux = sum(stash[id(g)] for g in gates.values()) / ALPHA
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            gf = {n: p.grad.detach().clone() for n, p in targets.items()}
            model.zero_grad(set_to_none=True)
            hb_aux.backward()
            for n, p in targets.items():
                res[tau][n]["gf"].append(gf[n].double().norm().item())
                res[tau][n]["gh"].append(p.grad.detach().double().norm().item())
            if b == 0:
                print(f"  [{time.time()-t0:6.1f}s] tau={tau:.2f} L_flow={loss.item():.5f} "
                      f"L_HB={hb_aux.item()/len(gates):.5f}")

    # ----------------------------------------------------------- report
    print("\n" + "=" * 100)
    print(f"lambda_HB*||g_HB||/||g_flow|| on the gate weight, vs flow timestep tau")
    print(f"(tau ~ Beta(1.5,1) in training: mean 0.6, density rising toward 1)")
    print("=" * 100)
    names = list(targets)
    print(f"  {'tau':>5} " + " ".join(f"L{l:>6}" for l in HB_LAYERS) + "     mean")
    ratios = {}
    for tau in TAUS:
        row = []
        for n in names:
            gf = np.mean(res[tau][n]["gf"]); gh = np.mean(res[tau][n]["gh"])
            row.append(ALPHA * gh / gf)
        ratios[tau] = row
        print(f"  {tau:5.2f} " + " ".join(f"{100*v:6.2f}%" for v in row) +
              f"  {100*np.mean(row):6.2f}%")
    print(f"\n  {'tau':>5} " + " ".join(f"L{l:>6}" for l in HB_LAYERS) + "   <- ||g_flow||")
    for tau in TAUS:
        print(f"  {tau:5.2f} " + " ".join(f"{np.mean(res[tau][n]['gf']):6.3f}"
                                          for n in names))
    print(f"\n  {'tau':>5} " + " ".join(f"L{l:>6}" for l in HB_LAYERS) + "   <- ||g_HB||")
    for tau in TAUS:
        print(f"  {tau:5.2f} " + " ".join(f"{np.mean(res[tau][n]['gh']):6.3f}"
                                          for n in names))
    # Beta(1.5,1) weighted average over the grid, pdf ~ 1.5 * x^0.5
    w = np.array([1.5 * t ** 0.5 for t in TAUS]); w /= w.sum()
    wm = np.array([np.mean(ratios[t]) for t in TAUS]) @ w
    print(f"\n  grid-mean ratio                 = {100*np.mean([np.mean(ratios[t]) for t in TAUS]):.2f}%")
    print(f"  Beta(1.5,1)-weighted mean ratio = {100*wm:.2f}%   "
          f"(training spends most mass at high tau)")


if __name__ == "__main__":
    main()
