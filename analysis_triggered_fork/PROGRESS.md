# Triggered-fork experiment — progress log

Goal: one LIBERO rollout, online MoE-routing detector fires mid-episode, fork K=8
continuations with fresh flow-noise at that exact step, ask whether any branch
succeeds while the unchanged-noise trunk fails.

## Status: IN PROGRESS

---

## 2026-08-30 — session start

### Facts established

- Remote A100 box reachable via SSH_ASKPASS recipe (verified):
  `root@ogx2i0xhbqlw4vpmakea100.funhpc.com:30726`, hostname
  `eehmpujduqeigsbk-make-665f87b544-8lh48`.
  A100-PCIE-40GB, 40960 MiB total, **1 MiB used** — GPU is free.
  `/data` 60G, 46G free (15G used = the checkpoint).
- Remote `/data/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/` contains
  **only `checkpoints/`**. No upstream HiMoE-VLA, no bridge, no route-capture code.
- Remote `/data/miniconda/envs/torch/bin/python`: torch 2.6.0+cu124, CUDA True.
- Remote has NO `/home/jovyan/work`, so the hardcoded
  `sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")` at
  serve_with_recorder.py:23 must be satisfied by creating that path on the remote
  (chosen over patching the file, to keep the file hash identical to the
  rolling-star provenance record).

### Server launch args (recovered from the rolling-star run's logs/server.log)

The previous (working) A100 run of the same server printed:
```
discovered 12 gates
  ...layers.{0,1}.mlp.gate   AS  experts=3   top_k=1
  ...layers.{2,3,4,5}.mlp.gate  HB experts=32 top_k=4
  ...layers.{12,13,14,15}.mlp.gate HB experts=32 top_k=4
  ...layers.{16,17}.mlp.gate AS  experts=3   top_k=1
n_suffix=11  n_denoise=10
serving on ws://127.0.0.1:8101
```
and PROVENANCE.json records `store_full_router_probabilities: true`,
`store_hidden_states: false`, legacy (non request-gated) capture mode.
=> launch with `--store-full-probs`, WITHOUT `--store-hidden`, WITHOUT
`--request-gated-capture`, WITHOUT `--no-route-capture`.

**Known trap found in that log**: the first server start CRASHED with
```
RuntimeError: Could not inspect HiMoE upstream ...: fatal: detected dubious
ownership in repository at '.../upstream/HiMoE-VLA'
```
=> must run `git config --global --add safe.directory <upstream>` on the remote
before starting the server.

### Local model env

`/home/jovyan/.cache/himoe-libero-bridge/envs/model` — 7.5 GB, Python 3.11.15
(clang-built = uv/python-build-standalone). No pip module inside it.
Contains himoe_libero_bridge 0.1.0, lerobot 0.1.0, moevla 0.1.0, jax+cuda,
deepspeed, numpy 1.26.4, ...

Exact pins that matter: torch==2.6.0 (+cu124), transformers==4.48.1,
numpy==1.26.4, zarr==3.0.6, websockets==15.0.1, msgpack==1.1.0,
imageio==2.37.0, tokenizers==0.21.1, jax==0.5.0, deepspeed==0.15.4.
Full 200-line pin list derived from site-packages `*.dist-info` names.

### Transport decision

ssh link is capped on TOTAL bandwidth (~5 MB/s single stream, ~2 MB/s
aggregate over 12 streams — parallelism makes it WORSE). So:
copying the 7.5 GB env is off the table. Instead **pip-install the public
deps on the remote** (its own internet is faster) and scp only the private trees.

Private trees shipped as one tar (`/tmp/himoe_private.tar`, 360,458,240 B,
md5 `0b1d24277eeb3ab8e2b58609f5b70cd3`):
- `HiMoE-VLA/` — the 184 MB upstream, which supplies BOTH the editable
  `moevla` and `openpi_client` packages plus the `.git` the bridge inspects
- `himoe-libero-wrist-fix/` — src + **patches/** (policies.py finds patches via
  `Path(__file__).parents[2]`, so the repo root, not just src, must be present)
- `himoe-libero-bridge/` — the pip-editable bridge copy
- `himoe-route-capture/` — all top-level `*.py`

Excluded from the remote pip list: `himoe_libero_bridge`, `moevla`,
`openpi_client` (all private/editable) and `lerobot` (git install,
huggingface/lerobot @ 6674e368249472c91382eb54bb8501c94c7f0c56).
`draccus==0.10.0` was a git install locally (dlwh/draccus) but the same
version is on pypi — taking the pypi one.

### Remote path requirements (both hardcoded in serve_with_recorder.py:23-24)

```
sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
```
Locally `/home/jovyan/work/himoe-route-capture` is a symlink to
`himoe-vla/himoe-route-capture` (contents verified identical). On the remote
both paths get created directly. Chose to create the paths rather than patch
the file, so the served file keeps the rolling-star provenance hash
`bd9524f7e101c7fada9f4930bdd8503c7cc3234b9d0c77cf2ce318f74f828d21`.

Python version: **must be 3.11** — upstream `HiMoE-VLA/pyproject.toml` has
`requires-python = "==3.11.*"`. The remote's existing conda `torch` env is
3.10.16, so a new env `/data/envs/model` is being created with conda at 3.11.15.
