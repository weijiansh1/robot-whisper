# First-control MoE output capture

This is a sidecar capture for same-observation K-draw experiments.  It does not
change or replace `routes.zarr`, `hidden.zarr`, the hub layout, or the LIBERO
client format.  The normal client runs full episodes for success labels; the
server enables output hooks only for control step 0 of each episode.

## What is stored

`first_control_outputs.zarr` has one candidate row per episode and explicit
identity/axis arrays:

- identity: `episode_id`, `task_id`, `init_state_id`, `flow_seed`,
  `control_step`, and the exact request's `flow_noise_sha256`;
- axes: `hb_layer_id`, `as_layer_id`, `denoise_id`, `action_token_id`;
- AS0/1: selected expert id and actual weighted selected-expert output on action
  tokens, `[candidate, 2, denoise, action_token, hidden]` fp16;
- HB: expert ids/weights plus routed/shared/post norms, branch cosine and angle,
  and candidate distance/cosine to the same-state post-MoE centroid;
- group: exact post-MoE RMS/relative dispersion, mean pair cosine, and
  contraction relative to denoise step 0.

The post-MoE vector means `routed + shared`, before the transformer residual
addition.  By default it is buffered for one K pool, used for the exact group
reductions, then discarded.  `--store-hb-vectors` additionally persists the
full routed and shared fp16 vectors.

For a LIBERO 16x32 task, the default raw array payload is about 214 MiB.  Full HB
vectors add about 1.56 GiB per task.  A K=32 group reduction temporarily uses
roughly 250-350 MiB of host RAM.

## Run one right-16x32-compatible task

The example below is Goal task 0.  Substitute the suite, benchmark, task id/name,
and checkpoint for the other tasks.  Use the same two process environments as
`run_corpus_capture.py`.

Server (model Python 3.11):

```bash
env -u CUDA_VISIBLE_DEVICES \
  PYTHONPATH=/home/jovyan/work/himoe-vla/himoe-libero-wrist-fix/src \
  MOEVLA_DATA_HOME=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/moevla-data \
  /home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/model/bin/python -u \
  /home/jovyan/work/himoe-vla/himoe-route-capture/serve_first_output_recorder.py \
  --port 8411 --gpu MIG-UUID --suite goal --benchmark libero_goal \
  --task-id 0 --task-name open_the_middle_drawer_of_the_cabinet \
  --checkpoint-dir /home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-Goal \
  --upstream-root /home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA \
  --libero-wrist-layout paper-right \
  --scene-ids 0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49 \
  --draws 32 --noise-seed-base 1000 \
  --out /path/to/task/server-output
```

Client (LIBERO Python 3.8), after the server prints `serving on ws`:

```bash
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python -u \
  /home/jovyan/work/himoe-vla/himoe-route-capture/rollout_with_routes.py \
  --port 8411 --benchmark libero_goal --task-id 0 \
  --init-state-ids 0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49 \
  --repeats 32 --noise-seed-base 1000 --max-steps 300 \
  --no-routing-capture --label first-output-16x32 \
  --libero-root /home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO \
  --out /path/to/task/client-output
```

The client still needs the EGL, `LD_LIBRARY_PATH`, and `PYTHONPATH` environment
from `run_corpus_capture.py:client_env()` when they are not already configured.
Stop the server with SIGTERM so the last group and summary are flushed.

The server independently regenerates the first noise draw from `flow_seed` and
compares SHA-256 digests.  A wrong scene/draw declaration or episode ordering
therefore fails instead of silently assigning the wrong identity.

## Join success labels

Join each Zarr candidate row to `client-output/summaries.json` with:

```text
zarr.episode_id == summary.episode_index
```

Then assert both `zarr.init_state_id == summary.init_state_id` and
`zarr.flow_seed == summary.flow_noise_seed` before using `summary.success`.
`group_id` and `group_init_state_id` identify the K candidates that share the
same task, initial state, and control row.

## Existing-data boundary

- `right-16x32/hidden.zarr` can reconstruct HB router inputs/probabilities with
  checkpoint gate weights.  It cannot invert nonlinear experts to recover the
  true AS or HB outputs.
- `routes.zarr` can reconstruct selected ids/weights, not expert outputs.
- Client episode files contain actions/state/success but not the exact input
  images needed to replay the first forward pass.
- `runs/state-tier/state.npz` has true reduced output norms, but no same-state K
  success/failure pool and no reliable episode identity.

Consequently the new AS outputs, HB branch geometry, and post-MoE convergence
require a new rollout capture.  They cannot be backfilled exactly from the
existing right-16x32 artifacts.
