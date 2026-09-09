# Coupling-evidence circuit, round 1

Which computation turns object-interaction evidence into a change in the action
chunk, and where does it stop working when the object is lost?

`PROTOCOL.md` is the frozen design, written before any observation was rendered.
Its amendments section records every change made since, each with the reason and
the measurement behind it.

## Where this sits

`moe-progress-ratio-v12-0906/results/object_readability` established that the
routing interface separates a coupled object from one that stayed behind. That
is a readout claim on an observational contrast, and its own artifacts carry the
caveat that every phantom event sits in a failed episode while the coupled pool
is 96% successful.

This study replaces matching with construction. One restored MuJoCo state
produces every condition, so the robot configuration, scene, camera and
instruction are identical across conditions and only the object differs.

The behaviour the study would most like to trace -- notices the object is gone,
revises the action -- does not occur in the corpus at all:
`belief_mismatch_successes` is 0 in both 16,000-episode runs, `active_retry` is
0/307 failures, and the horizon-recovery pilot recovered 0/23. So the reference
computation is supplied by a graded **pre-closure** positive control (displace
the target before the grasp), and post-closure coupling loss is the case to be
explained.

## Stages

| stage | script | state |
|---|---|---|
| 0 build counterfactual observations | `experiments/build_coupling_counterfactuals.py` | **done**, 240 units / 960 rows |
| 1 does the command follow the object | `experiments/measure_action_response.py` | written, blocked on GPU |
| 2 the response after closure | same script | written, blocked on GPU |
| 3 coarse localisation | not written | gated on stage 1 |
| 4 gate against expert content | not written | gated on stage 3 |
| 5 where it breaks | not written | gated on stage 4 |

Stages 3 and 4 are deliberately unwritten: `PROTOCOL.md` makes stage 1 a hard
gate, and if the policy does not re-aim to a displaced object there is no working
evidence-to-action computation in this behaviour to localise.

## Stage 0 result

Primary task `libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`,
capture `right-50x8-20260903`, 400/400 successes.

- **240 units, 960 rows: 120 approach anchors and 120 transport anchors.**
- 210 anchors dropped, all at the approach: 138 because no shift sign clears
  contact and support, 72 because the target bearing is undefined. Every
  transport anchor considered was admitted.
- Visibility, median fraction of pixels moved against `held`: `shift8` 0.098 in
  the wrist camera and 0.002 in agentview; `uncoupled` 0.105 and 0.006; `light`
  0.011 and 0.392. The object edits land in the wrist camera and the lighting
  control is a large agentview change, which is what makes it a control rather
  than a formality.

Rendering is CPU-only through osmesa. The 130 MB dataset is gitignored following
the convention for dense artifacts; `counterfactuals_object_pudding_audit.json`
carries its sha256 and every row's provenance, gates and drop reason.

## Running

```bash
# stage 0, CPU only, about 13 minutes
moe-coupling-circuit-0906/experiments/render_env.sh \
  moe-coupling-circuit-0906/experiments/build_coupling_counterfactuals.py \
  --client-dir "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket/right-50x8-20260903/client" \
  --suite libero_object \
  --libero-root /home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO \
  --out moe-coupling-circuit-0906/results/counterfactuals_object_pudding.npz \
  --audit moe-coupling-circuit-0906/results/counterfactuals_object_pudding_audit.json \
  --max-units-per-anchor 120

# stages 1 and 2, needs a GPU with about 17 GB free
moe-coupling-circuit-0906/experiments/model_env.sh \
  moe-coupling-circuit-0906/experiments/measure_action_response.py \
  --inputs moe-coupling-circuit-0906/results/counterfactuals_object_pudding.npz \
  --inputs-audit moe-coupling-circuit-0906/results/counterfactuals_object_pudding_audit.json \
  --checkpoint-dir /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Object \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --suite object --wrist-layout paper-right \
  --out-dir moe-coupling-circuit-0906/results/action_response

python3 -m pytest moe-coupling-circuit-0906/tests/ -q
```

## GPU blocker, as of 2026-09-06

All eight H20-3e cards hold 139430 of 143771 MiB, taken by eight processes
outside this container. The policy needs about 16.3 GB. `--allow-cpu` loads the
checkpoint and validates the observation and flow-noise contracts but cannot
produce an action: this checkpoint's AS gate casts its input to bfloat16 while
the weight stays float32, which only works under CUDA autocast
(`modeling_moe.py:132`). Working around that would change the frozen
computation, so stages 1 and 2 wait for a card.

## What stage 0 does not establish

Nothing about the action response, and nothing about any internal computation.
Edited states are physically-gated restorations, not states the simulator
stepped into: contact history and settling are absent. RGB is regenerated from
stored MuJoCo state rather than replayed from stored frames.
