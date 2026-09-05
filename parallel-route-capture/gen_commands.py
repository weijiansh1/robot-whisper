#!/usr/bin/env python3
"""Expand plan.json into per-process launch scripts under cmds/.

One server instance PER TASK so every task owns its routes.zarr, laid out in
the VLA_MUI_HUB convention (cache_new/<model>/<hub_dir>/<task>/<run_id>/):

    <run_dir>/meta.json
    <run_dir>/logs/{server.log, client-iNN.log}
    <run_dir>/server/routes.zarr
    <run_dir>/client/init-NN/snapshot_XXX/...

All 40 LIBERO tasks exceed GPU memory if served at once (~22 GiB/server), so
tasks are packed into WAVES: at most `servers_per_gpu` per GPU per wave
(GPU 7 keeps one slot for the CALVIN server, which spans all waves).
launch.sh runs waves sequentially; workers of one task get worker ids 0..N-1
(branch_episode_id = (worker_id+1)*1e8 + snapshot*100 + candidate).
"""
import json
import pathlib
import stat
import time

HERE = pathlib.Path(__file__).resolve().parent
PLAN = json.load(open(HERE / "plan.json"))
P = PLAN["paths"]
CMDS = HERE / "cmds"
CMDS.mkdir(exist_ok=True)
for old in CMDS.glob("*.sh"):
    old.unlink()

OPENPI = f"{P['upstream_himoe']}/packages/openpi-client/src"
# The container image lacks glvnd (libEGL.so.1); the bridge persisted it, and
# scripts/libero.sh prepends it to LD_LIBRARY_PATH — replicate that here.
SYSTEM_LIBS = "/home/jovyan/.cache/himoe-libero-bridge/system-libs/usr/lib/x86_64-linux-gnu"
LD_PATH = f"{SYSTEM_LIBS}:/usr/lib/x86_64-linux-gnu"


def write(name, body):
    path = CMDS / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def libero_server_script(gpu, port, suite, out):
    return f"""#!/bin/bash
# route-recorder server: suite={suite} gpu={gpu} port={port}
# serve_with_recorder sets CUDA_VISIBLE_DEVICES itself from --gpu, overriding
# any outer env var — so the PHYSICAL gpu id must go through --gpu.
exec {P['model_python']} -u {P['route_capture']}/serve_with_recorder.py \\
  --host 127.0.0.1 --port {port} --gpu {gpu} --suite {suite} \\
  --checkpoint-dir {P['checkpoints'][suite]} \\
  --upstream-root {P['upstream_himoe']} \\
  --libero-wrist-layout checkpoint-right \\
  --store-full-probs \\
  --out {out}
"""


def worker_script(sh):
    return f"""#!/bin/bash
# worker {sh['name']}: {sh['benchmark']} task {sh['task']} init {sh['init']}
cd {P['route_capture']}
exec env PYTHONPATH={P['wrist_fix_src']}:{OPENPI} \\
  LD_LIBRARY_PATH={LD_PATH} \\
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID={sh['gpu']} \\
  {P['libero_python']} -u rolling_star_collect.py \\
  --host 127.0.0.1 --port {sh['port']} \\
  --benchmark {sh['benchmark']} --task-id {sh['task']} \\
  --init-state-id {sh['init']} --worker-id {sh['wid']} \\
  --k {sh['k']} --seed {PLAN['seed']} \\
  --environment-seed {PLAN['environment_seed']} \\
  --settle-steps {PLAN['settle_steps']} --max-steps {PLAN['max_steps']} \\
  --replan-steps {PLAN['replan_steps']} \\
  --max-trunk-queries {sh['mtq']} \\
  --stop-before-unix ${{STOP_BEFORE_UNIX:-0}} \\
  --libero-root {P['libero_root']} \\
  --out {sh['out']}
"""


def meta_json(group, task_id, task_name, port, wave):
    n_init = len(group["init_states"])
    mtq = group["max_trunk_queries"]
    return {
        "schema_version": "route-corpus/1",
        "hub_model": "HiMoE-VLA",
        "benchmark": group["benchmark"],
        "suite": group["suite"],
        "task_id": task_id,
        "task_name": task_name,
        "run_id": PLAN["run_id"],
        "campaign": PLAN["campaign"],
        "goal_tag": group["goal_tag"],
        "port": port,
        "wave": wave,
        "wrist_layout": "checkpoint-right",
        "max_steps": PLAN["max_steps"],
        "replan_steps": PLAN["replan_steps"],
        "settle_steps": PLAN["settle_steps"],
        "environment_seed": PLAN["environment_seed"],
        "campaign_seed": PLAN["seed"],
        "sampling": {
            "design": f"rolling-star: every committed trunk query launches K={PLAN['k']} "
                      f"terminal branches; {n_init} init states x <= {mtq} snapshots each",
            "k_candidates_per_snapshot": PLAN["k"],
            "init_states": group["init_states"],
            "max_trunk_queries_per_worker": mtq,
            "designed_branches_max": n_init * mtq * PLAN["k"],
            "held_fixed": ["task", "wrist_layout", "environment_seed"],
            "varies": ["init_state", "snapshot", "branch_flow_noise"],
        },
        "protocol_reference": "himoe-route-capture/ROLLING_STAR_EXPERIMENT_PROTOCOL.md",
        "status": "prepared",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def calvin_scripts(gpu, port, run_dir, max_new):
    c = P["calvin_alignment_cache"]
    h = P["upstream_himoe"]
    src = f"{P['calvin_alignment_src']}/src"
    srv = f"""#!/bin/bash
# CALVIN-D route-recorder server: gpu={gpu} port={port}
exec env CUDA_VISIBLE_DEVICES={gpu} \\
  PYTHONPATH={src}:{P['wrist_fix_src']}:{OPENPI} \\
  {P['model_python']} -u {P['route_capture']}/serve_calvin_with_recorder.py \\
  --checkpoint-dir {P['checkpoints']['calvin']} \\
  --upstream-root {h} --calvin-root {c}/upstream/calvin \\
  --host 127.0.0.1 --port {port} \\
  --out {run_dir}/server
"""
    cli = f"""#!/bin/bash
# CALVIN client: fresh corpus, {max_new} sequences
cd {P['calvin_alignment_src']}
exec bash scripts/calvin.sh \\
  --dataset-root {c}/datasets/task_D_D --calvin-root {c}/upstream/calvin \\
  --himoe-root {h} --output-dir {run_dir}/client \\
  --host 127.0.0.1 --port {port} \\
  --max-new-sequences {max_new}
"""
    return srv, cli


# ---- flatten tasks and pack into waves (calvin pins one GPU-7 slot always) --
flat = []
for group in PLAN["task_groups"]:
    for task_id in group["tasks"]:
        flat.append((group, task_id))

CAP = PLAN["servers_per_gpu"]
caps_per_wave = {g: CAP for g in range(8)}
caps_per_wave[PLAN["calvin"]["gpu"]] = CAP - 1

waves = []
i = 0
while i < len(flat):
    slots = []
    for g in range(8):
        slots += [g] * caps_per_wave[g]
    take = flat[i:i + len(slots)]
    waves.append(list(zip(take, slots)))
    i += len(take)

manifest = {"campaign": PLAN["campaign"], "run_id": PLAN["run_id"],
            "n_waves": len(waves), "servers": [], "shards": []}
port = PLAN["base_port"]

for wave_idx, wave in enumerate(waves):
    for (group, task_id), gpu in wave:
        bench = group["benchmark"]
        hub = PLAN["hub_dir"][bench]
        task_name = PLAN["task_names"][bench][task_id]
        run_dir = pathlib.Path(P["cache_new_root"]) / hub / task_name / PLAN["run_id"]
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        json.dump(meta_json(group, task_id, task_name, port, wave_idx),
                  open(run_dir / "meta.json", "w"), indent=1)

        sname = f"server-p{port}"
        write(f"{sname}.sh", libero_server_script(gpu, port, group["suite"],
                                                  f"{run_dir}/server"))
        entry = {"name": sname, "gpu": gpu, "port": port, "kind": "libero",
                 "wave": wave_idx, "suite": group["suite"], "task_id": task_id,
                 "task_name": task_name, "script": f"{sname}.sh",
                 "log": f"{run_dir}/logs/server.log", "clients": [],
                 "run_dir": str(run_dir)}
        short = task_name[:30]
        for wid, init in enumerate(group["init_states"]):
            wname = f"worker-p{port}-i{init:02d}"
            sh = {"name": wname, "wid": wid, "gpu": gpu, "port": port,
                  "wave": wave_idx, "benchmark": bench, "task": task_id,
                  "task_name": task_name, "init": init, "k": PLAN["k"],
                  "mtq": group["max_trunk_queries"],
                  "out": f"{run_dir}/client/init-{init:02d}",
                  "log": f"{run_dir}/logs/client-i{init:02d}.log",
                  "label": f"w{wave_idx}:{group['goal_tag']}:{short}:i{init:02d}"}
            write(f"{wname}.sh", worker_script(sh))
            entry["clients"].append(wname)
            manifest["shards"].append(sh)
        manifest["servers"].append(entry)
        port += 1

cal = PLAN["calvin"]
cal_run = pathlib.Path(P["cache_new_root"]) / "calvin_d" / "task_D_D" / PLAN["calvin_run_id"]
(cal_run / "logs").mkdir(parents=True, exist_ok=True)
json.dump({
    "schema_version": "route-corpus/1", "hub_model": "HiMoE-VLA",
    "benchmark": "calvin_d", "task_name": "task_D_D",
    "run_id": PLAN["calvin_run_id"], "campaign": PLAN["campaign"],
    "port": cal["port"], "gpu": cal["gpu"],
    "sampling": {"design": f"D->D 5-subtask chains, official get_sequences universe, "
                           f"first {cal['max_new_sequences']} sequences",
                 "max_new_sequences": cal["max_new_sequences"]},
    "status": "prepared",
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(cal_run / "meta.json", "w"), indent=1)
srv, cli = calvin_scripts(cal["gpu"], cal["port"], cal_run, cal["max_new_sequences"])
write(f"server-p{cal['port']}.sh", srv)
write("calvin-client.sh", cli)
manifest["calvin"] = {
    "name": f"server-p{cal['port']}", "gpu": cal["gpu"], "port": cal["port"],
    "script": f"server-p{cal['port']}.sh", "log": f"{cal_run}/logs/server.log",
    "client_log": f"{cal_run}/logs/client.log", "run_dir": str(cal_run)}

s = PLAN["smoke"]
smoke_out = f"{P['run_root']}/smoke"
write("server-smoke.sh", libero_server_script(
    s["gpu"], s["port"], s["suite"], f"{smoke_out}/server"))
write("worker-smoke.sh", worker_script(
    {"name": "smoke", "wid": 0, "gpu": s["gpu"], "port": s["port"],
     "benchmark": s["benchmark"], "task": s["task"], "init": s["init_state"],
     "k": s["k"], "mtq": s["max_trunk_queries"], "out": f"{smoke_out}/shard"}))

json.dump(manifest, open(CMDS / "manifest.json", "w"), indent=1)
n_snap = sum(sh["mtq"] for sh in manifest["shards"])
for w in range(len(waves)):
    srvs = [e for e in manifest["servers"] if e["wave"] == w]
    per_gpu = {}
    for e in srvs:
        per_gpu[e["gpu"]] = per_gpu.get(e["gpu"], 0) + 1
    print(f"wave {w}: {len(srvs)} task servers (per gpu {dict(sorted(per_gpu.items()))})")
print(f"libero workers: {len(manifest['shards'])}  "
      f"snapshot budget: {n_snap} (= {n_snap * PLAN['k']} branches)")
print(f"calvin: +{cal['max_new_sequences']} sequences on gpu {cal['gpu']} (spans waves)")
