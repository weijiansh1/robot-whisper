"""Check simulator-boundary versus cached-observation kinematics without actions."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

from restore_alarm_physics import BASE, ROOT, sha, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['plus', 'pro'])
    parser.add_argument('--manifest', type=Path, default=BASE/'samples/v82-evaluation-20260914T144322Z/summary.json')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    source = BASE/'upstream'/('LIBERO-plus' if args.benchmark == 'plus' else 'LIBERO-PRO')
    sys.dont_write_bytecode = True
    sys.path.insert(0, '/data/srv/src')
    sys.path.insert(0, str(source))
    if args.benchmark == 'plus':
        sys.path.insert(0, str(BASE/'dependencies/libero-plus'))
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT/('physics-config-'+args.benchmark))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from himoe_libero_bridge.libero_runtime import _configure_libero
    _configure_libero(source)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco
    episodes = [e for e in json.loads(args.manifest.read_text())['episodes'] if e['benchmark'] == args.benchmark]
    if args.limit:
        episodes = episodes[:args.limit]
    directory = ROOT/'kinematic-audit'
    directory.mkdir(exist_ok=True)
    suites = {}
    for episode in episodes:
        target = directory/(episode['name']+'.npz')
        if target.with_suffix('.json').exists():
            assert sha(target) == json.loads(target.with_suffix('.json').read_text())['output_sha256']
            continue
        if episode['task_suite'] not in suites:
            with contextlib.redirect_stdout(io.StringIO()):
                suites[episode['task_suite']] = benchmark.get_benchmark_dict()[episode['task_suite']]()
        suite = suites[episode['task_suite']]
        task = suite.get_task(episode['task_id'])
        bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        env = ControlEnv(bddl_file_name=str(bddl), has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, horizon=531)
        try:
            env.seed(7)
            env.reset()
            env.set_init_state(suite.get_task_init_states(episode['task_id'])[episode['init_state_id']])
            sim = env.env.sim
            dt = float(sim.model.opt.timestep)
            before, after = [], []
            trace_path = Path(episode['source_artifact_dir'])/'episode-trace.npz'
            assert sha(trace_path) == episode['source_trace_sha256']
            with np.load(trace_path) as trace:
                for flat in trace['sim_states_before']:
                    env.set_state(flat)
                    sim.forward()
                    after.append(sim.data.site_xpos[env.env.robots[0].eef_site_id].copy())
                    # Reverse only the final position integration, retaining its ending velocity.
                    mujoco.mj_integratePos(sim.model._model, sim.data.qpos, sim.data.qvel, -dt)
                    sim.forward()
                    before.append(sim.data.site_xpos[env.env.robots[0].eef_site_id].copy())
                observed = trace['states'][:, :3]
            before, after = np.asarray(before), np.asarray(after)
            uncompensated = np.abs(after-observed)
            compensated = np.abs(before-observed)
            np.savez_compressed(target, boundary_eef=after, preintegration_eef=before, observed_eef=observed)
            record = dict(name=episode['name'], timestep_seconds=dt, integrator=int(sim.model.opt.integrator),
                source_trace_sha256=episode['source_trace_sha256'], output_sha256=sha(target),
                boundary_max_abs_error_m=float(uncompensated.max()),
                preintegration_max_abs_error_m=float(compensated.max()),
                cached_observation_recovered=bool(compensated.max() < 1e-6), new_environment_actions=0)
            save(target.with_suffix('.json'), record)
            print(json.dumps(record), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
