"""Read physical task predicates from saved states without reading MoE scores."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path('/data/libero-runtime')
ROOT = BASE / 'samples/v82-knn-trajectory-20260915'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def read_state(environment, flat, goals, names):
    env = environment.env
    environment.set_state(flat)
    env.sim.forward()
    environment._post_process()
    predicate = np.array([bool(env._eval_predicate(goal)) for goal in goals])
    assert bool(predicate.all()) == bool(environment.check_success())
    positions = np.array([env.object_states_dict[name].get_geom_state()['pos'].copy() for name in names])
    grasp = []
    for name in names:
        state = env.object_states_dict[name]
        grasp.append(bool(env._check_grasp(env.robots[0].gripper, env.get_object(name)))
                     if state.object_state_type == 'object' else False)
    distance = np.full(len(goals), np.nan)
    for i, goal in enumerate(goals):
        if len(goal) == 3:
            first, second = [env.object_states_dict[name].get_geom_state()['pos'] for name in goal[1:]]
            distance[i] = np.linalg.norm(first-second)
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id].copy()
    return predicate, positions, np.array(grasp), distance, eef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['plus', 'pro'])
    parser.add_argument('--manifest', type=Path, default=BASE / 'samples/v82-evaluation-20260914T144322Z/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'physical')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    source = BASE / 'upstream' / ('LIBERO-plus' if args.benchmark == 'plus' else 'LIBERO-PRO')
    sys.dont_write_bytecode = True
    sys.path.insert(0, '/data/srv/src')
    sys.path.insert(0, str(source))
    if args.benchmark == 'plus':
        sys.path.insert(0, str(BASE / 'dependencies/libero-plus'))
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / ('physics-config-'+args.benchmark))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from himoe_libero_bridge.libero_runtime import _configure_libero
    _configure_libero(source)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    episodes = json.loads(args.manifest.read_text())['episodes']
    episodes = [e for e in episodes if e['benchmark'] == args.benchmark]
    if args.limit:
        episodes = episodes[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    suites = {}
    for index, episode in enumerate(episodes):
        target = args.output / (episode['name'] + '.npz')
        if target.exists() and target.with_suffix('.json').exists():
            metadata = json.loads(target.with_suffix('.json').read_text())
            assert metadata['output_sha256'] == sha(target)
            continue
        directory = Path(episode['source_artifact_dir'])
        manifest = json.loads((directory / 'episode-trace.json').read_text())
        trace_path = directory / manifest['array_file']
        assert sha(trace_path) == episode['source_trace_sha256']
        with np.load(trace_path) as saved:
            trace = {key: saved[key] for key in ('sim_states_before', 'sim_states_after', 'final_sim_state',
                                                'executed_lengths', 'successes', 'replan_start_steps')}
        if episode['task_suite'] not in suites:
            with contextlib.redirect_stdout(io.StringIO()):
                suites[episode['task_suite']] = benchmark.get_benchmark_dict()[episode['task_suite']]()
        suite = suites[episode['task_suite']]
        task = suite.get_task(episode['task_id'])
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        environment = ControlEnv(bddl_file_name=str(bddl), has_renderer=False, has_offscreen_renderer=False,
                                 use_camera_obs=False, horizon=531)
        try:
            environment.seed(7)
            environment.reset()
            initial = suite.get_task_init_states(episode['task_id'])[episode['init_state_id']]
            environment.set_init_state(initial)
            goals = environment.env.parsed_problem['goal_state']
            names = sorted({name for goal in goals for name in goal[1:]})
            states = np.concatenate([trace['sim_states_before'], trace['final_sim_state'][None]], axis=0)
            reads = [read_state(environment, state, goals, names) for state in states]
            predicate, position, grasp, distance, eef = [np.stack([row[j] for row in reads]) for j in range(5)]
            recorded_success = np.r_[False, trace['successes'][np.cumsum(trace['executed_lengths'])-1]]
            np.testing.assert_array_equal(predicate.all(-1), recorded_success)
            # Boundary states must be exactly the same physical state stored after the previous chunk.
            np.testing.assert_array_equal(states[1:], trace['sim_states_after'][np.cumsum(trace['executed_lengths'])-1])
            np.savez_compressed(target, predicates=predicate, positions=position, grasp=grasp, goal_distance=distance,
                                eef=eef, names=np.asarray(names), goals=np.asarray([json.dumps(g) for g in goals]),
                                action_steps=np.r_[trace['replan_start_steps'], trace['executed_lengths'].sum()])
            save(target.with_suffix('.json'), dict(name=episode['name'], benchmark=args.benchmark,
                 source_trace_sha256=episode['source_trace_sha256'], output_sha256=sha(target),
                 source_bddl=str(environment.env.bddl_file_name),
                 source_bddl_sha256=sha(Path(environment.env.bddl_file_name)), goals=goals, names=names,
                 states=len(states), success_reconstruction_exact=True, new_environment_actions=0,
                 annotations_read_routing=False))
            print('Physical reconstruction %s %d/%d %s: %d states, success exact' %
                  (args.benchmark, index+1, len(episodes), episode['name'], len(states)), flush=True)
        finally:
            environment.close()


if __name__ == '__main__':
    main()
