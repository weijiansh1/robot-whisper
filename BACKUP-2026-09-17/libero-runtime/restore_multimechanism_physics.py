"""Add rotations and goal articulation to saved native boundary observations."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

from restore_alarm_physics import read_state, sha, save

BASE = Path('/data/libero-runtime')
ROOT = BASE / 'samples/moe-multimechanism-matching-20260915'
PREVIOUS = BASE / 'samples/v82-knn-trajectory-20260915'
CURRENT = BASE / 'samples/v82-evaluation-20260914T144322Z'


def interval_for(obj, verb, low, high):
    predicate = getattr(obj, 'is_close' if verb == 'close' else 'turn_on')
    grid = np.linspace(low, high, 33)
    truth = np.array([bool(predicate(float(x))) for x in grid])
    changes = np.flatnonzero(truth[1:] != truth[:-1])
    if truth.all():
        return None, None
    if len(changes) != 1:
        raise ValueError('Expected one monotone predicate boundary: %r' % truth)
    a, b = grid[changes[0]], grid[changes[0]+1]
    left_truth = bool(predicate(float(a)))
    for _ in range(50):
        mid = (a+b)/2
        if bool(predicate(float(mid))) == left_truth:
            a = mid
        else:
            b = mid
    boundary = (a+b)/2
    # MuJoCo's soft joint limits allow overshoot; predicate half-spaces remain valid.
    return (None, boundary) if left_truth else (boundary, None)


def descriptors(env, goals):
    joints, unary = {}, []
    for goal_index, goal in enumerate(goals):
        for name in goal[1:]:
            state = env.object_states_dict[name]
            if state.object_state_type == 'object':
                obj = env.get_object(name)
                names = obj.joints
            else:
                obj = env.get_object(state.parent_name)
                names = env.object_sites_dict[name].joints
            for joint in names or []:
                jid = env.sim.model.joint_name2id(joint)
                if int(env.sim.model.jnt_type[jid]) not in (2, 3):
                    continue
                low, high = map(float, env.sim.model.jnt_range[jid])
                if high <= low:
                    raise ValueError('A positive limited hinge/slide range is required')
                joints[joint] = dict(name=joint, low=low, high=high,
                    address=int(env.sim.model.get_joint_qpos_addr(joint)))
                if len(goal) == 2:
                    if goal[0] not in ('close', 'turnon'):
                        raise ValueError('Unsupported unary goal: %r' % goal)
                    good_low, good_high = interval_for(obj, goal[0], low, high)
                    unary.append(dict(goal_index=goal_index, joint=joint,
                        good_low=good_low, good_high=good_high, verb=goal[0]))
    return [joints[k] for k in sorted(joints)], unary


def extra_state(env, names, joints, unary, goal_count):
    rotations = []
    for name in names:
        state = env.object_states_dict[name]
        matrix = (env.sim.data.body_xmat[env.obj_body_id[name]]
                  if state.object_state_type == 'object' else env.sim.data.get_site_xmat(name))
        rotations.append(np.asarray(matrix).reshape(3, 3).copy())
    values = np.array([env.sim.data.qpos[j['address']] for j in joints])
    by_name = {j['name']: (j, x) for j, x in zip(joints, values)}
    distances = np.full(goal_count, np.nan)
    for item in unary:
        j, value = by_name[item['joint']]
        distance = max(item['good_low']-value if item['good_low'] is not None else 0,
                       value-item['good_high'] if item['good_high'] is not None else 0,
                       0)/(j['high']-j['low'])
        k = item['goal_index']
        if np.isnan(distances[k]):
            distances[k] = distance
        elif item['verb'] == 'close':
            distances[k] = max(distances[k], distance)
        else:
            distances[k] = min(distances[k], distance)
    return np.array(rotations), values, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['plus', 'pro'])
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
    episodes = []
    for manifest, folder in ((CURRENT / 'summary.json', 'physical'),
                             (PREVIOUS / 'fresh-manifest.json', 'fresh-physical')):
        episodes.extend(dict(e, physical_folder=folder) for e in json.loads(manifest.read_text())['episodes']
                        if e['benchmark'] == args.benchmark)
    if args.limit:
        episodes = episodes[:args.limit]
    output = ROOT / 'physical'
    output.mkdir(parents=True, exist_ok=True)
    suites = {}
    for index, episode in enumerate(episodes):
        target = output / (episode['name']+'.npz')
        if target.exists() and target.with_suffix('.json').exists():
            prior = json.loads(target.with_suffix('.json').read_text())
            assert sha(target) == prior['output_sha256']
            if prior.get('implementation_sha256') == sha(Path(__file__)):
                continue
        old_path = PREVIOUS / episode['physical_folder'] / (episode['name']+'.npz')
        old_metadata = json.loads(old_path.with_suffix('.json').read_text())
        assert sha(old_path) == old_metadata['output_sha256']
        trace_path = Path(episode['source_artifact_dir']) / 'episode-trace.npz'
        assert sha(trace_path) == episode['source_trace_sha256']
        with np.load(trace_path) as trace:
            states = np.concatenate([trace['sim_states_before'], trace['final_sim_state'][None]])
        with np.load(old_path) as saved:
            old = {k: saved[k] for k in saved.files}
        if episode['task_suite'] not in suites:
            with contextlib.redirect_stdout(io.StringIO()):
                suites[episode['task_suite']] = benchmark.get_benchmark_dict()[episode['task_suite']]()
        suite = suites[episode['task_suite']]
        task = suite.get_task(episode['task_id'])
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        environment = ControlEnv(bddl_file_name=str(bddl), has_renderer=False,
            has_offscreen_renderer=False, use_camera_obs=False, horizon=531)
        try:
            bddl = Path(environment.env.bddl_file_name)
            assert sha(bddl) == old_metadata['source_bddl_sha256']
            environment.seed(7)
            environment.reset()
            environment.set_init_state(suite.get_task_init_states(episode['task_id'])[episode['init_state_id']])
            goals = environment.env.parsed_problem['goal_state']
            names = old['names'].tolist()
            assert goals == old_metadata['goals']
            joints, unary = descriptors(environment.env, goals)
            extra = []
            for q, flat in enumerate(states):
                fields = read_state(environment, flat, goals, names)
                for key, value in zip(('predicates', 'positions', 'grasp', 'goal_distance', 'eef'), fields):
                    np.testing.assert_allclose(value, old[key][q], rtol=0, atol=1e-12, equal_nan=True)
                more = extra_state(environment.env, names, joints, unary, len(goals))
                unary_mask = np.isfinite(more[2])
                np.testing.assert_array_equal(more[2][unary_mask] <= 1e-10, fields[0][unary_mask])
                extra.append(more)
            rotations, joint_values, unary_distance = [np.stack([r[k] for r in extra]) for k in range(3)]
            np.savez_compressed(target, **old, rotations=rotations, joint_values=joint_values,
                joint_ranges=np.array([[j['low'], j['high']] for j in joints]).reshape(-1, 2),
                joint_names=np.array([j['name'] for j in joints], dtype='U128'), unary_distance=unary_distance)
            save(target.with_suffix('.json'), dict(name=episode['name'], benchmark=args.benchmark,
                states=len(states), old_physical_sha256=sha(old_path), source_trace_sha256=sha(trace_path),
                output_sha256=sha(target), old_readouts_exact=True, unary_truth_exact=True,
                joints=joints, unary=unary, source_bddl_sha256=sha(bddl),
                new_environment_actions=0, annotations_read_routing=False,
                implementation_sha256=sha(Path(__file__)),
                protocol_sha256=sha(ROOT / 'PROTOCOL.md')))
            print('%s %d/%d %s: %d states, %d articulation joints' %
                  (args.benchmark, index+1, len(episodes), episode['name'], len(states), len(joints)), flush=True)
        finally:
            environment.close()


if __name__ == '__main__':
    main()
