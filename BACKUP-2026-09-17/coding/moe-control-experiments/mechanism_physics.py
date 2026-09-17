"""Python 3.8: measure saved physical states without stepping the environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main(root, name):
    config = json.loads((root / 'config.json').read_text())
    parent, = [p for p in config['parents'] if p['name'] == name]
    output = root / 'physics' / name
    output.mkdir(parents=True)
    runtime = Path('/data/libero-runtime')
    source_root = runtime / 'upstream' / ('LIBERO-plus' if parent['benchmark'] == 'plus' else 'LIBERO-PRO')
    sys.dont_write_bytecode = True
    sys.path[:0] = ['/data/srv/src', '/data/srv/packages/openpi-client/src', str(source_root)]
    if parent['benchmark'] == 'plus':
        sys.path.insert(0, str(runtime / 'dependencies/libero-plus'))
    os.environ['LIBERO_CONFIG_PATH'] = str(output / 'libero-config')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['OMP_NUM_THREADS'] = '1'
    import numpy as np
    from himoe_libero_bridge.episode_trace import load_episode_trace
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task, _sim_state
    manifest, arrays = load_episode_trace(Path(parent['source']))
    environment = None
    started = time.monotonic()
    try:
        environment, _, task, prompt = _load_task(EpisodeConfig(libero_root=str(source_root), **manifest['config']))
        assert str(task.name) == manifest['task_name'] and prompt == parent['prompt']
        base = environment.env
        goals = base.parsed_problem['goal_state']
        objects = sorted(base.objects_dict)
        entities = sorted(base.object_states_dict)
        joints = list(base.sim.model.joint_names)
        report = dict(parent=name, goals=goals, objects=objects, entities=entities, joints=joints,
                      arms={}, new_environment_actions=0, method='restore saved state, sim.forward, read only')
        for arm in config['physics_arms']:
            source = Path(config['source_run']) / name / arm / 'rollout.npz'
            with np.load(source, allow_pickle=False) as saved:
                states = np.concatenate([arrays['sim_states_before'][:1], saved['sim_states_after']])
                expected_success = np.r_[False, saved['successes']]
                actions = saved['actions']
            values = {key: [] for key in ('goals', 'grasp', 'object_position', 'entity_position', 'eef_position', 'qpos', 'success')}
            for index, state in enumerate(states):
                environment.sim.set_state_from_flattened(state)
                environment.sim.forward()
                assert np.array_equal(_sim_state(environment), state), (arm, index, 'state roundtrip')
                predicates = [bool(base._eval_predicate(goal)) for goal in goals]
                success = bool(environment.check_success())
                assert all(predicates) == success == bool(expected_success[index]), (arm, index, 'goal conjunction')
                values['goals'].append(predicates)
                values['success'].append(success)
                values['grasp'].append([bool(base._check_grasp(base.robots[0].gripper, base.get_object(obj).contact_geoms)) for obj in objects])
                values['object_position'].append(np.stack([base.sim.data.body_xpos[base.obj_body_id[obj]].copy() for obj in objects]))
                values['entity_position'].append(np.stack([base.object_states_dict[entity].get_geom_state()['pos'].copy() for entity in entities]))
                values['eef_position'].append(base.sim.data.site_xpos[base.robots[0].eef_site_id].copy())
                values['qpos'].append(base.sim.data.qpos.copy())
                if index % 100 == 0:
                    print(json.dumps(dict(parent=name, arm=arm, restored=index, goals=predicates)), flush=True)
            destination = output / (arm + '.npz')
            with destination.open('xb') as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in values.items()}, actions=actions)
            report['arms'][arm] = dict(states=len(states), source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                all_state_roundtrips_exact=True, all_successes_exact=True,
                final_goals=values['goals'][-1], final_success=values['success'][-1])
        report.update(passed=True, elapsed_seconds=time.monotonic() - started)
        write(output / 'result.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        write(output / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.parent)
