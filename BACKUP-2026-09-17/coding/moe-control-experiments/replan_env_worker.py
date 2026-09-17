"""Python 3.8 worker with replay-state auditing and explicit prefix execution."""

import argparse
import faulthandler
import hashlib
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--benchmark', choices=('plus', 'pro'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runtime = Path('/data/libero-runtime')
    source_root = runtime / 'upstream' / ('LIBERO-plus' if args.benchmark == 'plus' else 'LIBERO-PRO')
    sys.dont_write_bytecode = True
    sys.path[:0] = ['/data/srv/src', '/data/srv/packages/openpi-client/src', str(source_root)]
    if args.benchmark == 'plus':
        sys.path.insert(0, str(runtime / 'dependencies/libero-plus'))
    os.environ.update(LIBERO_CONFIG_PATH=str(args.out / 'libero-config'), MUJOCO_GL='egl',
                      PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1')
    import imageio.v2 as imageio
    import numpy as np
    from openpi_client import msgpack_numpy as codec
    from himoe_libero_bridge.episode_trace import load_episode_trace
    from himoe_libero_bridge.libero_runtime import (
        EpisodeConfig, _load_task, _sim_state, _assert_exact_array, LIBERO_DUMMY_ACTION)
    from himoe_libero_bridge.preprocess import build_policy_observation, frame_from_observation

    def packable(value):
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value)
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [packable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): packable(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, np.random.RandomState):
            return packable(value.get_state())
        if isinstance(value, np.random.Generator):
            return packable(value.bit_generator.state)
        raise TypeError('Unsupported state field: ' + type(value).__name__)

    connection = Connection(args.fd)
    manifest, arrays = load_episode_trace(args.source)
    source_actions = np.concatenate([a[:int(n)] for a, n in zip(arrays['predicted_actions'], arrays['executed_lengths'])])
    environment = writer = None
    steps = exact_steps = exact_queries = frame_count = 0
    success = False
    action_rows, states, integration_rows, rewards, dones, successes, frame_hashes = [], [], [], [], [], [], []
    physics_rows = []
    started = time.monotonic()
    spec = None

    def send(value):
        connection.send_bytes(codec.packb(value))

    def integration():
        sim = environment.sim
        value = np.empty(mujoco.mj_stateSize(sim.model._model, spec), np.float64)
        mujoco.mj_getState(sim.model._model, sim.data._data, value, spec)
        return value

    def physics():
        base = environment.env
        robot = base.robots[0]
        goals = [bool(base._eval_predicate(g)) for g in base.parsed_problem['goal_state']]
        objects = sorted(base.objects_dict)
        grasp = [bool(base._check_grasp(robot.gripper, base.get_object(o).contact_geoms)) for o in objects]
        return dict(goals=goals, grasp=grasp,
                    eef=base.sim.data.site_xpos[robot.eef_site_id].copy(),
                    object_position=np.stack([base.sim.data.body_xpos[base.obj_body_id[o]].copy() for o in objects]))

    def state_packet():
        base = environment.env
        controllers, buffers, grippers = [], [], []
        for robot in base.robots:
            state = {}
            for key, value in vars(robot.controller).items():
                if key == 'sim':
                    continue
                if key.startswith('interpolator_') and value is not None:
                    state[key] = packable(vars(value))
                else:
                    state[key] = packable(value)
            controllers.append(state)
            buffers.append({k: packable(vars(v)) for k, v in vars(robot).items()
                            if k.startswith('recent_') and hasattr(v, '__dict__')})
            grippers.append(packable(robot.gripper.current_action))
        observables = {name: {k: packable(v) for k, v in vars(obs).items() if not callable(v)}
                       for name, obs in base._observables.items()}
        clocks = {k: packable(getattr(base, k)) for k in ('timestep', 'cur_time', 'done') if hasattr(base, k)}
        local_rngs = {}
        for label, owner in (('wrapper', environment), ('base', base)):
            for key, value in vars(owner).items():
                if isinstance(value, (np.random.RandomState, np.random.Generator)):
                    local_rngs[label + '.' + key] = packable(value)
        packet = packable(dict(integration=integration(), controllers=controllers, robot_buffers=buffers,
                               gripper_actions=grippers, clocks=clocks, observation_cache=base._obs_cache,
                               observables=observables, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                               environment_rngs=local_rngs, observation=build_policy_observation(observation, prompt)))
        parts = {k: hashlib.sha256(codec.packb(v)).hexdigest() for k, v in packet.items()}
        return packet, dict(sha256=hashlib.sha256(codec.packb(packet)).hexdigest(), sections=parts,
                            integration_spec=int(spec), controller_fields=[sorted(c) for c in controllers])

    def frame(compare):
        nonlocal frame_count
        value = frame_from_observation(observation)
        sha = hashlib.sha256(value.tobytes()).hexdigest()
        if compare and sha != manifest['source_render']['frame_sha256'][frame_count]:
            raise RuntimeError('Source frame mismatch at %d' % frame_count)
        writer.append_data(value)
        frame_hashes.append(sha)
        frame_count += 1

    def observed():
        _, audit = state_packet()
        return dict(observation=build_policy_observation(observation, prompt), sim_state=_sim_state(environment),
                    steps=steps, success=success, state_audit=audit, physics=physics())

    try:
        config = EpisodeConfig(libero_root=str(source_root), **manifest['config'])
        random.seed(config.seed)
        environment, observation, task, prompt = _load_task(config)
        import mujoco
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        assert str(task.name) == manifest['task_name'] and prompt == manifest['prompt']
        assert config.max_steps == 520 and config.replan_steps == 10
        writer = imageio.get_writer(str(args.out / 'episode.mp4'), fps=config.fps,
                                   codec='libx264', quality=6, macro_block_size=None)
        frame(True)
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frame(True)
        physics_rows.append(physics())
        send(dict(event='ready', **observed()))
        while True:
            command = codec.unpackb(connection.recv_bytes())
            if command['op'] == 'close':
                break
            if command['op'] == 'snapshot':
                packet, audit = state_packet()
                path = args.out / 'fork-state.msgpack'
                with path.open('xb') as stream:
                    stream.write(codec.packb(packet))
                send(dict(event='snapshot', state_audit=audit))
                continue
            if command['op'] != 'step':
                raise ValueError('Unknown operation')
            actions = np.asarray(command['actions'], np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 10:
                raise ValueError('Invalid explicit action prefix')
            if not np.isfinite(actions).all() or success or steps >= config.max_steps:
                raise ValueError('Cannot execute actions')
            compare_start = command.get('compare_start')
            query = command.get('compare_query')
            if query is not None:
                current = observed()
                assert steps == int(arrays['replan_start_steps'][query])
                for key, source_key in (('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'),
                                        ('observation/state', 'states')):
                    _assert_exact_array(current['observation'][key], arrays[source_key][query], key)
                _assert_exact_array(current['sim_state'], arrays['sim_states_before'][query], 'source sim before')
                _assert_exact_array(actions, arrays['predicted_actions'][query][:len(actions)], 'source actions')
                exact_queries += 1
                compare_start = steps
            if compare_start is not None:
                assert compare_start == steps
                _assert_exact_array(actions, source_actions[steps:steps + len(actions)], 'source flat actions')
            began, step_started = steps, time.monotonic()
            for action in actions:
                observation, reward, done, _ = environment.step(action.tolist())
                success = bool(environment.check_success())
                state = _sim_state(environment)
                if compare_start is not None:
                    _assert_exact_array(state, arrays['sim_states_after'][steps], 'source state after')
                    assert reward == arrays['rewards'][steps] and done == arrays['dones'][steps]
                    assert success == bool(arrays['successes'][steps])
                    exact_steps += 1
                action_rows.append(action.copy())
                states.append(state)
                integration_rows.append(integration())
                rewards.append(float(reward))
                dones.append(bool(done))
                successes.append(success)
                physics_rows.append(physics())
                frame(compare_start is not None)
                steps += 1
                if success or steps >= config.max_steps:
                    break
            send(dict(event='stepped', executed=steps - began, worker_seconds=time.monotonic() - step_started,
                      **observed()))
        final = observed()
        with (args.out / 'rollout.npz').open('xb') as stream:
            np.savez_compressed(stream, actions=np.asarray(action_rows, np.float32),
                                sim_states_after=np.asarray(states, np.float64), integration_after=np.asarray(integration_rows),
                                rewards=np.asarray(rewards), dones=np.asarray(dones), successes=np.asarray(successes),
                                final_image=final['observation']['observation/image'],
                                final_wrist_image=final['observation']['observation/wrist_image'],
                                final_state=final['observation']['observation/state'],
                                final_sim_state=final['sim_state'],
                                **{'physical_' + k: np.asarray([v[k] for v in physics_rows]) for k in physics_rows[0]})
        writer.close()
        writer = None
        report = dict(steps=steps, success=success, settle_steps=config.settle_steps,
                      exact_source_queries=exact_queries, exact_source_steps=exact_steps, frames=frame_count,
                      frame_sha256=frame_hashes, final_state_audit=final['state_audit'],
                      worker_pid=os.getpid(), elapsed_seconds=time.monotonic() - started)
        with (args.out / 'environment.json').open('x') as stream:
            json.dump(report, stream, indent=2)
        send(dict(event='closed', report=report))
    except BaseException as error:
        failure = dict(event='error', error=str(error), traceback=traceback.format_exc(), action_steps=steps)
        with (args.out / 'environment-failure.json').open('x') as stream:
            json.dump(failure, stream, indent=2)
        try:
            send(failure)
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if writer is not None:
            writer.close()
        if environment is not None:
            print('environment_cleanup_started', flush=True)
            environment.close()
            print('environment_cleanup_finished', flush=True)
        connection.close()


if __name__ == '__main__':
    main()
