"""Python 3.8 LIBERO worker; only explicit actions can advance the simulator."""

import argparse
import hashlib
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import traceback


def main():
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
    os.environ['LIBERO_CONFIG_PATH'] = str(args.out / 'libero-config')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['OMP_NUM_THREADS'] = '1'
    from openpi_client import msgpack_numpy as codec
    import imageio.v2 as imageio
    import numpy as np
    from himoe_libero_bridge.episode_trace import load_episode_trace
    from himoe_libero_bridge.libero_runtime import (
        EpisodeConfig, _load_task, _sim_state, _assert_exact_array, LIBERO_DUMMY_ACTION)
    from himoe_libero_bridge.preprocess import build_policy_observation, frame_from_observation

    connection = Connection(args.fd)
    environment = writer = None
    manifest, arrays = load_episode_trace(args.source)
    steps, success, frame_count, exact_queries, exact_steps = 0, False, 0, 0, 0
    action_rows, state_rows, rewards, dones, successes, frame_hashes = [], [], [], [], [], []

    def send(value):
        connection.send_bytes(codec.packb(value))

    def frame(observation, compare):
        nonlocal frame_count
        value = frame_from_observation(observation)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        if compare and digest != manifest['source_render']['frame_sha256'][frame_count]:
            raise RuntimeError('Source render mismatch at frame %d' % frame_count)
        writer.append_data(value)
        frame_hashes.append(digest)
        frame_count += 1

    def observed():
        return dict(observation=build_policy_observation(observation, prompt),
                    sim_state=_sim_state(environment), steps=steps, success=success)

    try:
        config = EpisodeConfig(libero_root=str(source_root), **manifest['config'])
        environment, observation, task, prompt = _load_task(config)
        if str(task.name) != manifest['task_name'] or prompt != manifest['prompt']:
            raise RuntimeError('Task identity or prompt differs from source')
        writer = imageio.get_writer(str(args.out / 'episode.mp4'), fps=config.fps,
                                   codec='libx264', quality=8, macro_block_size=None)
        frame(observation, True)
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frame(observation, True)
        send(dict(event='ready', **observed()))
        while True:
            command = codec.unpackb(connection.recv_bytes())
            if command['op'] == 'close':
                break
            if command['op'] != 'step':
                raise ValueError('Unknown worker operation')
            actions = np.asarray(command['actions'], np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 10:
                raise ValueError('Invalid action block')
            if not np.isfinite(actions).all() or success or steps >= config.max_steps:
                raise ValueError('Cannot advance this episode')
            query = command.get('compare_query')
            if query is not None:
                current = observed()
                if int(arrays['replan_start_steps'][query]) != steps:
                    raise RuntimeError('Source action step mismatch')
                for key, source_key in (('observation/image', 'images'),
                                        ('observation/wrist_image', 'wrist_images'),
                                        ('observation/state', 'states')):
                    _assert_exact_array(current['observation'][key], arrays[source_key][query], key)
                _assert_exact_array(current['sim_state'], arrays['sim_states_before'][query], 'sim before')
                _assert_exact_array(actions, arrays['predicted_actions'][query][:len(actions)], 'source actions')
                exact_queries += 1
            began = steps
            for action in actions:
                observation, reward, done, _ = environment.step(action.tolist())
                success = bool(environment.check_success())
                state = _sim_state(environment)
                if query is not None:
                    _assert_exact_array(state, arrays['sim_states_after'][steps], 'sim after step %d' % steps)
                    if reward != arrays['rewards'][steps] or done != arrays['dones'][steps] or success != arrays['successes'][steps]:
                        raise RuntimeError('Source reward, done, or success mismatch')
                    exact_steps += 1
                action_rows.append(action.copy())
                state_rows.append(state)
                rewards.append(float(reward))
                dones.append(bool(done))
                successes.append(success)
                frame(observation, query is not None)
                steps += 1
                if success or steps >= config.max_steps:
                    break
            send(dict(event='stepped', executed=steps - began, **observed()))
        final = observed()
        with (args.out / 'rollout.npz').open('xb') as stream:
            np.savez_compressed(stream, actions=np.asarray(action_rows, np.float32),
                                sim_states_after=np.asarray(state_rows, np.float64),
                                rewards=np.asarray(rewards, np.float64), dones=np.asarray(dones, bool),
                                successes=np.asarray(successes, bool),
                                final_image=final['observation']['observation/image'],
                                final_wrist_image=final['observation']['observation/wrist_image'],
                                final_state=final['observation']['observation/state'],
                                final_sim_state=final['sim_state'])
        writer.close()
        writer = None
        report = dict(steps=steps, success=success, settle_steps=config.settle_steps,
                      exact_source_queries=exact_queries, exact_source_steps=exact_steps,
                      frames=frame_count, frame_sha256=frame_hashes, worker_pid=os.getpid())
        with (args.out / 'environment.json').open('x') as stream:
            json.dump(report, stream, indent=2)
        send(dict(event='closed', report=report))
    except BaseException as error:
        failure = dict(event='error', error=str(error), traceback=traceback.format_exc(),
                       action_steps=steps, settle_steps=manifest['config']['settle_steps'])
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
            environment.close()
        connection.close()


if __name__ == '__main__':
    main()
