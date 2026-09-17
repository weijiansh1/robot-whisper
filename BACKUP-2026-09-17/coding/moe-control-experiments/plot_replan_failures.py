"""Decode completed failure-only videos and show their recorded final frames."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    audit = json.loads((root / 'failure-only-audit.json').read_text())
    assert audit['passed'] and audit['actual_completed_rollouts'] == 48
    parents = list(dict.fromkeys(p['parent'] for p in audit['pairs']))
    videos, figures = [], []
    for parent in parents:
        output = root / parent / 'failure-only-final-frames.png'
        assert not output.exists()
        fig, axes = plt.subplots(4, 2, figsize=(7.5, 12))
        for replicate in range(4):
            for column, arm in enumerate(('native10', 'window5')):
                directory = root / parent / ('r%d' % replicate) / arm
                result = json.loads((directory / 'result.json').read_text())
                assert not result['source_success']
                video = directory / 'episode.mp4'
                count, last = 0, None
                with imageio.get_reader(str(video)) as reader:
                    for frame in reader:
                        assert frame.shape == (256, 256, 3) and np.ptp(frame) > 0
                        count += 1
                        last = frame
                assert count == result['action_steps'] + 11
                videos.append(dict(path=str(video.relative_to(root)), decoded_frames=count, nonblank=True))
                ax = axes[replicate, column]
                ax.imshow(last)
                ax.set_title('%s, noise %d: %s at %d' %
                             (arm, replicate, 'success' if result['success'] else 'failure', result['action_steps']), fontsize=10)
                ax.axis('off')
        fig.suptitle(parent + ': recorded paired terminal frames', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(output, dpi=130)
        plt.close(fig)
        figures.append(str(output.relative_to(root)))
        print('%s: %d videos decoded' % (parent, len(videos)), flush=True)
    assert len(videos) == 48
    with (root / 'failure-only-visual-check.json').open('x') as stream:
        json.dump(dict(passed=True, scope='completed original-failure parents only', videos=videos, figures=figures,
                       total_decoded_frames=sum(v['decoded_frames'] for v in videos)), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
