"""Paired outcomes and physically aligned diagnostics, without new inference."""

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main(root):
    config, audit = read(root / 'config.json'), read(root / 'audit.json')
    assert audit['passed']
    pairs = audit['pairs']
    parents = config['parents']
    colors = ['#e5e7eb', '#398f78', '#2586bb', '#c95759']
    image = np.zeros((len(parents), 4), int)
    for i, p in enumerate(parents):
        for row in [r for r in pairs if r['parent'] == p['name']]:
            image[i, row['replicate']] = 2 if row['rescue'] else 3 if row['harm'] else 1 if row['native_success'] else 0
    fig, (ax, cost) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={'width_ratios': [2, 1]})
    ax.imshow(image, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect='auto')
    ax.set_yticks(range(8), ['%s q%d' % (p['name'], p['query']) for p in parents], fontsize=9)
    ax.set_xticks(range(4), ['original noise', 'new noise 1', 'new noise 2', 'new noise 3'], fontsize=9)
    ax.set_title('Paired terminal outcomes; one cell is two actual rollouts', fontsize=11)
    labels = ['both fail', 'both succeed', 'rescue', 'harm']
    for i in range(8):
        for j in range(4):
            ax.text(j, i, labels[image[i, j]], ha='center', va='center', fontsize=9,
                    color='#202020' if image[i, j] == 0 else 'white')
    native = [sum(r['native_calls'] for r in pairs if r['parent'] == p['name']) for p in parents]
    short = [sum(r['window_calls'] for r in pairs if r['parent'] == p['name']) for p in parents]
    y = np.arange(8)
    cost.barh(y - .18, native, .35, label='native10', color='#2586bb')
    cost.barh(y + .18, short, .35, label='20-step window5', color='#c95759')
    cost.set_yticks(y, [str(i + 1) for i in y])
    cost.invert_yaxis()
    cost.set_xlabel('Total suffix calls over four noise streams')
    cost.set_title('Actual inference cost', fontsize=11)
    cost.legend(fontsize=9, frameon=False)
    fig.suptitle('P3h: F=10, H=10; only executed prefix changes', fontsize=14)
    fig.tight_layout()
    fig.savefig(root / 'replan-window-overview.png', dpi=160)
    plt.close(fig)

    with (root / 'physical-time-mobility.csv').open() as stream:
        mobility = list(csv.DictReader(stream))
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for p, ax in zip(parents, axes.flat):
        ax.axvspan(0, 20, color='#f1dddd', alpha=.7)
        for arm, color in (('native10', '#2586bb'), ('window5', '#c95759')):
            for replicate in range(4):
                rows = [r for r in mobility if r['parent'] == p['name'] and r['arm'] == arm and
                        int(r['replicate']) == replicate and int(r['offset']) <= 100]
                ax.plot([int(r['offset']) for r in rows], [float(r['mobility_10steps']) for r in rows],
                        color=color, alpha=.45, linewidth=1.1, label=arm if replicate == 0 else None)
        ax.set_title('%s q%d' % (p['name'], p['query']), fontsize=10)
        ax.set_xlabel('Physical control steps after fork')
        ax.set_ylabel('Route mobility over 10 physical steps')
        ax.set_xlim(0, 100)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Diagnostic only: route normalization is not a recovery label', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(root / 'physical-time-routes.png', dpi=150)
    plt.close(fig)

    videos = []
    for p in parents:
        fig, axes = plt.subplots(4, 2, figsize=(8, 13))
        for replicate in range(4):
            for column, arm in enumerate(('native10', 'window5')):
                directory = root / p['name'] / ('r%d' % replicate) / arm
                result = read(directory / 'result.json')
                video = directory / 'episode.mp4'
                reader = imageio.get_reader(str(video))
                count, last = 0, None
                for frame in reader:
                    assert frame.shape == (256, 256, 3) and np.ptp(frame) > 0
                    last = frame
                    count += 1
                reader.close()
                assert count == result['action_steps'] + 11
                videos.append(dict(path=str(video.relative_to(root)), decoded_frames=count, nonblank=True))
                axes[replicate, column].imshow(last)
                axes[replicate, column].set_title('%s r%d: %s, step %d' %
                    (arm, replicate, 'success' if result['success'] else 'failure', result['action_steps']), fontsize=10)
                axes[replicate, column].axis('off')
        fig.suptitle('%s: four paired noise streams' % p['name'], fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(root / p['name'] / 'final-frames.png', dpi=130)
        plt.close(fig)
        print(json.dumps(dict(plotted=p['name'], videos_checked=len(videos))), flush=True)
    with (root / 'visual-check.json').open('x') as stream:
        json.dump(dict(passed=True, videos=videos, total_decoded_frames=sum(v['decoded_frames'] for v in videos)), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
