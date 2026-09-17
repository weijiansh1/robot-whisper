"""Render actual endpoints, accepted interventions and all five score paths."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARMS = ('native', 'mobility', 'mobility_balance', 'random_balance')
LABELS = ('Native', 'Mobility', 'Mobility + balance', 'Matched random')
COLORS = ('#565656', '#147d92', '#25814b', '#be4e65')
COMPONENTS = ('Freeze', 'Acceleration', 'Periodicity', 'Inversion', 'Curvature')


def read(path):
    return json.loads(path.read_text())


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def contact_sheet(root, parent):
    width, height = 244, 350
    panel = Image.new('RGB', (width * 5, height), 'white')
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((8, 5), parent['name'], fill='black', font=font)
    draw.text((8, 27), parent['head'] + ' alarm', fill='#555555', font=font)
    draw.text((8, 49), 'Start q%d / step %d' % (parent['query'], parent['step']), fill='black', font=font)
    first = root / parent['name'] / 'native' / 'queries' / ('q%03d-native.npz' % parent['query'])
    with np.load(first, allow_pickle=False) as data:
        panel.paste(Image.fromarray(data['observation/image']), (10, 116))
    for index, (arm, label) in enumerate(zip(ARMS, LABELS), 1):
        directory = root / parent['name'] / arm
        result = read(directory / 'result.json')
        x = index * width + 8
        draw.text((x, 5), label, fill=COLORS[index - 1], font=font)
        draw.text((x, 27), '%s / step %d' % ('SUCCESS' if result['success'] else 'FAIL', result['action_steps']),
                  fill='#25814b' if result['success'] else '#b43849', font=font)
        draw.text((x, 49), 'Accepted %d / %d' % (result['accepted'], result['candidates']), fill='black', font=font)
        draw.text((x, 71), 'Step change %+d' % (result['action_steps'] - parent['source_steps']), fill='#555555', font=font)
        with np.load(directory / 'rollout.npz', allow_pickle=False) as data:
            panel.paste(Image.fromarray(data['final_image']), (index * width + 10, 116))
    with (root / parent['name'] / 'endpoint-comparison.png').open('xb') as stream:
        panel.save(stream, format='PNG')


def trajectories(root, parent):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(13, 14))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        rows = records(root / parent['name'] / arm / 'queries.jsonl')
        x = np.array([r['query'] for r in rows])
        selected = np.array([r['decision']['accepted'] for r in rows])
        formal = np.array([r['selected']['normalized'] for r in rows])
        instant = np.array([np.array(r['selected']['instantaneous']) / r['selected']['margins'] for r in rows])
        for index in range(5):
            for column, data in enumerate((formal, instant)):
                axis = axes[index, column]
                axis.plot(x, data[:, index], color=color, label=label, lw=1.3, alpha=.9)
                axis.scatter(x[selected], data[selected, index], color=color, marker='o', s=16, zorder=3)
    for index, component in enumerate(COMPONENTS):
        for column in range(2):
            axis = axes[index, column]
            axis.axvline(parent['query'] + 11.5, color='#777777', linestyle='--', lw=.8)
            if column == 0:
                axis.axhline(0, color='#888888', linestyle=':', lw=.8)
            axis.set_title(component + (' / original severity' if column == 0 else ' / instantaneous component'), fontsize=10)
            axis.set_xlabel('Query', fontsize=9)
            axis.set_ylabel('Margin units' if column == 0 else 'Component / margin', fontsize=9)
            axis.grid(alpha=.15)
            axis.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(parent['name'] + ' | dots: executed gate candidates; dashed: budget end\n'
                 'Different observations after divergence, not paired local effects.\n'
                 'Instantaneous components have no alarm threshold.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for extension in ('png', 'pdf'):
        with (root / parent['name'] / ('component-trajectories.' + extension)).open('xb') as stream:
            fig.savefig(stream, format=extension, dpi=150)
    plt.close(fig)


def overview(root, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    table = [[read(root / p['name'] / arm / 'result.json') for arm in ARMS] for p in config['parents']]
    acceptance = np.array([[r['accepted'] for r in row] for row in table])
    steps = np.array([[r['action_steps'] - p['source_steps'] for r in row] for p, row in zip(config['parents'], table)])
    for axis, matrix, title, cmap in ((axes[0], acceptance, 'Executed gate candidates', 'Greens'),
                                       (axes[1], steps, 'Action steps relative to native', 'RdYlGn_r')):
        limit = max(10, np.abs(matrix).max())
        axis.imshow(matrix, cmap=cmap, vmin=0 if axis is axes[0] else -limit, vmax=12 if axis is axes[0] else limit, aspect='auto')
        for i, row in enumerate(table):
            for j, result in enumerate(row):
                label = ('%d / %d' % (result['accepted'], result['candidates'])) if axis is axes[0] else (
                    '%+d\n%s' % (steps[i, j], 'SUCCESS' if result['success'] else 'FAIL'))
                axis.text(j, i, label, ha='center', va='center', fontsize=9,
                          color='white' if axis is axes[0] and matrix[i, j] >= 8 else 'black')
        axis.set_xticks(range(4), ('Native', 'Mobility', 'Mobility\n+ balance', 'Matched\nrandom'), fontsize=9)
        axis.set_yticks(range(5), [p['name'] for p in config['parents']], fontsize=8)
        axis.set_title(title, fontsize=11)
    fig.suptitle('P3d: 5 development parents, 20 actual simulation branches', fontsize=12)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        with (root / ('rollout-overview.' + extension)).open('xb') as stream:
            fig.savefig(stream, format=extension, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent')
    parser.add_argument('--overview-only', action='store_true')
    parser.add_argument('--trajectories-only', action='store_true')
    args = parser.parse_args()
    if args.overview_only and (args.parent or args.trajectories_only):
        parser.error('--overview-only cannot be combined with --parent or --trajectories-only')
    config = read(args.run / 'config.json')
    if not args.overview_only:
        for parent in config['parents']:
            if args.parent is None or args.parent == parent['name']:
                if not args.trajectories_only:
                    contact_sheet(args.run, parent)
                trajectories(args.run, parent)
    if args.parent is None:
        overview(args.run, config)
