"""Inspect actual candidate endpoints and post-intervention head trajectories."""

import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(path.read_text())


def contact_sheet(root, parent):
    directory = root / parent['name']
    decisions = load(directory / 'decisions.json')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    panel = Image.new('RGB', (224 * 5, 340), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), parent['name'], font=font, fill='black')
    draw.text((6, 26), parent['head'], font=font, fill='black')
    draw.text((6, 49), 'Input q%d, step %d' % (parent['query'], parent['step']), font=font, fill='black')
    with np.load(directory / 'pool.npz', allow_pickle=False) as pool:
        panel.paste(Image.fromarray(pool['image']), (0, 116))
    for candidate in range(4):
        branch = directory / ('candidate-%d' % candidate)
        result = load(branch / 'result.json')
        x = 224 * (candidate + 1)
        draw.text((x + 6, 5), 'c%d: %s, step %d' %
                  (candidate, 'SUCCESS' if result['success'] else 'FAIL', result['action_steps']),
                  font=font, fill='#17623e' if result['success'] else '#aa2834')
        draw.text((x + 6, 26), 'head = %.4f' % decisions['target_severity'][candidate], font=font, fill='black')
        for line, label in enumerate(textwrap.wrap(', '.join(result['selected_by']) or 'unselected', width=25)):
            draw.text((x + 6, 49 + 19 * line), label, font=font, fill='#444444')
        with np.load(branch / 'rollout.npz', allow_pickle=False) as rollout:
            panel.paste(Image.fromarray(rollout['final_image']), (x, 116))
    with (directory / 'candidate-comparison.png').open('xb') as stream:
        panel.save(stream, format='PNG')


def trajectories(root, parents):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    colors = ['#555555', '#24887f', '#c24455', '#b48b20']
    for axis, parent in zip(axes.flat, parents):
        directory = root / parent['name']
        decision = load(directory / 'decisions.json')
        for candidate in range(4):
            with (directory / ('candidate-%d' % candidate) / 'queries.jsonl').open() as stream:
                rows = [json.loads(line) for line in stream][:7]
            tags = [name for name in ('native', 'edge', 'head_low', 'head_high') if decision['selected'][name] == candidate]
            axis.plot(range(len(rows)), [r['head_severity'] for r in rows], marker='o', ms=3,
                      color=colors[candidate], label='c%d %s' % (candidate, ','.join(tags)))
        axis.axhline(0, color='#888888', lw=.8, linestyle=':')
        axis.set_title(parent['name'] + ' / ' + parent['head'], fontsize=10)
        axis.set_xlabel('Queries after intervention (0 = changed query)')
        axis.set_ylabel('Target severity (margin units)')
        axis.grid(alpha=.15)
        axis.legend(fontsize=7)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        path = root / ('head-trajectories.' + extension)
        with path.open('xb') as stream:
            fig.savefig(stream, format=extension, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent')
    parser.add_argument('--trajectories-only', action='store_true')
    args = parser.parse_args()
    if args.trajectories_only and args.parent:
        parser.error('--trajectories-only and --parent cannot be combined')
    config = load(args.run / 'config.json')
    selected = [p for p in config['parents'] if args.parent is None or p['name'] == args.parent]
    if not args.trajectories_only:
        for parent in selected:
            contact_sheet(args.run, parent)
    if args.parent is None:
        trajectories(args.run, config['parents'])
