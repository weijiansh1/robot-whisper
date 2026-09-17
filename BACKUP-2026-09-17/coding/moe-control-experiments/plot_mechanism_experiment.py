"""Images and descriptive plots from recorded data; no model or environment."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def frames(root):
    from PIL import Image, ImageDraw, ImageFont
    config = read(root / 'config.json')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for parent in config['parents']:
        original = load(Path(parent['source']) / 'episode-trace.npz')
        cases = [c for c in config['cases'] if c['parent'] == parent['name']]
        points = [0, 6, cases[0]['query'], parent['first_alarm'], cases[-1]['query'], None]
        panel = Image.new('RGB', (6 * 224, 4 * 252 + 50), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), parent['name'] + ' | native and combo: agent / wrist views', fill='black', font=font)
        for arm_index, arm in enumerate(config['physics_arms']):
            directory = Path(config['source_run']) / parent['name'] / arm
            for col, query in enumerate(points):
                if query is None:
                    final = load(directory / 'rollout.npz')
                    images = [final['final_image'], final['final_wrist_image']]
                    label = 'final step %d' % len(final['actions'])
                elif query < parent['query']:
                    images = [original['images'][query], original['wrist_images'][query]]
                    label = 'q%d / step %d' % (query, query * 10)
                else:
                    saved = load(directory / 'queries' / ('q%03d-native.npz' % query))
                    images = [saved['observation/image'], saved['observation/wrist_image']]
                    label = 'q%d / step %d' % (query, query * 10)
                for camera, image in enumerate(images):
                    x, y = col * 224, 50 + (2 * arm_index + camera) * 252
                    draw.text((x + 4, y + 2), ('native' if arm_index == 0 else 'combo') + ' ' + label, font=font, fill='black')
                    panel.paste(Image.fromarray(image), (x, y + 26))
        destination = root / 'physics' / parent['name'] / 'frames.png'
        with destination.open('xb') as stream:
            panel.save(stream, format='PNG')
        print(destination, flush=True)


def save(fig, destination):
    for extension in ('png', 'pdf'):
        with destination.with_suffix('.' + extension).open('xb') as stream:
            fig.savefig(stream, format=extension, dpi=160, bbox_inches='tight')


def plots(root, timelines_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    config, summary = read(root / 'config.json'), read(root / 'summary.json')
    colors = dict(combo='#14835d', opposite='#bd435d', random='#297dae')
    fields = ['back_routed_relative', 'back_total_relative', 'back_block_relative', 'projection_input_relative', 'velocity_relative']
    labels = ['Routed sum\n(derived)', 'MoE total', 'Block output', 'Projection\ninput', 'Flow velocity']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
    for label in colors:
        values = [r for r in summary['rows'] if r['label'] == label]
        axes[0].plot(range(5), [100 * np.mean([r[k] for r in values]) for k in fields], marker='o', label=label, color=colors[label])
        for phase_index, phase in enumerate(('before_alarm', 'first_accepted', 'last_accepted')):
            selected = [r['action_rms'] for r in values if r['phase'] == phase]
            x = phase_index + {'combo': -.2, 'opposite': 0, 'random': .2}[label]
            axes[1].scatter([x] * 5, selected, s=24, alpha=.6, color=colors[label])
            axes[1].plot([x - .065, x + .065], [np.mean(selected)] * 2, color=colors[label], linewidth=2)
    axes[0].set_xticks(range(5), labels)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Paired change / native tensor norm (%)')
    axes[0].set_title('Descriptive response scales, 15 states per direction')
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(3), ['Before alarm', 'First accepted', 'Last accepted'])
    axes[1].set_ylabel('Action difference RMS / action std (6D)')
    axes[1].set_title('Action response; dots are five development parents')
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    if not timelines_only:
        save(fig, root / 'mechanism-response')
    plt.close(fig)
    for parent in config['parents']:
        directory = root / 'physics' / parent['name']
        metadata = read(directory / 'result.json')
        data = {arm: load(directory / (arm + '.npz')) for arm in config['physics_arms']}
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1]})
        longest = max(len(a['goals']) for a in data.values())
        matrix = np.full((4, longest), np.nan)
        for i, arm in enumerate(config['physics_arms']):
            matrix[2*i:2*i+2, :len(data[arm]['goals'])] = data[arm]['goals'].T
        axes[0].imshow(matrix, interpolation='nearest', aspect='auto', origin='upper',
                       extent=[-.5, longest - .5, 3.5, -.5], cmap=ListedColormap(['#eeeeee', '#14835d']), vmin=0, vmax=1)
        axes[0].set_yticks(range(4), ['native G1', 'native G2', 'combo G1', 'combo G2'])
        axes[0].set_title(parent['name'] + '\n' + '\n'.join('G%d: %s' % (i + 1, ' '.join(g)) for i, g in enumerate(metadata['goals'])), fontsize=10)
        targets = [name for name in metadata['objects'] if any(name in goal[1:] for goal in metadata['goals'])]
        for i, target in enumerate(targets):
            index = metadata['objects'].index(target)
            for arm, style in (('native', '-'), ('mobility_balance', '--')):
                a = data[arm]
                distance = np.linalg.norm(a['eef_position'] - a['object_position'][:, index], axis=-1)
                axes[1].plot(distance * 100, style, color=['#297dae', '#bd435d'][i % 2],
                             label=target + (' native' if arm == 'native' else ' combo'), linewidth=1.3)
        axes[1].set_ylabel('EEF to object\ncenter (cm)', fontsize=10)
        axes[1].legend(fontsize=8, frameon=False)
        a, b = data['native'], data['mobility_balance']
        length = min(len(a['goals']), len(b['goals']))
        separation = np.linalg.norm(a['eef_position'][:length] - b['eef_position'][:length], axis=1)
        axes[2].plot(separation * 1000, color='#14835d')
        axes[2].set_ylabel('EEF separation\nnative / combo (mm)', fontsize=10)
        axes[2].set_xlabel('Executed action steps (saved P3d trajectories, not new rollouts)')
        for ax in axes:
            ax.axvline(parent['first_alarm'] * 10, color='#555555', linestyle=':', linewidth=1)
            ax.set_xlim(0, longest - 1)
            ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        save(fig, directory / ('task-timeline-v2' if timelines_only else 'task-timeline'))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('frames', 'plots', 'timelines'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    frames(args.run.resolve()) if args.command == 'frames' else plots(args.run.resolve(), args.command == 'timelines')
