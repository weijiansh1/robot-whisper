"""Create image-only diagnostics from completed online simulator rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', required=True)
    args = parser.parse_args()
    root = args.run
    config = json.loads((root / 'config.json').read_text())
    parent, = [p for p in config['parents'] if p['name'] == args.parent]
    arms = config['arms']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    panel = Image.new('RGB', (224 * 5, 280), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), parent['name'], font=font, fill='black')
    with np.load(Path(parent['source']) / 'episode-trace.npz', allow_pickle=False) as source:
        query = parent['first_alarm'] + 1 if parent['first_alarm'] >= 0 else 0
        panel.paste(Image.fromarray(source['images'][query]), (0, 55))
    draw.text((8, 31), 'Branch input q%d' % query, font=font, fill='black')
    for column, arm in enumerate(arms, 1):
        result = json.loads((root / parent['name'] / arm / 'result.json').read_text())
        used_arm = result['alias_of'] or arm
        with np.load(root / parent['name'] / used_arm / 'rollout.npz', allow_pickle=False) as rollout:
            panel.paste(Image.fromarray(rollout['final_image']), (224 * column, 55))
        draw.text((224 * column + 7, 5), arm, font=font, fill='black')
        draw.text((224 * column + 7, 31), '%s, step %d' %
                  ('SUCCESS' if result['success'] else 'FAIL', result['action_steps']),
                  font=font, fill='#14643e' if result['success'] else '#a22f3b')
    destination = root / parent['name'] / 'state-comparison.png'
    with destination.open('xb') as stream:
        panel.save(stream, format='PNG')
    print(destination)


if __name__ == '__main__':
    main()
