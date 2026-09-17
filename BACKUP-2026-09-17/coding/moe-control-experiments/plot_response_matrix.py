"""Display all high-dose signs and random controls, without outcome selection."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config, collection = read(root / 'config.json'), read(root / 'collection.json')
    rows = [r for r in collection['rows'] if r['kind'] == 'probe']
    operators = config['operators']
    labels = ([op + '-high-plus' for op in operators] + [op + '-high-minus' for op in operators] +
              ['random-' + op + '-high' for op in operators])
    display = ([op.replace('_', ' + ') + ' : +' for op in operators] +
               [op.replace('_', ' + ') + ' : -' for op in operators] +
               ['random / ' + op.replace('_', ' + ') for op in operators])
    cohorts = [('all', 'Mean of five development states', rows)] + [
        (p['name'], p['name'] + ' / ' + p['head'], [r for r in rows if r['parent'] == p['name']])
        for p in config['parents']]
    for name, title, subset in cohorts:
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        for axis, field, heading in zip(axes, ('instantaneous_delta', 'delta'),
                                       ('Instantaneous components', 'Original v8.2 score components')):
            values = np.stack([np.mean([r[field] for r in subset if r['spec']['label'] == label], axis=0)
                               for label in labels])
            bound = max(.05, float(np.max(np.abs(values))))
            image = axis.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
            axis.set_xticks(range(5), config['components'], rotation=30, ha='right')
            axis.set_yticks(range(len(labels)), display, fontsize=8)
            axis.set_title(heading, fontsize=12)
            for i in range(len(labels)):
                for j in range(5):
                    axis.text(j, i, '%+.3f' % values[i, j], ha='center', va='center', fontsize=8,
                              color='white' if abs(values[i, j]) > .65 * bound else '#202020')
            for boundary in (4.5, 9.5):
                axis.axhline(boundary, color='#333333', lw=1)
            fig.colorbar(image, ax=axis, shrink=.75, label='Change / original margin')
        fig.suptitle(title + '\nHigh requested dose, equal L2 within each state; negative = lower component', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .93))
        directory = root if name == 'all' else root / 'states' / name
        extensions = ('png', 'pdf') if name == 'all' else ('png',)
        for extension in extensions:
            with (directory / ('response-matrix.' + extension)).open('xb') as stream:
                fig.savefig(stream, format=extension, dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    plot(parser.parse_args().run.resolve())
