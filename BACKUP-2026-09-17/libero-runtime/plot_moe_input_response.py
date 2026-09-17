"""Scientific response figures and recorded observations for the mechanism pilot."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path('/data/libero-runtime/samples/moe-input-response-20260915')
COLORS = {'observation': '#188576', 'noise': '#d38235', 'interaction': '#92979e',
          'success': '#247caa', 'failure': '#bb3e5f'}


def finish(figure, stem):
    figure.savefig(ROOT / (stem + '.png'), dpi=180, facecolor='white')
    figure.savefig(ROOT / (stem + '.pdf'), facecolor='white')
    plt.close(figure)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlepad': 10, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    existing = pd.read_csv(ROOT / 'existing-parent-summary.csv')
    existing = existing[existing.comparison.eq('within_flow') & existing.scope.eq('back_action')]
    fig, ax = plt.subplots(figsize=(8.7, 4.1), layout='constrained')
    x = np.arange(len(existing))
    value = existing.expert_term_norm_fraction.to_numpy() * 100
    colors = [COLORS['success'] if success else COLORS['failure'] for success in existing.success]
    bars = ax.bar(x, value, color=colors, width=.65)
    ax.set_xticks(x, [name.replace('plus-', '').replace('-init', '\ninit ') for name in existing.episode], fontsize=9)
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in value], padding=3)
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(handles=[Patch(color=COLORS[outcome], label='Eventual ' + outcome)
                       for outcome in ('success', 'failure')],
              loc='upper center', ncols=2, frameon=False, fontsize=9)
    ax.set_ylabel('Expert-output term / sum of term norms (%)')
    ax.set_title('Existing six parents: output changes even when the expert set stays fixed', fontsize=12)
    finish(fig, 'existing-functional-decomposition')

    if not (ROOT / 'controlled-results.json').exists():
        return
    energy = pd.read_csv(ROOT / 'factorial-energy.csv')
    parent = pd.read_csv(ROOT / 'controlled-parent-responses.csv')
    back = parent[parent.scope.eq('back_action')].sort_values('parent_id')
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 8.1), layout='constrained')
    for ax, field, title in ((axes[0, 0], 'router_sqrt', 'A  Router response: observation and noise'),
                             (axes[0, 1], 'total', 'B  HB block output: observation and noise')):
        table = energy[energy.field.eq(field) & energy.scope.eq('back_action')].sort_values('parent_id')
        x, bottom = np.arange(len(table)), np.zeros(len(table))
        for name in ('observation', 'noise', 'interaction'):
            values = table[name + '_energy_fraction'].to_numpy() * 100
            ax.bar(x, values, bottom=bottom, width=.76, color=COLORS[name], label=name.capitalize())
            bottom += values
        ax.set_xticks(x, [f'T{row.task:02d}\n' + ('S' if row.success else 'F') for row in table.itertuples()], fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel('Balanced 2x2 contrast energy (%)')
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.legend(loc='upper center', ncols=3, fontsize=8,
                  framealpha=.9, handlelength=1.3, columnspacing=.7)
    for ax, key, title, ylabel in (
        (axes[1, 0], 'routed_relative_gain', 'C  Routed output response with noise held fixed', 'Relative output change / relative input change'),
        (axes[1, 1], 'expert_term_norm_fraction', 'D  Expert-output term on unchanged expert sets', 'Expert-output term / sum of term norms')):
        for pair_id, group in back.groupby('pair_id'):
            s, f = group[group.success].iloc[0], group[~group.success].iloc[0]
            alpha = 1 if s.stable_matched else .38
            ax.plot([pair_id-.13, pair_id+.13], [s[key], f[key]], color='#8b8b8b', lw=1, alpha=alpha)
            ax.scatter(pair_id-.13, s[key], color=COLORS['success'], s=40, alpha=alpha,
                       label='Eventual success' if pair_id == 0 else None)
            ax.scatter(pair_id+.13, f[key], color=COLORS['failure'], s=40, marker='s', alpha=alpha,
                       label='Eventual failure' if pair_id == 0 else None)
        tasks = back.drop_duplicates('pair_id').sort_values('pair_id')
        ax.set_xticks(tasks.pair_id, [f'T{r.task:02d}' + ('*' if r.stable_matched else '') for r in tasks.itertuples()])
        ax.set_xlabel('* Meets the fixed stable-routing match criteria')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.13)
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_xlabel('52 / 1,280 sites retain all four experts; unavailable values omitted', fontsize=9)
    fig.suptitle('Back four HB layers, action tokens: 16 trajectories, four conditions each', fontsize=14, fontweight='bold')
    finish(fig, 'controlled-functional-response')

    comparisons = (
        ('router_sqrt', 'front_action', 'Front action router'),
        ('router_sqrt', 'back_action', 'Back action router'),
        ('router_sqrt', 'front_state', 'Front state router'),
        ('router_sqrt', 'back_state', 'Back state router'),
        ('total', 'front_action', 'Front action HB output'),
        ('total', 'back_action', 'Back action HB output'),
        ('action_translation', 'action_chunk', 'Final translation action'),
        ('action_rotation', 'action_chunk', 'Final rotation action'),
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.6), layout='constrained')
    for row, (field, scope, label) in enumerate(comparisons):
        table = energy[energy.field.eq(field) & energy.scope.eq(scope)].sort_values('parent_id')
        values = table.noise_energy_fraction.to_numpy() * 100
        jitter = np.linspace(-.18, .18, len(table))
        for success, marker in ((True, 'o'), (False, 's')):
            mask = table.success.to_numpy() == success
            ax.scatter(values[mask], row+jitter[mask], s=27, marker=marker, alpha=.7,
                       color=COLORS['success' if success else 'failure'],
                       label=('Eventual success' if success else 'Eventual failure') if row == 0 else None)
        ax.plot(np.median(values), row, marker='|', color='#222222', markersize=22, markeredgewidth=2,
                linestyle='none', label='Median' if row == 0 else None)
        median = np.median(values)
        ax.text(104, row, '<0.01%' if median < .01 else f'{median:.1f}%', va='center', fontsize=10)
    ax.set_yticks(np.arange(len(comparisons)), [item[2] for item in comparisons])
    ax.set_xlim(-2, 117)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(len(comparisons)-.5, -1.15)
    ax.set_xlabel('Noise main-effect energy / total balanced 2x2 contrast energy (%)')
    ax.set_title('Noise response depends on layer, token type and readout', fontsize=13, fontweight='bold')
    ax.legend(loc='upper center', ncols=3, frameon=False, fontsize=9)
    ax.grid(axis='x', alpha=.12)
    finish(fig, 'noise-response-by-scope')

    selection = json.loads((ROOT / 'selection.json').read_text())
    pairs = [p for p in selection['pairs'] if p['stable_matched']]
    fig, axes = plt.subplots(len(pairs), 4, figsize=(11.6, len(pairs)*2.65), layout='constrained', squeeze=False)
    for row, pair in enumerate(pairs):
        for side, outcome in enumerate(('success', 'failure')):
            p = next(p for p in selection['parents'] if p['name'] == pair[outcome])
            with np.load(Path(p['source_dir']) / 'episode-trace.npz') as z:
                images = z['images'][p['query']-1:p['query']+1]
            for oldnew in (0, 1):
                ax = axes[row, side*2+oldnew]
                ax.imshow(images[oldnew])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"T{p['task']:02d} / eventual {outcome} / q{p['query']-1+oldnew}", fontsize=9)
                if oldnew == 1:
                    ax.set_xlabel(f"Recorded next-chunk EEF motion: {p['eef_next_displacement_m']*1000:.1f} mm", fontsize=8)
    fig.suptitle('Recorded observations for the four stable-routing matched pairs', fontsize=13)
    finish(fig, 'matched-recorded-observations')


if __name__ == '__main__':
    main()
