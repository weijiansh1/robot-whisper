"""Standalone figures for branch coverage and selected contribution captures."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moe_multimechanism import ROOT, FEATURES


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 180})
    windows = pd.read_csv(ROOT / 'primary-with-later-progress.csv')
    support = pd.read_csv(ROOT / 'matching-support.csv')
    categories = ['verified_progress', 'approach_proxy', 'holding_or_static',
                  'return_motion', 'other_motion_no_gain', 'transient_or_regressing']
    colors = ['#26816a', '#71b8a7', '#9e9e9e', '#cf9a26', '#c34d55', '#56799d']
    labels = ['Verified progress', 'Approach proxy', 'Holding / static',
              'Return motion', 'Other motion, no gain', 'Transient / regressing']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    branch_labels = ['Freeze', 'Turbulence', 'Inversion', 'Curvature', 'No alarm']
    for y, branch in enumerate(['freeze', 'turbulence', 'inversion', 'curvature', 'none']):
        part = windows[~windows.v82_on if branch == 'none' else windows['on_'+branch]]
        left = 0.
        for category, color, label in zip(categories, colors, labels):
            fraction = part.category.eq(category).mean() if len(part) else 0
            axes[0].barh(y, fraction, left=left, height=.65, color=color, label=label if y == 0 else None)
            left += fraction
        axes[0].text(1.02, y, 'n=%d' % len(part), va='center', fontsize=9)
    axes[0].set(yticks=np.arange(5), yticklabels=branch_labels, xlim=(0, 1.24),
        xlabel='Fraction of complete H3 windows (overlapping branches)',
        title='External behavior across every v8.2 branch')
    axes[0].set_xticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
    axes[0].invert_yaxis()
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.19), ncol=2, frameon=False, fontsize=9)
    tiers = ['phase', 'geometry', 'all5_margin_1', 'same_variant_all5_margin_1']
    for offset, population, color, label in ((-.18, 'native', '#397d9b', '80 native parents'),
                                            (.18, 'functional', '#ba5266', 'Existing functional captures')):
        part = support[(support.population.eq(population)) & support.horizon.eq(3)].set_index('tier').loc[tiers]
        values = part.matched_windows.to_numpy()
        bars = axes[1].bar(np.arange(4)+offset, values, width=.34, color=color, label=label)
        axes[1].bar_label(bars, padding=3, fontsize=9)
    axes[1].set(xticks=np.arange(4), xticklabels=['Phase + time', '+ Geometry', '+ All 5 signals', '+ Same variant'],
        ylabel='Matched no-gain windows (controls may repeat)', title='Predefined matching support')
    axes[1].legend(frameon=False)
    fig.savefig(ROOT / 'branch-and-matching.png', bbox_inches='tight')
    fig.savefig(ROOT / 'branch-and-matching.pdf', bbox_inches='tight')
    plt.close(fig)
    table = pd.read_csv(ROOT / 'functional-features-with-context.csv')
    table = table[table.scope.eq('back_action')]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    titles = ['Total response / input response', 'Total direction change', 'Routed/shared update cosine',
        'Routed/shared norm deficit', 'Retained contribution fraction', 'Support-change contribution fraction',
        'Expert-update norm deficit', 'Negative expert attribution mass', 'Shared update attribution',
        'Largest absolute expert attribution']
    groups = ['verified_progress', 'no_observed_gain', 'transient_or_regressing']
    for ax, feature, title in zip(axes.flat, FEATURES, titles):
        for x, group, color in zip(range(3), groups, ['#26816a', '#c34d55', '#56799d']):
            values = table.loc[table.local_label.eq(group), feature].dropna().to_numpy()
            jitter = np.linspace(-.14, .14, len(values)) if len(values) else []
            ax.scatter(x+np.array(jitter), values, s=22, color=color, alpha=.75)
            if len(values):
                ax.plot([x-.22, x+.22], [np.median(values)]*2, color='#222222', linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(3), ['Progress', 'No gain', 'Mixed'], rotation=25)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Back-action tokens: selected captures, descriptive only; one point per anchor', fontsize=13)
    fig.savefig(ROOT / 'selected-contribution-features.png', bbox_inches='tight')
    fig.savefig(ROOT / 'selected-contribution-features.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
