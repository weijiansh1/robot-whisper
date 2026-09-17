"""Standalone figures for archived alarms, neighbor geometry and real observations."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v82_knn_research import ROOT, RUN_B, GROUPS, archive, combine_first

COLORS = {'v82': '#bf4766', 'knn': '#267daf', 'OR': '#198b70', 'AND_latched': '#747981',
          'failure': '#bf4766', 'success': '#267daf'}
FEATURE_COLORS = ['#379cb8', '#306592', '#d09037', '#8c689f']


def finish(fig, name):
    fig.savefig(ROOT / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(ROOT / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def archived_figure():
    frame = pd.read_csv(ROOT / 'archived-episode-decisions.csv')
    b = frame[frame.run_id.eq(RUN_B)].reset_index(drop=True)
    methods = combine_first(b.v82_frozen, b.knn20)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), layout='constrained')
    fractions = np.linspace(0, 1, 101)
    for ax, failure, title in ((axes[0, 0], True, 'A  Failure detection across the configured budget'),
                                (axes[0, 1], False, 'B  Successful-episode alarms, all retained')):
        eligible = b.failure.to_numpy() == failure
        for method, first in methods.items():
            event_fraction = first*10 / b.horizon_actions.to_numpy()
            curve = [((first >= 0) & (event_fraction <= fraction) & eligible).sum() / eligible.sum() * 100
                     for fraction in fractions]
            ax.plot(fractions*100, curve, color=COLORS[method], label=method.replace('_', ' '), lw=1.8)
        ax.set_xlabel('Configured action budget consumed (%)')
        ax.set_ylabel('Recall (%)' if failure else 'False-alarm rate (%)')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100 if failure else 1.8)
        ax.grid(alpha=.12)
        ax.legend(frameon=False, fontsize=8)
    groups = pd.read_csv(ROOT / 'archived-joint-groups.csv')
    groups = groups[groups.cohort.eq('B') & groups.cutoff.eq('all')].set_index('group')
    names = ['knn_only', 'v82_only', 'both']
    ax = axes[1, 0]
    bottom = np.zeros(3)
    for outcome in ('failures', 'successes'):
        values = groups.loc[names, outcome].to_numpy()
        ax.bar(np.arange(3), values, bottom=bottom, width=.65,
               color=COLORS['failure' if outcome == 'failures' else 'success'], label=outcome.capitalize())
        bottom += values
    labels = [label + f'\n{groups.loc[name, "failures"]} F / {groups.loc[name, "successes"]} S'
              for name, label in zip(names, ['kNN only', 'v8.2 only', 'Both'])]
    ax.set_xticks(np.arange(3), labels, fontsize=9)
    ax.set_ylabel('Parent episodes')
    ax.set_title('C  Whole-record groups among flagged B episodes', loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    timing = pd.read_csv(ROOT / 'archived-alarm-order.csv')
    timing = timing[timing.cohort.eq('B')].set_index('outcome')
    ax = axes[1, 1]
    for offset, outcome in ((-.18, 'failure'), (.18, 'success')):
        row = timing.loc[outcome]
        values = row[['v82_first', 'simultaneous', 'knn_first']].to_numpy(float) / row['both'] * 100
        bars = ax.bar(np.arange(3)+offset, values, width=.34, color=COLORS[outcome], label=f'Eventual {outcome}')
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], fontsize=8, padding=3)
    ax.set_xticks(np.arange(3), ['v8.2 earlier', 'Same query', 'kNN earlier'])
    ax.set_ylim(0, 80)
    ax.set_ylabel('Fraction of episodes flagged by both (%)')
    ax.set_title('D  Alarm order does not itself separate outcomes', loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Original frozen configurations: 16,000 B episodes', fontsize=14, fontweight='bold')
    finish(fig, 'original-v82-knn-comparison')


def geometry_figure():
    rank = pd.read_csv(ROOT / 'available-same-query-aucs.csv')
    mean = rank.groupby(['query', 'measurement']).auc.mean().unstack()
    points = pd.read_csv(ROOT / 'neighbor-explanations.csv', dtype={'identity': str})
    points = points[points.cohort.eq('B') & points.anchor.eq('knn_first')]
    deletion = pd.read_csv(ROOT / 'reference-deletion-summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8), layout='constrained')
    ax = axes[0, 0]
    for column, label, color in (
        ('knn_distance', 'kNN distance', COLORS['knn']),
        ('acceleration', 'Signed acceleration score', '#d09037'),
        ('acceleration_distance_term', 'Acceleration distance term', '#8c689f')):
        ax.plot(mean.index, mean[column], marker='o', color=color, label=label, lw=1.8)
    ax.axhline(.5, color='#777777', ls=':', lw=1)
    ax.set_xticks([9, 13, 20])
    ax.set_ylim(.45, .8)
    ax.set_xlabel('Query index')
    ax.set_ylabel('Task-mean same-query AUROC')
    ax.set_title('A  Direction and distance differ by observation stage', loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.12)
    ax = axes[0, 1]
    groups = [('both', True), ('both', False), ('knn_only', True), ('knn_only', False)]
    labels, means = [], []
    for group, failure in groups:
        part = points[points.joint_group.eq(group) & points.failure.eq(failure)]
        means.append(part[[name + '_fraction' for name in GROUPS]].mean().to_numpy())
        labels.append(('Both' if group == 'both' else 'kNN only') + '\n' + ('F' if failure else 'S') + f' (n={len(part)})')
    values, bottom = np.asarray(means)*100, np.zeros(len(groups))
    for j, name in enumerate(GROUPS):
        ax.bar(np.arange(len(groups)), values[:, j], bottom=bottom, color=FEATURE_COLORS[j], width=.67,
               label=name.replace('_', ' '))
        bottom += values[:, j]
    ax.set_xticks(np.arange(len(groups)), labels, fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Mean share of kNN distance (%)')
    ax.set_title('B  First-kNN-alarm distance attribution', loc='left', fontsize=11, fontweight='bold')
    ax.legend(loc='upper center', ncols=2, fontsize=7.5, frameon=False)
    ax.set_xlabel('Groups contain 4 / 4 / 1 / 1 axes; shares are not causal importance', fontsize=8)
    for ax, failure, title in ((axes[1, 0], False, 'C  Reference deletion: successful-episode alarms'),
                               (axes[1, 1], True, 'D  Reference deletion: failure detection')):
        rows = deletion[deletion.failure.eq(failure)].set_index('method')
        counts = np.array([rows.iloc[0].original_hits, rows.loc['random'].deletion_hits, rows.loc['same_task'].deletion_hits])
        n = int(rows.iloc[0].episodes)
        bars = ax.bar(np.arange(3), counts/n*100, color=['#747981', '#379cb8', '#d09037'], width=.6)
        ax.bar_label(bars, labels=[f'{v}/{n}' for v in counts], padding=4, fontsize=9)
        ax.set_xticks(np.arange(3), ['Full reference', 'Random removal', 'Same-task removal'], fontsize=9)
        ax.set_ylabel('Recall (%)' if failure else 'False-alarm rate (%)')
        ax.set_ylim(0, 105 if failure else 2.7)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=.12)
    fig.suptitle('Separate available-data kNN profile: mechanism diagnostics', fontsize=14, fontweight='bold')
    finish(fig, 'knn-mechanism-diagnostics')


def case_figure():
    frame = pd.read_csv(ROOT / 'current-decisions.csv')
    data = archive(ROOT / 'current-geometry.npz')
    tau = json.loads((ROOT / 'available-profile/parameters.json').read_text())['threshold']
    names = ['plus-task00-init039', 'plus-task00-init047', 'plus-task09-init039', 'plus-task09-init047']
    fig, axes = plt.subplots(4, 4, figsize=(14.5, 9.8), width_ratios=[2.0, 1, 1, 1], layout='constrained')
    chosen = []
    for i, name in enumerate(names):
        row = frame[frame['name'].eq(name)].iloc[0]
        index = int(frame.index[frame['name'].eq(name)][0])
        ax = axes[i, 0]
        x = np.arange(row.length)*10
        ratio = data['score'][index, :row.length] / tau
        ax.plot(x, ratio, color=COLORS['knn'], label='kNN / threshold', lw=1.5)
        ax.axhline(1, color='#777777', ls='--', lw=1)
        if row.v82_frozen >= 0:
            ax.axvline(row.v82_frozen*10, color=COLORS['v82'], label='First v8.2 alarm', lw=1.2)
        ax.scatter(row.knn_available_first*10, ratio[row.knn_available_first], color=COLORS['knn'], s=25, zorder=3)
        ax.set_xlim(0, 520)
        ax.set_ylim(0, max(1.5, np.nanmax(ratio)*1.12))
        ax.set_xlabel('Executed actions before query', fontsize=8)
        ax.set_ylabel('Distance / threshold', fontsize=8)
        outcome = 'failure' if row.failure else 'success'
        ax.set_title(f'{name}\nEventual {outcome}; {row.action_steps} actions', fontsize=9, loc='left')
        ax.grid(alpha=.12)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
        queries = [max(0, int(row.knn_available_first)-1), int(row.knn_available_first)]
        with np.load(Path(row.source_dir) / 'episode-trace.npz') as trace:
            images = [*trace['images'][queries], trace['final_image']]
        titles = [f'Before alarm\nq{queries[0]}, before action {queries[0]*10}',
                  f'First kNN alarm\nq{queries[1]}, before action {queries[1]*10}',
                  f'Episode end\nafter {row.action_steps} actions']
        for j, (title, image) in enumerate(zip(titles, images), 1):
            ax = axes[i, j]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        chosen.append(dict(name=name, queries=queries, terminal_image_key='final_image',
                           terminal_actions=int(row.action_steps), source_dir=row.source_dir,
                           exploratory_illustration=True))
    fig.suptitle('Recorded Plus observations: an alarm or distance return is not a local trap label', fontsize=13, fontweight='bold')
    finish(fig, 'current-alarm-cases')
    (ROOT / 'case-figure-index.json').write_text(json.dumps(chosen, indent=2) + '\n')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.titlepad': 10})
    archived_figure()
    geometry_figure()
    case_figure()


if __name__ == '__main__':
    main()
