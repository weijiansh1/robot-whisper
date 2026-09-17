"""Standalone scientific figures for the historical MoE expansion."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')
COLORS = {'failure': '#bc3d5a', 'success': '#218679', 'm': '#3458a1', 'a': '#d17325'}
LABELS = {'freeze_control': 'Freeze only', 'frozen_active': 'Frozen + internally active',
          'both_cooling': 'Both cooling', 'both_active': 'Both active',
          'acceleration_control': 'Internal acceleration only', 'v82_archived': 'Frozen v8.2'}


def finish(figure, name):
    figure.savefig(ROOT / f'{name}.png', dpi=180, facecolor='white')
    figure.savefig(ROOT / f'{name}.pdf', facecolor='white')
    plt.close(figure)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlepad': 12,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    events = pd.read_csv(ROOT / 'event-metrics.csv')
    b = events[events.cohort.eq('external_8b') & events.factor.eq(1.2)].groupby('method').sum(numeric_only=True)
    associations = pd.read_csv(ROOT / 'association-summary.csv')
    associations = associations[associations.cohort.eq('external_8b') &
                                associations.window.eq('early_q9_13') & associations.matching.eq('task_query')].set_index('method')
    matched = pd.read_csv(ROOT / 'completion-matched-summary.csv')
    matched = matched[matched.cohort.eq('external_8b')].set_index('method')
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), layout='constrained')
    names = ['freeze_control', 'frozen_active', 'v82_archived']
    x = np.arange(len(names))
    recall = b.loc[names, 'detected_failures'] / b.loc[names, 'failures'] * 100
    fpr = b.loc[names, 'flagged_successes'] / b.loc[names, 'successes'] * 100
    for ax, values, color, title in ((axes[0, 0], recall, COLORS['failure'], 'A  Failure coverage at episode end'),
                                    (axes[0, 1], fpr, COLORS['success'], 'B  Successful episodes flagged')):
        bars = ax.bar(x, values, width=.58, color=color)
        ax.set_xticks(x, ['Freeze only', 'Frozen + active', 'v8.2'])
        ax.set_ylabel('Episodes (%)')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    names = ['freeze_control', 'frozen_active', 'both_active', 'both_cooling', 'acceleration_control']
    y = np.arange(len(names))
    a = associations.loc[names]
    ax = axes[1, 0]
    ax.errorbar(a.auc, y, xerr=np.stack([a.auc-a.ci_low, a.ci_high-a.auc]),
                fmt='o', color=COLORS['m'], capsize=3)
    ax.axvline(.5, ls='--', color='#747474', lw=1)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(.25, .85)
    ax.set_xlabel('AUROC; task bootstrap 95% interval')
    ax.set_title('C  Same task + query, q9 to q13', loc='left', fontweight='bold', fontsize=12)
    names = ['freeze_control', 'frozen_active', 'both_active', 'acceleration_control', 'v82_archived']
    y = np.arange(len(names))
    m = matched.loc[names]
    ax = axes[1, 1]
    ax.errorbar(m.task_mean_hit_difference * 100, y,
                xerr=np.stack([m.task_mean_hit_difference-m.ci_low, m.ci_high-m.task_mean_hit_difference]) * 100,
                fmt='o', color=COLORS['a'], capsize=3)
    ax.axvline(0, ls='--', color='#747474', lw=1)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel('Failure minus success alarm rate (percentage points)')
    ax.set_title('D  Matched task + initial state + duration', loc='left', fontweight='bold', fontsize=12)
    fig.suptitle('Historical B: 15,600 trajectories, 564 failures, 15,036 successes', fontsize=15, fontweight='bold')
    finish(fig, 'historical-comparison')

    index = pd.read_csv(ROOT / 'episode-events.csv', low_memory=False)
    coordinates = np.load(ROOT / 'two-axis-coordinates.npz')['coordinates']
    scores = np.load(ROOT / 'supported-motif-scores.npz')
    names = scores['names'].astype(str).tolist()
    values = scores['scores']
    external = index[index.cohort.eq('external_8b')]
    s05 = external[external.task.str.contains('next_to_the_plate_and_place_it')]
    selectors = [
        (s05[~s05.failure & s05.first_frozen_active.ge(0)], 'frozen_active', 'Success with frozen + active routing'),
        (s05[s05.failure & s05.first_frozen_active.ge(0)], 'frozen_active', 'Failure with frozen + active routing'),
        (external[external.failure & external.first_both_active.ge(0) & external.first_frozen_active.lt(0)],
         'both_active', 'Failure without the frozen + active motif'),
        (external[~external.failure & external.first_both_cooling.ge(0)], 'both_cooling', 'Success with both coordinates cooling'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.8), layout='constrained')
    cases = []
    task_map = pd.read_csv(ROOT / 'inputs/safe&vlaconf/moe_trainfree/results/routing_dynamics_20260908/plot_task_map.csv')
    task_map = dict(zip(task_map.task, task_map.plot_id))
    for ax, (candidates, method, title) in zip(axes.flat, selectors):
        selected = candidates.sort_values(['first_' + method, 'identity']).iloc[0]
        row, length = selected.name, int(selected.length)
        q = np.arange(length)
        xx = q * 10
        ax.plot(xx, coordinates[row, :length, 0], color=COLORS['m'], label='Across-query change, m', lw=1.8)
        ax.plot(xx, coordinates[row, :length, 1], color=COLORS['a'], label='Within-query acceleration, a', lw=1.8)
        active = values[row, :length, names.index(method)] >= np.log(1.2)
        for start in xx[active]:
            ax.axvspan(start, min(start + 10, int(selected.action_steps)), color='#f1df8f', alpha=.55, lw=0)
        ax.axhline(np.log(1.2), color='#777777', ls=':', lw=1)
        ax.axhline(-np.log(1.2), color='#777777', ls=':', lw=1)
        ax.axhline(0, color='#aaaaaa', lw=.6)
        ax.set_xlim(0, int(selected.action_steps))
        task_id = task_map.get(selected.task, selected.suite)
        ax.set_title(f'{title}\n{task_id}, episode {selected.episode}', loc='left', fontsize=11)
        ax.set_ylabel('Log ratio to q1..q4 reference')
        ax.set_xlabel('Executed actions before query')
        ax.legend(fontsize=8, loc='best', frameon=False)
        cases.append(dict(identity=selected.identity, row=int(row), motif=method,
                          selection='earliest alarm then lexical identity within declared subset'))
    fig.suptitle('Observed routing trajectories: highlighted spans meet the fixed motif rule', fontsize=14)
    finish(fig, 'trajectory-counterexamples')
    (ROOT / 'figure-case-index.json').write_text(json.dumps(cases, indent=2) + '\n')


if __name__ == '__main__':
    main()
