"""Standalone figures for temporal-order evidence and physical interpretation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moe_offline_sequences import ROOT, archive

LABELS = {'freeze': 'Freeze', 'activity': 'Internal activity',
          'simultaneous': 'Simultaneous combination',
          'activity_then_freeze': 'Activity then freeze', 'reverse': 'Reverse order',
          'order_excess': 'Order excess', 'direction_gap': 'Forward minus reverse'}


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    order = ['freeze', 'activity', 'simultaneous', 'activity_then_freeze', 'order_excess']
    table = pd.read_csv(ROOT / 'same-initial-query-summary.csv').query('query == 13')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout='constrained')
    for ax, cohort, label in zip(axes, ['development_main', 'external_8b', 'legacy_16x32'],
                                 ['Historical A', 'Historical B', 'Legacy (checkpoint unverified)']):
        part = table[table.cohort.eq(cohort)].set_index('method').loc[order]
        y = np.arange(len(order))
        for i, row in enumerate(part.itertuples()):
            color = '#bd4d5f' if row.Index == 'activity_then_freeze' else '#2679b5'
            ax.errorbar(row.mean, i, xerr=[[row.mean-row.low], [row.high-row.mean]], fmt='o',
                        color=color, capsize=3, markersize=6)
        ax.axvline(.5, color='#777777', linestyle=':', linewidth=1)
        ax.set(xlim=(0, 1), yticks=y, yticklabels=[LABELS[k] for k in order],
               xlabel='Matched failure/success rank (0.5 = tie)',
               title=label+f'\n{int(part.iloc[0].tasks)} tasks, q13')
        ax.grid(axis='x', alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle('Same task, initial state and query; descriptive task-bootstrap intervals', fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT / ('offline-order-comparison.'+suffix), dpi=180)
    plt.close(fig)
    physical = pd.read_csv(ROOT / 'external-matched-summary.csv')
    branch = pd.read_csv(ROOT / 'branch-comparisons.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    part = physical.query('contrast == "static_vs_progress_proxy"').set_index('method').loc[order]
    for i, row in enumerate(part.itertuples()):
        axes[0].errorbar(row.mean, i, xerr=[[row.mean-row.low], [row.high-row.mean]], fmt='o',
                         color='#228c72', capsize=3)
    axes[0].set(yticks=np.arange(len(order)), yticklabels=[LABELS[k] for k in order], xlim=(0, 1),
                xlabel='Static/return vs progress-proxy rank',
                title='External alignment\n10 tasks, 184 matched windows, 417 pair occurrences')
    axes[0].invert_yaxis()
    axes[0].axvline(.5, color='#777777', linestyle=':', linewidth=1)
    arms = [('p3d-', 'mobility_balance', 'P3d combined'), ('p3g-', 'joint', 'P3g joint'),
            ('p3g-', 'route_only', 'P3g route only'), ('p3g-', 'output_only', 'P3g output only')]
    for i, (prefix, arm, label) in enumerate(arms):
        selected = branch[branch.run.str.startswith(prefix) & branch.arm.eq(arm) &
                          branch.method.eq('activity_then_freeze')]
        axes[1].scatter(selected.mean_delta_vs_native, np.full(len(selected), i)+np.linspace(-.1, .1, len(selected)),
                        color='#bd4d5f', s=32)
    axes[1].axvline(0, color='#777777', linestyle=':', linewidth=1)
    axes[1].set(yticks=np.arange(len(arms)), yticklabels=[r[2] for r in arms],
                xlabel='Mean sequence-score change versus native suffix',
                title='Existing intervention branches\nSame 5 parent states; rescue remains 0/4 per arm')
    axes[1].invert_yaxis()
    fig.suptitle('Lower routing scores and less motion do not certify task recovery', fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT / ('offline-physical-controls.'+suffix), dpi=180)
    plt.close(fig)
    frame = pd.read_csv(ROOT / 'episode-index.csv', low_memory=False)
    counter = pd.read_csv(ROOT / 'successful-counterexamples.csv')
    chosen = counter[counter.cohort.eq('external_8b')].sort_values('actions_after_event', ascending=False).iloc[0]
    row = frame.index[frame.identity.eq(chosen.identity)][0]
    coordinates = archive(ROOT / 'coordinates.npz')['coordinates'][row, :int(frame.loc[row, 'length'])]
    scores = archive(ROOT / 'sequence-scores.npz')['scores'][row, :len(coordinates)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.7), sharex=True, layout='constrained')
    q = np.arange(len(coordinates))
    axes[0].plot(q*10, coordinates[:, 0], label='Cross-query mobility (m)', color='#2679b5')
    axes[0].plot(q*10, coordinates[:, 1], label='Within-query activity (a)', color='#bd4d5f')
    axes[0].set(ylabel='Log ratio to own baseline')
    axes[0].legend(loc='best')
    axes[1].plot(q*10, scores[:, 3], label='Activity then freeze', color='#bd4d5f')
    axes[1].plot(q*10, scores[:, 5], label='Order excess', color='#228c72')
    axes[1].axhline(np.log(1.2), color='#777777', linestyle=':', label='Fixed event threshold (sequence only)')
    for ax in axes:
        ax.axvline(chosen.query*10, color='#bd4d5f', linestyle='--', alpha=.6)
        ax.axvline(chosen.action_steps, color='#228c72', linestyle='--', alpha=.6)
    axes[1].set(xlabel='Actions already executed', ylabel='Score', xlim=(0, chosen.action_steps+5))
    axes[1].legend(loc='best')
    fig.suptitle(f'Successful counterexample: event at {int(chosen.query*10)} actions, success at {int(chosen.action_steps)}\n'
                 'Historical B / cream cheese and butter / initial state 33 / episode 271', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT / ('offline-success-counterexample.'+suffix), dpi=180)
    plt.close(fig)
    print('Saved three standalone figures as PNG and PDF', flush=True)


if __name__ == '__main__':
    main()
