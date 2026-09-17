"""Publication-style comparison figures and recorded observation windows."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alarm_trajectory_study import ROOT

COLORS = {'v82': '#b94764', 'instantaneous': '#287fae', 'displacement': '#b98531', 'endpoint': '#269379'}
LABELS = {'v82': 'v8.2', 'instantaneous': 'Instantaneous kNN', 'displacement': 'Displacement error', 'endpoint': 'Fixed endpoint error'}


def finish(fig, name):
    fig.savefig(ROOT / (name+'.png'), dpi=180, facecolor='white')
    fig.savefig(ROOT / (name+'.pdf'), facecolor='white')
    plt.close(fig)


def routing_figure():
    metrics = pd.read_csv(ROOT / 'alarm-metrics.csv')
    b = metrics[metrics.cohort.eq('B') & metrics.cutoff.eq('all')].set_index('method')
    names = list(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), layout='constrained')
    for ax, column, count, title in ((axes[0, 0], 'recall', 'tp', 'A  Historical B failure detection'),
                                    (axes[0, 1], 'fpr', 'fp', 'B  Historical B successful-episode alarms')):
        values = b.loc[names, column].to_numpy()*100
        bars = ax.barh(np.arange(4), values, color=[COLORS[n] for n in names], height=.55)
        ax.bar_label(bars, labels=[str(int(v)) for v in b.loc[names, count]], padding=5, fontsize=9)
        ax.set_yticks(np.arange(4), [LABELS[n] for n in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 103 if column == 'recall' else 4.2)
        ax.set_xlabel('Recall (%)' if column == 'recall' else 'False-alarm rate (%)')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=.12)
    ranking = pd.read_csv(ROOT / 'same-query-ranking-summary.csv')
    ax = axes[1, 0]
    for name in names[1:]:
        part = ranking[ranking.method.eq(name)]
        ax.plot(part['query'], part.auc, '-o', color=COLORS[name], label=LABELS[name], lw=1.7)
    ax.axhline(.5, color='#777777', ls=':', lw=1)
    ax.set_xticks([13, 20, 30])
    ax.set_ylim(.4, .85)
    ax.set_xlabel('Query index (task risk set changes)')
    ax.set_ylabel('Task-mean AUROC')
    ax.set_title('C  Same-task, same-query outcome ranking', loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    matched = pd.read_csv(ROOT / 'distance-matched-summary.csv')
    ax = axes[1, 1]
    for offset, name in ((-.4, 'displacement'), (.4, 'endpoint')):
        part = matched[matched.method.eq(name)]
        mean = part.task_mean_failure_higher.to_numpy()
        error = np.stack([mean-part.bootstrap_low.to_numpy(), part.bootstrap_high.to_numpy()-mean])
        ax.errorbar(part['query']+offset, mean, yerr=error, fmt='o', capsize=3, color=COLORS[name], label=LABELS[name])
    ax.axhline(.5, color='#777777', ls=':', lw=1)
    ax.set_xticks([13, 20, 30])
    ax.set_ylim(.1, .9)
    ax.set_xlabel('Query index')
    ax.set_ylabel('Task-mean fraction: failure score > matched success')
    ax.set_title('D  After matching instantaneous distance', loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Frozen success-prefix continuation experiment: 15,600 B episodes', fontsize=14, fontweight='bold')
    finish(fig, 'continuation-comparison')


def functional_figure():
    parent = pd.read_csv(ROOT / 'functional-parent-responses.csv')
    pairs = pd.read_csv(ROOT / 'functional-paired-changes.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    for ax, metric, title in ((axes[0, 0], 'routed_relative_gain', 'A  Routed response to changed observation'),
                              (axes[0, 1], 'total_relative_gain', 'B  Routed + shared response')):
        for parent_id, part in parent[parent.scope.eq('back_action')].groupby('parent_id'):
            part = part.set_index('stage').loc[['before', 'anchor']]
            failure = not bool(part.iloc[0].success)
            color = COLORS['v82'] if failure else COLORS['instantaneous']
            offset = -.04 if failure else .04
            ax.plot(np.arange(2)+offset, part[metric], '-o', color=color, alpha=.6, lw=1.1, ms=4)
        for failure, label in ((True, 'Eventual failure'), (False, 'Eventual success')):
            ax.plot([], [], '-o', color=COLORS['v82'] if failure else COLORS['instantaneous'], label=label, ms=4)
        ax.set_xticks([0, 1], ['Three queries before', 'Alarm / matched anchor'])
        ax.set_xlim(-.18, 1.18)
        ax.set_ylabel('Normalized finite observation-response gain')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.12)
    ax = axes[1, 0]
    part = pairs[pairs.scope.eq('back_action') & pairs.measurement.eq('total_relative_gain')]
    colors = [COLORS['endpoint'] if m else '#93979d' for m in part.phase_matched]
    ax.bar(part.pair_id, part.difference_of_changes, color=colors, width=.65)
    ax.axhline(0, color='#777777', lw=1)
    ax.set_xticks(np.arange(8), ['T00', 'T01', 'T02', 'T03', 'T04', 'T05', 'T07', 'T09'])
    ax.set_ylabel('(Anchor - before) failure minus success')
    ax.set_title('C  No consistent paired gain reduction', loc='left', fontsize=11, fontweight='bold')
    ax.set_xlabel('Green: same predicate/grasp signature; grey: mismatch', fontsize=8)
    energy = pd.read_csv(ROOT / 'functional-energy.csv')
    energy = energy[energy.stage.eq('anchor') & energy.field.eq('router')]
    ax = axes[1, 1]
    scopes = ['front_state', 'back_state', 'front_action', 'back_action']
    values = energy.groupby('scope')[['observation_fraction', 'noise_fraction', 'interaction_fraction']].mean().loc[scopes]
    bottom = np.zeros(4)
    for key, color, label in [('observation_fraction', '#287fae', 'Observation'), ('noise_fraction', '#b98531', 'Noise'),
                              ('interaction_fraction', '#9c719b', 'Interaction')]:
        value = values[key].to_numpy()*100
        ax.bar(np.arange(4), value, bottom=bottom, color=color, label=label, width=.65)
        bottom += value
    ax.set_xticks(np.arange(4), ['Front\nstate', 'Back\nstate', 'Front\naction', 'Back\naction'])
    ax.set_ylim(0, 119)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Mean finite-contrast energy share (%)')
    ax.set_title('D  Routing responses at the selected anchors', loc='left', fontsize=11, fontweight='bold')
    ax.legend(ncols=3, fontsize=7, frameon=False, loc='upper center')
    fig.suptitle('Actual-alarm functional probes: 16 parents, 32 anchors, 162 forwards', fontsize=13, fontweight='bold')
    finish(fig, 'alarm-functional-response')


def observations():
    selection = json.loads((ROOT / 'functional-selection.json').read_text())
    parents = [p for p in selection['parents'] if p['task'] in (0, 1, 3, 9)]
    fig, axes = plt.subplots(len(parents), 3, figsize=(8.5, 17), layout='constrained')
    index = []
    for i, parent in enumerate(parents):
        with np.load(Path(parent['source_dir']) / 'episode-trace.npz') as trace:
            q = parent['query']
            selected = [q-3, q, min(q+3, len(trace['images']))]
            for j, t in enumerate(selected):
                terminal = t == len(trace['images'])
                image = trace['final_image'] if terminal else trace['images'][t]
                action = int(trace['executed_lengths'].sum()) if terminal else int(trace['replan_start_steps'][t])
                axes[i, j].imshow(image)
                axes[i, j].set_axis_off()
                axes[i, j].set_title('W%02d | T%02d | %s, action %d' % (i, parent['task'], 'terminal' if terminal else 'q%d' % t, action), fontsize=8)
        index.append(dict(window_id=i, name=parent['name'], source_dir=parent['source_dir'], queries=selected,
                           shown_without_scores_or_outcome_labels=True))
    fig.suptitle('Recorded observation windows; identifiers resolved in the companion index', fontsize=11, fontweight='bold')
    finish(fig, 'external-observation-windows')
    (ROOT / 'observation-figure-index.json').write_text(json.dumps(index, indent=2)+'\n')


def fresh_figure():
    metrics = pd.read_csv(ROOT / 'alarm-metrics.csv')
    subset = metrics[metrics.cohort.str.startswith('fresh_') & metrics.cutoff.eq('all')]
    if subset.empty:
        return
    names = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    for ax, cohort in zip(axes, ['fresh_plus', 'fresh_pro']):
        part = subset[subset.cohort.eq(cohort)].set_index('method').loc[names]
        x = np.arange(4)
        for offset, column, color, label in ((-.18, 'tp', COLORS['v82'], 'Failures detected'),
                                             (.18, 'fp', COLORS['instantaneous'], 'Successful episodes flagged')):
            bars = ax.bar(x+offset, part[column], width=.34, color=color, label=label)
            ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_xticks(x, ['v8.2', 'Instant.\nkNN', 'Displace.\nerror', 'Endpoint\nerror'])
        ax.set_ylim(0, 13)
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_ylabel('Parent episodes')
        ax.set_title('%s: %d failures, %d successes' % (cohort.replace('fresh_', '').title(), part.iloc[0].failures, part.iloc[0].successes), fontsize=11)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Fresh frozen evaluation: ten preselected tasks per benchmark', fontsize=13, fontweight='bold')
    finish(fig, 'fresh-frozen-comparison')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.titlepad': 10})
    routing_figure()
    functional_figure()
    observations()
    fresh_figure()


if __name__ == '__main__':
    main()
