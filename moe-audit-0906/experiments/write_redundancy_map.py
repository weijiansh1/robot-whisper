# -*- coding: utf-8 -*-
"""Redundancy map: distinct quantities vs the names they go by.
All correlations are READ from bundle artifacts, not retyped."""
import csv, os
ROOT='/home/jovyan/work/himoe-vla'; OUT=f'{ROOT}/moe-audit-0906/results'
red={r['feature']:r for r in csv.DictReader(open(f'{ROOT}/moe-circuit-analogy-0906/results/diagnostics/redundancy_vs_published.csv'))}
inter=list(csv.DictReader(open(f'{ROOT}/moe-circuit-analogy-0906/results/diagnostics/circuit_internal_spearman.csv')))
ikey=list(inter[0].keys())[0]
IM={r[ikey]:r for r in inter}
def rho(a,b):
    try: return abs(float(IM[a][b]))
    except Exception: return None
def best(f):
    r=red.get(f); return (r['best_reference'], float(r['best_abs_spearman'])) if r else (None,None)

rows=[]
def add(cls, name, ev, src, note=''):
    rows.append(dict(equivalence_class=cls, name=name, evidence=ev, evidence_source=src, note=note))

# ---- class 1: total residual routing energy (scale) ----
C1='E1_total_residual_routing_energy'
add(C1,'conditional_energy','reference member of the class','moe-hb-front-back-0905 METRIC_NAMES','The published name. = 1 - mean_t(<a_t,s>^2), verified to 2.54e-08 in moe-state-channel-0906/results/formation/formation_summary.json')
add(C1,'state_action_alignment','exact monotone (decreasing) transform; all external metrics bit-identical','moe-hb-front-back-0905/docs/FRAME_SURVEY_REPORT_ZH.md §5; dependence ratio 1253','Circuit redundancy table: rho_state_action_alignment = -rho_conditional_energy to 7 digits for all 31 circuit features.')
add(C1,'kernel_trace','= 10 x conditional_energy; identical survival AUC in 288/288 cells','moe-audit-0906/results/recheck_survival_auc.csv vs moe-token-geometry-0906/results/effect/survival_conditioned_auc.csv','THE GEOMETRY BUNDLE EVALUATED conditional_energy UNDER THIS NAME AND NEVER NOTICED.')
for f,label in [('ground_leak_mean','exact monotone transform'),('lam_max',''),('res_chain_series',''),('res_adjacent_mean',''),
                ('vol','total conductance'),('log_spanning_trees','matrix-tree theorem'),('log_kirchhoff','Kirchhoff index'),
                ('res_mean',''),('res_end2end',''),('fiedler','algebraic connectivity'),
                ('th_mean',''),('th_min',''),('th_max',''),('th_first',''),('th_last','')]:
    br,bs=best(f)
    if br: add(C1,f,f'|Spearman| = {bs:.4f} vs {br}','moe-circuit-analogy-0906/results/diagnostics/redundancy_vs_published.csv',label)
add(C1,'commute_time','= 2 * vol * R by definition','moe-circuit-analogy-0906 §3','Removed at implementation time; no new degree of freedom.')

# ---- class 2: relative edge shape ----
C2='E2_relative_edge_shape_scale_free'
add(C2,'partial_edge_std','reference member of the class','moe-hb-front-back-0905 METRIC_NAMES','')
for f in ['norm_fiedler','gap_ratio','spectral_erank','kirchhoff_efficiency','res_cv','dd_deficit']:
    br,bs=best(f)
    if br: add(C2,f,f'|Spearman| = {bs:.4f} vs {br}','moe-circuit-analogy-0906/results/diagnostics/redundancy_vs_published.csv','')
for a,b in [('spectral_erank','kirchhoff_efficiency'),('gap_ratio','spectral_erank'),('gap_ratio','kirchhoff_efficiency'),
            ('res_cv','spectral_erank'),('norm_fiedler','gap_ratio')]:
    r=rho(a,b)
    if r is not None: add(C2,f'{a} ~ {b}',f'internal |Spearman| = {r:.4f}','moe-circuit-analogy-0906/results/diagnostics/circuit_internal_spearman.csv',
                          'gap_ratio / spectral_erank / kirchhoff_efficiency / res_cv are ONE quantity (all pairwise >= 0.992).')

# ---- class 3: arc order / 1-D token ordering ----
C3='E3_token_arc_order'
add(C3,'bandedness (distance-lag correlation)','reference member','moe-token-geometry-0906','')
add(C3,'Robinson property / Spearman(d,|i-j|)','same construct, thresholded','moe-token-geometry-0906 §2','630/632 (task x layer) cells satisfy it.')
add(C3,'pc1_index_corr / PC1 vs token index','r = 0.958-0.998','moe-token-geometry-0906 §2','')
add(C3,'fiedler_index_rho (norm_fiedler VECTOR ordering)','schur_normalised median |Spearman| 0.8909-0.9879 across 8 layers','moe-state-channel-0906/results/controller_probes/probe_summary.json',
    'Same statistic as arc order in the BACK layers only. schur_combinatorial (unnormalised) is 0.3939-0.5515 in the front layers, so the equivalence is layer-dependent, NOT 1.0000 everywhere.')
add(C3,'arc_stretch / neighbour_ratio','declared in moe-state-channel-0906 PREREG as arc-order family','moe-state-channel-0906/results/PREREG.md §A.1','')

# ---- class 4: centred configuration size ----
C4='E4_centred_configuration_size'
add(C4,'size (sqrt tr(J C J))','reference member','moe-token-geometry-0906','')
add(C4,'centred_energy = 0.9*(1-action_consensus)','algebraic identity on the raw Gram, max abs error 2.29e-08','moe-state-channel-0906/results/formation/formation_summary.json',
    'Related to E1 by size^2 = 10*conditional_energy - (1/10)*1^T C 1. Not the same quantity, but not independent either.')
add(C4,'overlap with conditional_effective_rank / partial_edge_std','|Pearson| up to 0.755 / 0.739 at L12','moe-token-geometry-0906/results/geometry/overlap_with_existing.csv',
    'The geometry bundle itself flags size as not a new quantity.')

# ---- class 5: cross-query routing change ----
C5='E5_cross_query_routing_change'
add(C5,'mobility (final step)','reference member','moe-hb-front-back-0905','')
add(C5,'mobility_step s0..s9','step-0 vs step-9 within-task correlation only 0.24-0.35 at L2, 0.80-0.88 at L12-L15','moe-flow-semantics-0906 §3',
    'FRONT-LAYER early-step mobility is genuinely a DIFFERENT quantity; back-layer steps are one quantity.')
add(C5,'conditional_query_d1 / partial_query_d1','declared same "adjacent_query" frame','moe-hb-front-back-0905/results/frame_survey/survey.json','')
add(C5,'v7 relative_freeze / mob_ratio','mobility of L12-L15 divided by the rollout own q1-q4 baseline','moe-v7-0905/docs/SURVIVAL_BASELINE_REPORT_ZH.md §5','Same quantity, rollout-relative reference frame.')

# ---- class 6: expert load concentration ----
C6='E6_expert_load_concentration'
add(C6,'expert_load_effective_rank','reference member','moe-hb-front-back-0905','')
add(C6,'load_entropy_s9','effective rank is a strict monotone transform of the entropy; same L3/low/q0.85 winner, same 370/93','moe-flow-semantics-0906 §6 + tests/test_anchors.py','Confirmed identical TP/FP; lift differs only in the 5th decimal (see consistency C03).')

# ---- singletons that survived ----
C7='S_singletons'
for n,ev,src in [('flow_settling_log_ratio','distinct denoising-step velocity profile; but collapses to 3 TP under a global threshold','moe-hb-front-back-0905/docs/FRAME_SURVEY_REPORT_ZH.md §2'),
                 ('token_differentiation','= I(token;expert); no published equivalent found','moe-flow-semantics-0906 §2'),
                 ('state_mobility_step','carries ZERO step information (state route is step-invariant); NOT redundant, EMPTY','moe-flow-semantics-0906 §7.4'),
                 ('n_near_zero / foster_residual / neg_count / neg_mass','|rho| <= 0.020, near-constant; survival AUC 0.4951-0.5002','moe-circuit-analogy-0906 §6')]:
    add(C7,n,ev,src,'')

with open(f'{OUT}/redundancy_map.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['equivalence_class','name','evidence','evidence_source','note']); w.writeheader(); w.writerows(rows)
import collections
cnt=collections.Counter(r['equivalence_class'] for r in rows)
print('redundancy_map.csv rows', len(rows))
for k,v in cnt.items(): print(f'  {k}: {v} names')
