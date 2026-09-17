"""External labels and ID-aligned dispatched-contribution diagnostics."""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/libero-runtime/samples/moe-multimechanism-matching-20260915')
SIGNALS = ('freeze', 'acceleration', 'periodicity', 'frontback_inversion', 'curvature')
HEADS = ('freeze', 'turbulence', 'inversion', 'curvature')
FEATURES = ('total_relative_gain', 'total_direction_change', 'routed_shared_update_cosine',
    'routed_shared_update_cancellation', 'retained_update_fraction', 'support_change_update_fraction',
    'expert_update_cancellation', 'expert_negative_attribution', 'shared_update_attribution',
    'largest_expert_absolute_attribution')


def norm(x):
    return np.linalg.norm(x, axis=-1)


def ratio(a, b, minimum=1e-12):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan),
                     where=np.asarray(b) > minimum)


def rotation_angle(a, b):
    return np.arccos(np.clip((np.einsum('...ij,...ij->...', a, b)-1)/2, -1, 1))


def phase(data, q):
    goals = [json.loads(g) for g in data['goals']]
    names = data['names'].tolist()
    unmet = sorted({g[1] for g, truth in zip(goals, data['predicates'][q]) if not truth})
    if unmet:
        target = min(unmet, key=lambda name: (not data['grasp'][q, names.index(name)],
                     norm(data['positions'][q, names.index(name)]-data['eef'][q]), name))
    else:
        target = 'complete'
    return json.dumps([data['predicates'][q].astype(int).tolist(),
                       [name for name, held in zip(names, data['grasp'][q]) if held], target], separators=(',', ':'))


def physical_window(data, q, horizon):
    end = min(q+horizon, len(data['predicates'])-1)
    if not 0 <= q < end:
        raise ValueError('A nonempty saved boundary window is required')
    p = data['predicates'][q:end+1]
    gain_any = int((p[1:] & ~p[0]).any(0).sum())
    gain_end = int((p[-1] & ~p[0]).sum())
    loss_end = int((~p[-1] & p[0]).sum())
    local = ('verified_progress' if gain_end and not loss_end else
             'no_observed_gain' if not gain_any and not loss_end else 'transient_or_regressing')
    eef = data['eef'][q:end+1]
    positions = data['positions'][q:end+1]
    path, net = norm(np.diff(eef, axis=0)).sum(), norm(eef[-1]-eef[0])
    motion = norm(positions-positions[:1]).max()
    rotation = rotation_angle(data['rotations'][q:q+1], data['rotations'][q:end+1]).max()
    ranges = np.diff(data['joint_ranges'], axis=-1).ravel()
    joint_motion = (np.abs(data['joint_values'][q:end+1]-data['joint_values'][q:q+1])/ranges).max() if len(ranges) else 0.
    geometry = data['goal_distance'][q:q+1]-data['goal_distance'][q:end+1]
    geometry_gain = float(np.nanmax(geometry)) if np.isfinite(geometry).any() else 0.
    source_dist = norm(positions-eef[:, None])
    approach = (source_dist[:1]-source_dist).max()
    unary = data['unary_distance'][q:q+1]-data['unary_distance'][q:end+1]
    unary_gain = float(np.nanmax(unary)) if np.isfinite(unary).any() else 0.
    old_static = motion < .003 and path < .015
    internal_motion = rotation > np.deg2rad(5) or joint_motion > .02
    if local != 'no_observed_gain':
        category = local
    elif geometry_gain > .01 or approach > .01 or unary_gain > .02:
        category = 'approach_proxy'
    elif motion < .003 and not internal_motion and path > .03 and net < .008:
        category = 'return_motion'
    elif old_static and not internal_motion:
        category = 'holding_or_static'
    else:
        category = 'other_motion_no_gain'
    return dict(query=q, horizon=horizon, end_query=end, complete_window=end-q == horizon,
        local_label=local, category=category, gained_any=gain_any, gained_endpoint=gain_end,
        lost_endpoint=loss_end, terminal_success=bool(p[-1].all()), eef_path_m=float(path),
        eef_net_m=float(net), object_motion_m=float(motion), rotation_max_deg=float(np.rad2deg(rotation)),
        joint_motion_range=float(joint_motion), geometric_gain_m=geometry_gain,
        approach_m=float(approach), unary_gain_range=unary_gain,
        translation_static_but_internal_motion=bool(old_static and internal_motion))


def geometry_distance(a, aq, b, bq):
    if not np.array_equal(a['names'], b['names']) or not np.array_equal(a['joint_names'], b['joint_names']):
        return dict(geometry_rms_m=np.inf, geometry_max_m=np.inf, rotation_max_deg=np.inf, joint_max_range=np.inf)
    d = norm(np.concatenate([a['positions'][aq]-b['positions'][bq], (a['eef'][aq]-b['eef'][bq])[None]]))
    rotation = rotation_angle(a['rotations'][aq], b['rotations'][bq]).max()
    ranges = np.diff(a['joint_ranges'], axis=-1).ravel()
    joint = (np.abs(a['joint_values'][aq]-b['joint_values'][bq])/ranges).max() if len(ranges) else 0.
    return dict(geometry_rms_m=float(np.sqrt((d*d).mean())), geometry_max_m=float(d.max()),
                rotation_max_deg=float(np.rad2deg(rotation)), joint_max_range=float(joint))


def contributions(snapshot):
    ids = np.asarray(snapshot['ids'], int)
    if ids.ndim != 3 or ids.shape[:2] != (8, 11) or ids.shape[-1] != 4:
        raise ValueError('Expected eight HB layers, eleven tokens, and top-four IDs')
    if (ids < 0).any() or (ids >= 32).any() or (np.diff(np.sort(ids, axis=-1), axis=-1) == 0).any():
        raise ValueError('Invalid or duplicate expert IDs')
    c = np.asarray(snapshot['expert_output'], float)*np.asarray(snapshot['weights'], float)[..., None]
    dense = np.zeros(ids.shape[:2]+(32, c.shape[-1]), float)
    support = np.zeros(ids.shape[:2]+(32,), bool)
    i, j = np.indices(ids.shape[:2])
    dense[i[..., None], j[..., None], ids] = c
    support[i[..., None], j[..., None], ids] = True
    return dense, support


def contribution_contrast(a, b):
    c0, s0 = contributions(a)
    c1, s1 = contributions(b)
    dc = c1-c0
    retained, entered, exited = s0 & s1, ~s0 & s1, s0 & ~s1
    dr = dc.sum(-2)
    kept = (dc*retained[..., None]).sum(-2)
    added = (c1*entered[..., None]).sum(-2)
    removed = (c0*exited[..., None]).sum(-2)
    np.testing.assert_allclose(kept+added-removed, dr, rtol=1e-10, atol=1e-10)
    for snap, dense in ((a, c0), (b, c1)):
        error = norm(dense.sum(-2)-snap['routed'])/np.maximum(norm(snap['routed']), 1e-12)
        if error.max() >= 1e-5:
            raise ValueError('Dispatched contribution reconstruction failed')
        np.testing.assert_allclose(snap['routed']+snap['shared'], snap['total'], rtol=0, atol=0)
    ds = np.asarray(b['shared'], float)-np.asarray(a['shared'], float)
    # Use the reconstructed update for exact attribution; audit stored rounding below.
    dt = dr+ds
    t0, t1 = np.asarray(a['total'], float), np.asarray(b['total'], float)
    x0, x1 = np.asarray(a['input'], float), np.asarray(b['input'], float)
    relative_x = ratio(norm(x1-x0), (norm(x0)+norm(x1))/2)
    relative_t = ratio(norm(t1-t0), (norm(t0)+norm(t1))/2)
    mass = norm(dc)
    denominator = mass.sum(-1)
    attribution = ratio((dc*dt[..., None, :]).sum(-1), (dt*dt).sum(-1)[..., None])
    shared_attribution = ratio((ds*dt).sum(-1), (dt*dt).sum(-1))
    valid = np.isfinite(shared_attribution)
    attribution_error = np.abs(attribution.sum(-1)+shared_attribution-1)
    np.testing.assert_allclose(attribution_error[valid], 0, atol=1e-10)
    result = dict(
        total_relative_gain=ratio(relative_t, relative_x, minimum=1e-4),
        total_direction_change=1-ratio((t0*t1).sum(-1), norm(t0)*norm(t1)),
        routed_shared_update_cosine=ratio((dr*ds).sum(-1), norm(dr)*norm(ds)),
        routed_shared_update_cancellation=1-ratio(norm(dt), norm(dr)+norm(ds)),
        retained_update_fraction=ratio((mass*retained).sum(-1), denominator),
        support_change_update_fraction=ratio((mass*(entered | exited)).sum(-1), denominator),
        expert_update_cancellation=1-ratio(norm(dr), denominator),
        expert_negative_attribution=np.maximum(-attribution, 0).sum(-1),
        shared_update_attribution=shared_attribution,
        largest_expert_absolute_attribution=np.abs(attribution).max(-1),
        same_support=(s0 == s1).all(-1),
        input_relative_change=relative_x,
        stored_update_rounding_relative=norm(dt-(t1-t0))/np.maximum((norm(t0)+norm(t1))/2, 1e-12))
    return result, attribution, dict(decomposition_max_absolute=float(np.abs(kept+added-removed-dr).max()),
        attribution_max_absolute=float(attribution_error[valid].max()) if valid.any() else 0.,
        stored_update_rounding_relative=float(result['stored_update_rounding_relative'].max()))
