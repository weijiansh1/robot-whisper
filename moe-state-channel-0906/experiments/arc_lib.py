#!/usr/bin/env python3
"""Geometry of the ten action tokens relative to the state token, per flow step.

Notation, for one (query, layer, denoising step):

    a_1 .. a_10   unit-norm square roots of the ten action tokens' router
                  probability vectors, in R^32
    s             the same for the STATE token (router token 0)
    G_ij          = <a_i, a_j>, the Bhattacharyya coefficient, G_ii = 1
    c_t           = <a_t, s>, the state-alignment coordinate of token t

Three 10x10 matrices are in play and they are not the same object:

    G                    raw Gram.  Dominated by the common component; every
                         entry is close to 1.
    cond = G - c c^T     the published Schur complement.  Removes the state
                         DIRECTION from every token and keeps the centroid.
                         This is what every detector in the project reads.
    Gc   = H G H         the centred Gram, H = I - 11^T/10.  Removes the
                         CENTROID and keeps the state direction.  This is the
                         configuration of the ten tokens as a shape, and it is
                         the matrix whose leading eigenvector is the "arc".

Exact identities used below (verified against raw Zarr in
``tests/test_identities.py``):

    conditional_energy          = 1 - mean_t(c_t^2)
    trace(Gc)/10                = 0.9 * (1 - action_consensus)
    along-state centred energy  = var_t(c_t)
                                = 1 - conditional_energy - state_action_alignment^2

so the published ``conditional_energy`` is a function of the state-alignment
vector alone, and the centred configuration size is a function of the action
consensus alone.  Total centred energy splits exactly as

    trace(Gc)/10  =  var_t(c_t)               (along the state direction)
                  +  trace(H cond H)/10       (orthogonal to it)

and the second term is precisely what the Schur complement keeps.
"""

from __future__ import annotations

import numpy as np


N_TOKENS = 10
EPSILON = 1e-12
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")

ARC_NAMES = (
    "centred_energy",        # trace(Gc)/10                                [scale]
    "along_state_energy",    # var_t(c_t), the component the Schur complement discards
    "state_align_mean",      # mean_t c_t   (duplicate of state_action_alignment, kept as a check)
    "pc1_share",             # lambda_1 / trace(Gc), one-dimensionality of the shape
    "pc1_index_corr",        # |Pearson(v1, token index)|, "is the shape an ordered arc"
    "pc1_state_cos",         # |<w1, s>| = |v1 . c| / sqrt(lambda_1), arc axis vs state axis
    "fiedler_index_rho",     # |Spearman(Fiedler vector of L_norm(cond+), token index)|
    "arc_stretch",           # d2(token1, token10) / mean_{i<j} d2   (1 = clique, ~4.9 = uniform arc)
    "neighbour_ratio",       # mean_i d2(i, i+1) / mean_{i<j} d2      (1 = clique, ~0.06 = uniform arc)
    "state_index_corr",      # Pearson(c, token index), signed
    "erank_centred",         # entropy effective rank of eig(Gc), over 9
    "norm_fiedler",          # lambda_2 of L_norm(cond+); published circuit quantity
)

_IU = np.triu_indices(N_TOKENS, 1)
_INDEX = np.arange(N_TOKENS, dtype=np.float64)
_INDEX_C = _INDEX - _INDEX.mean()
_INDEX_N = float(np.sqrt((_INDEX_C ** 2).sum()))
_H = np.eye(N_TOKENS) - np.full((N_TOKENS, N_TOKENS), 1.0 / N_TOKENS)


def normalize(raw: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(raw, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), EPSILON)


def _abs_corr_with_index(vectors: np.ndarray, signed: bool = False) -> np.ndarray:
    """|Pearson| between each length-10 row and the token index 0..9."""
    centred = vectors - vectors.mean(axis=-1, keepdims=True)
    norm = np.sqrt((centred ** 2).sum(axis=-1))
    value = (centred * _INDEX_C).sum(axis=-1) / np.maximum(norm * _INDEX_N, EPSILON)
    return value if signed else np.abs(value)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.broadcast_to(np.arange(values.shape[-1]), order.shape), axis=-1
    )
    return ranks.astype(np.float64)


def _abs_spearman_with_index(vectors: np.ndarray) -> np.ndarray:
    """|Spearman| between each length-10 row and the token index, no ties assumed."""
    ranks = _rank(vectors)
    squared = ((ranks - _INDEX) ** 2).sum(axis=-1)
    n = float(N_TOKENS)
    return np.abs(1.0 - 6.0 * squared / (n * (n * n - 1.0)))


def _entropy_effective_rank(mass: np.ndarray, maximum: int) -> np.ndarray:
    mass = np.maximum(mass, 0.0)
    share = mass / np.maximum(mass.sum(axis=-1, keepdims=True), EPSILON)
    entropy = -(share * np.log(np.maximum(share, EPSILON))).sum(axis=-1)
    return np.exp(entropy) / float(maximum)


def arc_features(root: np.ndarray) -> np.ndarray:
    """root: [..., 11, 32] square-root router probabilities -> [..., 12] features.

    Everything is computed in float64; the leading axes are flattened internally
    so the batched LAPACK calls see one contiguous stack of 10x10 matrices.
    """
    root = np.asarray(root, dtype=np.float64)
    lead = root.shape[:-2]
    flat = root.reshape(-1, root.shape[-2], root.shape[-1])
    action = flat[:, 1:, :]                          # (N, 10, 32)
    state = flat[:, 0, :]                            # (N, 32)

    gram = action @ np.swapaxes(action, -1, -2)      # (N, 10, 10)
    c = np.einsum("ne,nte->nt", state, action, optimize=True)

    # -- centred configuration -------------------------------------------------
    centred_gram = _H @ gram @ _H
    trace_centred = np.einsum("nii->n", centred_gram)
    lam, vec = np.linalg.eigh(centred_gram)
    lam = np.clip(lam, 0.0, None)
    lam1 = lam[:, -1]
    v1 = vec[:, :, -1]

    centred_energy = trace_centred / N_TOKENS
    pc1_share = lam1 / np.maximum(trace_centred, EPSILON)
    pc1_index_corr = _abs_corr_with_index(v1)
    c_centred = c - c.mean(axis=-1, keepdims=True)
    pc1_state_cos = np.abs((v1 * c_centred).sum(axis=-1)) / np.maximum(
        np.sqrt(lam1), EPSILON
    )
    # centring kills exactly one eigenvalue, so the shape lives in 9 dimensions
    erank_centred = _entropy_effective_rank(lam[:, 1:], N_TOKENS - 1)

    along_state_energy = c.var(axis=-1)
    state_align_mean = c.mean(axis=-1)
    state_index_corr = _abs_corr_with_index(c, signed=True)

    # -- arc shape from squared distances, d2_ij = 2 - 2 G_ij ------------------
    upper_mean = gram[:, _IU[0], _IU[1]].mean(axis=-1)
    denominator = np.maximum(1.0 - upper_mean, EPSILON)
    arc_stretch = (1.0 - gram[:, 0, N_TOKENS - 1]) / denominator
    neighbour = 1.0 - gram[:, np.arange(N_TOKENS - 1), np.arange(1, N_TOKENS)].mean(
        axis=-1
    )
    neighbour_ratio = neighbour / denominator

    # -- normalised Laplacian of the published Schur-complement graph ----------
    conditional = gram - c[:, :, None] * c[:, None, :]
    weight = np.clip(conditional, 0.0, None)
    idx = np.arange(N_TOKENS)
    weight[:, idx, idx] = 0.0
    degree = weight.sum(axis=-1)
    inv_sqrt = np.where(degree > 0.0, 1.0 / np.sqrt(np.where(degree > 0.0, degree, 1.0)), 0.0)
    lnorm = np.eye(N_TOKENS) - weight * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
    lam_n, vec_n = np.linalg.eigh(lnorm)
    norm_fiedler = lam_n[:, 1]
    fiedler_index_rho = _abs_spearman_with_index(vec_n[:, :, 1])

    stacked = np.stack(
        (
            centred_energy,
            along_state_energy,
            state_align_mean,
            pc1_share,
            pc1_index_corr,
            pc1_state_cos,
            fiedler_index_rho,
            arc_stretch,
            neighbour_ratio,
            state_index_corr,
            erank_centred,
            norm_fiedler,
        ),
        axis=-1,
    )
    return stacked.reshape(*lead, len(ARC_NAMES)).astype(np.float32, copy=False)


def ols_slope(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """OLS slope of `values` against 0..n-1 along `axis`.  NaN-free inputs only."""
    values = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    n = values.shape[-1]
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    slope = (values * xc).sum(axis=-1) / (xc ** 2).sum()
    return slope.astype(np.float32, copy=False)
