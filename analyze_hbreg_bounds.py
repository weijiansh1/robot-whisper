#!/usr/bin/env python3
"""
(A) Reachable Top-K-consistent counterexamples to "L_HB = 1 is the lower bound".
(B) Corrected naming + closed form for the equal-mass K-hot regime.
(E) Is exactly-flat routing a per-sequence stationary point?  (No.)

Every score matrix here is a strictly valid softmax (all entries > 0) whose top-K
sets are exactly the K largest entries, so all of these are REACHABLE router states,
not counterfactual (c, S) pairs.
"""
import numpy as np

N, K, U = 32, 4, 11
EPS = 1e-9


def L_HB(s, check=True):
    """s: (U,N) rows sum to 1.  Returns L_HB with c from the true top-K of s."""
    idx = np.argsort(-s, axis=1)[:, :K]
    if check:
        srt = np.sort(s, axis=1)[:, ::-1]
        assert (srt[:, K - 1] > srt[:, K]).all(), "top-K not strict (tie)"
        assert (s > 0).all() and np.allclose(s.sum(1), 1), "not a valid softmax"
    c = np.bincount(idx.ravel(), minlength=N).astype(float)
    S = s.sum(0)
    return (N / (K * U ** 2)) * (c * S).sum(), c, S


def report(name, s, expect=None):
    L, c, S = L_HB(s)
    used = (c > 0).sum()
    print(f"  {name:<46} L_HB = {L:.6f}   experts used {used:>2}/{N}   "
          f"Var(c) = {c.var():6.3f}", end="")
    if expect is not None:
        print(f"   (closed form {expect:.6f}, diff {abs(L-expect):.2e})")
    else:
        print()
    return L


print("=" * 100)
print("(A) REACHABLE configurations, strict top-4, all probabilities > 0")
print("=" * 100)

# --- flat reference
s = np.full((U, N), 1.0 / N)
s[:, 0] += 1e-7 * np.arange(1, U + 1)[:, None].ravel()[0]   # break the tie minimally
s = np.full((U, N), (1 - 1e-6) / N)
s[:, :K] += 1e-6 / K
s /= s.sum(1, keepdims=True)
report("near-flat, all tokens share one top-4", s, 1.0)

# --- the construction from the review: A for 10 tokens, B for token 11
for delta in [1e-3, 1e-5, 1e-7]:
    s = np.empty((U, N))
    a = 1 / N + delta
    b = 1 / N - delta * K / (N - K)
    s[:U - 1] = b
    s[:U - 1, 0:K] = a
    # token 11: ~all mass uniformly on B = experts 4..7
    tiny = delta ** 2 / (N - K)
    s[U - 1] = tiny
    s[U - 1, K:2 * K] = (1 - (N - K) * tiny) / K
    s /= s.sum(1, keepdims=True)
    report(f"review's A/B split, delta={delta:g}", s, 118 / 121)

# --- the infimum construction: 11 private experts + 3 shared dead fillers
for eps in [1e-3, 1e-5, 1e-7]:
    s = np.empty((U, N))
    dead = [N - 3, N - 2, N - 1]
    for u in range(U):
        s[u] = eps ** 2
        s[u, dead] = eps                      # always ranks 2-4, carries ~no mass
        s[u, u] = 1.0                         # private expert, ranks 1
    s /= s.sum(1, keepdims=True)
    report(f"11 private + 3 dead fillers, eps={eps:g}", s, N / (K * U))

print(f"\n  closed forms:  flat = 1.000000   review A/B = 118/121 = {118/121:.6f}"
      f"   infimum construction = N/(KU) = 8/11 = {N/(K*U):.6f}")

# --- numerical search: can anything reachable go below N/(KU)?
print("\n  numerical minimisation of L_HB over the 11x32 softmax simplex")
print("  (Adam on free logits, f_i stop-gradient exactly as in training):")
rng = np.random.default_rng(0)
best = np.inf
for restart in range(24):
    z = rng.normal(0, 1.0 + 3 * rng.random(), size=(U, N))
    m = np.zeros_like(z); v = np.zeros_like(z)
    for it in range(4000):
        s = np.exp(z - z.max(1, keepdims=True)); s /= s.sum(1, keepdims=True)
        idx = np.argsort(-s, axis=1)[:, :K]
        c = np.bincount(idx.ravel(), minlength=N).astype(float)
        f = (N / (K * U)) * c
        ell = s @ f
        g = (s * (f[None, :] - ell[:, None])) / U          # dL/dz, f stop-grad
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        z -= 0.05 * m / (np.sqrt(v) + 1e-12)
    s = np.exp(z - z.max(1, keepdims=True)); s /= s.sum(1, keepdims=True)
    L, c, S = L_HB(s, check=False)
    best = min(best, L)
print(f"    best over 24 restarts: L_HB = {best:.6f}   "
      f"(N/(KU) = {N/(K*U):.6f})")

print("\n" + "=" * 100)
print("(B) equal-mass K-hot regime  [NOT 'maximally sharp' -- corrected]")
print("=" * 100)
print("  valid only when s_{i,u} = r_{i,u}/K, i.e. each token splits its mass")
print("  EQUALLY over its K selected experts.  Then L-1 = (N^2/(K^2 U^2)) Var(c),")
print("  and with the balanced integer load Var_min = r(N-r)/N^2:")
print(f"\n    L_min^Khot = 1 + r(N-r)/(K^2 U^2)\n")
print(f"  {'U':>7} {'M=KU':>7} {'q':>4} {'r':>4} {'Var_min':>9} "
      f"{'L_min^Khot':>11} {'integer decisiveness tax':>25}")
for u in [11, 22, 55, 110, 550, 1100, 11000]:
    M = K * u
    q, r = divmod(M, N)
    vmin = r * (N - r) / N ** 2
    Lk = 1 + r * (N - r) / (K ** 2 * u ** 2)
    print(f"  {u:>7} {M:>7} {q:>4} {r:>4} {vmin:9.6f} {Lk:11.7f} "
          f"{100*(Lk-1):>24.4f}%")
print("\n  NOTE: a router with ~1 on its top-1 and ~0 on the other three selected")
print("  experts is SHARPER yet violates s = r/K, so 1.1240 is not a floor over all")
print("  sharp routers.  The infimum over ALL reachable states is N/(KU) = 0.7273.")

print("\n" + "=" * 100)
print("(E) is exactly-flat routing a per-sequence stationary point?")
print("=" * 100)
s = np.full((U, N), 1.0 / N)
# exact ties -> top-K decided by tie-break; take the first K indices
c = np.zeros(N); c[:K] = U
f = (N / (K * U)) * c
ell = s @ f
g = (s * (f[None, :] - ell[:, None])) / U
print(f"  at s_i = 1/N exactly:  ell_u = sum_i f_i s_{{i,u}} = {ell[0]:.6f}")
print(f"  f_i takes values {sorted(set(np.round(f,4)))} -- and f_i = 1 needs "
      f"c_i = KU/N = {K*U/N} (non-integer)")
print(f"  => dL/dz has max |g| = {np.abs(g).max():.3e}, ||g||_F = "
      f"{np.linalg.norm(g):.3e}  (NOT zero)")
print("  so exact equal logits is NOT a smooth per-sequence stationary point;")
print("  near-flat routing is a loss-VALUE refuge, and whether it is an attractor")
print("  of the dynamics is exactly what the lambda x scope ablation must decide.")
