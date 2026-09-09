import numpy as np, pandas as pd
import token_axis_sweep as S

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)

ORDER = ([f"tok{k:02d}" for k in range(11)] +
         ["act1_3", "act4_7", "act8_10", "act_all_1_10", "all11",
          "disp11_hell", "disp11_bhat", "dispAct_hell", "dispAct_bhat",
          "state_vs_actmean_hell"])
FEATORDER = [f"{o}|{q}" for o in ORDER for q in ("hell", "ent")]

store = {}
for tag in ("A", "B"):
    for d in (9, 5, 0):
        df, P = S.sweep(tag, d)
        df = df.loc[FEATORDER]
        store[(tag, d)] = (df, P)
        df.to_csv(f".tokaxis_grid_{tag}_d{d}.csv")

print("############ RAW within-group AUC  = P(feature_fail > feature_succ) ############")
print("#  <0.5 means LOWER feature value on failures; >0.5 means HIGHER on failures.\n")
for tag in ("A", "B"):
    df, P = store[(tag, 9)]
    print(f"===== CORPUS {tag}, layers 12-15, denoise 9, 8-step window =====")
    print("risk set: " + ", ".join(
        f"t={t}: n={P[t]['n']} (S{P[t]['nsucc']}/F{P[t]['nfail']}) pairs={P[t]['npairs']}"
        for t in S.TS))
    print(df.round(3).to_string())
    print()

print("############ DETECTION AUC = max(a, 1-a) ############\n")
best = {}
for tag in ("A", "B"):
    df, P = store[(tag, 9)]
    dd = np.maximum(df, 1 - df)
    print(f"===== CORPUS {tag} detection AUC (denoise 9) =====")
    print(dd.round(3).to_string())
    m = dd.values.max()
    ij = np.unravel_index(dd.values.argmax(), dd.shape)
    best[tag] = (dd.index[ij[0]], dd.columns[ij[1]], m, df.values[ij])
    print(f"  MAX: {best[tag][0]} @ t={best[tag][1]}  det={m:.4f} (raw {df.values[ij]:.4f})")
    print("  top 12 cells:")
    flat = dd.stack().sort_values(ascending=False)[:12]
    for (f, t), v in flat.items():
        print(f"    {f:28s} t={t:2d}  det={v:.4f}  raw={df.loc[f, t]:.4f}")
    print()

print("############ PERMUTATION FAMILY-WISE NULL (200 perms, labels shuffled within group) ############")
for tag in ("A", "B"):
    df, P = store[(tag, 9)]
    obs = np.maximum(df, 1 - df).values.max()
    p, null = S.permutation_family_p(P, obs, n_perm=200, two_sided=True, seed=12345)
    print(f"CORPUS {tag}: family = 42 orgs x 5 t = 210 cells (denoise 9)")
    print(f"  observed family max det-AUC = {obs:.4f}")
    print(f"  null max: mean={null.mean():.4f} p50={np.percentile(null,50):.4f} "
          f"p95={np.percentile(null,95):.4f} max={null.max():.4f}")
    print(f"  family-wise p = {p:.4f}\n")
    np.save(f".tokaxis_null_{tag}.npy", null)

print("############ DENOISE ROBUSTNESS (best orgs re-tested at denoise 0 and 5) ############\n")
CHECK = ["tok00|hell", "all11|hell", "act4_7|hell", "act_all_1_10|hell",
         "tok08|ent", "tok09|ent", "act8_10|ent", "disp11_hell|ent", "dispAct_hell|ent",
         "state_vs_actmean_hell|ent"]
for tag in ("A", "B"):
    print(f"===== CORPUS {tag}: raw AUC by denoise =====")
    tbl = {}
    for d in (0, 5, 9):
        df, _ = store[(tag, d)]
        for f in CHECK:
            tbl[(f, d)] = df.loc[f]
    out = pd.DataFrame(tbl).T
    out.index.names = ["org", "denoise"]
    print(out.round(3).to_string())
    print()

print("############ SIGN AGREEMENT A vs B (denoise 9) ############")
dfa, _ = store[("A", 9)]; dfb, _ = store[("B", 9)]
rows = []
for f in FEATORDER:
    for t in S.TS:
        a, b = dfa.loc[f, t], dfb.loc[f, t]
        rows.append(dict(org=f, t=t, A=a, B=b,
                         sgnA=np.sign(a - .5), sgnB=np.sign(b - .5),
                         disagree=(np.sign(a - .5) != np.sign(b - .5)),
                         both_strong=(abs(a - .5) > .08 and abs(b - .5) > .08)))
R = pd.DataFrame(rows)
bad = R[R.disagree & R.both_strong]
print(f"cells where both corpora deviate >0.08 from 0.5 but in OPPOSITE directions: {len(bad)}")
print(bad[["org", "t", "A", "B"]].round(3).to_string(index=False))
print()
print("cells with |A-0.5|>0.08 and |B-0.5|>0.08 and SAME sign:",
      int((R.both_strong & ~R.disagree).sum()), "/", int(R.both_strong.sum()))
R.to_csv(".tokaxis_signcheck.csv", index=False)
