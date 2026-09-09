# data/ 里是什么

全部从 `VLA_MUI_HUB/.../KITCHEN_SCENE8_.../right-16x32/server/routes.zarr` 派生。
索引一律是 `summaries.json` 里的 episode 顺序,0..511。

原始张量每次读约 4 分钟;这些缓存让 `scripts/make_tables.py` 三分钟跑完全部表。

---

## 距离序列(核心)

### `distB.pkl` — 8 种距离 × 2 通道,逐 chunk (1.7 MB)

```python
D = pickle.load(open("distB.pkl", "rb"))
D["H"]["st"]["hellinger"][i]   # (T_i - 1,) float32,分支 i 的逐 chunk 距离
D["y"]                          # (512,) 0=成功 1=失败
D["g"]                          # (512,) init_state_id
D["T"]                          # (512,) inference_calls
```

通道:`st` = 后缀 token 0;`ac` = 后缀 token 1–10。
距离:`hellinger` `wjaccard` `tv` `js` `cos` `l2` `top4_jac` `top4_mass`。

均在 HB 层 12–15(zarr 的 `4:8`)、去噪第 9 轮上,逐 (层, token) 位点算完再对位点取平均。

- `wjaccard` = Ruzicka 距离 `1 − Σmin(p,q)/Σmax(p,q)`,**不是 f-散度**
- `top4_jac` = 硬 top-4 集合的 Jaccard 距离 —— **受「路由 ID 没用」禁令约束**,
  仅作对照保留(第 4/5 名精确平票率在本库为 27.7%)
- `top4_mass` = `½[Σ p·1(∉ top4(q)) + Σ q·1(∉ top4(p))]`,用**概率质量**不用身份,
  故不受该禁令约束

### `tokB.pkl` — 逐 token,不对 token 取平均 (2.0 MB)

```python
D["hel"][i]   # (T_i - 1, 11) —— 11 个后缀 token 各自的 Hellinger(已对 4 层取平均)
D["l2"][i]    # 同上,L2
```

列 0 = state token,列 1–10 = action token。`token_axis.csv` 与 `token_correlations.csv` 的来源。

### `serB.pkl` — 只有 lag-1 Hellinger 的精简版 (0.2 MB)

`{"st": [...], "ac": [...], "y", "g", "T"}`。`distB.pkl` 的子集,早期脚本用。

---

## 派生特征

### `causalF.pkl` — 48 个离线统计量的因果化版本 (11 MB,本目录最大)

```python
D["F"]["st"]["tail_ratio"]   # (512, 52) —— [i, t] = 只用 c[0:t] 算出的该统计量
```

26 个统计量 × 2 通道,t 从 16 到 34。**「逐步放出前缀」实验的产物**,支撑
`CORRECTIONS.md` §H(离线名次不能预测在线名次)。

### `wtB.pkl` — 同样 26 个统计量的**离线**版(用整条轨迹算)

`{"FT": {"st·tail_ratio": (512,), ...}, "y", "grp", "T"}`。与 `causalF.pkl` 配对使用,
差值即「离线偷到了多少」。

### `amp30.pkl` — 610 个放大器格子在第 30 步的取值

`{"C": {名称: (512,)}, "res": [(判对率, 名称, 判定, q)]}`。自参照 z 分、CUSUM、
差分窗、EWMA、窗口前非线性跨通道 —— 无一胜过朴素窗均值。

### `online.pkl` — 早期在线回放的分数与报警时刻

`{"S": [...], "y", "grp", "TH", "FIRE", "T"}`,`S[i][t]` 为
`r(t) = mean(c[t-8:t]) / mean(c[:8])`。**其判定窗口为 t∈[16,52),含 `CORRECTIONS.md` §A
所述泄漏**,仅供追溯,不要用来出数。

---

## 标签

### `labels_B_derived.csv` — 语料 B 的失败分类(216 行,只有失败)

由冻结物理判据移植而来。列:`episode_id` `init_state_id` `seed` `success` `n_q`
`stag` `loop` `q_late_static_window_fraction` `q_terminal_static_window_steps`
`loop_return_count` `loop_return_fraction` `cls` `cls_strict` `cls_strict_is_stag`。

`cls` ∈ {stagnation, loop, other} = **128 / 41 / 47**。

**已知的移植偏差**:loop 判据本身是 query 级,精确移植;停滞判据原本是 20 个 dense 步的
滚动窗,语料 B 只有 query 级 `sim_state`,故用了 query 代理。该代理在语料 A 上
召回 1.00 / 精确 0.82(60 触发 vs 49 真,11 假阳 0 假阴)。**因此 B 的停滞类偏宽约 20%。**

### `labels_A_rederived.csv` — 语料 A 的判据重推,用于验证移植

重推**逐条复现了语料 A 随包发布的标签**:停滞 49/49、loop 163/163。移植代码由此确信。
