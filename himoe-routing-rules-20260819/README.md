# himoe-routing-rules — 包里有什么

HiMoE-VLA 已发布检查点上，"MoE 路由状态能不能读出结局"这一段的完整记录：报告、脚本、每次运行的原始日志，以及一份能脱离原始语料重跑的派生数据。

2026-08-19。

```
REPORT.md          结论与全部数字，按实验顺序
scripts/           每个实验的脚本，与日志一一对应
scripts/context/   报告末尾「语境」一节引用的更早的脚本
logs/              上面每个脚本的完整 stdout（09 是脱离语料的复现）
data/              派生数据，43 MB，5 个任务各一个 npz
```

## 数据

原始语料是 76 GB（`VLA_MUI_HUB/cache/HiMoE-VLA/*/*/right-16x32`，每任务 16 个初始
场景 × 32 个流匹配噪声抽样 = 512 局），装不进包。但这些实验只碰它很小一块，
`scripts/bundle_extract.py` 把那一块抽了出来：

| 键 | 形状 | 说明 |
|---|---|---|
| `n_rows` | (512,) | 每局的控制步数，也是下面各数组的切分点 |
| `success` `scene` | (512,) | 结局、初始场景 id |
| `proprio` | (总步数, 8) | 机器人自己的 8 个数 |
| `sim_state` | (总步数, 47/79/92) | 完整仿真状态，宽度随任务 |
| `actions` | (总步数, 70) | 该控制步发出的动作块 10×7 |
| `state_token_top4` | (总步数, 8, 4) | 8 个 HB 层、状态 token、去噪步 0 的 top-4 专家 id |
| `state_token_probs` | (总步数, 8, 32) | 同上的完整门控分布，float16 |

行按 (局, 控制步) 平铺，局序即 `episode_index` 序。

`state_token_probs` 存 float16 是因为它只被承诺那一节的回归读到，第三位小数远在场景
bootstrap 噪声之下，而这样能把包缩掉一半。

## 重跑

```bash
python3 scripts/reproduce_from_bundle.py data/     # 只用 43 MB 派生数据
```

复现四块头条数字：k=0 承诺表、规则搜索与置换零假设、诚实划分的配对差、加动作块之后
的配对差。输出见 `logs/09_reproduce_from_bundle.log`，可与 `logs/01/02/07` 对照。

它 import `analyze_rules` 里的 `cooccur` 和 `score`，不重写，所以两边对不上就是真的
对不上，而不是抄错。

`scripts/` 下其余脚本按当时运行的样子原样保留，**它们读的是完整语料**，需要
`VLA_MUI_HUB/` 在位。日志就是它们的输出。

依赖：numpy、scipy、scikit-learn、zarr（只有读原始语料的脚本需要 zarr）。

## 从哪儿开始读

`REPORT.md`。一句话版：不存在"某某路由 ⇒ 必失败"的硬规则（置换检验说这不是没找够）；
有一条能跨场景转移的软规则，在最难的任务上比完整世界状态多 1.3 个百分点，另外三个任务
上是零。

报告最后一节「方法学」记了三个踩过的坑——留一场景会让结局基线失准、残差在任何子集上都
有正偏、生存偏差——这三个都足以单独造出一个假结论。
