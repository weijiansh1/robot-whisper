# HiMoE-VLA Section 4.1 复现汇报与操作指南

> **2026-08-15 更正：本文的结果表与 `--libero-wrist-layout` 推荐值已被取代。**
> 本文推荐的 `checkpoint-right`（两个 wrist mask 都为 true）**不符合论文**——
> example.tex:591 写明 left 槽位「zero-padded **with masks**」，即 mask 应为 false。
> 当时选定它的依据只是一次 20 局比较（第 6.2 节），样本不足以支撑该决定。
> 改用严格 `paper-right` 后：Goal 98.0%、Object 99.0%（**两套与论文统计相容**）、
> Spatial 95.4%、Long 92.4%。
>
> 另有两处已过时：第 2 节「Long 未运行」——权重已于 2026-08-14 下载并校验
> （sha256 `cdc2b21f…`）；第 6.8 节「work 总容量只有 8 GB」——实测可用 131.5 GB。
>
> 当前结果、证据链与结论请以
> [`himoe-vla-libero-alignment-2026-08-15.md`](himoe-vla-libero-alignment-2026-08-15.md) 为准。
> **本文第 4–6 节的环境固定、验收标准与操作注意事项仍然有效**，只需把
> `checkpoint-right` 换成 `paper-right`。

更新时间：2026-08-04  
论文：HiMoE-VLA, arXiv:2512.05693v2  
范围：论文 Section 4.1 的 LIBERO 仿真评测  
结论依据：公开代码、公开 checkpoint、论文 TeX、本地张量审计、配对实验及 500-episode 正式评测

## 1. 结论摘要

原始公开代码和公开 checkpoint 可以正常运行，但直接使用公开 `LiberoInputs` 时，Goal、Spatial、
Object 只能得到约 80% 的成功率，与论文约 98% 的结果明显不一致。

主要原因已经定位为 **LIBERO wrist 图像所在的模型槽位不一致**：

- 论文写明单臂输入映射到 right-arm channel，left-arm channel 使用零填充；
- 公开 `LiberoInputs` 却把真实 wrist 图像放入 left wrist 槽位，把 right wrist 置零；
- 发布 checkpoint 的闭环行为明确偏好真实 wrist 位于 right wrist 槽位；
- Goal 固定噪声配对实验中，right-slot 相比公开 left-slot 从 `40/50` 提升到 `48/50`，
  exact McNemar `p=0.0386`；
- 修复后 Goal 从 `82.0%` 提升到 `97.8%`，Spatial 从 `81.2%` 提升到 `94.2%`，Object 从
  `79.2%` 提升到 `96.6%`。

最终推荐的输入模式是：

```text
--libero-wrist-layout checkpoint-right
```

这里的 `checkpoint-right` 表示：

```text
left wrist  = 全零图像，mask=true
right wrist = 真实 wrist 图像，mask=true
```

这不是严格照搬论文 mask 描述，而是与发布 checkpoint 已验证行为一致的模式。严格论文 mask
`left=false/right=true` 被单独命名为 `paper-right`，只作为消融，不建议用于正式复现。

## 2. 当前复现结果

主结果均使用相同协议：10 个任务、每任务 50 个官方初始状态、环境 seed 7、模型 seed 42、
`top-K=4`、`replan=10`、settling 10 步、EGL。

| Suite | 论文结果 | 公开输入旧基线 | 修复后结果 | Wilson 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Goal | 98.6% | 410/500 = 82.0% | **489/500 = 97.8%** | [96.10%, 98.77%] | 统计上与论文相容，未逐字面命中 |
| Spatial | 98.2% | 406/500 = 81.2% | **471/500 = 94.2%** | [91.79%, 95.93%] | 主差距已修复，仍未对齐论文 |
| Object | 99.4% | 396/500 = 79.2% | **483/500 = 96.6%** | [94.62%, 97.87%] | 主差距已修复，仍未对齐论文 |
| Long | 95.8% | 未运行 | 未运行 | 不适用 | 本地只有 135-byte LFS pointer，缺少完整权重 |

Spatial 另有 `replan=5` 的诊断运行，结果为 `478/500 = 95.6%`。论文没有公布该执行长度，而且
95% 区间仍不包含论文的 98.2%，所以它不能替代上表的公开默认 `replan=10` 主结果。

可以准确声明：

> 已找到并修复公开 LIBERO 输入实现与发布 checkpoint 之间的 wrist 槽位错配。修复解释了原结果
> 的绝大部分差距，使 Goal 达到与论文统计相容的 97.8%，并使 Spatial/Object 分别提升到 94.2%
> 和 96.6%。由于 checkpoint 训练身份、正式策略噪声序列和完整训练数据 lineage 未公开，
> Spatial、Object 及 Long 目前不能声明严格复现论文数字。

## 3. 复现层次

本指南完成的是 **公开 checkpoint 的评测复现**，不是从头训练复现。

| 层次 | 含义 | 当前状态 |
|---|---|---|
| 评测复现 | 使用作者发布的 suite checkpoint 重跑 LIBERO | Goal/Spatial/Object 已完成 |
| 微调复现 | 从公开 base checkpoint 重新训练 LIBERO suite | 未完成，缺完整数据转换 lineage |
| 从零训练 | 重建 OXE/ALOHA 预训练及 LIBERO 微调 | 公开材料不足以严格逐制品复现 |

因此，“复现成功”应限定为评测管线和公开 checkpoint 行为的复现，不能写成已完整复现论文训练过程。

## 4. 固定环境和文件

### 4.1 代码身份

| 项目 | 固定值 |
|---|---|
| HiMoE-VLA 上游 | `27a2c46932d8b6373ca0074eb997f299bcd4f6f5` |
| LIBERO 上游 | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| 冻结旧基线 bridge | `f5dc26bbf4d878ad6621c7be85335eb61f494dc0` |
| wrist 对齐 bridge | `d7797415d0fce6b4fd25393dbc0d1923ac28ed3b` |
| wrist 对齐分支 | `fix/libero-paper-wrist-layout` |
| wrist 对齐 worktree | `/home/jovyan/work/himoe-libero-wrist-fix` |

注意：HiMoE 上游目录在固定 commit 之上还有三个已审计的本地运行时修改，不能只记录 Git commit：

```text
src/moevla/models/paligemma_with_expert.py
src/moevla/policies/policy.py
src/moevla/policies/policy_config.py
```

这些修改用于修正模型构造/加载兼容性和支持显式 flow noise。现有正式运行没有传入显式 noise，
仍使用发布策略的内部噪声流。运行时文件与补丁哈希见：

```text
/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json
```

### 4.2 依赖协议

| 项目 | 固定值 |
|---|---|
| 模型环境 Python | 3.11.15 |
| 模型环境 Torch | 2.6.0 + CUDA 12.4 |
| Transformers | 4.48.1，实际 `modeling_gemma.py` 已记录 SHA |
| LIBERO 环境 Python | 3.8.20 |
| robosuite | 1.4.1 |
| MuJoCo | 3.2.3 |
| NumPy | 1.22.4 |
| 渲染 | EGL |
| 图像输入 | 224 x 224 |
| 环境 seed | 7 |
| 模型 seed | 42，由新 server 初始化 |
| flow integration steps | 10 |
| action chunk | 10 x 7 |
| replan steps | 10 |

不要把早期 README 中的 OSMesa 诊断命令与本次正式 EGL 结果混在一起。OSMesa 可用于无模型环境检查，
但上表的正式数字来自 EGL。

### 4.3 Checkpoint 与 normalization

| Suite | checkpoint SHA-256 | stats SHA-256 |
|---|---|---|
| Goal | `98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953` | `0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf` |
| Spatial | `1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137` | `4b06853e1320af2a05614f14bdc79e441b4f175c9c18970657c80c7ed779d266` |
| Object | `f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa` | `50719783647bee5adb7d936863eff02e0500da6e13867ddbcbfbc551b3a06487` |

每个完整权重文件应为 `8,138,322,389` bytes。不能混用不同 suite 的权重和 stats。

## 5. 正式复现步骤

以下步骤针对当前机器已有环境。建议先跑 Goal；确认 500 episodes 完整后，再分别运行 Spatial 和 Object。

### 5.1 预检查

```bash
test -x /home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python
test -x /home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python
test -d /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA
test -d /home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO
test -f /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal/pytorch_model.pth

git -C /home/jovyan/work/himoe-libero-wrist-fix rev-parse HEAD
git -C /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA rev-parse HEAD
git -C /home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO rev-parse HEAD

df -h /home/jovyan/work /home/jovyan/.cache
nvidia-smi
```

预期 bridge HEAD 为 `d779741...`，HiMoE 为 `27a2c469...`，LIBERO 为 `8f1084e...`。
如果环境、上游目录或权重不存在，说明非持久 cache 已丢失，需要重新构建或下载后再运行。

核验 Goal 权重：

```bash
sha256sum \
  /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal/pytorch_model.pth \
  /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal/libero_goal_no_noops/meta/stats.json
```

### 5.2 运行 bridge 测试

```bash
cd /home/jovyan/work/himoe-libero-wrist-fix

PYTHONPATH=src /opt/conda/bin/pytest -q
```

当前预期为：

```text
155 passed
```

测试使用工作区基础环境中的 pytest；模型推理环境只保留运行时依赖，当前没有安装 pytest。不要为了跑测试
而改装模型环境。

### 5.3 启动全新的模型服务

终端 A：

```bash
cd /home/jovyan/work/himoe-libero-wrist-fix

SUITE=goal
PORT=8063

PYTHONPATH=src \
MOEVLA_DATA_HOME=/home/jovyan/.cache/himoe-libero-bridge/moevla-data \
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  -m himoe_libero_bridge.cli serve \
  --backend himoe \
  --suite "$SUITE" \
  --gpu 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-dir "/home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-${SUITE^}" \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right
```

`--gpu 0` 是当前机器可见 GPU 的索引。运行前必须用 `nvidia-smi` 确认它空闲；如需换卡，只修改
server 的 `--gpu`，不要让 LIBERO 客户端加载 CUDA。

服务启动后不要发送任何模型 inference 请求。可以进行一次只读 metadata handshake，它不会消费模型
noise：

```bash
cd /home/jovyan/work/himoe-libero-wrist-fix

PYTHONPATH=src \
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python - <<'PY'
import json
from himoe_libero_bridge.client import PolicyClient

with PolicyClient(host="127.0.0.1", port=8063) as client:
    print(json.dumps(client.metadata, indent=2, sort_keys=True))
PY
```

至少检查：

```json
{
  "suite": "goal",
  "libero_wrist_layout": "checkpoint-right",
  "libero_wrist_layout_transform_anchor": "DropStateAndImage",
  "checkpoint_sha256": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
  "normalization_stats_sha256": "0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf"
}
```

### 5.4 运行官方 Goal 10 x 50

终端 B：

```bash
SUITE=goal
PORT=8063
RUN_ID="manual-checkpoint-right-${SUITE}-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/paper-alignment/$RUN_ID"
mkdir -p "$OUT/videos"

CUDA_VISIBLE_DEVICES= \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=/home/jovyan/.cache/himoe-libero-bridge/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO:/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src \
/home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python \
  /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/examples/libero/main.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --resize-size 224 \
  --replan-steps 10 \
  --task-suite-name "libero_${SUITE}" \
  --num-steps-wait 10 \
  --num-trials-per-task 50 \
  --video-out-path "$OUT/videos" \
  --seed 7 2>&1 | tee "$OUT/eval.log"

rg "Total success rate|Total episodes" "$OUT/eval.log"
sha256sum "$OUT/eval.log" > "$OUT/eval.log.sha256"
```

成功的完整运行必须同时出现：

```text
Total success rate: ...
Total episodes: 500
```

Goal 当前参考结果为 `0.978`。正式运行完成后，在终端 A 使用 `Ctrl-C` 停止 server。

### 5.5 切换 Spatial 或 Object

每个 suite 都必须先停止旧 server，再启动一个全新 server。只需要同步修改 `SUITE`：

| Suite | `SUITE` | checkpoint 目录 | benchmark | 参考结果 |
|---|---|---|---|---:|
| Goal | `goal` | `HiMoE-VLA-Libero-Goal` | `libero_goal` | 489/500 |
| Spatial | `spatial` | `HiMoE-VLA-Libero-Spatial` | `libero_spatial` | 471/500 |
| Object | `object` | `HiMoE-VLA-Libero-Object` | `libero_object` | 483/500 |

终端 A 和终端 B 中同时改为：

```bash
SUITE=spatial
```

或：

```bash
SUITE=object
```

`${SUITE^}` 会生成 checkpoint 目录所需的首字母大写名称，官方评测脚本会根据 benchmark 自动采用
Spatial 220、Object 280、Goal 300 的 horizon。

## 6. 最重要的注意事项

### 6.1 必须显式指定 `checkpoint-right`

最终 bridge 为了保留冻结旧基线，把默认值留为 `released-left`。因此漏写参数不会报错，但会悄悄回到
约 80% 的旧行为：

```text
--libero-wrist-layout checkpoint-right
```

这是整个复现中最重要的运行参数。

### 6.2 不要把 `paper-right` 当成正式修复

最终代码中：

- `checkpoint-right`：真实 wrist 在 right，两个 wrist mask 都为 true，推荐；
- `paper-right`：真实 wrist 在 right，仅 right mask 为 true，严格 mask 消融；
- `released-left`：公开旧输入，冻结基线；
- `both`：两个槽位都放真实 wrist，只用于消融。

历史正式结果目录名带有 `paper-right`，是旧提交 `927640c` 当时的 CLI 名称。由于旧 transform 后续会把
两个 mask 都覆盖成 true，这些历史运行的有效语义其实是现在的 `checkpoint-right`。不要根据历史目录名
误用最终 `paper-right`。

### 6.3 每次正式评测都必须使用全新 server

模型 server 初始化时固定 seed 42，随后使用进程级 flow-noise RNG。任何提前 inference 都会消费噪声，
改变之后 500 episodes 的策略随机流。

- metadata handshake 安全；
- inference probe 不安全；
- smoke episode 不安全；
- 切换 suite 或重跑正式结果前必须重启 server；
- 不要在同一个 server 上先调试再跑正式 500 episodes。

### 6.4 不要混用 suite 资产

Goal、Spatial、Object 必须同时匹配以下三项：

```text
--suite
--checkpoint-dir
--task-suite-name
```

normalization 由 suite registry 从对应 checkpoint 读取。不要复制 Goal stats 到 Spatial/Object，也不要用
一个 checkpoint server 连续评测多个 suite。

### 6.5 正式协议保持 `replan=10`

`replan=5` 只在 Spatial 上做过诊断，得到 95.6%，论文未公布该配置，且仍未达到论文 98.2%。正式主表
必须使用公开默认 `replan=10`；如果报告 replan 5，要单列为消融或诊断结果。

### 6.6 EGL 退出时的清理警告

正式日志在打印 `Total episodes: 500` 后，可能出现 Python 解释器退出阶段的：

```text
EGLError: EGL_NOT_INITIALIZED
```

这是已观察到的 EGL context 析构清理警告。只有在 500 episodes 和 Total success rate 已完整写出、过程中
没有 episode 基础设施异常时，才可把它视为不影响统计结果的退出警告。若 Total 行缺失，则该 run 不完整。

### 6.7 不要覆盖历史证据

每次都创建新的 `RUN_ID`。下列目录是只读证据，不应复用为新输出目录：

```text
/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/paper-alignment/
/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/formal-artifacts/
```

可以在它们下面新建唯一目录，但不能覆盖已有 `eval.log`、summary、NPZ、视频或 manifest。

### 6.8 Cache 与重启风险

不可替代的日志和元数据已持久化到 `/home/jovyan/work/himoe-vla-cache`，原 cache 结果路径是指向它的
软链接。它们重启后仍在。

但以下约几十 GB 的可重建大文件仍在 `/home/jovyan/.cache`，不是持久化保证的一部分：

```text
checkpoints/
envs/
upstream/
system-libs/
```

当前 `/home/jovyan/work` 总容量只有 8 GB，不能直接迁入约 98 GB 的全部模型与环境。重启后应先执行
5.1 的预检查；如果这些目录丢失，需要重新下载 checkpoint 和重建双环境。不要把“大权重未持久化”
误写成“全部 cache 已经不会丢失”。

### 6.9 不要清理已审计的运行时修改

HiMoE 上游目录显示三个 modified runtime files 是当前已审计环境的一部分。不要对该目录执行
`git reset --hard`、不要重新安装覆盖 Transformers/LeRobot，也不要只凭 `strict=True` 加载成功就认为
推理语义没有变化。若重建环境，必须重新生成 runtime source audit 并比较 SHA。

### 6.10 Long 当前不能替代运行

本地 Long 文件只是 Git LFS pointer，不是 8.14 GB 完整权重。不能拿 Goal/Object/Spatial checkpoint
替代 Long，也不能据此补一个推测结果。正式报告应写“Long 因完整 checkpoint 不可用而未复现”。

## 7. 为什么仍不能严格命中全部论文结果

完成 wrist 槽位修复后，仍有以下不可消除的信息缺口：

1. 论文没有发布正式评测所用的逐 episode 策略噪声 manifest；
2. 公开 checkpoint 不含 global step、run ID、训练代码 commit 或 dataset manifest；
3. 论文写 Goal/Object 45k、Spatial 35k、Long 40k，但公开配置对四套统一写 50k；
4. 缺少足以逐制品重建训练集的完整 demonstration 转换、过滤、采样和 normalization lineage；
5. 当前无法取得完整 Long checkpoint；
6. Spatial/Object 修复后结果的 95% 区间仍不包含论文点估计。

因此不能把剩余差距归因于简单的 MuJoCo 小版本、horizon 或 top-K，也不能直接断言作者上传了错误权重。
最严谨的判断是：公开 checkpoint 有效且主要输入错配已定位，但公开制品不足以证明它们就是论文表格所用
的最终训练状态，也不足以唯一化论文的完整正式随机流。

## 8. 结果验收标准

一次可纳入报告的正式运行至少应满足：

- bridge commit、HiMoE commit、LIBERO commit 已记录；
- checkpoint 和 normalization SHA 与本指南一致；
- metadata 明确为 `checkpoint-right`；
- 新 server 启动后没有发生 inference probe；
- 10 个任务各使用 50 个官方 init states；
- `seed=7`、`replan=10`、settling 10、224 输入、EGL；
- 日志包含 `Total episodes: 500`；
- suite 总成功率和各 task 成功率已保存；
- `eval.log` 已计算 SHA-256；
- 输出目录唯一且位于持久化研究目录；
- 报告中明确区分论文值、公开旧基线、修复值和诊断值。

只要任一项不满足，该运行最多算 smoke test 或诊断实验，不应加入正式对齐表。

## 9. 文档与证据索引

| 内容 | 路径 |
|---|---|
| 完整原因、统计与证据报告 | [`himoe-vla-section4.1-alignment-2026-08-03.md`](himoe-vla-section4.1-alignment-2026-08-03.md) |
| 机器可读总清单 | [`section4.1-alignment-20260803.json`](himoe-vla-cache/himoe-libero-bridge/metadata/section4.1-alignment-20260803.json) |
| 运行时源码审计 | [`runtime-source-audit-20260803.json`](himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json) |
| checkpoint lineage 审计 | [`checkpoint-lineage-audit-20260803.json`](himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json) |
| 最终/旧版 right-slot 一致性 | [`checkpoint-right-compatibility-audit-20260803.json`](himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-right-compatibility-audit-20260803.json) |
| 冻结旧基线 | [`released-checkpoint-baseline-v1.json`](himoe-vla-cache/himoe-libero-bridge/metadata/released-checkpoint-baseline-v1.json) |
| 最终修复代码 | [`himoe-libero-wrist-fix`](himoe-libero-wrist-fix/) |
| 持久化 cache 说明 | [`himoe-vla-cache/README.md`](himoe-vla-cache/README.md) |

正式结果日志：

```text
Goal:
himoe-vla-cache/himoe-libero-bridge/paper-alignment/
  paper-right-libero-goal-rs141-egl-20260803T1437Z/eval.log

Spatial:
himoe-vla-cache/himoe-libero-bridge/paper-alignment/
  paper-right-libero-spatial-rs141-egl-20260803T1624Z/eval.log

Object:
himoe-vla-cache/himoe-libero-bridge/paper-alignment/
  paper-right-libero-object-rs141-egl-20260803T1752Z/eval.log
```

这些历史目录名中的 `paper-right` 必须按第 6.2 节解释为旧实现下的有效 `checkpoint-right` 语义。
