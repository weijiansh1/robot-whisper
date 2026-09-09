# HiMoE-VLA CALVIN D→D 复现报告（2026-08-04）

对照论文 arXiv-2512.05693v2 与官方仓库 [ZhiyingDu/HiMoE-VLA](https://github.com/ZhiyingDu/HiMoE-VLA)，
用官方发布的 CALVIN-D checkpoint 按官方协议完成了完整 1000 序列长程评测。

## 结论

**复现成功。** 五个等级的成功率全部与论文吻合（论文值全部落在本次复跑的 Wilson 95% 区间内），
Sum 得分 4.008 vs 论文 3.98（+0.028）。

| 指标 | 本次复跑 (1000 条) | Wilson 95% | 论文 Table 1(a) | 差值 |
|---|---|---|---|---|
| ≥1 任务 | 0.939 | [0.922, 0.952] | 0.938 | +0.001 |
| ≥2 任务 | 0.869 | [0.847, 0.889] | 0.866 | +0.003 |
| ≥3 任务 | 0.799 | [0.773, 0.823] | 0.794 | +0.005 |
| ≥4 任务 | 0.734 | [0.706, 0.761] | 0.723 | +0.011 |
| ≥5 任务 | 0.667 | [0.637, 0.696] | 0.659 | +0.008 |
| **Sum** | **4.008** | — | **3.98** | **+0.028** |

共完成 4008 个子任务；1000 条序列全覆盖（`coverage_complete: true`），零基础设施失败。

## 协议对齐（对照官方 `examples/calvin/main.py`）

| 项 | 值 | 与官方一致性 |
|---|---|---|
| 评测设置 | D→D 长程，1000 条五连任务链 | ✓ |
| 序列宇宙 | `get_sequences(1000)`，SHA-256 `e0b61a93…ba49e52`，每次启动重建并校验 | ✓（生成器对 worker 数不敏感，逐项确认） |
| 初始状态映射 | `get_env_state_for_initial_condition` 重实现（pyhash fnv1_32 UTF-16LE 语义，测试向量锁定） | ✓ |
| 单子任务步数上限 | 360（EP_LEN） | ✓ |
| 图像 | static + gripper，`resize_with_pad` 至 224 | ✓ |
| 重规划 | 每 10 步重新推理（10×8 关节动作块） | ✓ |
| 动作类型 | `joint_abs`，夹爪二值化（<0→−1，否则 +1） | ✓ |
| 环境配置 | `task_D_D/validation/.hydra/merged_config.yaml`（与官方硬编码的 `task_ABCD_D` 版本逐字节相同），`use_egl: true`，tactile 相机保留 | ✓ |
| 环境种子 | `np.random.seed(7)` | ✓ |
| wrist 布局 | `released-left`（上游 CalvinInputs 原样路径，无重映射） | ✓ |

## 身份锁定

| 构件 | 标识 |
|---|---|
| Checkpoint | `ZhiyingDu/HiMoE-VLA-CALVIN-D` @ `7809e66`，`pytorch_model.pth` 8,138,322,389 字节，SHA-256 `1ea466b4…3c9ebfa` |
| 归一化 stats | `calvin_d_joint/meta/stats.json`，SHA-256 `d371cc5c…7bafaad` |
| 训练/数据配置 | `calvin_d_joint`（flow 步数 10，内部动作维 24） |
| HiMoE-VLA | `27a2c46932d8b6373ca0074eb997f299bcd4f6f5` |
| calvin | `fa03f01f19c65920e18cf37398a9ce859274af76`；calvin_env 子模块 `1431a46bd36bde5903fb6345e68b5ccc30def666` |
| 官方 CALVIN 运行时补丁 | HiMoE 发布的 `examples/calvin/{robot.py,play_table_env.py}` 按 SHA-256 校验后覆盖安装 |
| 评测代码 | `himoe-calvin-alignment` worktree，分支 `align/calvin-released-checkpoint` @ `edf9076`（163 tests passed） |
| run-config | SHA-256 `ecfc7199…0983726f`（协议+来源+服务器元数据的规范化哈希，断点续跑强校验） |

上游 HiMoE-VLA 工作树带 3 处已知兼容补丁（LIBERO 阶段引入，diff SHA 已录入服务器元数据）：
修复损坏的 `from pytest import Cache` 导入、离线构造 PaliGemma（权重完全来自 checkpoint，`strict=True`）、
安全 `torch.load`。另有仅在客户端显式传 `flow/noise` 时才生效的显式噪声注入——CALVIN 评测不传，
推理数值路径与官方一致。

## 运行时布局

```
GPU 1 (H20-3e MIG 4g.71gb):  py3.11 模型服务器（websocket :8000，CUDA_VISIBLE_DEVICES=1）
GPU 0 (H20-3e 完整卡):        pybullet EGL 硬件渲染（EGL_VISIBLE_DEVICES=0；MIG 分区不支持图形）
CPU:                          py3.8 CALVIN 客户端（calvin_env + pybullet + hydra）
```

数据集未下载（177 GB）：环境所需的 `.hydra` 配置经 HTTP Range 请求从官方 zip 精准抽取并 SHA 记录。

## 运行时间线

- 08:24 UTC 服务器启动（checkpoint SHA 校验 + 模型加载）
- 08:26–08:30 冒烟 2 条序列通过（EGL、推理、oracle 全链路）
- 08:31 全量启动；11:04 在第 243 条后进程被静默杀掉（无 Python 栈、无 OOM 记录，疑似 pybullet/EGL 偶发段错误）
- 11:06 监督循环（`supervise-eval.sh`）断点续跑，此后零崩溃
- 19:54 UTC 全部 1000 条完成；均速 ~41 秒/条，总墙钟 ~11.5 小时

## 证据位置（已持久化）

```
/home/jovyan/work/himoe-vla-cache/himoe-calvin-alignment/
├── formal-artifacts/calvin-d-released-1000/   正式评测（4.6 MB）
│   ├── run-config.json       协议+来源+服务器元数据，run_config_sha256
│   ├── sequences.json        1000 条序列宇宙全文
│   ├── sequences.jsonl       逐序列结果（子任务级成功/步数/推理次数、首块动作 SHA）
│   ├── summary.json          成功率、Wilson 区间、与论文差值
│   └── failure-videos/       前 20 个失败子任务视频
└── metadata/                 服务器/评测/监督日志、checkpoint.env、
                              数据集 Range 抽取清单、supervise-eval.sh
```

2026-08-04 晚已把**整个 CALVIN cache**（checkpoint 8.1 GB、打好补丁的 calvin 仓库、py3.8
环境、数据集配置）从 ephemeral overlay 迁入
`work/himoe-vla-cache/himoe-calvin-alignment/cache/`，`/home/jovyan/.cache/himoe-calvin-alignment`
现在只是软链接；LIBERO bridge cache、uv 解释器仓库、OpenVLA 对照同样已迁入 work。
**重启后先运行 `work/himoe-vla-cache/restore-cache-symlinks.sh` 重建软链接**，随后所有
命令按原路径继续可用（已验证：迁移后 calvin/model 两环境导入正常、163 tests passed、
checkpoint SHA 复核通过）。

## 复跑方法

```bash
cd /home/jovyan/work/himoe-calvin-alignment

# 1. 服务器（GPU 1）
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python -m himoe_calvin_alignment.policy_server \
  --checkpoint-dir /home/jovyan/.cache/himoe-calvin-alignment/checkpoints/HiMoE-VLA-CALVIN-D \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --calvin-root /home/jovyan/.cache/himoe-calvin-alignment/upstream/calvin \
  --host 127.0.0.1 --port 8000

# 2. 评测（可断点续跑，output-dir 不变即续）
bash scripts/calvin.sh \
  --dataset-root /home/jovyan/.cache/himoe-calvin-alignment/datasets/task_D_D \
  --calvin-root /home/jovyan/.cache/himoe-calvin-alignment/upstream/calvin \
  --himoe-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --output-dir <输出目录> --host 127.0.0.1 --port 8000

# 或直接用带自动重启的监督循环
/home/jovyan/work/himoe-vla-cache/himoe-calvin-alignment/metadata/supervise-eval.sh
```

环境重建：`scripts/setup_calvin_env.sh`（py3.8 客户端）、`scripts/download_calvin_checkpoint.sh`、
`scripts/prepare_calvin_runtime.sh`（官方运行时补丁安装+校验）、`scripts/extract_calvin_config.py`
（Range 抽取数据集配置）。

## 备注

- 论文表 1(a) 另有 FLOWER+HiMoE 4.49 一行，属 FLOWER 训练配方（非本 checkpoint），不在本次范围。
- LIBERO 四套件的复现见 `himoe-vla-section4.1-reproduction-guide-2026-08-04.md`：公开
  `LiberoInputs` 存在 wrist 槽位错配，修复后 Goal 97.8%（与论文统计相容）、Spatial 94.2%、
  Object 96.6%。CALVIN 这边**不需要任何输入重映射**——上游原样 `released-left` 路径直接复现
  论文数字，与"LIBERO 差距来自其输入实现而非 checkpoint 本身"的前期定位一致。`wrist.py`
  中的 CALVIN 槽位 A/B 开关仅作为后续消融预留，本次正式运行未启用。
