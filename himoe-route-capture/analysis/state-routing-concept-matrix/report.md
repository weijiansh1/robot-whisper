# State-token 概念可读出矩阵

## 直接结论

这里分开回答两个问题：

1. `state_soft_routing` 超过 chance：概念可以从 state-token 的 soft gate 分布线性读出；
2. `geometry+state - geometry` 的 cluster-bootstrap CI 高于 0：routing 在当前 raw kinematics 之外仍有增量。

共分析 51,308 个 query、2,560 条 rollout、80 个 task×init cluster。35/35 个概念满足第一个探索性门槛，8/35 个满足第二个更严格门槛。两者不能互换：routing 可读出一个概念，常常只说明 gate 是物理/视觉状态的低维影子。

## 验证协议

- 每个 task 分别拟合；一折完整留出一个 init，其 32 个 flow seed 和全部 query 不进入训练。
- 指标先在 held-out task×init 内计算，再 task-macro；CI 分层重采样 task 和 init cluster。
- 连续概念指标是相对该折训练均值的 OOF skill，0 为均值基线，1 为完美；二元概念是 within-init AUC，chance=0.5。
- `geometry` 是 raw proprio、所选目标 free-joint pose、相关 drawer qpos、一步差分和 phase；它不含 RGB。
- `state_soft_routing` 只有 token 0 的 `8×32` soft probabilities。没有读取 hard expert IDs；抽查 denoise 轴最大概率差为 0.00000000。
- `MDE80` 是固定 OOF 预测下，用 cluster-bootstrap delta 标准差乘 2.80 的正态近似；它是 metric-point 检出尺度，不是事前功效保证。

## 主矩阵

| concept | kind | tasks | geometry | state soft | joint | joint−geometry [95% CI] | MDE80 | hidden ref |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eef_x_m | continuous | 5 | 0.998 | 0.988 | 0.996 | -0.002 [-0.004, -0.001] | 0.002 | NA |
| eef_z_m | continuous | 5 | 0.998 | 0.987 | 0.996 | -0.002 [-0.003, -0.001] | 0.001 | NA |
| eef_y_m | continuous | 5 | 0.998 | 0.985 | 0.995 | -0.003 [-0.004, -0.001] | 0.002 | NA |
| target_minus_eef_z_m | continuous | 4 | 0.997 | 0.973 | 0.995 | -0.002 [-0.003, -0.001] | 0.002 | NA |
| eef_axis_angle_y | continuous | 5 | 0.997 | 0.973 | 0.994 | -0.003 [-0.006, -0.000] | 0.004 | NA |
| drawer_goal_distance_native | continuous | 2 | 0.967 | 0.970 | 0.982 | +0.016 [-0.011, +0.050] | 0.049 | NA |
| drawer_position_native | continuous | 2 | 0.998 | 0.963 | 0.989 | -0.009 [-0.015, -0.005] | 0.008 | NA |
| eef_axis_angle_z | continuous | 5 | 0.999 | 0.963 | 0.994 | -0.005 [-0.008, -0.002] | 0.004 | NA |
| task_progress | continuous | 5 | 0.977 | 0.954 | 0.985 | +0.008 [-0.006, +0.030] | 0.026 | NA |
| target_z_m | continuous | 4 | 0.997 | 0.949 | 0.995 | -0.002 [-0.004, -0.001] | 0.002 | NA |
| target_speed_m_per_query | continuous | 4 | 0.937 | 0.946 | 0.969 | +0.032 [+0.024, +0.040] | 0.012 | NA |
| phase_episode_posthoc | continuous | 5 | 0.997 | 0.944 | 0.995 | -0.002 [-0.003, -0.001] | 0.001 | NA |
| phase_budget | continuous | 5 | 0.997 | 0.944 | 0.993 | -0.005 [-0.007, -0.002] | 0.004 | NA |
| eef_target_distance_m | continuous | 4 | 0.980 | 0.943 | 0.989 | +0.009 [-0.003, +0.029] | 0.025 | NA |
| gripper_aperture_m | continuous | 5 | 0.999 | 0.943 | 0.989 | -0.011 [-0.030, -0.001] | 0.024 | NA |
| task_progress_speed | continuous | 5 | 0.933 | 0.937 | 0.969 | +0.036 [+0.020, +0.054] | 0.026 | NA |
| target_goal_distance_m | continuous | 4 | 0.990 | 0.936 | 0.993 | +0.002 [-0.001, +0.006] | 0.005 | NA |
| target_x_m | continuous | 4 | 0.998 | 0.924 | 0.996 | -0.002 [-0.003, -0.001] | 0.002 | NA |
| eef_axis_angle_x | continuous | 5 | 0.998 | 0.901 | 0.994 | -0.004 [-0.010, -0.001] | 0.006 | NA |
| target_minus_eef_x_m | continuous | 4 | 0.980 | 0.890 | 0.978 | -0.002 [-0.012, +0.010] | 0.015 | NA |
| eef_speed_m_per_query | continuous | 5 | 0.804 | 0.888 | 0.936 | +0.132 [+0.067, +0.206] | 0.098 | NA |
| eef_axis_angle_norm_rad | continuous | 5 | 0.938 | 0.840 | 0.972 | +0.035 [+0.022, +0.048] | 0.020 | NA |
| gripper_command_mean | continuous | 5 | 0.777 | 0.828 | 0.828 | +0.051 [-0.018, +0.110] | 0.086 | NA |
| target_minus_eef_y_m | continuous | 4 | 0.991 | 0.814 | 0.983 | -0.008 [-0.014, -0.004] | 0.008 | NA |
| target_y_m | continuous | 4 | 0.999 | 0.751 | 0.995 | -0.003 [-0.006, -0.001] | 0.003 | NA |
| target_quat_w | continuous | 4 | 0.998 | 0.636 | 0.997 | -0.001 [-0.001, -0.000] | 0.001 | NA |
| target_tilt_rad | continuous | 4 | 0.677 | 0.560 | 0.684 | +0.008 [-0.025, +0.034] | 0.043 | NA |
| target_quat_z | continuous | 4 | 0.998 | 0.544 | 0.997 | -0.001 [-0.002, -0.000] | 0.001 | NA |
| target_quat_y | continuous | 4 | 0.999 | 0.518 | 0.998 | -0.002 [-0.004, -0.000] | 0.002 | NA |
| target_quat_x | continuous | 4 | 0.999 | 0.513 | 0.998 | -0.001 [-0.002, -0.000] | 0.002 | NA |
| gripper_closed_proxy | binary | 4 | 0.999 | 1.000 | 1.000 | +0.000 [-0.000, +0.001] | 0.001 | NA |
| close_command | binary | 4 | 0.993 | 0.998 | 0.998 | +0.005 [+0.002, +0.009] | 0.005 | NA |
| near_target_close_command_proxy | binary | 4 | 0.995 | 0.998 | 0.998 | +0.004 [+0.001, +0.008] | 0.006 | NA |
| target_lifted_1cm_proxy | binary | 4 | 0.996 | 0.997 | 0.998 | +0.002 [+0.000, +0.005] | 0.003 | NA |
| hold_proxy | binary | 4 | 0.996 | 0.997 | 0.998 | +0.002 [+0.000, +0.006] | 0.004 | NA |

## Geometry 定义

- EEF `state[0:3]` 是位置，`state[3:6]` 是由 bridge 明确定义的 axis-angle，`state[6:8]` 是两指 qpos；aperture 是两指位置差的绝对值。
- transport task 的 target 是当前离 EEF 最近的任务目标；long task 因而可能在两个 moka pot 之间切换。target free-joint quaternion 按 MuJoCo `wxyz` 解释并规范到 `w≥0`，tilt 是局部 z 轴相对世界 z 的夹角。
- target goal 使用同 task 全部成功 rollout 的 terminal position 均值；这是描述性的 transductive reference，不应用于声称 unseen-init 在线泛化。
- task progress 是各 transport target 的归一化 goal-distance progress，与相关 drawer opening progress 的均值。hold/grasp 均为闭爪、邻近、物体和 EEF 共动的运动学 proxy，不是接触或真抓取标签。
- `phase_budget` 使用任务已知 query budget；`phase_episode_posthoc` 使用最终 rollout 长度，只作后验对照。

## Hidden 参考

hidden store 存在且键可核对，但本次以 `--skip-hidden` 运行；主矩阵没有 hidden 上限列。

## 解释边界

- state token 结构上能看 image/language prefix 与 proprio，本分析无法把视觉来源和 proprio 来源拆开。
- geometry baseline 是线性模型；routing 对非线性距离/阈值概念的增量可能只是另一种非线性编码，不等于含有物理状态之外的信息。
- query 高度时序相关；task×init cluster CI 比逐 query CI 更保守，但只有 5 个 task，仍是探索性矩阵。
- soft routing 的观察性读出不证明某个 expert 实现该概念，也不证明 routing 对行为有因果作用。

## 产物

- `concept_matrix.csv`: task-macro 主矩阵、增量 CI 与 MDE。
- `concept_task_metrics.csv`: 每 task 的 held-init 指标。
- `concept_cluster_metrics.csv`: 每个 task×init 的 OOF 指标。
- `query_concepts.csv.gz`: query 元数据与物理概念，不含大路由张量。
- `compact_features.npz`: float16 state-soft/geometry 与可选 hidden 投影。
- `concept_matrix.png`: chance-centered 主矩阵。
- `summary.json`: 数据、路由契约、cross-fitting 协议和计数。
