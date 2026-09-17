# P3h 准备记录

正式采集前未执行模型前向，先完成控制器测试与模拟器预检。

- `p3h-env-preflight-bckhxgu6`：Plus 原前缀及下一个 chunk 的 190 个动作均精确复现，文件写完后 worker 清理退出出现 `free(): invalid pointer`。本批未通过，不计入正式结果。
- `p3h-cleanup-diagnostic-o3ukf6q4`：仅初始化及原规定 10 个 settle 动作，0 个任务动作，复现同一退出错误。
- `p3h-cleanup-diagnostic-hpv0mvk1`：将直接导入 mujoco 的位置移到既有 `_load_task` 初始化之后，保留原运行时的导入顺序；0 个任务动作，正常清理退出。
- `p3h-env-preflight-7bdwhzfj`：使用修正后的同一 worker，Plus/Pro 各执行 ten 和 five_plus_five 两种提交，共 4 条前检轨迹。任务动作合计 740，另有 settle 40；全部原前缀精确复现，同一父环境两种提交方式的所有保存数组、分叉及终点状态审计完全一致。

因此失败预检及诊断共有额外任务动作 190、settle 30，未隐藏、不删除。正式第一层仍独立为 64 条固定续跑，不使用预检作为统计样本。

冻结准备中还修正了旧 P1a 清单字段兼容和 SHA 接口的 Path 类型检查；这些检查错误发生在正式 run 创建及模型调用前。正式配置和计算代码随后一次冻结，运行时不调参数、不覆盖结果。
