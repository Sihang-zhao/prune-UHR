# 源代码调用位置与整理记录

以下原始路径相对于工作目录中的 `prune/`。行号对应提取时的原文件。

## 原评测入口

| 原文件 | 作用 |
|---|---|
| `testv.py:15,73–77` | Q-FlowCover 主入口；导入 testx 并替换 canonical worker |
| `testx.py:24–37` | 动态加载 testp，复用参数和评测基础设施 |
| `testp.py:583,782` | spawn 多进程、数据、断点、逐题与汇总输出 |
| `compression/testx_worker.py:199,257` | 构造 compression runner，调用 answer |
| `compression/qwen3_vl_modes.py:415–530,636–652` | 捕获第18层原生 RoPE Q/K，调用 Q-FlowCover selector |
| `compression/q_flowcover.py:265,373` | 单视图实现和公开多视图入口 |

其他调用 `compression/` 的实验入口是 `testn.py`（direct/seeded/QCAC 扩展）、`testww.py`（direct_residual）、`testm.py → test.oy`（旧 bridge 实验）。本次按 `testv.py` 范围提取，未包含这些后续变体。`test.py` 与 `testp.py` 本身不直接调用 Q-FlowCover。

原单元测试为 `tests/test_qwen3_vl_modes.py`、`tests/test_readout_verified_bridge.py`，原目录中没有 Q-FlowCover 专用算法单元测试。

## 新文件映射

| 新文件 | 原来源与修改 |
|---|---|
| `testv.py` | 统一 testv/testx/testp 的实际 CLI 参数，取消动态载入和 worker 替换 |
| `evaluator.py` | 从 testp 提取数据评测和进程管理；直接调用 compression.worker |
| `compression/q_flowcover.py` | 原样保留，逐字节一致；参考哈希见 source_manifest.json |
| `compression/qwen3_vl_modes.py` | 保留原五种方法和三种执行方式，去掉后续实验分支 |
| `compression/worker.py` | 原 compression/testx_worker.py，去掉后续实验参数和空诊断 |
| `compression/readout_verified_bridge.py` | 仅保留 runner 实际使用的 compare_rows |
| `model/qwen_vl.py` / `model/training_free_pruning.py` | 保留推理与 token/DeepStack 基础设施，去掉无关选择策略和旧 pruner 依赖 |
| `utils/*.py` | 数据、图像、任务与评分实现原样复制；包 init 改为轻量导入 |
| `tests/*.py` | 独立新增 dense reference、native block、完整视觉路径和接口回归测试 |

## 整理时修正的问题

1. 帮助信息：完整展示数据、模型、GPU 和算法参数；`python -S testv.py --help` 无需安装推理依赖。
2. 可移植性：去掉本机数据绝对路径；必须传 `--data` 或设置 `XLRS_DATA_ROOT`。模型显式使用用户传入的 ID 或路径，不再从邻近目录隐式挑选 checkpoint。
3. 决策层：固定人类层18 / block17，拒绝与原入口语义矛盾的层号。
4. 结果标注：报告实际 `prune_method` 和执行模式，不再将 Q-FlowCover 标为 text cosine。
5. 执行轨迹：hard18 的完整输入块数为18，full19/bridge 为19；按实际模式校验。
6. 多视图 bridge：原 `_raw_indices` 把全局 raw 索引作为每视图局部索引，第二视图可能越界；现在分别维护局部索引和拼接索引。
7. DeepStack debug：原比较路径把 raw 索引用于 merged 特征；现在按合并组索引 gather。
8. 续跑：检查旧 config，拒绝混用不同数据、模型、方法、预算和生成设置的结果。

原始 `prune/` 保持不变。完整视觉测试使用本地随机 tiny 模型验证接口与数值关系，不是 8B 模型的 GPU 基准测试。
