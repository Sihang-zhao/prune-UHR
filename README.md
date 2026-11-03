# prune-UHR — Q-FlowCover

用于 Qwen3-VL 的免训练视觉 token 剪枝与 VQA 评测代码，从原工程的 `prune/testv.py` 整理而来。该目录可直接作为 `Sihang-zhao/prune-UHR` 仓库根目录使用，不依赖原 `prune/` 目录。

默认在第 18 个视觉块结束后（从零计数为 block 17）执行一次 Q-FlowCover，保留每张图像视图 50% 的合并 token，再运行剩余视觉块和语言模型。核心 `compression/q_flowcover.py` 与原文件逐字节一致。

## 安装

已验证环境为 Python 3.10、PyTorch 2.6.0、Transformers 4.57.6。建议使用独立环境，并根据设备安装匹配的 PyTorch / torchvision 构建。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# 使用本地 HuggingFace 数据集或 Parquet / Feather 时额外安装：
python -m pip install -r requirements-data.txt
```

模型权重和数据需要自行准备，不随源码分发。`--model` 接受模型 ID 或本地 checkpoint 目录；本地离线运行使用 `--offline`。本项目依赖 Qwen3-VL 的视觉内部接口，运行其他架构或替换 Transformers 版本需要重新验证。

## 运行

```bash
python testv.py \
  --data /path/to/XLRS-Bench-lite \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --offline \
  --gpus 0 \
  --dtype bf16 \
  --prune-method q_flowcover \
  --exec-mode hard18 \
  --visual-prune-percent 70 \
  --output-dir outputs/q_flowcover_70
```

`--visual-prune-percent 70` 表示删除 70%、保留 30%。也可以使用 `--prune-fraction 0.7`，二者不能同时传入。固定每视图预算使用 `--visual-keep-tokens K`，不能再与删除比例参数同时使用。默认按 `K=max(1, round(N×keep_ratio))` 计算预算，输出索引按原图顺序排列。

第一次验证可增加 `--limit 2 --max-new-tokens 8`。多卡使用 `--gpus 0,1`；每张卡单独加载一份模型并处理不同样本，不是模型切分。无 CUDA 时使用 CPU。默认每个样本最多一个全局视图、总视觉预算 4,194,304 像素；需要复现 16M 像素配置时显式传 `--max-image-pixels 16777216`。

查看全部数据、模型、图像、GPU、断点续跑和算法参数：

```bash
python testv.py --help
```

`--data` 必须提供，也可以通过环境变量 `XLRS_DATA_ROOT` 设置。支持 JSON、JSONL、Parquet、Feather 及本地 HuggingFace 数据集。相对图像路径默认以标注文件所在目录为根，也可以传 `--image-root`。一个最小 JSONL 样本如下，需自行提供 `images/example.jpg`：

```json
{"id":"example","image":"images/example.jpg","question":"Which object is visible?","options":{"A":"An aircraft","B":"A ship"},"answer":"A"}
```

## 方法与执行模式

| `--prune-method` | 含义 |
|---|---|
| `text_cos` | 第 18 层投影特征与问题文本的余弦 Top-K 基线 |
| `flow_topk` | 文本质量与 attention flow 支持度混合后取 Top-K |
| `text_local_cover` | 只使用文本质量权重的局部贪心覆盖 |
| `q_flowcover` | 文本质量与 flow 支持度混合后的局部贪心覆盖（默认） |
| `q_flowcover_global` | 使用全局余弦覆盖图的对照方法 |

Q-FlowCover 先以文本余弦 Top-K 确定查询组 P，在 P 内 softmax 得到质量分布 π。使用第 18 层原生 RoPE Q/K、分块 attention 汇总源组支持度，得到 `ω=0.5×(π+support)`，再以合并组内部有序的隐藏特征做贪心加权覆盖。评分文本与原 `testv.py` 一致，仅使用问题；生成提示仍包含题目、选项和回答要求。

| `--exec-mode` | 决策之后的执行 |
|---|---|
| `hard18` | 第 19 个视觉块开始只保留选中 token（默认） |
| `full19_gather` | 第 19 个视觉块完整执行后再 gather |
| `att_bridge19_fullkv` | 第 19 个视觉块只计算选中查询，读取该视图全部 K/V，再执行紧凑后缀 |

后三种执行方式复用同一组选择结果，但计算量不同。单块完整 K/V bridge 应与完整 block 的对应行数值一致；`hard18` 的后续 self-attention 使用压缩后的 K/V。

`--qflow-radius` 默认为 1；`--qflow-query-chunk` 默认为 256；`--qflow-candidate-chunk` 默认为 64，作用于全局覆盖候选分块。局部方法存储稀疏邻接边，原始 attention 按查询分块计算；全局覆盖仍有较高计算成本。决策层固定为 18。

## 输出与续跑

每次运行写入 `--output-dir` 下的新时间戳目录：

- `config.json`：参数、模型、设备和数据配置。
- `results.jsonl`：逐题答案、正确性、固定预算、索引、执行轨迹和诊断。
- `summary.json` / `results.md`：准确率、token 数量、延迟和 FLOPs 估算。
- `test.log`：运行进度和错误。

FLOPs 为架构估算，不能替代硬件 profiler；随机 tiny 模型测试不代表真实模型准确率。需要 bridge 诊断时使用 `--exec-mode att_bridge19_fullkv --att-stats summary --save-att-stats`；`debug` 会增加参考计算开销。

同一配置中断后，可在原命令上增加 `--resume`，或使用 `--resume-from /path/to/run/results.jsonl`。续跑必须保留同目录 `config.json`，并匹配数据、模型、方法、预算、图像处理和生成设置。不同实验请使用不同输出目录。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用 CPU 张量和本地初始化的小型 Qwen3-VL 视觉模型，不下载权重或数据。覆盖 dense attention / 贪心参考、固定预算、多视图隔离、三种完整视觉执行路径、DeepStack、RoPE 恢复、CLI 和续跑配置。缺少 PyTorch 或 Transformers 时部分测试会跳过，实际运行应确认无意外跳过。

整理时实测 26 项全部通过、无跳过。另用本地随机 tiny Qwen3-VL 完成真实 CLI → spawn worker → 生成 → 五类结果文件的离线 CPU 验证，64 组保留 2 组、0 个运行错误；尚未进行 8B 模型的完整 GPU 数据集评测。

## 文件结构与来源

```text
testv.py                         统一命令行入口
evaluator.py                     数据评测、进程管理、续跑和汇总
compression/q_flowcover.py       原始算法核心
compression/qwen3_vl_modes.py    hard / full / bridge 视觉执行
compression/worker.py            可被 spawn 导入的推理 worker
model/                          Qwen 模型封装与剪枝基础设施
utils/                          图像、数据、提示和评分工具
tests/                          CPU 数值与接口回归测试
```

原工程调用位置、文件映射及清理期间修复的问题见 [SOURCE_MAP.md](SOURCE_MAP.md)。保留了模型封装内部部分通用多层/grid helper，以减少提取过程对推理行为的影响；公开入口只支持上表中的方法。
