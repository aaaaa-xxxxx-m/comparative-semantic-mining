# 黄酒比较语义挖掘

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow)](https://huggingface.co/docs/transformers/)

从中文消费文本中识别与黄酒相关的比较语义，抽取“主体—客体—方面—得分”四元组，并按原始文本聚合为可分析的比较结论。

仓库包含完整的预处理、合成数据生成、句子标注、筛选、抽取、规范化和训练入口，并已附带当前训练数据。模型权重尚未训练，下载后可以直接开始训练；训练完成前不能得到真实的筛选或抽取结果。

## 当前状态

| 阶段 | 状态 |
|---|---|
| 原始表拆分 | 已完成，188,084 条句级窗口 |
| 合成文章 | 已完成，650 篇 |
| 正例文章标注 | 已完成，488 篇 |
| Filter 数据 | 5,155 / 2,983 / 1,493 |
| GTS 数据 | 1,078 / 625 / 298 |
| 训练数据 | 已随仓库提供 |
| 筛选器权重 | 未训练 |
| GTS 权重 | 未训练 |

数据是合成数据，不是人工金标。用于训练或评测前仍需抽检。

## 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/aaaaa-xxxxx-m/comparative-semantic-mining.git
cd comparative-semantic-mining
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 训练请按机器 CUDA 版本单独安装 PyTorch。训练和推理设备由 `--device` 显式指定，支持 `cpu`、`cuda`、`cuda:0`，默认 `cpu`；指定 CUDA 但环境不可用时会直接报错，不会自动回退。

### 2. 直接训练

训练数据已经放在 `data/filter/` 和 `data/gts/`，训练不需要 GLM key，也不需要重新生成数据。

```powershell
python train.py filter --device cuda `
  --train-data data/filter/filter_train.jsonl `
  --valid-data data/filter/filter_valid.jsonl

python train.py gts --device cuda `
  --train-data data/gts/gts_train.csv
```

没有 GPU 时把 `--device cuda` 改成 `--device cpu`。显存不足时先降低 `--batch-size`。

训练产物：

```text
models/filter/filter_model.pt
models/filter/filter_config.json
models/filter/tokenizer files

models/gts/gts_model.pt
models/gts/gts_config.json
models/gts/tokenizer files
```

### 3. 推理

训练完成后运行：

```powershell
python run.py filter --device cuda
python run.py extract --device cuda
python run.py normalize
```

也可以一次执行后三步：

```powershell
python run.py all --device cuda
```

`filter` 和 `extract` 需要对应的模型权重。当前仓库不包含权重，直接运行会明确报错。

### 4. 重新生成数据（可选）

只有在需要修改合成数据或重新标注时才需要 GLM。创建 `.env`：

```text
ZHIPU_API_KEY=你的key
ZHIPU_MODEL=glm-5.3
GLM_REASONING_EFFORT=low
```

然后执行：

```powershell
python generate_articles.py --workers 4
python annotate_articles.py --limit 488
```

`.env` 已被 `.gitignore` 忽略，不要提交到仓库。`run.py prepare` 只在需要从原始表重新生成 `data/sentences.csv` 时使用。

## 项目结构

```text
compare/
├── run.py                       # 预处理、筛选、抽取、规范化入口
├── filter.py                    # 层级句向量筛选器
├── gts.py                       # GTS 模型、Dataset 与解码
├── train.py                     # 筛选器 / GTS 训练入口
├── generate_articles.py         # 生成合成文章
├── annotate_articles.py         # 句子标注与训练数据组装
├── README.md
├── requirements.txt
├── .env                         # 本地配置，不提交
├── input/
│   ├── raw/                     # 9 张原始表，git 忽略
│   └── lexicons/                # 品牌、方面、品类词典
├── data/
│   ├── sentences.csv            # 原始表拆分结果，git 忽略
│   ├── articles.jsonl           # 650 篇合成文章
│   ├── annotations.jsonl        # 488 篇正例标注
│   ├── manifest.json            # 数据数量与 ID 规则
│   ├── filter/                  # filter_train / valid / test.jsonl
│   └── gts/                     # gts_train / valid / test.csv
├── output/                      # 运行时产物，git 忽略
└── models/
    ├── filter/
    └── gts/
```

### 文件职责

| 路径 | 职责 |
|---|---|
| `run.py` | 预处理与推理阶段调度 |
| `filter.py` | 比较句筛选与链式分段 |
| `gts.py` | 四元组抽取与解码 |
| `train.py` | 筛选器和 GTS 训练 |
| `generate_articles.py` | 调用 GLM 生成文章 |
| `annotate_articles.py` | 生成句子标签、filter 数据和 GTS 数据 |
| `data/articles.jsonl` | 合成文章，ID 为 `a_90000001..a_90000650` |
| `data/annotations.jsonl` | 正例文章逐句标签与四元组 |
| `data/filter/` | `u/l/t` 三类筛选训练样本 |
| `data/gts/` | 带字符级 span 的四元组训练数据 |
| `output/` | 筛选、抽取、规范化结果 |

## 数据

### ID 规则

原始表拆分后：

```text
a_00012345_cut01
s_00012346_cut01
```

合成文章：

```text
a_90000001 ... a_90000650
```

合成句子单元、GTS mention 和最终聚合：

```text
a_90000004_cut02
a_90000004_cut02_01
a_90000004_quad0000001
```

`text_group_id` 取文章级 ID，如 `a_90000004`。同一篇文章的不同 `cut` 会在最终阶段合并，因此最终 `id` 是文章级编号，而不是句子级编号。

### 数据规模

| 数据集 | 数量 |
|---|---:|
| 原始拆句 | 188,084 |
| 合成文章 | 650 |
| 正例标注 | 488 |
| Filter train / valid / test | 5,155 / 2,983 / 1,493 |
| GTS train / valid / test | 1,078 / 625 / 298 |

`data/manifest.json` 记录当前数量、ID 规则和数据文件清单。

### 仓库包含的数据

| 文件 | 是否随仓库提供 | 用途 |
|---|---|---|
| `data/filter/*.jsonl` | 是 | 筛选器训练与验证 |
| `data/gts/*.csv` | 是 | GTS 训练与验证 |
| `data/articles.jsonl` | 是 | 文章源数据与回溯 |
| `data/annotations.jsonl` | 是 | 句子标签与四元组回溯 |
| `data/manifest.json` | 是 | 数据数量与 ID 说明 |
| `data/sentences.csv` | 否 | 原始表拆分中间产物，可重新生成 |
| `input/raw/*.xlsx` | 否 | 原始采集数据 |
| `output/*` | 否 | 推理运行时产物 |
| `models/*` | 否 | 训练后生成的权重 |

训练只需要 `data/filter/` 和 `data/gts/`；`articles` 与 `annotations` 用于检查和重新组装数据。

### Filter 数据格式

`data/filter/*.jsonl` 每行一个训练样本：

```json
{"task": "u", "text": "当前句", "label": 1}
{"task": "l", "context": ["上一句"], "next": "下一句", "label": 1}
{"task": "t", "title": "标题", "current": "当前句", "label": 0}
```

### GTS 数据格式

`data/gts/*.csv` 至少包含：

```text
text, subject_start, subject_end, object_start, object_end, aspect_theme, score
```

`text` 使用 `[SEP]` 分隔标题和正文句子，`subject_start/end`、`object_start/end` 是字符级偏移，`score` 表示主体相对客体的得分，范围为 `-1..1`。

## 工作流

### 1. 预处理

```powershell
python run.py prepare
```

读取 `input/raw/` 下的 9 张原始表，过滤过短文本，按 `。？！` 切句，输出 `data/sentences.csv`。

### 2. 生成文章

```powershell
python generate_articles.py --workers 4
```

按训练 350、验证 200、测试 100 生成 650 篇文章，支持断点续跑。完成后可继续生成缺失文章，不会重复处理已有 ID。

### 3. 标注并组装数据

```powershell
python annotate_articles.py --limit 488
```

给正例文章补逐句比较标签和四元组，生成 `data/annotations.jsonl`、`data/filter/` 和 `data/gts/`。

### 4. 训练

```powershell
python train.py filter --device cpu `
  --train-data data/filter/filter_train.jsonl `
  --valid-data data/filter/filter_valid.jsonl

python train.py gts --device cuda `
  --train-data data/gts/gts_train.csv
```

训练产物：

```text
models/filter/filter_model.pt
models/filter/filter_config.json
models/filter/tokenizer files

models/gts/gts_model.pt
models/gts/gts_config.json
models/gts/tokenizer files
```

### 5. 推理

```powershell
python run.py filter --device cuda
python run.py extract --device cuda
python run.py normalize
```

筛选结果写入 `output/comparison_sentences.csv`，字段为：

```text
id, compare_sentence
```

最终结果写入 `output/quads_final.xlsx`，字段为：

```text
id, 比较客体, 比较方面, 比较得分, origin_id
```

## 设计要点

- 筛选器使用共享 BERT，包含 U、L、T 三个判断头；T 只在 U/L 都不成立时作为标题兜底。
- 链式分段最多 5 句、512 tokens；超长比较在块边界截断，不把句子截断。
- 标题型块保留标题，显式块保留空的标题段，块内用 `[SEP]` 保持结构。
- GTS 一次处理 `title + chain`，按出现顺序定位主体和客体，输出 5 类方面和主体相对得分。
- 规范化阶段处理实体、方面、方向和得分，按 `text_group_id` 合并；同一组最多使用 5 条 mention 计算均值。
- 最终 `origin_id` 取组内最早的原始 mention ID。

## 配置

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `ZHIPU_API_KEY` | 智谱 API key | 无，生成/标注必填 |
| `ZHIPU_MODEL` | 生成模型 | `glm-5.3` |
| `ZHIPU_BASE_URL` | 智谱兼容接口 | `https://open.bigmodel.cn/api/paas/v4/` |
| `GLM_REASONING_EFFORT` | 思考强度 | `low` |
| `GLM_MAX_TOKENS` | 单次生成上限 | `4096` |
| `HF_ENDPOINT` | Hugging Face 镜像 | `https://hf-mirror.com` |

## Windows 说明

PowerShell 中激活虚拟环境：

```powershell
.venv\Scripts\activate
```

如果 `python` 不在 `PATH` 中，使用完整解释器路径。训练和推理命令中的 `--device` 写在子命令之后：

```powershell
python train.py filter --device cuda --train-data ...
python run.py filter --device cuda
```

## AI 数据说明

本项目的文章、句子标签和四元组包含 GLM 生成内容，属于合成数据，不是人工金标。

- 生成数据可以用于训练和流程验证，但不能直接当作真实评测结论。
- 用于正式训练前应抽取样本检查文章结构、span、方面和得分方向。
- 使用 LLM 修改代码或数据时，提交者需要理解改动并对结果负责。
- 不要把 API key、原始数据和模型权重提交到公开仓库。

## 已知限制

- 筛选器和 GTS 尚未训练，当前没有可用的性能指标。
- `filter` 和 `extract` 在缺少权重时会报错，这是预期行为。
- 合成数据存在标题重复、句式相似和标注偏差，需要后续人工抽检。
- 原始表、`data/sentences.csv`、模型权重和 `output/` 默认不进入 Git。
- 当前未附许可证文件，对外使用或再分发前请先确认授权方式。

## 贡献

1. Fork 仓库并创建分支。
2. 修改代码或数据后，至少运行一次编译检查：

   ```powershell
   python -m py_compile run.py filter.py gts.py train.py generate_articles.py annotate_articles.py
   ```

3. 如果修改了数据结构，同步更新 `data/manifest.json` 和 README。
4. 提交 Pull Request，并说明数据来源、改动范围和验证方式。

## 问题反馈

如果发现数据错误、ID 不一致或流程无法运行，请在仓库 Issues 中提交，附上：

- 复现命令
- 输入文件路径
- 错误日志
- 相关 ID 或样本

README 结构参考 [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt)。
