# 黄酒比较语义挖掘

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow)](https://huggingface.co/docs/transformers/)

从中文平台文本中识别黄酒相关的比较语义，抽取“主体—客体—方面—得分”四元组，并按原始文本聚合，供后续统计分析使用。

训练数据以 9 个平台的原始数据为主体，LLM 生成数据只用于补充训练量。验证集和测试集只使用原始数据。

## 当前状态

| 项目 | 数量 |
|---|---:|
| 原始标注单元 | 1,000 |
| 生成训练单元 | 599 |
| GLM 5.3 Flash 标注 | 1,599 |
| Filter train / valid / test | 1,384 / 903 / 1,132 |
| GTS train / valid / test | 1,722 / 182 / 168 |
| 训练集 raw : generated | 5 : 5 |
| GTS 正 : 负 pair | 534 : 1,188（约 1 : 2.22） |
| 模型权重 | 未训练，不包含在仓库中 |

测试集只使用原始数据，生成数据只进入训练集。

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

GPU 训练请按 CUDA 版本安装对应 PyTorch。

### 2. 训练

训练数据已经在仓库中，训练不需要 GLM key。

```powershell
python train.py filter --device cuda `
  --train-data data/filter/filter_train.jsonl `
  --valid-data data/filter/filter_valid.jsonl

python train.py gts --device cuda `
  --train-data data/gts/gts_train.csv
```

没有 GPU 时使用 `--device cpu`。显存不足时降低 `--batch-size`。

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

```powershell
python run.py filter --device cuda
python run.py extract --device cuda
python run.py normalize
```

也可以：

```powershell
python run.py all --device cuda
```

`filter` 和 `extract` 需要对应模型权重；当前仓库不包含权重。

Filter 阈值需要在原始验证集上校准，命令行支持：

```powershell
python run.py filter --device cuda `
  --u-threshold 0.5 `
  --l-continue-threshold 0.4 `
  --title-gate-mode recall
```

`--title-gate-mode recall` 会扩大 T 的 gate 覆盖，但需要同时检查 precision，不能直接在测试集上调。

### 4. 评估

训练完成后，在原始测试集上评估：

```powershell
python evaluate.py --device cuda
```

结果写入 `output/evaluation.json`。

评估口径：

- Filter：U/L/T 的 precision、recall、F1
- GTS：relation precision/recall/F1、方面准确率、得分 MAE、得分符号准确率、exact、partial
- exact：主体、客体、方面、得分全部正确
- 得分 exact 容差：`±0.05`
- 测试集不参与训练或调参

## 验收目标

以下是目标值，不是当前结果。测试集只用于最终评估，不参与调参。

### Filter

| 指标 | 目标 |
|---|---:|
| U/L precision | ≥ 0.85 |
| U/L recall | ≥ 0.80 |
| U/L F1 | ≥ 0.82 |
| 当前句锚定准确率 | ≥ 0.85 |
| T precision | ≥ 0.90 |
| T FPR | ≤ 0.10 |
| T FDR | ≤ 0.10 |
| T implicit recall | ≥ 0.70 |

### GTS

| 指标 | 目标 |
|---|---:|
| pair relation precision | ≥ 0.85 |
| pair relation recall | ≥ 0.80 |
| pair relation F1 | ≥ 0.82 |
| 方面准确率 | ≥ 0.85 |
| 方面 Macro-F1 | ≥ 0.80 |
| 得分 MAE | ≤ 0.20 |
| 得分符号准确率 | ≥ 0.80 |
| exact（±0.05） | ≥ 0.60 |
| partial | ≥ 0.75 |

测试集不参与训练或调参；exact 要求主体、客体、方面、得分全部正确，得分容差为 `±0.05`。

## 训练后回传模型

训练完成后，回传整个模型目录，不要只传单个 `.pt` 文件：

```text
models/filter/filter_model.pt
models/filter/filter_config.json
models/filter/tokenizer files

models/gts/gts_model.pt
models/gts/gts_config.json
models/gts/tokenizer files
```

模型文件通常较大，优先使用 GitHub Release：

```powershell
Compress-Archive -Path models -DestinationPath trained-models.zip
gh release create models-v1 trained-models.zip --title "Trained models v1"
```

主设备下载：

```powershell
gh release download models-v1 --pattern trained-models.zip
Expand-Archive trained-models.zip -DestinationPath .
```

如果每个文件都小于 100MB，可以直接提交：

```powershell
git add -f models/filter models/gts
git commit -m "chore: add trained models"
git push
```

`models/` 默认被 `.gitignore` 忽略，所以需要 `git add -f`；超过 100MB 的单个文件不要直接提交。

## 项目结构

```text
compare/
├── run.py                       # 实际推理入口
├── filter.py                    # 层级句向量筛选器
├── gts.py                       # pair 级 GTS
├── train.py                     # Filter / GTS 训练入口
├── evaluate.py                  # 原始测试集评估
├── text_utils.py                # 共享清洗和切句
├── curate_raw.py                # 原始数据筛选、去重、分组、拆分
├── generate_supplement.py       # 按原始风格生成训练补充数据
├── annotate_units.py            # raw/generated 标注
├── build_datasets.py            # Filter/GTS 数据组装
├── llm_client.py                # GLM 5.3 Flash 客户端
├── README.md
├── requirements.txt
├── input/
│   ├── raw/                     # 9 个平台原始表，git 忽略
│   └── lexicons/                # 品牌、方面、品类词典
├── data/
│   ├── raw/                     # raw train / valid / test
│   ├── generated/               # generated train
│   ├── annotations/             # raw / generated 标注
│   ├── filter/                  # Filter 训练数据
│   ├── gts/                     # pair 级 GTS 训练数据
│   └── manifest.json
├── output/                      # 推理产物，git 忽略
└── models/
    ├── filter/
    └── gts/
```

## 数据来源与 ID

正式数据只来自以下 9 个平台：

```text
BZ  DY  JD  PDD  TB  WB  XHS  XW  ZH
```

原始数据按文章、帖子或回复线程分组，再划分：

```text
raw train : raw valid : raw test = 600 : 200 : 200
```

生成数据只进入 train。

ID 规则：

```text
raw article/unit:      a_80000001 ...
generated unit:        a_81000001 ...
sentence unit:         a_80000001_cut01
GTS mention:           a_80000001_cut01_01
final aggregate id:    a_80000001_quad0000001
```

`text_group_id` 取文章级 ID。

## 标注

- 标注模型：GLM 5.3 Flash
- 原始数据和生成数据都标注
- 每条标注包含句子级 `comparison`、`title_comparison` 和四元组
- 为中和过度抽取，限制每篇最多 3 条四元组、每句最多 2 条
- 相同 subject/object/aspect 去重
- 保留 `annotator_model` 和 `annotation_version`

## Filter 数据

训练集 raw : generated = 5 : 5。

Filter 任务包含：

```text
U：当前句是否比较
L：当前链上下文下，下一句是否继续
T：标题 + 当前句是否比较
```

验证集和测试集保持原始数据自然分布。

## GTS 数据

GTS 使用 pair 级数据：

```text
正 pair : 负 pair = 1 : 2
```

训练集的 raw : generated = 5 : 5。

负 pair 来自同一文章内的交叉实体对和词典实体对；负 pair 的方面为“其他”，得分为 0。

GTS 模型保留 BIO span 抽取，并按 pair 分别预测 relation、aspect 和 score。`max_pairs=20`，在 relation 过滤和排序后截断。

不做实体名规范化。

## 配置

生成本仓库数据时使用 GLM 5.3 Flash：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | 生成/标注提供方 | `glm` |
| `ZHIPU_API_KEY` | 智谱 API key | 无，重新生成数据时必填 |
| `ZHIPU_MODEL` | 标注模型 | `glm-5.3-flash` |
| `ZHIPU_BASE_URL` | 智谱兼容接口 | `https://open.bigmodel.cn/api/paas/v4/` |
| `GLM_REASONING_EFFORT` | 思考强度 | `low` |
| `GLM_MAX_TOKENS` | 单次生成上限 | `8192` |
| `HF_ENDPOINT` | Hugging Face 镜像 | `https://hf-mirror.com` |

训练不需要 `.env`。重新生成数据时才需要配置。

## 重新生成数据（可选）

```powershell
python curate_raw.py
python generate_supplement.py --workers 2
python annotate_units.py --limit 1600 --workers 2
python build_datasets.py
```

原始文件不上传仓库；`data/` 中是处理后的训练数据。

## 已知限制

- 模型尚未训练，仓库不提供性能指标或权重。
- 合成数据存在标签噪声，正式使用前应抽检。
- 原始测试集中缺少无标题或跨句样本时，对应分组指标标记为“不支持”。
- `data/sentences.csv` 是可重新生成的推理预处理中间产物，不随训练数据上传。
- 当前未附许可证文件。

## 贡献

1. Fork 仓库并创建分支。
2. 修改后运行编译检查：

   ```powershell
   python -m py_compile run.py filter.py gts.py train.py evaluate.py text_utils.py curate_raw.py generate_supplement.py annotate_units.py build_datasets.py llm_client.py
   ```

3. 修改数据结构时同步更新 `data/manifest.json` 和 README。
4. 提交 Pull Request，说明数据来源、改动范围和验证方式。

## 问题反馈

请在仓库 Issues 中提交：

- 复现命令
- 输入文件
- 错误日志
- 相关 ID 或样本

README 结构参考 [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt)。
