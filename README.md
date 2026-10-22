# 黄酒比较语义筛选与四元组抽取

从多平台黄酒文本中筛选具有比较语义的句子，抽取 `(主体, 客体, 方面, 得分)`，再规范化输出“黄酒 vs 明确酒类主体”的比较四元组。

本 README 按**新方案**编写。核心代码已部署，模型权重与数据补齐尚未完成；在训练完成前，不把未验证结果当作已有指标。

## 项目特点

- 预处理阶段统一编号，保留文章级上下文窗口。
- 筛选器使用层级句向量编码 + TransformerEncoder 融合上下文，再用下一句 token 查询上下文 token。
- 标题只作为 `U/L` 失败后的 T 兜底，不参与每轮循环。
- 筛选输出只保留比较句，不输出 `is_compare` 字段。
- GTS 从上下文窗口中抽取主体、客体、方面和主体相对得分。
- 方面固定为 5 类：文化认知、消费场景、产品体验、竞争压力、其他。
- 规范化全部放在四元组抽取之后。
- 最终输出只保留 `id`、比较客体、比较方面、比较得分、`origin_id`。

## 目标流程

```text
9 张原始表
  ↓
预处理：切句、编号、拼接上下文窗口
  ↓
筛选：层级句向量 + 交叉注意力判断比较语义，标题只做 T-last 兜底，只输出正例
  ↓
GTS：从窗口文本抽取四元组
  ↓
规范化：错别字、品牌归一、跨品类筛选、方面 5 类、黄酒主体
  ↓
最终四元组表
```

各阶段只读取上一阶段产物，不跨阶段修改文件。

## 数据契约

### 单一 ID

```text
a_{8位文本序号}_{2位子句序号}
s_{8位文本序号}_{2位子句序号}
```

- `a`：article，做过切分的文章级文本；
- `s`：sentence，没有切分的句级文本；
- 文本序号在长度过滤后按处理顺序分配，补零；
- 子句序号从 `01` 开始；
- 没有切分的文本也编号 `_01`。

GTS 原始 mention ID：

```text
{预处理id}_{2位mention序号}
```

示例：

```text
a_00012345_01_01
a_00012345_01_02
a_00012345_02_01
s_00012346_01_01
```

合并后 ID：

```text
{text_group_id}_quad{7位序号}
```

示例：

```text
a_00012345_quad0000001
a_00012345_quad0000002
s_00012346_quad0000001
```

- `text_group_id` = 预处理 ID 的前两段，如 `a_00012345`；
- `quad` 序号不足 7 位补零；
- 合并后 ID 按组内最早的原始 mention ID 升序编号。

### 各阶段字段

| 阶段 | 输出字段 |
|---|---|
| 预处理 | `id, sentence, origin_id` |
| 筛选/分段 | `id, simple_sentence` |
| GTS | `id, 比较主体, 比较客体, 比较方面, 比较得分` |
| 合并/规范化 | `id, 比较客体, 比较方面, 比较得分, origin_id` |

字段含义：

- 预处理 `origin_id`：原始数据 `编号`；
- 最终 `origin_id`：合并组内最早的原始 mention ID；
- 两者同名但阶段不同，最终来源可沿 `origin_id → quads_raw → 预处理 id` 回查。

## 预处理

### 输入

| 数据 | 说明 |
|---|---|
| `input/raw/*.xlsx` | 9 张原始表 |
| `编号` | 原始数据编号 |
| `标题` | 文章标题，部分来源为空 |
| `文本内容` | 原始正文或评论 |
| `来源` / `平台` / `品牌` | 元数据 |

### 输出

`sentences.csv`：

| 字段 | 说明 |
|---|---|
| `id` | 句子主键：`a/s_{文本序号}_{子句序号}` |
| `sentence` | 标题、上下文、后句组成的单字段窗口 |
| `origin_id` | 原始 `编号` |

`sentence` 的格式：

```text
a 类：[title] [SEP] [context] [SEP] [next]
s 类：[SEP] [context] [SEP]
```

- `context`：当前句；
- `next`：后一句；
- `title`：文章标题，可为空；
- 不记录前一句；
- 句边界统一用 `[SEP]`。

### 1. 加载与初筛

1. 加载 9 张原始表；
2. 丢弃原始文本长度小于 6 的记录；
3. 长度过滤只做一次，不对切分后的子句再次过滤。

### 2. 判断文章级

先合并连续的 `。？！`，再统计剩余句子结束标记：

```text
>= 2 → article
< 2  → sentence
```

连续的情绪标点不单独算多句：

```text
说的对！！！   → sentence
好喝！？       → sentence
第一段。第二段。 → article
```

### 3. 切分与编号

- article 按 `。？！` 切句，保留原文标点；
- 每个子句分配 `a_{8位文本序号}_{2位子句序号}`；
- 未切分文本分配 `s_{8位文本序号}_01`；
- 文本序号在长度过滤后按处理顺序分配，补零；
- 同一来源文本的子句共享 `origin_id`。

示例：

```text
PDD384 是第 12345 个处理的文本，切出两句：
a_00012345_01
a_00012345_02
origin_id = PDD384
```

### 4. 窗口构造

article 类：

```text
[title] [SEP] [当前句] [SEP] [后一句]
```

- 第一句没有前句；
- 最后一句：`next` 为空；
- 标题为空时，第一段留空；
- 标题只出现一次，所有子句共享。

sentence 类：

```text
[SEP] [当前句] [SEP]
```

### 5. 上下文 mask

筛选器解析 `[SEP]` 后得到：

```text
context_mask = [has_title, has_next]
```

- `[1,1]`：有标题、有后句；
- `[1,0]`：有标题、无后句；
- `[0,1]`：无标题、有后句；
- `[0,0]`：句级短文本。

`context_mask` 不写入数据文件，只作为筛选器内部输入。

空标题和空后句不进行 BERT 编码，直接使用零向量；对应 gate 为 0。

## 筛选器与链式分段

### 输入

| 数据 | 说明 |
|---|---|
| `sentences.csv` | 预处理输出 |
| `id` | 句子主键 |
| `sentence` | `[title] [SEP] [context] [SEP] [next]` |
| `context_mask` | 内部解析得到 `[has_title, has_next]` |

筛选器按 `[SEP]` 解析出：

```text
title, context, next
```

- `title` 不参与每次循环，只在 U/L 都失败后作为 T 兜底；
- `context` 是当前句；
- `next` 是候选后句；
- 空段不编码，gate=0。

### 输出

`comparison_sentences.csv`：

| 字段 | 说明 |
|---|---|
| `id` | 块首句 ID |
| `simple_sentence` | 链式分段后的比较文本 |

只保留比较块，不输出负例；不保留 `sentence` 和 `is_compare`。

### 句子编码与上下文融合

```text
H_i = BERT(s_i)
v_i = Pool(H_i)

C = [v_start, ..., v_end] + PositionEncoding
c = TransformerEncoder(C)
```

- 每句独立编码；
- 句向量进入 TransformerEncoder；
- 保留句边界和位置编码；
- `context` 为已纳入链的句子集合。

### 与下一句计算

```text
Q = H_next
K/V = H_context
u = CrossAttention(Q, K, V)
link = MLP([u, c, v_next])
```

- 下一句 token 查询 context token；
- 句向量 `c` 提供全局上下文；
- `link` 输出 0/1。

### 链式分段

定义：

```text
U(i)：第 i 句本身是否有比较语义
L(context, next)：累计 context 与下一句是否仍属于同一比较
T(i)：第 i 句是否需要结合标题才构成比较
```

起链：

```text
U(i) = 1
或 L(i, i+1) = 1
或（U/L 均失败，且存在标题，规则门控通过，T(i) >= 0.9）
```

续链同时满足：

1. 存在后句；
2. 当前链少于 5 句；
3. 融合后不超过 512 tokens；
4. `L(context, next) = 1`。

算法：

```text
i = 1

while i <= n:
    start = None

    if U(i):
        start = i
    elif i < n and L(i, i + 1):
        start = i
    elif has_title and rule_gate(i) and T(i) >= 0.9:
        start = i

    if start is None:
        i += 1
        continue

    end = start

    while end < n:
        if (end - start + 1) >= 5:
            break
        if token_len(context(start, end) + next) > 512:
            break
        if L(context(start, end), end + 1) != 1:
            break
        end += 1

    emit(start, end)
    i = end + 1
```

- 最多 5 句，不含标题；
- 达到 5 句或 512 tokens 时停止；
- 不纳入第 6 句；
- 每句只消费一次；
- 块 ID 用引起句 ID。

### 标题兜底判断 T

T 只在 `U` 和 `L` 都失败后启用，避免标题在每轮循环中造成假阳性。

启用条件：

```text
U(i) = 0
L(i, i+1) = 0
标题非空
规则门控通过
T(i) >= 0.9
```

#### 规则门控

规则只做宽泛词类判断，不针对具体品牌或标题内容。

| 规则 | 条件 | 动作 |
|---|---|---|
| A | 标题有实体 + 本句有代词/省略 | 进入 T |
| B | 本句有比较词/评价词 + 标题有实体 | 进入 T |
| C | 两者都不满足 | 拒绝 T |

- 实体：名词、专名、品牌、品类等宽泛实体；
- 代词：它、这款、该品牌、其、这个、该产品、他们；
- 省略：缺少明确主体或客体；
- 比较词：比、不如、更、相比、较、没有…那么、胜过、优于；
- 评价词：柔和、刺激、便宜、贵、上头、清爽、醇厚、难喝、好喝等。

不使用“行业/周报/观察”黑名单。

#### 模型

```text
共享 BERT
 ├─ U 头：本句是否有比较语义
 ├─ L 头：context + next 是否仍构成比较
 └─ T 头：title + current 是否构成比较
```

T 输入：

```text
[title] [SEP] [current]
```

- 当前句 token 查询标题 token；
- 与 U/L 共用编码器；
- T 只在规则门控通过后调用；
- 句级数据没有标题，跳过 T。

#### 数据

| 数据集 | 隐式比较正例 | 负例 | 合计 |
|---|---:|---:|---:|
| 训练 | 500 | 800 | 1,300 |
| 验证 | 300 | 800 | 1,100 |
| 测试 | 150 | 400 | 550 |

- 按标题/文章分组划分，禁止同一标题跨集；
- 正例允许 LLM 生成 + 人工抽检；
- 每篇正例文章至少提供 2 个标题型正例；
- 验证/测试也使用生成数据；
- 原始 : 生成 = 5:5，最低原始占比 40%；
- 额外报告原始-only 子集；
- 负例覆盖标题有比较但本句无关、仅共享泛化词、无实体关联等情况；
- 负例不足时，从正例文章补硬负例；
- 句级数据没有标题，不进入 T 模块。

#### 阈值与指标

初始阈值：

```text
T >= 0.9
```

指标：

```text
T_precision
T_FPR = FP / (FP + TN)
T_FDR = FP / (TP + FP)
implicit_recall
```

目标：

```text
precision >= 0.90
FPR <= 0.10
FDR <= 0.10
```

如果 precision 不达标：

1. 先补难负例；
2. 仍不达标再提高阈值；
3. 不把阈值降到 0.9 以下。

### 输出文本

标题是否保留，取决于它在筛选过程中有没有真正起作用。

#### 标题被使用

起链由 `T` 触发，即：

```text
U = 0
L = 0
规则门控通过
T >= 0.9
```

输出保留标题：

```text
[title] [SEP] [s_start] [SEP] [s_start+1] ... [SEP] [s_end]
```

#### 标题未被使用

起链由 `U` 或 `L` 触发，标题没有参与判断。  
输出不保留标题，但保留空的标题段，维持固定结构：

```text
[SEP] [s_start] [SEP] [s_start+1] ... [SEP] [s_end]
```

#### 其他规则

- 标题只出现一次；
- 标题为空时，第一段保持为空；
- 标题被 `T` 使用过才保留；
- 标题型块的标题随 `simple_sentence` 进入 GTS；
- 显式块的标题段为空；
- `simple_sentence` 保留 `[SEP]` 句边界。

## 数据生成

### 文章级生成

使用 LLM 学习原始数据结构，生成 **650 篇**文章：

| 数据集 | 文章数 | 正例 75% | 负例 25% |
|---|---:|---:|---:|
| 训练 | 350 | 263 | 87 |
| 验证 | 200 | 150 | 50 |
| 测试 | 100 | 75 | 25 |

- 按 `article_id` 分组划分；
- 标题不跨集；
- 每篇 3–6 句，单篇不超过 512 tokens；
- 正例文章每篇至少 2 个标题型正例；
- 负例文章不进入 GTS 正例训练，只用于 filter/T 负例。

生成文章需要覆盖：

- 标题显式比较 / 标题仅主题；
- 正文显式比较 / 隐性比较；
- 有代词 / 无代词；
- 跨品类 / 品牌间 / 产品间；
- 新闻 / 论坛 / 社媒风格。

### 四元组生成

文章生成后，再由 LLM 基于文章生成四元组：

```text
subject_text
object_text
subject_start/end
object_start/end
aspect_theme
score
sentence_index
source: title / body
is_implicit
```

校验：

- span 必须能在原文定位；
- 方面必须在 5 类内；
- 得分在 `-1..1`；
- 代词必须可解析；
- 标题缺失字段先到正文补；
- 正文也没有则丢弃该四元组。

### 句级生成

句级数据沿用旧生成策略：

```text
generated_filter.csv：1,501
generated_quad.csv：1,200
```

- 标记 `is_synthetic=1`；
- 只用于 filter/GTS 训练；
- 不进入 T；
- 按训练比例采样，不全部灌入。

### 比例

训练集统一：

```text
原始 : 生成 = 5 : 5
```

最低要求：

```text
原始 >= 40%
```

- 生成数据不能覆盖原始数据；
- 按原始来源、品牌、比较类型分层生成；
- 验证/测试也使用生成数据时，额外报告原始-only 子集；
- GTS 主测试集仍以真实数据为主，生成测试集只作辅助诊断。

### 抽检

650 篇抽 **65 篇**：

- 按来源风格、比较类型、标题显式/主题、正/负例分层；
- 抽中文章的全部四元组都检查；
- 质量门槛：
  - 文章结构错误率 ≤10%
  - span 错误率 ≤10%
  - 方面错误率 ≤15%
  - 得分方向错误率 ≤10%

不达标时修改提示词、重新生成、重新抽检。

## GTS

### 输入

```text
simple_sentence
```

- 一次处理整段 `title + chain`；
- 标题不做特殊标签；
- 不单独跑 `GTS(title)`；
- 不做标题/链两次抽取；
- 显式块第一段为空；
- 标题型块第一段为标题。

### 抽取目标

```text
(比较主体, 比较客体, 比较方面, 比较得分)
```

- 主体/客体按在文本中的出现先后确定；
- 得分表示前者相对后者的得分；
- 交换主体/客体时，得分取反；
- 方面固定 5 类；
- 允许跨段四元组：
  - 主体来自标题；
  - 客体来自链；
  - 方面/得分来自链。

### 字段补全

逐字段补全：

```text
subject / object / relation / aspect / score
  ↓
标题有 → 标题
标题没有 → 链里找
链也没有 → 丢弃该四元组
```

- 补全时要求同一方面、同一方向；
- 标题失败只丢标题四元组，不丢整块；
- 链四元组继续保留；
- 链也抽不出四元组时，才丢弃整块。

### 模型结构

```text
BERT 编码
 ├─ BIO 头：主体 span、客体 span
 ├─ 方向头：S→O / O→S，保留为辅助
 ├─ 方面头：5 分类
 └─ 得分头：-1 到 1 回归
```

- 方向头保留，但不改变最终主体/客体排序；
- 主体/客体仍按出现顺序输出；
- 方向头用于辅助关系判断。

### 抽取阶段简化

GTS 只做候选生成：

- 不做句内去重；
- 不做标题/链四元组合并；
- 不做原始文本级合并；
- 不做分数平均；
- 不生成 canonical key。

保留：

- span 有效性；
- 主体/客体不能完全相同；
- pair 阈值，先跑 0.1；
- 每块最多 30 个主体-客体 pair，超过取前 30；
- `source=title/chain`。

### 训练数据

- 真实数据与生成数据按约 5:5 配比；
- 生成数据来自 350 篇训练文章和旧 `generated_quad.csv`；
- 生成数据覆盖跨品类、品牌间、产品间；
- 主测试集仍以真实数据为主；
- 生成测试集只作辅助诊断；
- 训练集需要包含跨句比较样本；
- 比例在真实数据完成回收和重标后重新计算。

### 输出

原始候选四元组：

```text
id
block_id
subject
object
aspect
score
subject_source = title / body
object_source = title / body
source = title / chain
mention_id
```

不在 GTS 阶段去重、合并或平均。

## 规范化

规范化全部放在 GTS 抽取之后，顺序固定：

```text
1. 错别字纠正
2. 黄酒品牌名统一替换为“黄酒”
3. 只保留“黄酒 vs 明确酒类主体”
4. 比较方面固定为 5 类
5. 统一黄酒为比较主体
```

### 第 1 步：错别字纠正

- 只纠正实体名和方面词；
- 不随意改写整句；
- 原始文本与修正文本分开保留，便于追溯。

### 第 2 步：黄酒品牌名 → 黄酒

- 17 个主要品牌及别名替换为 `黄酒`；
- 绍兴酒、老酒、花雕等黄酒别称替换为 `黄酒`；
- 句级数据中指代无法解析时默认 `黄酒`；
- 文章级数据中指代无法确定时直接舍弃。

### 第 3 步：只保留黄酒 vs 明确酒类主体

保留：

- 白酒、啤酒、红酒、葡萄酒、米酒、清酒、洋酒、果酒、鸡尾酒、RIO；
- 具体酒类品牌。

排除：

- 白酒行业、白酒股、白酒市场、白酒企业、白酒板块；
- 产业、赛道、概念等衍生对象；
- 两边都是黄酒的品牌间比较；
- 料酒、气泡水、饮料等非目标酒类。

### 第 4 步：方面固定 5 类

```text
文化认知 / 消费场景 / 产品体验 / 竞争压力 / 其他
```

GTS 已输出 5 类，这一步只做合法性校验和兜底。

### 第 5 步：统一黄酒为主体

```text
(白酒, 黄酒, 方面, s)
→ (黄酒, 白酒, 方面, -s)
```

- 交换主体/客体；
- 得分取反；
- 最终主体固定为黄酒。

### 最终输出

```text
id, 比较客体, 比较方面, 比较得分, origin_id
```

- `id`：合并后的 `{text_group_id}_quad{7位序号}`；
- `origin_id`：组内最早的原始 mention ID；
- 不输出主体、span 和 raw 文本。

## 原始文本级去重与聚合

GTS 阶段允许重复抽取，去重放在规范化之后。

### text_group_id

```text
text_group_id = id 的前两段
```

示例：

```text
a_00012345_01_01 → a_00012345
s_00001234_01_01 → s_00001234
```

### key

```text
text_group_id
+ subject_norm
+ object_norm
+ aspect_theme
+ direction_norm
```

- 同一原始文本的不同子句合并；
- 同一子句的多个四元组合并；
- 标题四元组和链四元组合并；
- 不同原始文本不合并；
- 实体和方面先规范化，再去重；
- 强度不进入 key。

### origin_id

合并组内最早的原始四元组 ID：

```text
a_00012345_01_01
a_00012345_01_02
a_00012345_02_01
→ origin_id = a_00012345_01_01
```

- 不是句子 ID；
- 不是原始 `编号`；
- 可按该 ID 回查 `quads_raw`，再回查预处理记录。

### 合并后 ID

按 `origin_id` 升序排序，再编号：

```text
a_00012345_quad0000001
a_00012345_quad0000002
```

- 7 位序号，不足补零；
- 超过 7 位自然扩展；
- 标题/链合并后仍取组内最早原始 ID。

### 分数聚合

1. 先按唯一 mention 去重；
2. 同一 canonical key 的分数取平均；
3. 同一重复最多使用 5 条：

```text
n_raw_mentions
n_used_mentions = min(n_raw_mentions, 5)
mean_score
overflow = n_raw_mentions - n_used_mentions
```

4. 保留 `std_score / min_score / max_score`；
5. 保留来源统计 `source=title/chain`。

### 评测

- 预测和 gold 都按 `text_group_id + canonical_key` 去重；
- 主指标是原始文本级唯一四元组 recall；
- 重复率只作诊断，不作为主要惩罚项；
- precision/F1 也按去重后计算；
- 训练集同样按 canonical key 去重，避免重复样本加权。

## 项目结构

当前结构：

```text
compare/
├── run.py                       # CLI 入口：预处理、筛选调度、抽取、规范化
├── filter.py                    # 层级句向量筛选器
├── gts.py                       # GTS 模型、Dataset 与解码
├── train.py                     # 筛选器 / GTS 训练入口，--device 选择 CPU/GPU
├── README.md
├── sentences.csv                # 预处理后的句级窗口
├── comparison_sentences.csv     # 筛选后的比较句（空表占位）
├── quads_raw.csv                # GTS 原始四元组（空表占位）
├── quads_final.xlsx             # 规范化后的最终四元组（空表占位）
├── run.json                     # 配置、指标、文件哈希
├── run.log                      # 运行日志
├── input/
│   ├── raw/                     # 9 张原始表
│   ├── supplement/              # 数据补齐阶段使用，当前为空
│   └── lexicons/                # 品牌、方面、品类词典
└── models/
    ├── filter/                  # 筛选器权重，当前为空
    └── gts/                     # GTS 权重，当前为空
```

### 文件职责

| 路径 | 职责 |
|---|---|
| `run.py` | 命令行入口与阶段调度 |
| `filter.py` | 层级句向量筛选模型 |
| `gts.py` | 四元组抽取模型 |
| `train.py` | 筛选器与 GTS 训练入口 |
| `sentences.csv` | 预处理后的句级窗口 |
| `comparison_sentences.csv` | 筛选后的比较句 |
| `quads_raw.csv` | 原始抽取结果 |
| `quads_final.xlsx` | 规范化后的最终结果 |
| `input/raw/` | 原始数据 |
| `input/supplement/` | 数据补齐阶段使用 |
| `input/lexicons/` | 品牌、方面、品类词典 |
| `models/filter/` | 筛选器权重 |
| `models/gts/` | GTS 权重 |

## 环境要求

- Python 3.12
- `pandas`
- `numpy`
- `openpyxl`
- `scikit-learn`
- `torch`
- `transformers`
- `openai`

训练和推理设备由 `--device` 显式指定，支持 `cpu`、`cuda`、`cuda:N`，默认 `cpu`。指定 CUDA 但当前环境不可用时直接报错，不自动回退到 CPU。首次训练会下载 `hfl/chinese-macbert-base`。

## 安装

```powershell
cd C:\Users\xx\Desktop\ai-lab\huangjiu\文档\compare

# 使用项目已有虚拟环境（可选）
..\..\.venv\Scripts\activate

pip install pandas numpy openpyxl scikit-learn torch transformers openai
```

设置 GLM 环境变量，不要把 key 写入代码：

```powershell
$env:ZHIPU_API_KEY = "<你的智谱 key>"
$env:ZHIPU_MODEL = "glm-4.7-flash"
```

## 配置说明

| 环境变量 | 含义 | 默认值 | 是否必需 |
|---|---|---|---|
| `ZHIPU_API_KEY` | 智谱 API key | 无 | 调用 LLM 时必需 |
| `ZHIPU_MODEL` | GLM 模型名 | `glm-4.7-flash` | 否 |
| `ZHIPU_BASE_URL` | 智谱兼容接口地址 | `https://open.bigmodel.cn/api/paas/v4/` | 否 |
| `GLM_THINKING` | GLM-4.x 是否关闭思考 | `disabled` | 否 |
| `GLM_REASONING_EFFORT` | GLM-5.3 推理强度：`low/high/max` | `low` | 否 |
| `HF_ENDPOINT` | 模型下载镜像 | `https://hf-mirror.com` | 否 |

## 使用方式

```powershell
# 1. 预处理
python run.py prepare

# 2. 筛选比较句；设备显式指定
python run.py filter --device cpu

# 3. GTS 抽取
python run.py extract --device cuda

# 4. 规范化
python run.py normalize
```

训练入口已部署，但当前没有训练数据，尚未执行训练：

```powershell
python train.py filter --device cpu --train-data data/filter_train.jsonl --valid-data data/filter_valid.jsonl
python train.py gts --device cuda --train-data data/gts_train.csv
```

`--device` 写在子命令之后，可取 `cpu`、`cuda`、`cuda:1`。筛选器训练数据为 JSONL，每条样本带 `task=u/l/t` 和 `label`；GTS 训练数据为 CSV，包含 `text`、四类 span 偏移、`aspect_theme`、`score`。

数据补齐与 `generate/annotate` 尚未部署，`input/supplement/` 当前为空。补齐完成后再补对应命令。

## 评测方案

### 筛选器

- U/L：精确率、召回率、F1；
- 当前句锚定准确率；
- 无上下文 vs 层级上下文融合的对照；
- 文章级与句级分组统计；
- T：precision、FPR、FDR、implicit recall；
- T-last 与无标题兜底的对照；
- 标题可用/不可用子集分别统计。

### GTS

- 四元组 exact / partial F1；
- 方面准确率 / Macro-F1；
- 得分 MAE；
- 得分符号准确率；
- 方向准确率（辅助）；
- 跨句样本单独统计；
- 标题型块 vs 显式块分别统计；
- 原始文本级唯一四元组 recall / precision / F1；
- 重复率只作诊断，不作主要惩罚；
- `max_pairs=30` 与阈值 0.1 的召回/精度对照。

### 端到端

- 重复输出率；
- 原始文本级覆盖率；
- 最终四元组数量；
- 软上限 5 前后的均值差异；
- 跨品类子集指标。

GTS 主测试集使用真实数据，生成测试集只作辅助诊断。  
T 的验证/测试可以使用生成数据，但需要额外报告原始-only 子集。  
生成数据不能掩盖原始数据分布。

## 当前状态

- 核心代码已迁移到新方案：`run.py`、`filter.py`、`gts.py`、`train.py`；
- `prepare` 已跑通，当前 `sentences.csv` 共 188,084 条句级窗口；
- `comparison_sentences.csv`、`quads_raw.csv`、`quads_final.xlsx` 已创建为空表占位；
- 未做数据补齐，`input/supplement/` 当前为空；
- 训练入口支持 `--device cpu/cuda/cuda:N`，但未准备训练数据、未执行训练；`models/filter/`、`models/gts/` 当前为空；
- `filter` 和 `extract` 在模型权重缺失时会明确报错；
- 在模型训练完成前，不提供新方案性能指标。

## 已知限制

- 筛选器依赖标题与后句，不再需要前句；预处理保留标题/当前句/后句窗口；
- 当前不做抽取式简化，`simple_sentence` 是链式块文本；
- 链式分段最多 5 句、512 tokens，超长比较会被截断在块边界；
- 标题覆盖率低，T 兜底主要影响新闻/论坛数据；
- GTS 阶段允许重复抽取，重复在原始文本级去重；
- 同一重复最多使用 5 条计算均值；
- 真实四元组数据仍需回收、重标和人工复核；
- GTS 主测试集使用真实数据；T 验证/测试可使用生成数据，但需报告原始-only 子集；
- 当前未附许可证文件。

## 路线图

1. [已完成] 实现新的预处理与单一 ID；
2. [已完成] 实现层级句向量筛选器代码；
3. [已完成] 实现 T-last 标题兜底与规则门控；
4. [已完成] 实现筛选阶段的指代消解与标题保留；
5. [已完成] 实现 GTS 一次处理 title + chain；
6. [已完成] 实现逐字段补全与标题/链保留；
7. [已完成] 将得分统一为主体相对客体；
8. [已完成] 实现规范化 1–5 步；
9. [已完成] 实现原始文本级软上限 5、去重与导出；
10. [待完成] 生成 650 篇文章并抽检 65 篇；
11. [待完成] 回收并重标真实四元组；
12. [已完成] 部署 `train.py` 与 CPU/GPU `--device` 入口；
13. [待完成] 准备训练数据并训练筛选器与 GTS；
14. [待完成] 跑真实测试集与上下文消融。

## 许可证

本目录当前未附许可证文件。对外使用或再分发前，请先确认项目整体的授权方式。
