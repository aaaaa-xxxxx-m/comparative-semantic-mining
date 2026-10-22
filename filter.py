"""层级句向量筛选器与链式分段。

输入：sentences.csv，字段 id / sentence / origin_id
输出：comparison_sentences.csv，字段 id / simple_sentence

模型结构：
- 共享 BERT 编码句子；
- TransformerEncoder 融合累计 context；
- 下一句 token 查询 context token；
- U / L / T 三个二分类头。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

SEP = "[SEP]"
PRONOUNS = ["它", "这款", "这个", "该品牌", "该产品", "其", "此款", "这瓶", "那瓶", "他们家", "这家"]
COMPARISON_CUES = ["比", "不如", "更", "相比", "较", "没有", "胜过", "优于", "赶不上", "逊于"]
EVALUATIVE_CUES = [
    "好喝", "难喝", "柔和", "刺激", "便宜", "贵", "上头", "清爽", "醇厚", "甜", "酸",
    "苦涩", "顺滑", "香气", "口感", "包装", "物流", "破损", "性价比", "养生", "健康",
]
MAX_SENTENCES = 5
MAX_TOKENS = 512
T_THRESHOLD = 0.9
START_THRESHOLD = 0.5


def resolve_device(requested: str) -> torch.device:
    value = str(requested or "").strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU。")
        if value.startswith("cuda:"):
            index = value.split(":", 1)[1]
            if not index.isdigit() or int(index) >= torch.cuda.device_count():
                raise ValueError(f"无效的 CUDA 设备：{requested}")
        return torch.device(value)
    raise ValueError(f"无效的 device：{requested}；只支持 cpu、cuda、cuda:N。")


@dataclass
class Unit:
    id: str
    title: str
    current: str
    next: str
    text_group: str
    index: int


def parse_sentence(sentence: str) -> Tuple[str, str, str]:
    parts = str(sentence).split(SEP)
    if len(parts) < 3:
        return "", str(sentence), ""
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def text_group_of(unit_id: str) -> str:
    parts = unit_id.split("_")
    return "_".join(parts[:2])


def build_units(frame: pd.DataFrame) -> List[Unit]:
    units: List[Unit] = []
    for idx, row in frame.iterrows():
        title, current, next_sentence = parse_sentence(row["sentence"])
        units.append(Unit(
            id=str(row["id"]),
            title=title,
            current=current,
            next=next_sentence,
            text_group=text_group_of(str(row["id"])),
            index=idx,
        ))
    return units


def find_next(units: List[Unit], position: int) -> Optional[Unit]:
    current = units[position]
    for j in range(position + 1, len(units)):
        candidate = units[j]
        if candidate.text_group != current.text_group:
            break
        return candidate
    return None


def load_lexicons(lexicon_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    brand_path = lexicon_dir / "brand_alias.csv"
    category_path = lexicon_dir / "category_terms.csv"
    brands: List[str] = []
    categories: List[str] = []
    if brand_path.exists():
        df = pd.read_csv(brand_path, encoding="utf-8-sig", dtype=str).fillna("")
        brands = [str(value) for value in df["alias"].tolist() if str(value).strip()]
    if category_path.exists():
        df = pd.read_csv(category_path, encoding="utf-8-sig", dtype=str).fillna("")
        df = df[df["is_alcohol"].astype(str).eq("1") & ~df["category"].eq("黄酒")]
        categories = [str(value) for value in df["term"].tolist() if str(value).strip()]
    return sorted(set(brands), key=len, reverse=True), sorted(set(categories), key=len, reverse=True)


def has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term and term in text for term in terms)


def title_entities(title: str, brands: Sequence[str], categories: Sequence[str]) -> List[str]:
    found: List[str] = []
    for term in list(brands) + list(categories):
        if term and term in title and term not in found:
            found.append(term)
    return found


def rule_gate(title: str, current: str, brands: Sequence[str], categories: Sequence[str]) -> bool:
    entities = title_entities(title, brands, categories)
    if not entities:
        return False

    has_pronoun = has_any(current, PRONOUNS)
    has_comparison = has_any(current, COMPARISON_CUES)
    has_evaluative = has_any(current, EVALUATIVE_CUES)
    # 省略：本句没有品牌/品类实体，也没有明确比较对象
    has_entity = has_any(current, brands) or has_any(current, categories)
    return has_pronoun or (has_comparison and has_evaluative) or (has_comparison and not has_entity)


def resolve_pronouns(title: str, sentences: Sequence[str], brands: Sequence[str], categories: Sequence[str]) -> Optional[List[str]]:
    entities = title_entities(title, brands, categories)
    if len(entities) != 1:
        return None
    entity = entities[0]
    resolved: List[str] = []
    for sentence in sentences:
        text = sentence
        for pronoun in PRONOUNS:
            text = text.replace(pronoun, entity)
        if has_any(text, PRONOUNS):
            return None
        resolved.append(text)
    return resolved


def has_pronoun(sentences: Sequence[str]) -> bool:
    return any(has_any(sentence, PRONOUNS) for sentence in sentences)


def _torch():
    import torch
    return torch


class FilterModel(torch.nn.Module):
    """共享编码器 + 上下文 Transformer + 三分类头。

    模型权重保存为 models/filter/filter_model.pt，tokenizer 保存在同目录。
    """

    def __init__(self, model_name: str = "hfl/chinese-macbert-base"):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.hidden = hidden
        self.pos_embedding = torch.nn.Embedding(MAX_SENTENCES + 1, hidden)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True,
        )
        self.context_encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.cross_attention = torch.nn.MultiheadAttention(hidden, 8, dropout=0.1, batch_first=True)
        self.u_head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2))
        self.l_head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden * 2, 2))
        self.t_head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden * 2, 2))

    def encode(self, input_ids: Any, attention_mask: Any) -> Tuple[Any, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state
        pooled = tokens[:, 0]
        return tokens, pooled

    def encode_one(self, input_ids: Any, attention_mask: Any) -> Tuple[Any, Any]:
        tokens, pooled = self.encode(input_ids, attention_mask)
        return tokens[0], pooled[0]

    def logits_u(self, current_ids: Any, current_mask: Any) -> Any:
        _, pooled = self.encode(current_ids, current_mask)
        return self.u_head(pooled)

    def logits_t(self, title_ids: Any, title_mask: Any, current_ids: Any, current_mask: Any) -> Any:
        title_tokens, title_pooled = self.encode(title_ids, title_mask)
        current_tokens, current_pooled = self.encode(current_ids, current_mask)
        attended, _ = self.cross_attention(
            current_tokens, title_tokens, title_tokens,
            key_padding_mask=~title_mask.bool(),
        )
        attended = attended.mean(dim=1)
        fused = torch.cat([attended, title_pooled + current_pooled], dim=-1)
        return self.t_head(fused)

    def logits_l(
        self,
        context_ids: Any,
        context_mask: Any,
        next_ids: Any,
        next_mask: Any,
    ) -> Any:
        batch, count, length = context_ids.shape
        flat_ids = context_ids.reshape(batch * count, length)
        flat_mask = context_mask.reshape(batch * count, length).bool()
        tokens, pooled = self.encode(flat_ids, flat_mask)
        tokens = tokens.reshape(batch, count, length, self.hidden)
        pooled = pooled.reshape(batch, count, self.hidden)

        positions = torch.arange(count, device=context_ids.device)
        context_vectors = pooled + self.pos_embedding(positions).unsqueeze(0)
        context_mask_bool = context_mask.bool()
        sample_mask = context_mask_bool.any(dim=-1)
        context_vectors = self.context_encoder(
            context_vectors, src_key_padding_mask=~sample_mask
        )
        weights = sample_mask.unsqueeze(-1).to(context_vectors.dtype)
        context_summary = (context_vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        key_value = tokens.reshape(batch, count * length, self.hidden)
        key_mask = context_mask_bool.reshape(batch, count * length)
        next_tokens, next_pooled = self.encode(next_ids, next_mask)
        attended, _ = self.cross_attention(
            next_tokens, key_value, key_value,
            key_padding_mask=~key_mask,
        )
        attended = attended.mean(dim=1)
        fused = torch.cat([attended, context_summary + next_pooled], dim=-1)
        return self.l_head(fused)

    def score_u(self, current_ids: Any, current_mask: Any) -> float:
        torch = _torch()
        return float(torch.softmax(self.logits_u(current_ids, current_mask), dim=-1)[0, 1].item())

    def score_l(self, context_items: Sequence[Tuple[Any, Any]], next_ids: Any, next_mask: Any) -> float:
        torch = _torch()
        if not context_items:
            return 0.0
        max_len = max(ids.size(-1) for ids, _ in context_items)
        batch = 1
        count = len(context_items)
        context_ids = torch.zeros((batch, count, max_len), dtype=torch.long, device=next_ids.device)
        context_mask = torch.zeros((batch, count, max_len), dtype=torch.bool, device=next_ids.device)
        for index, (ids, mask) in enumerate(context_items):
            length = ids.size(-1)
            context_ids[0, index, :length] = ids[0].to(next_ids.device)
            context_mask[0, index, :length] = mask[0].bool().to(next_ids.device)
        return float(torch.softmax(self.logits_l(context_ids, context_mask, next_ids, next_mask), dim=-1)[0, 1].item())

    def score_t(self, title_ids: Any, title_mask: Any, current_ids: Any, current_mask: Any) -> float:
        torch = _torch()
        return float(torch.softmax(self.logits_t(title_ids, title_mask, current_ids, current_mask), dim=-1)[0, 1].item())


def _tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def _encode(tokenizer: Any, text: str, max_length: int = 256, device: Any = "cpu") -> Tuple[Any, Any]:
    encoded = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def load_model(model_dir: Path, device: str = "cpu") -> Tuple[FilterModel, Any]:
    torch = _torch()
    target = resolve_device(device)
    weight_path = model_dir / "filter_model.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"筛选器权重不存在：{weight_path}；先训练筛选器。")
    config_path = model_dir / "filter_config.json"
    model_name = "hfl/chinese-macbert-base"
    if config_path.exists():
        import json
        with open(config_path, "r", encoding="utf-8") as fh:
            model_name = json.load(fh).get("model_name", model_name)
    tokenizer = _tokenizer(model_dir)
    model = FilterModel(model_name)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(target)
    model.eval()
    return model, tokenizer


def token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def run_filter(
    sentences_path: Path,
    output_path: Path,
    model_dir: Path,
    lexicon_dir: Path,
    max_sentences: int = MAX_SENTENCES,
    max_tokens: int = MAX_TOKENS,
    title_threshold: float = T_THRESHOLD,
    device: str = "cpu",
) -> pd.DataFrame:
    frame = pd.read_csv(sentences_path, encoding="utf-8-sig", dtype=str).fillna("")
    units = build_units(frame)
    target = resolve_device(device)
    model, tokenizer = load_model(model_dir, device)
    brands, categories = load_lexicons(lexicon_dir)
    torch = _torch()
    rows: List[Dict[str, str]] = []

    with torch.no_grad():
        position = 0
        while position < len(units):
            unit = units[position]
            current_ids, current_mask = _encode(tokenizer, unit.current, device=target)
            u_score = model.score_u(current_ids, current_mask)

            next_unit = find_next(units, position)
            l_score = 0.0
            if next_unit is not None:
                next_ids, next_mask = _encode(tokenizer, next_unit.current, device=target)
                l_score = model.score_l([(current_ids, current_mask)], next_ids, next_mask)

            title_score = 0.0
            if unit.title and rule_gate(unit.title, unit.current, brands, categories):
                title_ids, title_mask = _encode(tokenizer, unit.title, device=target)
                title_score = model.score_t(title_ids, title_mask, current_ids, current_mask)

            start_by_title = False
            if u_score >= START_THRESHOLD or l_score >= START_THRESHOLD:
                start = True
            elif unit.title and title_score >= title_threshold:
                start = True
                start_by_title = True
            else:
                start = False

            if not start:
                position += 1
                continue

            chain: List[str] = [unit.current]
            context_items: List[Tuple[Any, Any]] = [(current_ids, current_mask)]
            end_position = position
            while len(chain) < max_sentences:
                candidate = find_next(units, end_position)
                if candidate is None:
                    break
                if token_length(tokenizer, SEP.join(chain + [candidate.current])) > max_tokens:
                    break
                candidate_ids, candidate_mask = _encode(tokenizer, candidate.current, device=target)
                link = model.score_l(context_items, candidate_ids, candidate_mask)
                if link < START_THRESHOLD:
                    break
                chain.append(candidate.current)
                context_items.append((candidate_ids, candidate_mask))
                end_position += 1

            if start_by_title and has_pronoun(chain):
                resolved = resolve_pronouns(unit.title, chain, brands, categories)
                if resolved is None:
                    position = end_position + 1
                    continue
                chain = resolved
                title_prefix = ""
            else:
                if has_pronoun(chain):
                    position = end_position + 1
                    continue
                title_prefix = unit.title if start_by_title else ""

            simple_sentence = f"{title_prefix} {SEP} {SEP.join(chain)}"
            rows.append({"id": unit.id, "simple_sentence": simple_sentence})
            position = end_position + 1

    output = pd.DataFrame(rows, columns=["id", "simple_sentence"])
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output
