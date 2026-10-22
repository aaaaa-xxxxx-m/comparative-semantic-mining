"""GTS 改造版四元组抽取。

输入：comparison_sentences.csv 的 id / compare_sentence
输出：quads_raw.csv，字段见 QUAD_OUTPUT_COLUMNS

模型：
- 共享 BERT；
- BIO 头抽主体/客体 span；
- 双线性方向头判断 S→O / O→S，作为辅助；
- 方面 5 分类；
- 得分回归，表示前者相对后者。

抽取阶段不去重、不合并、不平均；重复留给原始文本级聚合。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ASPECT_THEMES = ["文化认知", "消费场景", "产品体验", "竞争压力", "其他"]
BIO_LABELS = ["O", "B-S", "I-S", "B-O", "I-O"]
BIO_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
SEP = "[SEP]"
MAX_PAIRS = 30
PAIR_THRESHOLD = 0.1

QUAD_OUTPUT_COLUMNS = [
    "mention_id", "text_group_id", "text",
    "subject_raw", "subject_start", "subject_end",
    "object_raw", "object_start", "object_end",
    "direction_raw", "aspect_theme", "score_raw",
    "subject_source", "object_source", "source", "confidence",
]


@dataclass
class GTSConfig:
    model_name: str = "hfl/chinese-macbert-base"
    max_length: int = 256
    pair_dim: int = 128
    dropout: float = 0.1


def _torch():
    import torch
    return torch


def resolve_device(requested: str) -> Any:
    torch = _torch()
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


def _set_threads() -> None:
    torch = _torch()
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))


class QuadDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int = 256):
        self.rows = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        torch = _torch()
        row = self.rows.iloc[index]
        text = str(row["text"])
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length",
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
        bio = [0] * len(offsets)
        subj = (int(row["subject_start"]), int(row["subject_end"]))
        obj = (int(row["object_start"]), int(row["object_end"]))
        subj_tokens: List[int] = []
        obj_tokens: List[int] = []
        for token_idx, (start, end) in enumerate(offsets):
            if start == end == 0:
                continue
            in_subj = start < subj[1] and end > subj[0]
            in_obj = start < obj[1] and end > obj[0]
            if in_subj:
                bio[token_idx] = BIO_TO_ID["B-S"] if not subj_tokens else BIO_TO_ID["I-S"]
                subj_tokens.append(token_idx)
            elif in_obj:
                bio[token_idx] = BIO_TO_ID["B-O"] if not obj_tokens else BIO_TO_ID["I-O"]
                obj_tokens.append(token_idx)

        seq_len = len(offsets)
        pair_s2o = torch.zeros((seq_len, seq_len), dtype=torch.float)
        pair_o2s = torch.zeros((seq_len, seq_len), dtype=torch.float)
        for i in subj_tokens:
            for j in obj_tokens:
                pair_s2o[i, j] = 1.0
        for i in obj_tokens:
            for j in subj_tokens:
                pair_o2s[i, j] = 1.0

        aspect = str(row.get("aspect_theme", "其他"))
        aspect_id = ASPECT_THEMES.index(aspect) if aspect in ASPECT_THEMES else ASPECT_THEMES.index("其他")
        try:
            score = float(row.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bio_labels": torch.tensor(bio, dtype=torch.long),
            "pair_s2o": pair_s2o,
            "pair_o2s": pair_o2s,
            "aspect_label": torch.tensor(aspect_id, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float),
        }


def _model_classes():
    torch = _torch()
    from transformers import AutoModel

    class GridTagger(torch.nn.Module):
        def __init__(self, model_name: str, config: GTSConfig):
            super().__init__()
            self.config = config
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.hidden = hidden
            self.bio_head = torch.nn.Linear(hidden, len(BIO_LABELS))
            self.s2o_u = torch.nn.Linear(hidden, config.pair_dim)
            self.s2o_v = torch.nn.Linear(hidden, config.pair_dim)
            self.o2s_u = torch.nn.Linear(hidden, config.pair_dim)
            self.o2s_v = torch.nn.Linear(hidden, config.pair_dim)
            self.aspect_head = torch.nn.Linear(hidden, len(ASPECT_THEMES))
            self.score_head = torch.nn.Linear(hidden, 1)
            self.dropout = torch.nn.Dropout(config.dropout)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence = self.dropout(outputs.last_hidden_state)
            bio_logits = self.bio_head(sequence)
            su = self.s2o_u(sequence)
            sv = self.s2o_v(sequence)
            ou = self.o2s_u(sequence)
            ov = self.o2s_v(sequence)
            pair_s2o = torch.matmul(su, sv.transpose(1, 2)) / (self.config.pair_dim ** 0.5)
            pair_o2s = torch.matmul(ou, ov.transpose(1, 2)) / (self.config.pair_dim ** 0.5)
            pooled = sequence[:, 0]
            aspect_logits = self.aspect_head(pooled)
            score = torch.tanh(self.score_head(pooled)).squeeze(-1)
            return bio_logits, pair_s2o, pair_o2s, aspect_logits, score

    return GridTagger


def train_or_load(
    train_df: pd.DataFrame,
    model_dir: Path,
    model_name: str = "hfl/chinese-macbert-base",
    max_length: int = 256,
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    seed: int = 42,
    force_train: bool = False,
    device: str = "cpu",
) -> None:
    torch = _torch()
    target = resolve_device(device)
    _set_threads()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "gts_config.json"
    weight_path = model_dir / "gts_model.pt"
    if not force_train and config_path.exists() and weight_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("model_name") == model_name:
            return
        force_train = True

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = GTSConfig(model_name=model_name, max_length=max_length)
    GridTagger = _model_classes()
    model = GridTagger(model_name, config).to(target)

    frame = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = max(1, int(len(frame) * 0.9))
    train_part = frame.iloc[:split].reset_index(drop=True)
    valid_part = frame.iloc[split:].reset_index(drop=True)
    if valid_part.empty:
        valid_part = train_part.copy()

    train_loader = DataLoader(QuadDataset(train_part, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(QuadDataset(valid_part, tokenizer, max_length), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bio_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0, 6.0, 6.0, 6.0], device=target))
    pair_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=target))
    aspect_loss_fn = torch.nn.CrossEntropyLoss()
    score_loss_fn = torch.nn.SmoothL1Loss()

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(target)
            attention_mask = batch["attention_mask"].to(target)
            bio_logits, pair_s2o, pair_o2s, aspect_logits, score = model(input_ids, attention_mask)
            bio_loss = bio_loss_fn(bio_logits.view(-1, len(BIO_LABELS)), batch["bio_labels"].to(target).view(-1))
            pair_loss = pair_loss_fn(pair_s2o, batch["pair_s2o"].to(target)) + pair_loss_fn(pair_o2s, batch["pair_o2s"].to(target))
            aspect_loss = aspect_loss_fn(aspect_logits, batch["aspect_label"].to(target))
            score_loss = score_loss_fn(score, batch["score"].to(target))
            loss = bio_loss + 0.3 * pair_loss + aspect_loss + 0.5 * score_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(target)
                attention_mask = batch["attention_mask"].to(target)
                bio_logits, pair_s2o, pair_o2s, aspect_logits, score = model(input_ids, attention_mask)
                bio_loss = bio_loss_fn(bio_logits.view(-1, len(BIO_LABELS)), batch["bio_labels"].to(target).view(-1))
                pair_loss = pair_loss_fn(pair_s2o, batch["pair_s2o"].to(target)) + pair_loss_fn(pair_o2s, batch["pair_o2s"].to(target))
                aspect_loss = aspect_loss_fn(aspect_logits, batch["aspect_label"].to(target))
                score_loss = score_loss_fn(score, batch["score"].to(target))
                valid_loss += float((bio_loss + 0.3 * pair_loss + aspect_loss + 0.5 * score_loss).item())
        avg_valid = valid_loss / max(1, len(valid_loader))
        print(f"[gts] epoch={epoch} train_loss={running / max(1, len(train_loader)):.4f} valid_loss={avg_valid:.4f}", flush=True)
        if avg_valid < best_loss:
            best_loss = avg_valid
            torch.save(model.state_dict(), weight_path)
            tokenizer.save_pretrained(model_dir)
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "model_name": model_name,
                    "max_length": max_length,
                    "pair_dim": config.pair_dim,
                    "aspect_themes": ASPECT_THEMES,
                    "bio_labels": BIO_LABELS,
                    "best_valid_loss": best_loss,
                }, fh, ensure_ascii=False, indent=2)


def _load_model(model_dir: Path, device: str = "cpu"):
    torch = _torch()
    target = resolve_device(device)
    from transformers import AutoTokenizer
    with open(model_dir / "gts_config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = GTSConfig(model_name=cfg["model_name"], max_length=cfg["max_length"], pair_dim=cfg.get("pair_dim", 128))
    GridTagger = _model_classes()
    model = GridTagger(cfg["model_name"], config)
    model.load_state_dict(torch.load(model_dir / "gts_model.pt", map_location="cpu"))
    model.to(target)
    model.eval()
    return model, tokenizer, config


def _decode_spans(bio_ids: Sequence[int], offsets: Sequence[Sequence[int]]) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    current_type: Optional[str] = None
    current_start = 0
    current_end = 0
    for idx, label_id in enumerate(bio_ids):
        label = BIO_LABELS[label_id] if 0 <= label_id < len(BIO_LABELS) else "O"
        start, end = offsets[idx]
        if start == end == 0:
            continue
        if label == "O":
            if current_type is not None:
                spans.append((current_type, current_start, current_end))
                current_type = None
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "B" or entity != current_type:
            if current_type is not None:
                spans.append((current_type, current_start, current_end))
            current_type = entity
            current_start = start
            current_end = end
        else:
            current_end = end
    if current_type is not None:
        spans.append((current_type, current_start, current_end))
    return [(kind, start, end) for kind, start, end in spans if end > start]


def _text_group(unit_id: str) -> str:
    parts = unit_id.split("_")
    return "_".join(parts[:2])


def _title_end(text: str) -> int:
    parts = text.split(SEP)
    if len(parts) < 2:
        return 0
    return len(parts[0]) + len(SEP) if parts[0].strip() else 0


def predict_dataframe(
    frame: pd.DataFrame,
    model_dir: Path,
    max_length: int = 256,
    max_pairs: int = MAX_PAIRS,
    pair_threshold: float = PAIR_THRESHOLD,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    torch = _torch()
    target = resolve_device(device)
    _set_threads()
    model, tokenizer, config = _load_model(model_dir, device)
    rows: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        unit_id = str(row["id"])
        text = str(row["sentence"] if "sentence" in row else row.get("compare_sentence", row.get("simple_sentence", "")))
        if not text:
            continue
        encoded = tokenizer(
            text, max_length=max_length, truncation=True, padding="max_length",
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
        input_ids = encoded["input_ids"].to(target)
        attention_mask = encoded["attention_mask"].to(target)
        with torch.no_grad():
            bio_logits, pair_s2o, pair_o2s, aspect_logits, score = model(input_ids, attention_mask)
        bio_ids = torch.argmax(bio_logits[0], dim=-1).cpu().tolist()
        spans = _decode_spans(bio_ids, offsets)
        subjects = [(s, e) for kind, s, e in spans if kind == "S"]
        objects = [(s, e) for kind, s, e in spans if kind == "O"]
        if not subjects or not objects:
            continue

        s2o = torch.sigmoid(pair_s2o[0]).cpu().numpy()
        o2s = torch.sigmoid(pair_o2s[0]).cpu().numpy()
        aspect = ASPECT_THEMES[int(torch.argmax(aspect_logits[0]).item())]
        score_value = float(score[0].item())
        title_end = _title_end(text)
        candidates: List[Dict[str, Any]] = []

        for subj in subjects[:5]:
            for obj in objects[:5]:
                subj_tokens = [i for i, (s, e) in enumerate(offsets) if s < subj[1] and e > subj[0]]
                obj_tokens = [i for i, (s, e) in enumerate(offsets) if s < obj[1] and e > obj[0]]
                if not subj_tokens or not obj_tokens:
                    continue
                forward = float(s2o[np.ix_(subj_tokens, obj_tokens)].max())
                backward = float(o2s[np.ix_(obj_tokens, subj_tokens)].max())
                confidence = max(forward, backward)
                if confidence < pair_threshold:
                    continue
                direction = "S→O" if forward >= backward else "O→S"
                subject_source = "title" if title_end and subj[0] < title_end else "body"
                object_source = "title" if title_end and obj[0] < title_end else "body"
                candidates.append({
                    "unit_id": unit_id,
                    "text_group_id": _text_group(unit_id),
                    "text": text,
                    "subject_raw": text[subj[0]:subj[1]],
                    "subject_start": subj[0],
                    "subject_end": subj[1],
                    "object_raw": text[obj[0]:obj[1]],
                    "object_start": obj[0],
                    "object_end": obj[1],
                    "direction_raw": direction,
                    "aspect_theme": aspect,
                    "score_raw": score_value,
                    "subject_source": subject_source,
                    "object_source": object_source,
                    "source": "title" if "title" in (subject_source, object_source) else "chain",
                    "confidence": confidence,
                })

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        for number, candidate in enumerate(candidates[:max_pairs], start=1):
            candidate["mention_id"] = f"{unit_id}_{number:02d}"
            rows.append(candidate)

    return rows
