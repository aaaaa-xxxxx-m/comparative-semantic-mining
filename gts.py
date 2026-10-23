"""GTS 改造版：BIO 实体 span + pair 级关系/方面/得分。"""

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
SCORE_VALUES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEP = "[SEP]"
MAX_PAIRS = 20
PAIR_THRESHOLD = 0.5


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


def _span_mask(offsets: Sequence[Sequence[int]], start: int, end: int) -> torch.Tensor:
    torch = _torch()
    mask = torch.zeros(len(offsets), dtype=torch.bool)
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end == 0:
            continue
        if token_start < end and token_end > start:
            mask[index] = True
    return mask


class PairDataset:
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
        subj = (int(row["subject_start"]), int(row["subject_end"]))
        obj = (int(row["object_start"]), int(row["object_end"]))
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length",
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
        bio = [BIO_TO_ID["O"]] * len(offsets)
        subj_mask = _span_mask(offsets, *subj)
        obj_mask = _span_mask(offsets, *obj)
        subj_seen = False
        obj_seen = False
        for token_index, (start, end) in enumerate(offsets):
            if start == end == 0:
                continue
            if subj_mask[token_index]:
                bio[token_index] = BIO_TO_ID["B-S"] if not subj_seen else BIO_TO_ID["I-S"]
                subj_seen = True
            elif obj_mask[token_index]:
                bio[token_index] = BIO_TO_ID["B-O"] if not obj_seen else BIO_TO_ID["I-O"]
                obj_seen = True
        aspect = str(row.get("aspect_theme", "其他"))
        aspect_id = ASPECT_THEMES.index(aspect) if aspect in ASPECT_THEMES else ASPECT_THEMES.index("其他")
        try:
            score = float(row.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        relation = int(row.get("relation", 1) or 1)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bio_labels": torch.tensor(bio, dtype=torch.long),
            "subject_mask": subj_mask,
            "object_mask": obj_mask,
            "relation": torch.tensor(relation, dtype=torch.long),
            "aspect_label": torch.tensor(aspect_id, dtype=torch.long),
            "score": torch.tensor(max(-1.0, min(1.0, score)), dtype=torch.float),
            "score_label": torch.tensor(min(range(len(SCORE_VALUES)), key=lambda i: abs(SCORE_VALUES[i] - score)), dtype=torch.long),
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
            self.dropout = torch.nn.Dropout(config.dropout)
            self.bio_head = torch.nn.Linear(hidden, len(BIO_LABELS))
            self.relation_head = torch.nn.Sequential(
                torch.nn.Linear(hidden * 4, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 2)
            )
            self.aspect_head = torch.nn.Sequential(
                torch.nn.Linear(hidden * 4, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, len(ASPECT_THEMES))
            )
            self.score_head = torch.nn.Sequential(
                torch.nn.Linear(hidden * 4, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, len(SCORE_VALUES))
            )

        def encode(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.dropout(outputs.last_hidden_state)

        def pair_heads(self, sequence, subject_mask, object_mask):
            torch = _torch()
            mask_s = subject_mask.unsqueeze(-1).to(sequence.dtype)
            mask_o = object_mask.unsqueeze(-1).to(sequence.dtype)
            subject = (sequence * mask_s).sum(1) / mask_s.sum(1).clamp(min=1.0)
            obj = (sequence * mask_o).sum(1) / mask_o.sum(1).clamp(min=1.0)
            pair = torch.cat([subject, obj, subject * obj, subject - obj], dim=-1)
            return self.relation_head(pair), self.aspect_head(pair), self.score_head(pair)

        def forward(self, input_ids, attention_mask, subject_mask, object_mask):
            sequence = self.encode(input_ids, attention_mask)
            bio_logits = self.bio_head(sequence)
            relation_logits, aspect_logits, score_logits = self.pair_heads(sequence, subject_mask, object_mask)
            return bio_logits, relation_logits, aspect_logits, score_logits

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
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "gts_config.json"
    weight_path = model_dir / "gts_model.pt"
    if not force_train and config_path.exists() and weight_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("model_name") == model_name:
            return

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = GTSConfig(model_name=model_name, max_length=max_length)
    model = _model_classes()(model_name, config).to(target)
    frame = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = max(1, int(len(frame) * 0.9))
    train_part = frame.iloc[:split].reset_index(drop=True)
    valid_part = frame.iloc[split:].reset_index(drop=True) if split < len(frame) else train_part.copy()
    train_loader = DataLoader(PairDataset(train_part, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(PairDataset(valid_part, tokenizer, max_length), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bio_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0, 6.0, 6.0, 6.0], device=target))
    relation_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=target))
    positive_frame = train_part[train_part["relation"].astype(int) == 1]
    aspect_counts = positive_frame["aspect_theme"].value_counts()
    total = max(1, int(len(positive_frame)))
    weights = []
    for theme in ASPECT_THEMES:
        count = int(aspect_counts.get(theme, 0))
        weights.append((total / (len(ASPECT_THEMES) * max(1, count))) ** 0.5)
    aspect_weights = torch.tensor(weights, dtype=torch.float, device=target)
    aspect_weights = torch.clamp(aspect_weights, min=0.5, max=5.0)
    aspect_loss_fn = torch.nn.CrossEntropyLoss(weight=aspect_weights)
    aspect_loss_fn_eval = torch.nn.CrossEntropyLoss(weight=aspect_weights)
    score_counts = positive_frame["score"].apply(lambda value: min(range(len(SCORE_VALUES)), key=lambda i: abs(SCORE_VALUES[i] - float(value)))).value_counts()
    score_weights = []
    for index in range(len(SCORE_VALUES)):
        count = int(score_counts.get(index, 0))
        score_weights.append((total / (len(SCORE_VALUES) * max(1, count))) ** 0.5)
    score_weights = torch.tensor(score_weights, dtype=torch.float, device=target)
    score_weights = torch.clamp(score_weights, min=0.5, max=5.0)
    score_loss_fn = torch.nn.CrossEntropyLoss(weight=score_weights)
    score_loss_fn_eval = torch.nn.CrossEntropyLoss(weight=score_weights)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(target)
            mask = batch["attention_mask"].to(target)
            subj = batch["subject_mask"].to(target)
            obj = batch["object_mask"].to(target)
            bio_logits, relation_logits, aspect_logits, score_logits = model(ids, mask, subj, obj)
            bio_loss = bio_loss_fn(bio_logits.view(-1, len(BIO_LABELS)), batch["bio_labels"].to(target).view(-1))
            relation = batch["relation"].to(target).float()
            relation_loss = relation_loss_fn(relation_logits[:, 1], relation)
            positive = relation > 0.5
            if positive.any():
                aspect_loss = aspect_loss_fn(aspect_logits[positive], batch["aspect_label"].to(target)[positive])
                score_loss = score_loss_fn(score_logits[positive], batch["score_label"].to(target)[positive])
            else:
                aspect_loss = score_loss = torch.tensor(0.0, device=target)
            loss = bio_loss + relation_loss + aspect_loss + 0.5 * score_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(target)
                mask = batch["attention_mask"].to(target)
                subj = batch["subject_mask"].to(target)
                obj = batch["object_mask"].to(target)
                bio_logits, relation_logits, aspect_logits, score_logits = model(ids, mask, subj, obj)
                relation = batch["relation"].to(target).float()
                loss = bio_loss_fn(bio_logits.view(-1, len(BIO_LABELS)), batch["bio_labels"].to(target).view(-1))
                loss = loss + relation_loss_fn(relation_logits[:, 1], relation)
                positive = relation > 0.5
                if positive.any():
                    loss = loss + aspect_loss_fn_eval(aspect_logits[positive], batch["aspect_label"].to(target)[positive])
                    score_loss = score_loss_fn_eval(score_logits[positive], batch["score_label"].to(target)[positive])
                    loss = loss + 0.5 * score_loss
                valid_loss += float(loss.item())
        avg_valid = valid_loss / max(1, len(valid_loader))
        print(f"[gts] epoch={epoch} train_loss={running / max(1, len(train_loader)):.4f} valid_loss={avg_valid:.4f}", flush=True)
        if avg_valid < best_loss:
            best_loss = avg_valid
            torch.save(model.state_dict(), weight_path)
            tokenizer.save_pretrained(model_dir)
            config_path.write_text(json.dumps({
                "model_name": model_name,
                "max_length": max_length,
                "pair_dim": config.pair_dim,
                "aspect_themes": ASPECT_THEMES,
                "bio_labels": BIO_LABELS,
                "score_values": SCORE_VALUES,
                "max_pairs": MAX_PAIRS,
                "best_valid_loss": best_loss,
                "aspect_weights": aspect_weights.cpu().tolist(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model(model_dir: Path, device: str = "cpu"):
    torch = _torch()
    target = resolve_device(device)
    cfg = json.loads((model_dir / "gts_config.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = GTSConfig(model_name=cfg["model_name"], max_length=cfg["max_length"], pair_dim=cfg.get("pair_dim", 128))
    model = _model_classes()(cfg["model_name"], config)
    model.load_state_dict(torch.load(model_dir / "gts_model.pt", map_location="cpu"))
    model.to(target)
    model.eval()
    return model, tokenizer, config


def _decode_spans(bio_ids: Sequence[int], offsets: Sequence[Sequence[int]]) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    current_type: Optional[str] = None
    current_start = 0
    current_end = 0
    for index, label_id in enumerate(bio_ids):
        label = BIO_LABELS[label_id] if 0 <= label_id < len(BIO_LABELS) else "O"
        start, end = offsets[index]
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
    return "_".join(unit_id.split("_")[:2])


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
        ids = encoded["input_ids"].to(target)
        mask = encoded["attention_mask"].to(target)
        with torch.no_grad():
            sequence = model.encode(ids, mask)
            bio_logits = model.bio_head(sequence)
        bio_ids = torch.argmax(bio_logits[0], dim=-1).cpu().tolist()
        spans = _decode_spans(bio_ids, offsets)
        subjects = [(s, e) for kind, s, e in spans if kind == "S"][:5]
        objects = [(s, e) for kind, s, e in spans if kind == "O"][:5]
        if not subjects or not objects:
            continue
        candidates = []
        for subj in subjects:
            for obj in objects:
                subj_mask = _span_mask(offsets, *subj).unsqueeze(0).to(target)
                obj_mask = _span_mask(offsets, *obj).unsqueeze(0).to(target)
                with torch.no_grad():
                    relation_logits, aspect_logits, score_logits = model.pair_heads(sequence, subj_mask, obj_mask)
                confidence = float(torch.softmax(relation_logits, dim=-1)[0, 1].item())
                if confidence < pair_threshold:
                    continue
                aspect = ASPECT_THEMES[int(torch.argmax(aspect_logits[0]).item())]
                score_index = int(torch.argmax(score_logits[0]).item())
                score_value = SCORE_VALUES[score_index]
                title_end = _title_end(text)
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
                    "direction_raw": "S→O",
                    "aspect_theme": aspect,
                    "score_raw": score_value,
                    "subject_source": "title" if title_end and subj[0] < title_end else "body",
                    "object_source": "title" if title_end and obj[0] < title_end else "body",
                    "source": "pair",
                    "confidence": confidence,
                })
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        for number, candidate in enumerate(candidates[:max_pairs], start=1):
            candidate["mention_id"] = f"{unit_id}_{number:02d}"
            rows.append(candidate)
    return rows
