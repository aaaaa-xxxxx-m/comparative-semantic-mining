"""筛选器与 GTS 的训练入口。

设备完全由 --device 决定：
    --device cpu
    --device cuda
    --device cuda:1

不会在未指定 CUDA 时自动使用 GPU，也不会在指定 CUDA 但不可用时回退到 CPU。

筛选器数据为 JSONL，每行一个训练样本：
    {"task": "u", "text": "当前句", "label": 1}
    {"task": "l", "context": ["上句"], "next": "下句", "label": 1}
    {"task": "t", "title": "标题", "current": "当前句", "label": 1}

GTS 数据为 CSV，至少包含：
    text, subject_start, subject_end, object_start, object_end, aspect_theme, score
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import filter as filter_module
import gts


class FilterDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int = 256):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            str(text), max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        task = str(example.get("task", "")).strip()
        label = int(example.get("label", 0))
        if label not in (0, 1):
            raise ValueError(f"label 必须是 0 或 1：{example}")
        if task == "u":
            ids, mask = self._encode(example["text"])
            return {"task": task, "current_ids": ids, "current_mask": mask, "label": label}
        if task == "t":
            title_ids, title_mask = self._encode(example["title"])
            current_ids, current_mask = self._encode(example["current"])
            return {
                "task": task,
                "title_ids": title_ids,
                "title_mask": title_mask,
                "current_ids": current_ids,
                "current_mask": current_mask,
                "label": label,
            }
        if task == "l":
            contexts = list(example.get("context", []))
            if not contexts:
                raise ValueError(f"l 样本缺少 context：{example}")
            if len(contexts) > filter_module.MAX_SENTENCES:
                raise ValueError(f"l 样本 context 超过 {filter_module.MAX_SENTENCES} 句：{example}")
            encoded = [self._encode(text) for text in contexts]
            next_ids, next_mask = self._encode(example["next"])
            return {
                "task": task,
                "context_ids": [item[0] for item in encoded],
                "context_mask": [item[1] for item in encoded],
                "next_ids": next_ids,
                "next_mask": next_mask,
                "label": label,
            }
        raise ValueError(f"未知 task：{task}")


def _pad(sequences: Sequence[torch.Tensor], value: int = 0) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(sequences), batch_first=True, padding_value=value
    )


def _pad_mask(sequences: Sequence[torch.Tensor]) -> torch.Tensor:
    return _pad(sequences, value=0).bool()


def collate_filter(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    task = batch[0]["task"]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    if task == "u":
        return {
            "task": task,
            "current_ids": _pad([item["current_ids"] for item in batch]),
            "current_mask": _pad_mask([item["current_mask"] for item in batch]),
            "label": labels,
        }
    if task == "t":
        return {
            "task": task,
            "title_ids": _pad([item["title_ids"] for item in batch]),
            "title_mask": _pad_mask([item["title_mask"] for item in batch]),
            "current_ids": _pad([item["current_ids"] for item in batch]),
            "current_mask": _pad_mask([item["current_mask"] for item in batch]),
            "label": labels,
        }
    if task == "l":
        max_context = max(len(item["context_ids"]) for item in batch)
        max_length = max(ids.size(0) for item in batch for ids in item["context_ids"])
        context_ids = torch.zeros((len(batch), max_context, max_length), dtype=torch.long)
        context_mask = torch.zeros((len(batch), max_context, max_length), dtype=torch.bool)
        for row, item in enumerate(batch):
            for index, ids in enumerate(item["context_ids"]):
                length = ids.size(0)
                context_ids[row, index, :length] = ids
                context_mask[row, index, :length] = item["context_mask"][index].bool()
        return {
            "task": task,
            "context_ids": context_ids,
            "context_mask": context_mask,
            "next_ids": _pad([item["next_ids"] for item in batch]),
            "next_mask": _pad_mask([item["next_mask"] for item in batch]),
            "label": labels,
        }
    raise ValueError(f"未知 task：{task}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            rows.append(row)
    if not rows:
        raise ValueError(f"训练数据为空：{path}")
    return rows


def split_by_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    tasks: Dict[str, List[Dict[str, Any]]] = {"u": [], "l": [], "t": []}
    for row in rows:
        task = str(row.get("task", "")).strip()
        if task not in tasks:
            raise ValueError(f"未知 task：{task}")
        tasks[task].append(row)
    return tasks


def _task_batches(
    model: Any,
    batch: Dict[str, Any],
    device: torch.device,
) -> Any:
    task = batch["task"]
    if task == "u":
        return model.logits_u(
            batch["current_ids"].to(device),
            batch["current_mask"].to(device),
        )
    if task == "t":
        return model.logits_t(
            batch["title_ids"].to(device),
            batch["title_mask"].to(device),
            batch["current_ids"].to(device),
            batch["current_mask"].to(device),
        )
    if task == "l":
        return model.logits_l(
            batch["context_ids"].to(device),
            batch["context_mask"].to(device),
            batch["next_ids"].to(device),
            batch["next_mask"].to(device),
        )
    raise ValueError(f"未知 task：{task}")


def evaluate_filter(
    model: Any,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> tuple[float, Dict[str, float]]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    task_losses: Dict[str, float] = {}
    task_accs: Dict[str, float] = {}
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for task, loader in loaders.items():
            if len(loader) == 0:
                continue
            loss_sum = 0.0
            correct = 0
            count = 0
            for batch in loader:
                labels = batch["label"].to(device)
                logits = _task_batches(model, batch, device)
                loss = loss_fn(logits, labels)
                loss_sum += float(loss.item())
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                count += labels.size(0)
            task_losses[task] = loss_sum / len(loader)
            task_accs[task] = correct / max(1, count)
            total_loss += loss_sum
            total_batches += len(loader)
    return total_loss / max(1, total_batches), task_accs


def _filter_loaders(
    tasks: Dict[str, List[Dict[str, Any]]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for task in ("u", "l", "t"):
        examples = tasks.get(task, [])
        if not examples:
            continue
        dataset = FilterDataset(examples, tokenizer, max_length)
        loaders[task] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_filter,
        )
    return loaders


def train_filter(args: argparse.Namespace) -> None:
    target = filter_module.resolve_device(args.device)
    train_rows = load_jsonl(args.train_data)
    if args.valid_data:
        valid_rows = load_jsonl(args.valid_data)
    else:
        random.Random(args.seed).shuffle(train_rows)
        split = max(1, int(len(train_rows) * 0.9))
        valid_rows = train_rows[split:]
        train_rows = train_rows[:split]
    train_tasks = split_by_task(train_rows)
    valid_tasks = split_by_task(valid_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loaders = _filter_loaders(
        train_tasks, tokenizer, args.max_length, args.batch_size, shuffle=True
    )
    valid_loaders = _filter_loaders(
        valid_tasks, tokenizer, args.max_length, args.batch_size, shuffle=False
    )
    if not train_loaders:
        raise ValueError("训练数据没有任何 u/l/t 样本")

    model = filter_module.FilterModel(args.model_name).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")
    patience_left = args.patience
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for task, loader in train_loaders.items():
            for batch in loader:
                optimizer.zero_grad()
                labels = batch["label"].to(target)
                logits = _task_batches(model, batch, target)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += float(loss.item())
                batches += 1
        valid_loss, accuracy = evaluate_filter(model, valid_loaders, target)
        print(
            f"[filter] epoch={epoch} train_loss={running / max(1, batches):.4f} "
            f"valid_loss={valid_loss:.4f} acc={accuracy}",
            flush=True,
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_left = args.patience
            torch.save(model.state_dict(), model_dir / "filter_model.pt")
            tokenizer.save_pretrained(model_dir)
            with open(model_dir / "filter_config.json", "w", encoding="utf-8") as fh:
                json.dump({
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "max_sentences": filter_module.MAX_SENTENCES,
                    "t_threshold": filter_module.T_THRESHOLD,
                    "start_threshold": filter_module.START_THRESHOLD,
                    "best_valid_loss": best_loss,
                }, fh, ensure_ascii=False, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[filter] early stop at epoch={epoch}", flush=True)
                break


def train_gts(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.train_data, encoding="utf-8-sig")
    required = {"text", "subject_start", "subject_end", "object_start", "object_end", "aspect_theme", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GTS 训练数据缺少字段：{sorted(missing)}")
    if frame.empty:
        raise ValueError(f"GTS 训练数据为空：{args.train_data}")
    gts.train_or_load(
        train_df=frame,
        model_dir=Path(args.model_dir),
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        force_train=args.force_train,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--device",
        default="cpu",
        help="训练设备：cpu、cuda 或 cuda:N；默认 cpu。",
    )

    parser = argparse.ArgumentParser(description="筛选器与 GTS 训练入口")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("filter", parents=[common], help="训练筛选器")
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--valid-data", type=Path)
    p.add_argument("--model-dir", type=Path, default=BASE_DIR / "models" / "filter")
    p.add_argument("--model-name", default="hfl/chinese-macbert-base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    p.set_defaults(func=train_filter)

    p = sub.add_parser("gts", parents=[common], help="训练 GTS")
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=BASE_DIR / "models" / "gts")
    p.add_argument("--model-name", default="hfl/chinese-macbert-base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-train", action="store_true")
    p.set_defaults(func=train_gts)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
