"""在原始测试集上评估 Filter 和 pair 级 GTS。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

import filter as filter_module
import gts
import train as train_module

BASE_DIR = Path(__file__).resolve().parent


def binary_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_filter(model_dir: Path, data_path: Path, device: str) -> Dict[str, Dict[str, float]]:
    model, tokenizer = filter_module.load_model(model_dir, device)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    target = filter_module.resolve_device(device)
    result = {}
    for task in ("u", "l", "t"):
        examples = [row for row in rows if row["task"] == task]
        if not examples:
            continue
        dataset = train_module.FilterDataset(examples, tokenizer, max_length=256)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=train_module.collate_filter)
        gold, pred = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch["label"].to(target)
                logits = train_module._task_batches(model, batch, target)
                gold.extend(labels.cpu().tolist())
                pred.extend(logits.argmax(dim=-1).cpu().tolist())
        result[task] = binary_metrics(gold, pred)
        result[task]["n"] = len(gold)
    return result


def evaluate_gts(model_dir: Path, data_path: Path, device: str, threshold: float) -> Dict[str, object]:
    model, tokenizer, config = gts._load_model(model_dir, device)
    frame = pd.read_csv(data_path, encoding="utf-8-sig")
    dataset = gts.PairDataset(frame, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    target = gts.resolve_device(device)
    gold_rel, pred_rel = [], []
    aspect_hits = 0
    aspect_total = 0
    score_abs = []
    score_sign_hits = 0
    score_sign_total = 0
    exact_hits = 0
    partial_hits = 0
    positive_total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(target)
            mask = batch["attention_mask"].to(target)
            subj = batch["subject_mask"].to(target)
            obj = batch["object_mask"].to(target)
            _, rel_logits, aspect_logits, score = model(ids, mask, subj, obj)
            rel_pred = (torch.softmax(rel_logits, dim=-1)[:, 1] >= threshold).long()
            aspect_pred = torch.argmax(aspect_logits, dim=-1)
            gold_rel.extend(batch["relation"].tolist())
            pred_rel.extend(rel_pred.cpu().tolist())
            positive = batch["relation"] == 1
            if positive.any():
                a_gold = batch["aspect_label"][positive].to(target)
                a_pred = aspect_pred[positive]
                s_gold = batch["score"][positive].to(target)
                s_pred = score[positive]
                aspect_hits += int((a_gold == a_pred).sum().item())
                aspect_total += int(positive.sum().item())
                score_abs.extend(torch.abs(s_gold - s_pred).cpu().tolist())
                score_sign_hits += int(((s_gold >= 0) == (s_pred >= 0)).sum().item())
                score_sign_total += int(positive.sum().item())
                correct_score = torch.abs(s_gold - s_pred) <= 0.05
                correct_aspect = a_gold == a_pred
                exact_hits += int(((rel_pred[positive] == 1) & correct_aspect & correct_score).sum().item())
                partial_hits += int(((rel_pred[positive] == 1) & (correct_aspect | correct_score)).sum().item())
                positive_total += int(positive.sum().item())
    metrics = binary_metrics(gold_rel, pred_rel)
    metrics.update({
        "n": len(gold_rel),
        "positive_n": positive_total,
        "aspect_accuracy": aspect_hits / max(1, aspect_total),
        "score_mae": sum(score_abs) / max(1, len(score_abs)),
        "score_sign_accuracy": score_sign_hits / max(1, score_sign_total),
        "exact_tolerance_0.05": exact_hits / max(1, positive_total),
        "partial": partial_hits / max(1, positive_total),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="在原始测试集上评估模型")
    parser.add_argument("--filter-model-dir", type=Path, default=BASE_DIR / "models" / "filter")
    parser.add_argument("--gts-model-dir", type=Path, default=BASE_DIR / "models" / "gts")
    parser.add_argument("--filter-test", type=Path, default=BASE_DIR / "data" / "filter" / "filter_test.jsonl")
    parser.add_argument("--gts-test", type=Path, default=BASE_DIR / "data" / "gts" / "gts_test.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gts-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output" / "evaluation.json")
    args = parser.parse_args()
    report = {
        "filter": evaluate_filter(args.filter_model_dir, args.filter_test, args.device) if args.filter_test.exists() else {},
        "gts": evaluate_gts(args.gts_model_dir, args.gts_test, args.device, args.gts_threshold) if args.gts_test.exists() else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
