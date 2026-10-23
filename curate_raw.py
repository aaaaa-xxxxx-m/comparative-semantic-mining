"""从 9 个平台原始表中筛选、去重、分组并划分训练数据。

只在训练数据构建阶段使用；实际推理走 run.py prepare。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from text_utils import clean_text, clip_to_sentences, has_comparison_cue, is_article, split_article

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "input" / "raw"
DATA_DIR = BASE_DIR / "data" / "raw"
PLATFORMS = ["BZ", "DY", "JD", "PDD", "TB", "WB", "XHS", "XW", "ZH"]
QUOTAS = {"BZ": 80, "DY": 100, "JD": 120, "PDD": 100, "TB": 100, "WB": 100, "XHS": 80, "XW": 220, "ZH": 100}


def pick(row: pd.Series, names: List[str], default: str = "") -> str:
    for name in names:
        if name in row.index and not pd.isna(row.get(name)):
            value = str(row.get(name)).strip()
            if value:
                return value
    return default


def load_file(path: Path, platform: str) -> List[Dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name=0)
    frame.columns = [str(c).strip() for c in frame.columns]
    rows: List[Dict[str, Any]] = []
    for order, (_, row) in enumerate(frame.iterrows()):
        raw_id = pick(row, ["编号", "id", "ID"])
        text = pick(row, ["文本内容", "内容", "评论内容"])
        if not raw_id or not text:
            continue
        rows.append({
            "raw_id": raw_id,
            "platform": platform,
            "source": pick(row, ["来源"]),
            "brand": pick(row, ["品牌"]),
            "title": pick(row, ["标题"]),
            "text": clean_text(text),
            "parent_id": pick(row, ["对应上级文本", "上级文本", "父级"]),
            "order": order,
        })
    return rows


def build_units(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if rows and rows[0]["platform"] == "XHS":
        return [{
            "raw_ids": [row["raw_id"]],
            "platform": row["platform"],
            "source": row["source"],
            "brand": row["brand"],
            "title": row["title"],
            "text": clip_to_sentences(row["text"], max_sentences=8),
            "unit_type": "article" if is_article(row["text"]) else "comment",
            "has_cue": int(has_comparison_cue(row["text"])),
        } for row in rows if row["text"]]
    by_id = {row["raw_id"]: row for row in rows}
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    roots: List[Dict[str, Any]] = []
    for row in rows:
        parent = row["parent_id"]
        if parent and parent in by_id and parent != row["raw_id"]:
            children[parent].append(row)
        else:
            roots.append(row)
    units: List[Dict[str, Any]] = []
    for root in roots:
        child_rows = sorted(children.get(root["raw_id"], []), key=lambda x: x["order"])[:5]
        pieces = [root["text"]] + [child["text"] for child in child_rows]
        text = clip_to_sentences(" ".join(piece for piece in pieces if piece), max_sentences=8)
        if not text:
            continue
        title = root["title"] or next((child["title"] for child in child_rows if child["title"]), "")
        units.append({
            "raw_ids": [root["raw_id"]] + [child["raw_id"] for child in child_rows],
            "platform": root["platform"],
            "source": root["source"],
            "brand": root["brand"],
            "title": title,
            "text": text,
            "unit_type": "article" if is_article(text) else "comment",
            "has_cue": int(has_comparison_cue(text)),
        })
    return units


def dedupe(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for unit in units:
        key = unit["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(unit)
    return output


def sample_units(units: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_platform: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_platform[unit["platform"]].append(unit)
    selected: List[Dict[str, Any]] = []
    for platform in PLATFORMS:
        candidates = by_platform.get(platform, [])
        rng.shuffle(candidates)
        cue = [u for u in candidates if u["has_cue"]]
        no_cue = [u for u in candidates if not u["has_cue"]]
        quota = QUOTAS[platform]
        target_cue = min(len(cue), max(1, quota // 2))
        target_no = min(len(no_cue), quota - target_cue)
        chosen = cue[:target_cue] + no_cue[:target_no]
        remaining = [u for u in candidates if u not in chosen]
        chosen += remaining[:max(0, quota - len(chosen))]
        selected.extend(chosen[:quota])
    rng.shuffle(selected)
    if len(selected) < 1000:
        chosen_ids = {id(unit) for unit in selected}
        remaining = [unit for unit in units if id(unit) not in chosen_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[:1000 - len(selected)])
    selected = selected[:1000]
    for index, unit in enumerate(selected, start=1):
        unit["article_id"] = f"a_{80000000 + index:08d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选原始平台数据并生成 raw train/valid/test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_units: List[Dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.xlsx")):
        platform = path.stem.upper()
        if platform not in PLATFORMS:
            continue
        all_units.extend(build_units(load_file(path, platform)))
    all_units = dedupe(all_units)
    selected = sample_units(all_units, args.seed)
    for index, unit in enumerate(selected):
        split = "train" if index < 600 else "valid" if index < 800 else "test"
        unit["split"] = split
        unit["source"] = "raw"

    with open(DATA_DIR / "units.jsonl", "w", encoding="utf-8") as fh:
        for unit in selected:
            fh.write(json.dumps(unit, ensure_ascii=False) + "\n")
    for split in ("train", "valid", "test"):
        rows = [unit for unit in selected if unit["split"] == split]
        with open(DATA_DIR / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for unit in rows:
                fh.write(json.dumps(unit, ensure_ascii=False) + "\n")

    print(json.dumps({
        "total": len(selected),
        "splits": {split: sum(1 for unit in selected if unit["split"] == split) for split in ("train", "valid", "test")},
        "platforms": {platform: sum(1 for unit in selected if unit["platform"] == platform) for platform in PLATFORMS},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
