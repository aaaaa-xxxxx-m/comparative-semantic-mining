"""黄酒比较语义筛选、四元组抽取与规范化入口。

流程：
    prepare  -> sentences.csv
    filter   -> comparison_sentences.csv
    extract  -> quads_raw.csv
    normalize-> quads_final.xlsx

本文件负责数据读取、阶段调度、规范化与聚合；筛选模型在 filter.py，
GTS 模型在 gts.py。当前未包含数据补齐与模型训练命令。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
RAW_DIR = INPUT_DIR / "raw"
LEXICON_DIR = INPUT_DIR / "lexicons"

SENTENCES_CSV = BASE_DIR / "sentences.csv"
COMPARISON_CSV = BASE_DIR / "comparison_sentences.csv"
QUADS_RAW_CSV = BASE_DIR / "quads_raw.csv"
QUADS_FINAL_XLSX = BASE_DIR / "quads_final.xlsx"
RUN_JSON = BASE_DIR / "run.json"
RUN_LOG = BASE_DIR / "run.log"

FILTER_DIR = BASE_DIR / "models" / "filter"
GTS_DIR = BASE_DIR / "models" / "gts"

SEP = "[SEP]"
ASPECT_THEMES = ["文化认知", "消费场景", "产品体验", "竞争压力", "其他"]
BLOCKED_SUFFIXES = ["行业", "股", "股票", "市场", "板块", "企业", "公司", "产业", "赛道", "概念"]

QUAD_RAW_COLUMNS = [
    "mention_id", "text_group_id", "text",
    "subject_raw", "subject_start", "subject_end",
    "object_raw", "object_start", "object_end",
    "direction_raw", "aspect_theme", "score_raw",
    "subject_source", "object_source", "source", "confidence",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, level: str = "INFO", **fields: Any) -> None:
    line = f"[{now()}] {level} {message}"
    if fields:
        line += " " + json.dumps(fields, ensure_ascii=False, default=str)
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)


def update_run_json(**updates: Any) -> None:
    data = read_json(RUN_JSON, {}) or {}
    data.setdefault("created_at", now())
    data["updated_at"] = now()
    for key, value in updates.items():
        if isinstance(value, dict):
            data.setdefault(key, {})
            data[key].update(value)
        else:
            data[key] = value
    write_json(RUN_JSON, data)


def ensure_dirs() -> None:
    for directory in [INPUT_DIR, RAW_DIR, LEXICON_DIR, FILTER_DIR, GTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"回复\s*@[^:：]*[:：]", "", text)
    text = re.sub(r"@[\w\-\u4e00-\u9fff]+", "", text)
    text = re.sub(r"#[^#]{1,30}#", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_article(text: str) -> bool:
    collapsed = re.sub(r"[。？！]+", "。", text)
    return len(re.findall(r"[。？！]", collapsed)) >= 2


def split_article(text: str) -> List[str]:
    pieces = re.split(r"(?<=[。？！])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {str(column).strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_raw_tables(raw_dir: Path) -> List[Dict[str, str]]:
    files = sorted(path for path in raw_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"没有找到原始表：{raw_dir}")

    records: List[Dict[str, str]] = []
    for path in files:
        frame = pd.read_excel(path, sheet_name=0)
        frame.columns = [str(column).strip() for column in frame.columns]
        id_col = pick_column(frame.columns, ["编号", "id", "ID"])
        title_col = pick_column(frame.columns, ["标题"])
        text_col = pick_column(frame.columns, ["文本内容", "内容", "评论内容"])
        if id_col is None or text_col is None:
            raise ValueError(f"{path.name} 缺少 编号 或 文本内容 字段")
        for _, row in frame.iterrows():
            raw_id = "" if pd.isna(row.get(id_col, "")) else str(row.get(id_col, "")).strip()
            raw_text = "" if pd.isna(row.get(text_col, "")) else str(row.get(text_col, "")).strip()
            title_value = row.get(title_col, "") if title_col else ""
            title = "" if pd.isna(title_value) else str(title_value).strip()
            if not raw_id or not raw_text:
                continue
            records.append({"origin_id": raw_id, "title": title, "text": raw_text})
    return records


def prepare(args: argparse.Namespace) -> None:
    ensure_dirs()
    records = load_raw_tables(args.raw_dir)
    output_rows: List[Dict[str, str]] = []
    text_number = 0
    kept_texts = 0
    dropped_short = 0

    for record in records:
        text = clean_text(record["text"])
        if len(text) < args.min_len:
            dropped_short += 1
            continue
        kept_texts += 1
        text_number += 1
        article = is_article(text)
        pieces = split_article(text) if article else [text]
        if not pieces:
            continue
        prefix = "a" if article else "s"
        title = record["title"] if article else ""
        for index, piece in enumerate(pieces, start=1):
            next_piece = pieces[index] if index < len(pieces) else ""
            if article:
                sentence = f"{title} {SEP} {piece} {SEP} {next_piece}".strip()
            else:
                sentence = f"{SEP} {piece} {SEP}"
            output_rows.append({
                "id": f"{prefix}_{text_number:08d}_{index:02d}",
                "sentence": sentence,
                "origin_id": record["origin_id"],
            })

    frame = pd.DataFrame(output_rows, columns=["id", "sentence", "origin_id"])
    frame.to_csv(SENTENCES_CSV, index=False, encoding="utf-8-sig")
    update_run_json(prepare={
        "raw_records": len(records),
        "kept_texts": kept_texts,
        "dropped_short": dropped_short,
        "sentence_units": len(frame),
        "output": SENTENCES_CSV.name,
        "output_sha256": sha256_file(SENTENCES_CSV),
    })
    log("prepare 完成", raw=len(records), kept=kept_texts, units=len(frame), output=str(SENTENCES_CSV))


def filter_units(args: argparse.Namespace) -> None:
    ensure_dirs()
    import filter as filter_module
    result = filter_module.run_filter(
        sentences_path=SENTENCES_CSV,
        output_path=COMPARISON_CSV,
        model_dir=args.filter_model_dir,
        lexicon_dir=LEXICON_DIR,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        title_threshold=args.title_threshold,
        device=args.device,
    )
    update_run_json(filter={
        "rows": len(result),
        "output": COMPARISON_CSV.name,
        "output_sha256": sha256_file(COMPARISON_CSV),
    })
    log("filter 完成", rows=len(result), output=str(COMPARISON_CSV))


def _complete_missing_quads(frame: pd.DataFrame) -> pd.DataFrame:
    """同一原始文本内按字段补全；补不到的四元组丢弃，不丢整块。"""
    if frame.empty:
        return frame
    frame = frame.copy()
    for column in ["subject_raw", "object_raw", "aspect_theme", "score_raw"]:
        if column not in frame.columns:
            frame[column] = ""
    groups = frame.groupby("text_group_id", sort=False)
    completed: List[pd.Series] = []
    for _, group in groups:
        group = group.copy()
        fallback = {
            "subject_raw": next((str(value) for value in group["subject_raw"] if str(value).strip()), ""),
            "object_raw": next((str(value) for value in group["object_raw"] if str(value).strip()), ""),
            "aspect_theme": next((str(value) for value in group["aspect_theme"] if str(value).strip()), "其他"),
            "score_raw": next((value for value in group["score_raw"] if str(value).strip()), 0.0),
        }
        for _, row in group.iterrows():
            row = row.copy()
            for column, value in fallback.items():
                if not str(row.get(column, "")).strip():
                    row[column] = value
            if not str(row["subject_raw"]).strip() or not str(row["object_raw"]).strip():
                continue
            completed.append(row)
    if not completed:
        return pd.DataFrame(columns=QUAD_RAW_COLUMNS)
    return pd.DataFrame(completed).reset_index(drop=True)


def extract_quads(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError("先运行 filter 生成 comparison_sentences.csv")
    import gts
    frame = pd.read_csv(COMPARISON_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    if frame.empty:
        pd.DataFrame(columns=QUAD_RAW_COLUMNS).to_csv(QUADS_RAW_CSV, index=False, encoding="utf-8-sig")
        update_run_json(extract={"rows": 0, "output": QUADS_RAW_CSV.name})
        log("extract 完成", quads=0, output=str(QUADS_RAW_CSV))
        return
    rows = gts.predict_dataframe(
        frame=frame,
        model_dir=args.gts_model_dir,
        max_length=args.max_length,
        max_pairs=args.max_pairs,
        pair_threshold=args.pair_threshold,
        device=args.device,
    )
    result = pd.DataFrame(rows, columns=QUAD_RAW_COLUMNS)
    result = _complete_missing_quads(result)
    result = result[QUAD_RAW_COLUMNS]
    result.to_csv(QUADS_RAW_CSV, index=False, encoding="utf-8-sig")
    update_run_json(extract={
        "rows": len(result),
        "output": QUADS_RAW_CSV.name,
        "output_sha256": sha256_file(QUADS_RAW_CSV),
    })
    log("extract 完成", quads=len(result), output=str(QUADS_RAW_CSV))


def load_alias_map() -> Dict[str, str]:
    path = LEXICON_DIR / "brand_alias.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return {str(row["alias"]).strip(): str(row["brand_std"]).strip() for _, row in frame.iterrows() if str(row["alias"]).strip()}


def load_category_terms() -> List[str]:
    path = LEXICON_DIR / "category_terms.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    frame = frame[frame["is_alcohol"].astype(str).eq("1") & ~frame["category"].eq("黄酒")]
    return sorted({str(value).strip() for value in frame["term"] if str(value).strip()}, key=len, reverse=True)


def load_aspect_map() -> List[Tuple[str, str]]:
    path = LEXICON_DIR / "aspect_map.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    records = []
    for _, row in frame.iterrows():
        keyword = str(row["keyword"]).strip()
        theme = str(row["aspect_theme"]).strip()
        if keyword and theme in ASPECT_THEMES:
            records.append((keyword, theme))
    return sorted(records, key=lambda item: len(item[0]), reverse=True)


def normalize_entity(value: str, alias_map: Dict[str, str]) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    for alias in sorted(alias_map, key=len, reverse=True):
        if alias in text:
            return "黄酒"
    return text


def normalize_aspect(value: str, aspect_map: List[Tuple[str, str]]) -> str:
    text = str(value or "").strip()
    if text in ASPECT_THEMES:
        return text
    for keyword, theme in aspect_map:
        if keyword in text:
            return theme
    return "其他"


def is_explicit_category(value: str, category_terms: Sequence[str]) -> bool:
    text = str(value or "")
    if any(suffix in text for suffix in BLOCKED_SUFFIXES):
        return False
    return any(term and term in text for term in category_terms)


def text_group_of(mention_id: str) -> str:
    return "_".join(mention_id.split("_")[:2])


def normalize(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not QUADS_RAW_CSV.exists():
        raise FileNotFoundError("先运行 extract 生成 quads_raw.csv")
    raw = pd.read_csv(QUADS_RAW_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    alias_map = load_alias_map()
    category_terms = load_category_terms()
    aspect_map = load_aspect_map()

    rows: List[Dict[str, Any]] = []
    for _, row in raw.iterrows():
        mention_id = str(row.get("mention_id", "")).strip()
        if not mention_id:
            continue
        group_id = text_group_of(mention_id)
        subject_raw = str(row.get("subject_raw", "")).strip()
        object_raw = str(row.get("object_raw", "")).strip()
        subject_norm = normalize_entity(subject_raw, alias_map)
        object_norm = normalize_entity(object_raw, alias_map)
        try:
            score = float(row.get("score_raw", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0

        if object_norm == "黄酒" and subject_norm != "黄酒":
            subject_norm, object_norm = object_norm, subject_norm
            subject_raw, object_raw = object_raw, subject_raw
            score = -score
        elif subject_norm == "黄酒":
            pass
        else:
            continue

        if subject_norm != "黄酒":
            continue
        if object_norm == "黄酒":
            continue
        if not is_explicit_category(object_norm, category_terms):
            continue

        aspect_theme = normalize_aspect(row.get("aspect_theme", ""), aspect_map)
        rows.append({
            "mention_id": mention_id,
            "text_group_id": group_id,
            "subject_norm": subject_norm,
            "object_norm": object_norm,
            "aspect_theme": aspect_theme,
            "direction_norm": "黄酒→其他",
            "score": score,
            "source": str(row.get("source", "")),
        })

    normalized = pd.DataFrame(rows)
    if normalized.empty:
        final = pd.DataFrame(columns=["id", "比较客体", "比较方面", "比较得分", "origin_id"])
        final.to_excel(QUADS_FINAL_XLSX, index=False)
        update_run_json(normalize={"rows": 0, "output": QUADS_FINAL_XLSX.name})
        log("normalize 完成", rows=0, output=str(QUADS_FINAL_XLSX))
        return

    grouped_rows: List[Dict[str, Any]] = []
    for (group_id, subject_norm, object_norm, aspect_theme, direction_norm), group in normalized.groupby(
        ["text_group_id", "subject_norm", "object_norm", "aspect_theme", "direction_norm"], sort=False
    ):
        # 唯一 mention 去重；同一重复最多使用 5 条
        unique = group.drop_duplicates("mention_id", keep="first").copy()
        unique = unique.head(5)
        grouped_rows.append({
            "text_group_id": group_id,
            "subject_norm": subject_norm,
            "object_norm": object_norm,
            "aspect_theme": aspect_theme,
            "direction_norm": direction_norm,
            "score": float(pd.to_numeric(unique["score"], errors="coerce").fillna(0).mean()),
            "origin_id": str(unique["mention_id"].min()),
            "n_raw_mentions": int(len(group)),
            "n_used_mentions": int(len(unique)),
        })

    merged = pd.DataFrame(grouped_rows).sort_values("origin_id").reset_index(drop=True)
    merged["id"] = ""
    for group_id, group in merged.groupby("text_group_id", sort=False):
        ordered = group.sort_values("origin_id")
        for index, (row_index, _) in enumerate(ordered.iterrows(), start=1):
            merged.at[row_index, "id"] = f"{group_id}_quad{index:07d}"
    merged = merged.sort_values("origin_id").reset_index(drop=True)
    final = merged.rename(columns={
        "object_norm": "比较客体",
        "aspect_theme": "比较方面",
        "score": "比较得分",
    })[["id", "比较客体", "比较方面", "比较得分", "origin_id"]]
    final.to_excel(QUADS_FINAL_XLSX, index=False)
    update_run_json(normalize={
        "rows": len(final),
        "output": QUADS_FINAL_XLSX.name,
        "output_sha256": sha256_file(QUADS_FINAL_XLSX),
    })
    log("normalize 完成", rows=len(final), output=str(QUADS_FINAL_XLSX))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="黄酒比较语言筛选与四元组抽取")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="原始表 -> sentences.csv")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--min-len", type=int, default=6)
    p.set_defaults(func=prepare)

    p = sub.add_parser("filter", help="筛选比较句并链式分段")
    p.add_argument("--filter-model-dir", type=Path, default=FILTER_DIR)
    p.add_argument("--max-sentences", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--title-threshold", type=float, default=0.9)
    p.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:N；默认 cpu")
    p.set_defaults(func=filter_units)

    p = sub.add_parser("extract", help="GTS 抽取四元组")
    p.add_argument("--gts-model-dir", type=Path, default=GTS_DIR)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=30)
    p.add_argument("--pair-threshold", type=float, default=0.1)
    p.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:N；默认 cpu")
    p.set_defaults(func=extract_quads)

    p = sub.add_parser("normalize", help="规范化、合并并导出 quads_final.xlsx")
    p.set_defaults(func=normalize)

    p = sub.add_parser("all", help="filter -> extract -> normalize")
    p.add_argument("--filter-model-dir", type=Path, default=FILTER_DIR)
    p.add_argument("--gts-model-dir", type=Path, default=GTS_DIR)
    p.add_argument("--max-sentences", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--title-threshold", type=float, default=0.9)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=30)
    p.add_argument("--pair-threshold", type=float, default=0.1)
    p.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:N；默认 cpu")
    p.set_defaults(func=lambda args: (filter_units(args), extract_quads(args), normalize(args)))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        log("命令失败", level="ERROR", command=args.command, error=str(exc))
        raise


if __name__ == "__main__":
    main()
