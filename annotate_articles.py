"""给已生成文章补句子级标签与四元组，并组装 filter/GTS 数据。

输入：
    data/articles.jsonl

输出：
    data/annotations.jsonl
    data/filter/filter_train.jsonl
    data/filter/filter_valid.jsonl
    data/filter/filter_test.jsonl
    data/gts/gts_train.csv
    data/gts/gts_valid.csv
    data/gts/gts_test.csv

只使用正例文章生成四元组；负例文章仍进入 filter 负例。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from generate_articles import call_glm, extract_json, load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTICLES = BASE_DIR / "data" / "articles.jsonl"
DEFAULT_ANNOTATIONS = BASE_DIR / "data"
DEFAULT_FILTER_DIR = BASE_DIR / "data" / "filter"
DEFAULT_GTS_DIR = BASE_DIR / "data" / "gts"
ASPECTS = ["文化认知", "消费场景", "产品体验", "竞争压力", "其他"]
SEP = "[SEP]"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def split_sentences(text: str) -> List[str]:
    return [item.strip() for item in re.split(r"(?<=[。？！])", str(text)) if item.strip()]


def load_articles(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(article: Dict[str, Any], sentences: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences))
    return f"""你是黄酒比较语义标注员。请只根据给定文章标注，不要补充文章外的事实。

标题：{article["title"]}
文章标签：{"正例" if int(article["label"]) == 1 else "负例"}
正文句子：
{numbered}

逐句判断并只输出一个 JSON 对象，不要 Markdown，不要解释：
{{
  "sentences": [
    {{
      "index": 0,
      "comparison": 0,
      "title_comparison": 0,
      "quads": [
        {{
          "subject": "文中原文主体",
          "object": "文中原文客体",
          "aspect_theme": "文化认知|消费场景|产品体验|竞争压力|其他",
          "score": 0.0,
          "subject_source": "title|body",
          "object_source": "title|body"
        }}
      ]
    }}
  ]
}}

要求：
- sentences 必须覆盖全部 {len(sentences)} 个句子，index 必须与输入编号一一对应。
- comparison：该句本身是否包含黄酒相关的比较语义；没有比较语义填 0。
- title_comparison：标题与该句合起来是否形成比较语义；没有填 0。
- 只有正例文章才允许填写 quads；负例文章的 quads 必须为空数组。
- subject 是“黄酒这一侧”的主体，object 是参照对象；score 表示 subject 相对 object 的好坏，范围 -1 到 1。
- subject/object 必须是标题或正文中真实出现的连续原文，不要改写。
- 每个四元组只能归到一个最直接的句子；同一句可有多条，但不要重复。
- aspect_theme 只能从五个值中选一个。
- 如果某句确实没有可抽取的四元组，quads 填空数组。"""


def normalize_annotation(
    payload: Dict[str, Any],
    article: Dict[str, Any],
    sentences: Sequence[str],
    attempts: int,
) -> Dict[str, Any]:
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list) or len(raw_sentences) != len(sentences):
        raise ValueError("标注句数与原文不一致")
    normalized: List[Dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        raw = raw_sentences[index]
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 句标注不是对象")
        comparison = 1 if int(raw.get("comparison", 0)) == 1 else 0
        title_comparison = 1 if int(raw.get("title_comparison", 0)) == 1 else 0
        quads: List[Dict[str, Any]] = []
        raw_quads = raw.get("quads", [])
        if int(article["label"]) == 1 and isinstance(raw_quads, list):
            for item in raw_quads[:5]:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                obj = str(item.get("object", "")).strip()
                if not subject or not obj or subject == obj:
                    continue
                if subject not in article["title"] + article["text"] or obj not in article["title"] + article["text"]:
                    continue
                aspect = str(item.get("aspect_theme", "其他")).strip()
                if aspect not in ASPECTS:
                    aspect = "其他"
                try:
                    score = float(item.get("score", 0))
                except (TypeError, ValueError):
                    score = 0.0
                score = max(-1.0, min(1.0, score))
                quads.append({
                    "subject": subject,
                    "object": obj,
                    "aspect_theme": aspect,
                    "score": score,
                    "sentence_index": index,
                    "subject_source": str(item.get("subject_source", "body")).strip() or "body",
                    "object_source": str(item.get("object_source", "body")).strip() or "body",
                })
        normalized.append({
            "index": index,
            "unit_id": f"{article['article_id']}_cut{index + 1:02d}",
            "comparison": comparison,
            "title_comparison": title_comparison,
            "quads": quads,
        })
    if int(article["label"]) == 1 and not any(item["quads"] for item in normalized):
        raise ValueError("正例文章没有有效四元组")
    return {
        "article_id": article["article_id"],
        "split": article["split"],
        "label": int(article["label"]),
        "sentences": normalized,
        "attempts": attempts,
        "annotated_at": now(),
    }


def annotate_one(
    article: Dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_attempts: int,
) -> Dict[str, Any]:
    sentences = split_sentences(article["text"])
    prompt = build_prompt(article, sentences)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_glm(prompt, api_key, base_url, model, timeout)
            return normalize_annotation(extract_json(raw), article, sentences, attempt)
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_attempts:
                prompt += "\n\n上一次输出不合格。请严格按句数、原文 span 和 JSON 结构重写。"
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def load_done(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            article_id = str(item.get("article_id", ""))
            if article_id:
                done[article_id] = item
    return done


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_filter_data(
    articles: Sequence[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "valid", "test"):
        rows: List[Dict[str, Any]] = []
        for article in articles:
            if article["split"] != split:
                continue
            sentences = split_sentences(article["text"])
            annotation = annotations.get(article["article_id"])
            flags = [0] * len(sentences)
            title_flags = [0] * len(sentences)
            if annotation:
                for item in annotation["sentences"]:
                    index = int(item["index"])
                    if 0 <= index < len(sentences):
                        flags[index] = int(item["comparison"])
                        title_flags[index] = int(item["title_comparison"])
            for index, sentence in enumerate(sentences):
                unit_id = f"{article['article_id']}_cut{index + 1:02d}"
                rows.append({
                    "article_id": article["article_id"],
                    "unit_id": unit_id,
                    "task": "u",
                    "text": sentence,
                    "label": flags[index],
                })
                rows.append({
                    "article_id": article["article_id"],
                    "unit_id": unit_id,
                    "task": "t",
                    "title": article["title"],
                    "current": sentence,
                    "label": title_flags[index],
                })
                if index > 0:
                    context = sentences[max(0, index - 5):index]
                    label = 1 if flags[index] and flags[index - 1] else 0
                    rows.append({
                        "article_id": article["article_id"],
                        "unit_id": unit_id,
                        "task": "l",
                        "context": context,
                        "next": sentence,
                        "label": label,
                    })
        path = output_dir / f"filter_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def find_span(text: str, value: str, start: int = 0) -> Tuple[int, int]:
    index = text.find(value, start)
    if index < 0:
        index = text.find(value)
    if index < 0:
        raise ValueError(f"span 无法定位：{value}")
    return index, index + len(value)


def build_gts_data(
    articles: Sequence[Dict[str, Any]],
    annotations: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "valid", "test"):
        rows: List[Dict[str, Any]] = []
        for article in articles:
            if article["split"] != split or int(article["label"]) != 1:
                continue
            annotation = annotations.get(article["article_id"])
            if not annotation:
                continue
            sentences = split_sentences(article["text"])
            for sentence_item in annotation["sentences"]:
                sentence_index = int(sentence_item["index"])
                if not 0 <= sentence_index < len(sentences):
                    continue
                sentence = sentences[sentence_index]
                text = f"{article['title']} {SEP} {sentence}"
                seen: set[Tuple[str, str, str]] = set()
                quad_number = 1
                for quad in sentence_item["quads"]:
                    key = (quad["subject"], quad["object"], quad["aspect_theme"])
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        subject_start, subject_end = find_span(text, quad["subject"])
                        object_start, object_end = find_span(text, quad["object"])
                    except ValueError:
                        continue
                    rows.append({
                        "mention_id": f"{article['article_id']}_cut{sentence_index + 1:02d}_{quad_number:02d}",
                        "text_group_id": article["article_id"],
                        "text": text,
                        "subject_raw": quad["subject"],
                        "subject_start": subject_start,
                        "subject_end": subject_end,
                        "object_raw": quad["object"],
                        "object_start": object_start,
                        "object_end": object_end,
                        "aspect_theme": quad["aspect_theme"],
                        "score": quad["score"],
                        "subject_source": quad["subject_source"],
                        "object_source": quad["object_source"],
                        "source": "annotation",
                    })
                    quad_number += 1
        path = output_dir / f"gts_{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        counts[split] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="给文章补句子标签与四元组")
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--filter-dir", type=Path, default=DEFAULT_FILTER_DIR)
    parser.add_argument("--gts-dir", type=Path, default=DEFAULT_GTS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("缺少 ZHIPU_API_KEY；请写入 compare/.env。")
    model = (args.model or os.environ.get("ZHIPU_MODEL", "glm-5.3")).strip() or "glm-5.3"
    base_url = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

    articles = load_articles(args.articles)
    positives = [article for article in articles if int(article["label"]) == 1]
    if args.limit is not None:
        positives = positives[:args.limit]
    annotations_path = args.output_dir / "annotations.jsonl"
    failed_path = args.output_dir / "annotations.failed.jsonl"
    done = load_done(annotations_path)
    pending = [article for article in positives if article["article_id"] not in done]
    print(f"正例文章 {len(positives)} 篇，已完成 {len(done)} 篇，待标注 {len(pending)} 篇。", flush=True)
    if not pending:
        pass
    else:
        lock = threading.Lock()
        completed = len(done)
        try:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        annotate_one,
                        article,
                        api_key,
                        base_url,
                        model,
                        args.timeout,
                        args.max_attempts,
                    ): article
                    for article in pending
                }
                for future in as_completed(futures):
                    article = futures[future]
                    try:
                        item = future.result()
                    except Exception as exc:
                        append_jsonl(failed_path, {
                            "article_id": article["article_id"],
                            "error": str(exc),
                            "failed_at": now(),
                        })
                        print(f"FAILED {article['article_id']}: {exc}", flush=True)
                        continue
                    with lock:
                        append_jsonl(annotations_path, item)
                        done[item["article_id"]] = item
                        completed += 1
                    print(f"[{completed}/{len(positives)}] {item['article_id']} quads={sum(len(x['quads']) for x in item['sentences'])}", flush=True)
        except KeyboardInterrupt:
            print("收到暂停，已完成标注已落盘。", flush=True)

    filter_counts = build_filter_data(articles, done, args.filter_dir)
    gts_counts = build_gts_data(articles, done, args.gts_dir)
    summary = {
        "articles": len(articles),
        "positive_articles": len(positives),
        "annotated": len(done),
        "filter": filter_counts,
        "gts": gts_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
