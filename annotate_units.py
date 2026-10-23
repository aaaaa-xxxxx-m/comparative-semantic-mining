"""标注原始/生成训练单元，输出句子标签和四元组。"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from llm_client import call_glm, extract_json, load_dotenv
from text_utils import split_article

BASE_DIR = Path(__file__).resolve().parent
ASPECTS = ["文化认知", "消费场景", "产品体验", "竞争压力", "其他"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_for(unit: Dict[str, Any], sentences: List[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    label = unit.get("label")
    label_text = "" if label is None else ("目标标签：正例" if int(label) == 1 else "目标标签：负例")
    return f"""你是黄酒比较语义标注员。只根据给定文本标注，不要补充外部事实。

平台：{unit.get("platform","")}
标题：{unit.get("title","") or "无"}
{label_text}
正文：
{numbered}

逐句判断，只输出一个 JSON 对象，不要 Markdown：
{{"sentences":[{{"index":0,"comparison":0,"title_comparison":0,"quads":[{{"subject":"原文主体","object":"原文客体","aspect_theme":"文化认知|消费场景|产品体验|竞争压力|其他","score":0.0,"subject_source":"title|body","object_source":"title|body"}}]}}]}}

要求：
- sentences 必须正好 {len(sentences)} 项，index 对应输入。
- comparison：该句是否有黄酒相关比较语义。
- title_comparison：标题与该句合起来是否形成比较；无标题填 0。
- quads 为空数组或有效四元组；正例至少给出一条有效四元组，负例 quads 为空。
- subject 表示黄酒一侧，object 是参照对象；score 为 subject 相对 object，范围 -1 到 1。
- subject/object 必须是标题或正文中连续出现的原文，不要改写。
- 只抽取明确的比较关系；仅是共现、联想、行业描述或单纯推荐，不要抽成四元组。
- 每句最多 2 条，全篇最多 4 条；优先保留不同方面、不同客体的四元组。
- 相同 subject/object/aspect 只保留一次；不要把同一比较拆成近义重复。
- 隐式比较可以抽，但必须能从原句直接推出主体、客体、方面和倾向。
- aspect_theme 只能从五个值中选。"""


def normalize(unit: Dict[str, Any], payload: Dict[str, Any], sentences: List[str]) -> Dict[str, Any]:
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list) or len(raw_sentences) != len(sentences):
        raise ValueError("句数不一致")
    source_text = str(unit.get("title", "")) + str(unit.get("text", ""))
    out = []
    for i, sentence in enumerate(sentences):
        raw = raw_sentences[i]
        if not isinstance(raw, dict):
            raise ValueError("句子标注不是对象")
        quads = []
        seen = set()
        for item in raw.get("quads", []) if isinstance(raw.get("quads"), list) else []:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subject or not obj or subject == obj:
                continue
            if subject not in source_text or obj not in source_text:
                continue
            aspect = str(item.get("aspect_theme", "其他")).strip()
            if aspect not in ASPECTS:
                aspect = "其他"
            key = (subject, obj, aspect)
            if key in seen:
                continue
            seen.add(key)
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            quads.append({
                "subject": subject,
                "object": obj,
                "aspect_theme": aspect,
                "score": max(-1.0, min(1.0, score)),
                "subject_source": str(item.get("subject_source", "body")).strip() or "body",
                "object_source": str(item.get("object_source", "body")).strip() or "body",
            })
        quads = quads[:2]
        out.append({
            "index": i,
            "unit_id": f"{unit['article_id']}_cut{i + 1:02d}",
            "comparison": 1 if int(raw.get("comparison", 0)) == 1 else 0,
            "title_comparison": 1 if int(raw.get("title_comparison", 0)) == 1 else 0,
            "quads": quads,
        })
    remaining = 4
    for item in out:
        item["quads"] = item["quads"][:remaining]
        remaining -= len(item["quads"])
    if unit.get("label") is not None and int(unit["label"]) == 1 and not any(x["quads"] for x in out):
        raise ValueError("正例没有有效四元组")
    return {
        "article_id": unit["article_id"],
        "split": unit.get("split", "train"),
        "source": unit.get("source", "raw"),
        "platform": unit.get("platform", ""),
        "label": unit.get("label"),
        "annotator_model": "glm-5.3-flash",
        "annotation_version": "balanced-v2",
        "sentences": out,
    }


def annotate_one(unit: Dict[str, Any], timeout: int, attempts: int) -> Dict[str, Any]:
    sentences = split_article(unit["text"])
    if not sentences:
        raise ValueError("没有句子")
    prompt = prompt_for(unit, sentences)
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            return normalize(unit, extract_json(call_glm(prompt, timeout=timeout)), sentences)
        except Exception as exc:
            last = str(exc)
            if attempt < attempts:
                prompt += "\n\n上一次输出不合格，请严格按句数和原文 span 重写。"
                time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(last)


def main() -> None:
    parser = argparse.ArgumentParser(description="标注原始和生成训练单元")
    parser.add_argument("--raw", type=Path, default=BASE_DIR / "data" / "raw" / "units.jsonl")
    parser.add_argument("--generated", type=Path, default=BASE_DIR / "data" / "generated" / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "annotations")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_dotenv()
    raw = load_jsonl(args.raw)
    generated = load_jsonl(args.generated)
    units = raw + generated
    if args.limit:
        units = units[:args.limit]
    outputs = {"raw": args.output_dir / "raw.jsonl", "generated": args.output_dir / "generated.jsonl"}
    done = {row["article_id"] for rows in (load_jsonl(p) for p in outputs.values()) for row in rows}
    pending = [u for u in units if u["article_id"] not in done]
    print(f"units={len(units)} done={len(done)} pending={len(pending)}", flush=True)
    lock = threading.Lock()
    completed = len(done)
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(annotate_one, u, args.timeout, args.max_attempts): u for u in pending}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    with open(args.output_dir / "failed.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"article_id": unit["article_id"], "error": str(exc)}, ensure_ascii=False) + "\n")
                    print("FAILED", unit["article_id"], exc, flush=True)
                    continue
                with lock:
                    append_jsonl(outputs[item["source"]], item)
                    completed += 1
                print(f"[{completed}/{len(units)}] {item['article_id']} {item['source']} quads={sum(len(x['quads']) for x in item['sentences'])}", flush=True)
    except KeyboardInterrupt:
        print("收到暂停，已落盘。", flush=True)
    print(f"done={completed} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
