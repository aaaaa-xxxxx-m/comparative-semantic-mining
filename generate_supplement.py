"""按原始平台风格生成训练补充数据，不生成验证/测试数据。"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from llm_client import call_glm, extract_json, load_dotenv
from text_utils import clean_text, has_comparison_cue, is_article, split_article

BASE_DIR = Path(__file__).resolve().parent
RAW_UNITS = BASE_DIR / "data" / "raw" / "units.jsonl"
OUTPUT = BASE_DIR / "data" / "generated" / "train.jsonl"
FAILED = BASE_DIR / "data" / "generated" / "failed.jsonl"
QUOTAS = {"XW": 120, "ZH": 60, "WB": 80, "XHS": 80, "JD": 80, "PDD": 60, "TB": 60, "BZ": 30, "DY": 30}
LONG_FORM = {"XW", "ZH"}
TARGET_SENTENCES = {"XW": "4-8", "ZH": "4-8", "WB": "2-5", "XHS": "2-5", "JD": "1-4", "PDD": "1-4", "TB": "1-4", "BZ": "1-3", "DY": "1-3"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_refs(units: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    refs: Dict[str, List[Dict[str, Any]]] = {}
    for platform in QUOTAS:
        rows = [unit for unit in units if unit["platform"] == platform and unit["split"] == "train"]
        rng.shuffle(rows)
        refs[platform] = rows[:4]
    return refs


def build_prompt(platform: str, label: int, refs: List[Dict[str, Any]], seed: int) -> str:
    examples = []
    for unit in refs[:3]:
        text = unit["text"][:220]
        examples.append(f"标题：{unit.get('title','') or '无'}\n正文：{text}")
    style_text = "\n\n".join(examples)
    label_text = "正例：必须形成黄酒与另一对象的有效比较" if label == 1 else "负例：不要形成有效的黄酒比较四元组"
    title_rule = "必须填写标题" if platform in LONG_FORM else "标题留空"
    return f"""你要模仿下面的真实中文平台文本风格，但不要照抄句子或事实。

平台：{platform}
风格参考：
{style_text}

生成一篇{label_text}的文本。
- 正文 {TARGET_SENTENCES[platform]} 句，语言自然，像真实用户/媒体，不要模板化。
- {title_rule}。
- 不要编造检测数据、奖项或不存在的新闻。
- 只输出 JSON：{{"title":"标题或空字符串","text":"正文"}}。"""


def validate(payload: Dict[str, Any], platform: str, label: int) -> Dict[str, str]:
    title = clean_text(str(payload.get("title", "")))
    text = clean_text(str(payload.get("text", "")))
    if len(text) < 20 or len(text) > 1500:
        raise ValueError("正文长度不合格")
    count = len(split_article(text))
    if platform in LONG_FORM:
        if not title:
            raise ValueError("长文缺少标题")
        if not 4 <= count <= 8:
            raise ValueError(f"长文句数 {count} 不合格")
    else:
        if count < 1 or count > 5:
            raise ValueError(f"短文本句数 {count} 不合格")
    if label == 1 and not has_comparison_cue(text):
        raise ValueError("正例缺少比较词")
    return {"title": title, "text": text}


def generate_one(task: Dict[str, Any], refs: List[Dict[str, Any]], timeout: int, attempts: int) -> Dict[str, Any]:
    prompt = build_prompt(task["platform"], task["label"], refs, task["index"])
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            payload = extract_json(call_glm(prompt, timeout=timeout))
            data = validate(payload, task["platform"], task["label"])
            return {
                "article_id": task["article_id"],
                "split": "train",
                "source": "generated",
                "platform": task["platform"],
                "source_style": "news" if task["platform"] == "XW" else "forum" if task["platform"] == "ZH" else "social",
                "style_ref_id": task["style_ref_id"],
                "label": task["label"],
                "title": data["title"],
                "text": data["text"],
                "unit_type": "article" if is_article(data["text"]) else "comment",
                "has_cue": int(has_comparison_cue(data["text"])),
            }
        except Exception as exc:
            last = str(exc)
            if attempt < attempts:
                time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(last)


def main() -> None:
    parser = argparse.ArgumentParser(description="按原始平台风格生成补充训练数据")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    load_dotenv()
    units = load_jsonl(RAW_UNITS)
    refs = style_refs(units, args.seed)
    done = {}
    if OUTPUT.exists():
        for row in load_jsonl(OUTPUT):
            done[row["article_id"]] = row
    tasks = []
    offset = 0
    for platform, quota in QUOTAS.items():
        for i in range(quota):
            offset += 1
            tasks.append({
                "article_id": f"a_{81000000 + offset:08d}",
                "platform": platform,
                "label": 1 if i % 2 == 0 else 0,
                "index": offset,
                "style_ref_id": refs[platform][i % max(1, len(refs[platform]))]["article_id"] if refs[platform] else "",
            })
    if args.limit:
        tasks = tasks[:args.limit]
    pending = [task for task in tasks if task["article_id"] not in done]
    print(f"planned={len(tasks)} done={len(done)} pending={len(pending)}", flush=True)
    lock = threading.Lock()
    completed = len(done)
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(generate_one, task, refs[task["platform"]], args.timeout, args.max_attempts): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    with open(FAILED, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"article_id": task["article_id"], "error": str(exc)}, ensure_ascii=False) + "\n")
                    print("FAILED", task["article_id"], exc, flush=True)
                    continue
                with lock:
                    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                    with open(OUTPUT, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    done[item["article_id"]] = item
                    completed += 1
                print(f"[{completed}/{len(tasks)}] {item['article_id']} {item['platform']} label={item['label']} {item['text'][:30]}", flush=True)
    except KeyboardInterrupt:
        print("收到暂停，已完成数据已落盘。", flush=True)
    print(f"done={len(done)} output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
