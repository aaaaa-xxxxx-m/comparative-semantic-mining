"""用 GLM 生成黄酒比较文章，支持断点续跑。

默认输出：
    data/articles.jsonl
    data/articles.failed.jsonl

运行前在 compare/.env 中设置：
    ZHIPU_API_KEY=...
    ZHIPU_MODEL=glm-5.3

示例：
    python generate_articles.py --dry-run
    python generate_articles.py --workers 4
    python generate_articles.py --limit 10 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-5.3"
SPLITS = [("train", 350, 263), ("valid", 200, 150), ("test", 100, 75)]
SPLIT_OFFSETS = {"train": 90000000, "valid": 90000350, "test": 90000550}
STYLES = ["新闻", "论坛", "社媒", "测评", "问答"]
TOPICS = [
    "黄酒与白酒的口感差异",
    "黄酒与啤酒的饮用场景",
    "黄酒品牌之间的比较",
    "黄酒产品之间的比较",
    "黄酒与葡萄酒的文化认知",
    "黄酒价格与性价比",
    "黄酒年轻化与气泡黄酒",
    "黄酒养生与健康认知",
    "黄酒包装与送礼场景",
    "黄酒与料酒的区别",
]
POSITIVE_CUES = ["比", "不如", "更", "相比", "较", "没有", "胜过", "优于", "赶不上", "逊于"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value and key not in os.environ:
            os.environ[key] = value


def build_plan(seed: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for split, total, positives in SPLITS:
        labels = [1] * positives + [0] * (total - positives)
        rng.shuffle(labels)
        for index, label in enumerate(labels, start=1):
            tasks.append({
                "article_id": f"a_{SPLIT_OFFSETS[split] + index:08d}",
                "split": split,
                "label": label,
                "topic": rng.choice(TOPICS),
                "style": rng.choice(STYLES),
            })
    if limit is not None:
        tasks = tasks[:max(0, limit)]
    return tasks


def build_prompt(task: Dict[str, Any]) -> str:
    label_text = "正例：文章必须包含与黄酒相关的比较语义" if task["label"] == 1 else "负例：文章不要形成有效的黄酒比较四元组"
    comparison = "跨品类/品牌间/产品间均可" if task["label"] == 1 else "可以出现比较词，但不要给出可抽取的黄酒比较结论"
    return f"""你是中文消费内容编辑。请生成一篇自然、具体、可直接阅读的中文文章，只输出一个 JSON 对象，不要 Markdown，不要解释。

主题：{task["topic"]}
风格：{task["style"]}
标签：{label_text}
比较范围：{comparison}

JSON 字段：
- title：文章标题，8-40 个中文字符
- text：正文，3-6 句，250-500 个中文字符，句子之间可用换行
- source_style：{task["style"]}
- comparison_type：正例填“跨品类”“品牌间”“产品间”之一；负例填“无”

写作要求：
- 像真实用户或媒体写的内容，不要模板化，不要出现“作为AI”。
- 正例要让读者清楚看出黄酒与另一个对象在某个方面谁更好，或至少形成明确比较；至少有一处可定位的主体、客体、方面和倾向。
- 负例不要形成有效的“黄酒 vs 其他对象”的四元组；可以只谈黄酒本身、只提行业，或比较与黄酒无关的对象。
- 不要编造具体检测数据、奖项或不存在的新闻来源。
- 只输出 JSON，字段值不要用 Markdown 代码块包裹。"""


def extract_json(text: str) -> Dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出没有 JSON 对象")
    data = json.loads(value[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return data


def validate_article(data: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    title = str(data.get("title", "")).strip()
    text = str(data.get("text", "")).strip()
    if not title or len(title) > 80:
        raise ValueError("标题为空或过长")
    if len(text) < 80 or len(text) > 1500:
        raise ValueError("正文长度不在 80-1500 字符内")
    sentences = [item for item in re.split(r"(?<=[。？！])", text) if item.strip()]
    max_sentences = 8 if task["style"] == "问答" else 6
    if not 3 <= len(sentences) <= max_sentences:
        raise ValueError(f"正文句数为 {len(sentences)}，要求 3-{max_sentences} 句")
    if task["label"] == 1 and not any(cue in text for cue in POSITIVE_CUES):
        raise ValueError("正例没有比较词")
    source_style = str(data.get("source_style", task["style"])).strip() or task["style"]
    comparison_type = str(data.get("comparison_type", "")).strip()
    if task["label"] == 1 and comparison_type not in {"跨品类", "品牌间", "产品间"}:
        comparison_type = "跨品类"
    if task["label"] == 0:
        comparison_type = "无"
    return {
        "article_id": task["article_id"],
        "split": task["split"],
        "label": task["label"],
        "title": title,
        "text": text,
        "source_style": source_style,
        "comparison_type": comparison_type,
        "topic": task["topic"],
        "is_synthetic": 1,
        "model": os.environ.get("ZHIPU_MODEL", DEFAULT_MODEL),
        "generated_at": now(),
        "prompt_version": "v1",
    }


def call_glm(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出合法 JSON，不输出 Markdown 或解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": int(os.environ.get("GLM_MAX_TOKENS", "4096")),
        "reasoning_effort": os.environ.get("GLM_REASONING_EFFORT", "low").strip().lower(),
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise RuntimeError(f"可重试的 HTTP {response.status_code}: {response.text[:200]}")
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


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


def generate_one(
    task: Dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_attempts: int,
) -> Dict[str, Any]:
    prompt = build_prompt(task)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_glm(prompt, api_key, base_url, model, timeout)
            return validate_article(extract_json(raw), task)
        except Exception as exc:
            last_error = str(exc)
            if "句数" in last_error:
                prompt = build_prompt(task) + "\n\n重要：正文必须恰好 5 句，只能用 5 个句号、问号或感叹号结尾，不能多也不能少。"
            if attempt < max_attempts:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 GLM 生成黄酒比较文章")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    tasks = build_plan(args.seed, args.limit)
    output_dir = args.output_dir
    articles_path = output_dir / "articles.jsonl"
    failed_path = output_dir / "articles.failed.jsonl"

    if args.dry_run:
        counts: Dict[str, Dict[str, int]] = {}
        for task in tasks:
            counts.setdefault(task["split"], {"total": 0, "positive": 0, "negative": 0})
            counts[task["split"]]["total"] += 1
            counts[task["split"]]["positive" if task["label"] == 1 else "negative"] += 1
        print(json.dumps({"planned": len(tasks), "counts": counts}, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("缺少 ZHIPU_API_KEY；请写入 compare/.env，或先设置环境变量。")
    model = (args.model or os.environ.get("ZHIPU_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    base_url = os.environ.get("ZHIPU_BASE_URL", DEFAULT_BASE_URL)

    done = load_done(articles_path)
    pending = [task for task in tasks if task["article_id"] not in done]
    if not pending:
        print("所有计划文章已存在，无需生成。")
        return

    print(f"计划 {len(tasks)} 篇，已完成 {len(done)} 篇，待生成 {len(pending)} 篇。", flush=True)
    lock = threading.Lock()
    completed = len(done)
    total = len(tasks)
    stop = False

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    generate_one,
                    task,
                    api_key,
                    base_url,
                    model,
                    args.timeout,
                    args.max_attempts,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    append_jsonl(failed_path, {
                        "article_id": task["article_id"],
                        "error": str(exc),
                        "failed_at": now(),
                    })
                    print(f"FAILED {task['article_id']}: {exc}", flush=True)
                    continue
                with lock:
                    append_jsonl(articles_path, item)
                    done[item["article_id"]] = item
                    completed += 1
                print(
                    f"[{completed}/{total}] {item['article_id']} "
                    f"{item['split']} label={item['label']} {item['title']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        stop = True
        print("收到暂停，已保留已完成文章；下次运行会从断点继续。", flush=True)
    finally:
        if stop:
            print(f"已完成 {len(done)} 篇，文件：{articles_path}", flush=True)


if __name__ == "__main__":
    main()
