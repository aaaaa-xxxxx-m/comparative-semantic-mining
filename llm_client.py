"""OpenAI-compatible LLM client for GLM / DeepSeek data scripts."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import requests

BASE_DIR = Path(__file__).resolve().parent
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
GLM_MODEL = "glm-5.3"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-flash"


def load_dotenv(path: Path | None = None) -> None:
    path = path or (BASE_DIR / ".env")
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


def call_glm(prompt: str, timeout: int = 180) -> str:
    provider = os.environ.get("LLM_PROVIDER", "deepseek").strip().lower()
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
            if auth_path.exists():
                try:
                    api_key = str(json.loads(auth_path.read_text(encoding="utf-8")).get("deepseek", {}).get("key", "")).strip()
                except Exception:
                    api_key = ""
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
        model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
        reasoning = os.environ.get("DEEPSEEK_REASONING_EFFORT", "low").strip().lower()
        max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096"))
    else:
        api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 ZHIPU_API_KEY")
        base_url = os.environ.get("ZHIPU_BASE_URL", GLM_BASE_URL)
        model = os.environ.get("ZHIPU_MODEL", GLM_MODEL)
        reasoning = os.environ.get("GLM_REASONING_EFFORT", "low").strip().lower()
        max_tokens = int(os.environ.get("GLM_MAX_TOKENS", "4096"))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出合法 JSON，不输出 Markdown 或解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning,
    }
    last_error = ""
    for attempt in range(1, 6):
        try:
            response = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 15 * attempt)
                print(f"[deepseek] 429 rate limit, sleep {wait}s (attempt {attempt}/5)", flush=True)
                time.sleep(wait)
                last_error = "429 Too Many Requests"
                continue
            if response.status_code >= 500:
                time.sleep(min(30, 5 * attempt))
                last_error = f"HTTP {response.status_code}"
                continue
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            return str(content)
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")
