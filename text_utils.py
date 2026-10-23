"""共享文本清洗与切句工具，训练和推理共用。"""

from __future__ import annotations

import re
from typing import List, Sequence

COMPARISON_CUES = ["比", "不如", "更", "相比", "较", "没有", "胜过", "优于", "赶不上", "逊于"]


def clean_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"https?://\S+|www\.\S+", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"回复\s*@[^:：]*[:：]", "", value)
    value = re.sub(r"@[\w\-\u4e00-\u9fff]+", "", value)
    value = re.sub(r"#[^#]{1,30}#", "", value)
    return re.sub(r"\s+", " ", value).strip()


def is_article(text: str) -> bool:
    collapsed = re.sub(r"[。？！]+", "。", text)
    return len(re.findall(r"[。？！]", collapsed)) >= 2


def split_article(text: str) -> List[str]:
    return [piece.strip() for piece in re.split(r"(?<=[。？！])", text) if piece.strip()]


def has_comparison_cue(text: str) -> bool:
    return any(cue in str(text or "") for cue in COMPARISON_CUES)


def clip_to_sentences(text: str, max_sentences: int = 8) -> str:
    pieces = split_article(text)
    if len(pieces) <= max_sentences:
        return text
    cue_index = next((i for i, piece in enumerate(pieces) if has_comparison_cue(piece)), 0)
    start = max(0, min(cue_index - 1, len(pieces) - max_sentences))
    return "".join(pieces[start:start + max_sentences])
