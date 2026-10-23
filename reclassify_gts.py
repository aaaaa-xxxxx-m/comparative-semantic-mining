"""根据人工抽检规则修正 GTS 关系标签。

原则：
- 主客体顺序和方向不在这里处理，交给规范化阶段；
- 明确不成立的 pair 保持 relation=0；
- 实际构成比较但被标成 0 的 pair 改为 relation=1；
- 明确不是实体关系的正例改回 relation=0；
- test 不自动改，避免训练后再修改测试金标。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "gts"
OUTPUT = BASE_DIR / "output"
AUDIT = OUTPUT / "gts_reclassification_report.csv"

# 人工抽检明确为“假负例”的样本。
# score 按当前 pair 顺序给出；规范化阶段会处理主客体交换。
FALSE_NEGATIVE_CORRECTIONS = {
    "a_81000193_cut02_n466": ("产品体验", 0.5),
    "a_81000599_cut02_n967": ("产品体验", -0.5),
    "a_81000247_cut03_n608": ("产品体验", 0.4),
    "a_81000211_cut02_n497": ("产品体验", -0.6),
    "a_81000245_lex1571": ("产品体验", -0.5),
    "a_80000601_cut04_n012": ("产品体验", 0.4),
}
DUPLICATE_NEGATIVE_IDS = {"a_80000601_cut02_n002"}
GENERIC_TERMS = ["这个", "这个牌子", "这种", "那瓶", "这东西", "你", "三大巨头", "便宜那瓶", "这次", "每一次", "超市只有八年的", "这种在超市直播卖的", "真正喝黄酒的", "洋气货", "老底子的东西", "这个品牌的黄酒"]
AGGREGATE_TERMS = ["全行业", "行业", "市场", "营收", "利润", "销量", "净利润", "持平"]
ATTRIBUTE_RE = re.compile(r"\d+年以上|年酒零|酒零\d+年|纯年份原浆")


def invalid_positive(row: pd.Series) -> str:
    subject = str(row["subject_raw"])
    obj = str(row["object_raw"])
    joined = subject + obj
    if subject == obj:
        return "same_entity"
    if any(term in joined for term in GENERIC_TERMS):
        return "generic_entity"
    if any(term in joined for term in AGGREGATE_TERMS):
        return "aggregate_entity"
    if ATTRIBUTE_RE.search(joined):
        return "attribute_entity"
    return ""


def reclass_split(split: str) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    path = DATA / f"gts_{split}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    frame["relation"] = frame["relation"].astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    for column in ["reclassified_to_positive", "reclassified_to_negative", "reclass_reason"]:
        frame[column] = ""
    audit: List[Dict[str, Any]] = []
    drop_index = []
    for index, row in frame.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id in DUPLICATE_NEGATIVE_IDS and split != "test":
            drop_index.append(index)
            audit.append({"split": split, "pair_id": pair_id, "action": "drop_duplicate", "reason": "人工抽检重复"})
            continue
        if split != "test" and pair_id in FALSE_NEGATIVE_CORRECTIONS:
            aspect, score = FALSE_NEGATIVE_CORRECTIONS[pair_id]
            frame.at[index, "relation"] = 1
            frame.at[index, "aspect_theme"] = aspect
            frame.at[index, "score"] = score
            frame.at[index, "reclassified_to_positive"] = "1"
            frame.at[index, "reclass_reason"] = "review_false_negative"
            audit.append({"split": split, "pair_id": pair_id, "action": "to_positive", "reason": "人工抽检确认构成比较"})
            continue
        if split != "test" and int(row["relation"]) == 1:
            reason = invalid_positive(row)
            if reason:
                frame.at[index, "relation"] = 0
                frame.at[index, "aspect_theme"] = "其他"
                frame.at[index, "score"] = 0.0
                frame.at[index, "reclassified_to_negative"] = "1"
                frame.at[index, "reclass_reason"] = reason
                audit.append({"split": split, "pair_id": pair_id, "action": "to_negative", "reason": reason})
    if drop_index:
        frame = frame.drop(index=drop_index).reset_index(drop=True)
    return frame, audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_audit: List[Dict[str, Any]] = []
    counts = {}
    for split in ("train", "valid", "test"):
        if split == "test":
            frame = pd.read_csv(DATA / "gts_test.csv", encoding="utf-8-sig", dtype=str).fillna("")
            frame["relation"] = frame["relation"].astype(int)
            counts[split] = {
                "rows": len(frame),
                "positive": int((frame["relation"] == 1).sum()),
                "negative": int((frame["relation"] == 0).sum()),
                "to_positive": 0,
                "to_negative": 0,
            }
            all_audit.append({"split": split, "pair_id": "", "action": "untouched", "reason": "测试集不自动改"})
            continue
        frame, audit = reclass_split(split)
        frame.to_csv(DATA / f"gts_{split}.csv", index=False, encoding="utf-8-sig")
        counts[split] = {
            "rows": len(frame),
            "positive": int((frame["relation"] == 1).sum()),
            "negative": int((frame["relation"] == 0).sum()),
            "to_positive": int(pd.to_numeric(frame["reclassified_to_positive"], errors="coerce").fillna(0).sum()),
            "to_negative": int(pd.to_numeric(frame["reclassified_to_negative"], errors="coerce").fillna(0).sum()),
        }
        all_audit.extend(audit)
    pd.DataFrame(all_audit).to_csv(AUDIT, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(counts).T.to_string())
    print("audit", AUDIT)


if __name__ == "__main__":
    main()
