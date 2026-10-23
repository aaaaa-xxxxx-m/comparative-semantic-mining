"""应用 test 25 条人工复核结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "gts" / "gts_test.csv"
REVIEW = BASE_DIR.parent / "test_review_25.xlsx"
AUDIT = BASE_DIR / "output" / "test_review_25_applied.csv"

POSITIVE_FIXES = {
    "a_80000815_cut03_p02": {"aspect_theme": "文化认知"},
    "a_80000814_cut03_p01": {"score": 0.3},
}
TO_POSITIVE = {
    "a_80000806_cut03_n007": {"relation": 1, "aspect_theme": "消费场景", "score": "-0.7"},
}
TO_NEGATIVE = {
    "a_80000966_cut01_p01",
    "a_80000936_cut07_p03",
    "a_80000856_cut05_p01",
    "a_80000930_cut08_p01",
}


def main() -> None:
    review = pd.read_excel(REVIEW, dtype=str).fillna("")
    frame = pd.read_csv(DATA, encoding="utf-8-sig", dtype=str).fillna("")
    frame["relation"] = frame["relation"].astype(int)
    for column in ["review_applied", "reclass_reason"]:
        frame[column] = ""
    audit: List[Dict[str, Any]] = []
    for _, decision in review.iterrows():
        valid = str(decision.get("review_valid", "")).strip()
        if valid not in {"是", "否"}:
            continue
        pair_id = str(decision["pair_id"])
        matches = frame.index[frame["pair_id"] == pair_id].tolist()
        if not matches:
            continue
        index = matches[0]
        before = int(frame.at[index, "relation"])
        action = "no_change"
        if pair_id in POSITIVE_FIXES:
            for column, value in POSITIVE_FIXES[pair_id].items():
                frame.at[index, column] = str(value)
            action = "fix_positive"
        elif pair_id in TO_POSITIVE:
            for column, value in TO_POSITIVE[pair_id].items():
                frame.at[index, column] = value if column == "relation" else str(value)
            action = "to_positive"
        elif pair_id in TO_NEGATIVE:
            frame.at[index, "relation"] = 0
            frame.at[index, "aspect_theme"] = "其他"
            frame.at[index, "score"] = "0"
            action = "to_negative"
        if action != "no_change":
            frame.at[index, "review_applied"] = valid
            frame.at[index, "reclass_reason"] = "human_test_review"
        audit.append({
            "pair_id": pair_id, "review_valid": valid, "action": action,
            "before": before, "after": int(frame.at[index, "relation"]),
            "note": decision.get("review_note", ""),
        })
    frame.to_csv(DATA, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(AUDIT, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_excel(BASE_DIR / "output" / "test_review_25_applied.xlsx", index=False)
    counts = {
        "rows": len(frame),
        "positive": int((frame["relation"] == 1).sum()),
        "negative": int((frame["relation"] == 0).sum()),
    }
    print(counts)
    manifest_path = BASE_DIR / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gts_labels"]["test"] = {"positive": counts["positive"], "negative": counts["negative"]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
