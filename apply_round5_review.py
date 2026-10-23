"""应用第五轮人工复核结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "gts"
REVIEW = BASE_DIR.parent / "quad_review_round5.xlsx"
AUDIT = BASE_DIR / "output" / "gts_review_round5_applied.csv"

POSITIVE_FIXES = {
    "a_80000200_cut01_pf01": {"score": 0.0},
    "a_80000450_cut04_pf01": {"aspect_theme": "竞争压力"},
    "a_80000136_cut03_pf01": {"aspect_theme": "文化认知", "score": 0.6},
    "a_80000709_cut01_p01": {"aspect_theme": "消费场景", "score": 0.3},
    "a_80000759_cut02_p01": {"score": -0.6},
    "a_80000782_cut01_p01": {"aspect_theme": "文化认知"},
    "a_80000602_cut04_p02": {"aspect_theme": "竞争压力", "score": 0.2},
    "a_80000792_cut05_p02": {"aspect_theme": "竞争压力"},
}

TO_NEGATIVE = {
    "a_80000026_cut01_p01",
    "a_80000557_cut01_p01",
    "a_80000773_cut01_p01",
    "a_80000662_cut04_p02",
    "a_80000766_cut05_p02",
    "a_80000632_cut01_p01",
    "a_80000777_cut01_p01",
}


def main() -> None:
    review = pd.read_excel(REVIEW, dtype=str).fillna("")
    audit: List[Dict[str, Any]] = []
    counts = {}
    for split in ("train", "valid"):
        path = DATA / f"gts_{split}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        frame["relation"] = frame["relation"].astype(int)
        for _, decision in review[review["split"] == split].iterrows():
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
            elif valid == "是" and pair_id in TO_NEGATIVE:
                frame.at[index, "relation"] = 0
                frame.at[index, "aspect_theme"] = "其他"
                frame.at[index, "score"] = "0"
                frame.at[index, "reclassified_to_positive"] = ""
                frame.at[index, "reclassified_to_negative"] = "1"
                frame.at[index, "reclass_reason"] = "human_review_invalid"
                action = "to_negative"
            audit.append({
                "split": split, "pair_id": pair_id, "review_valid": valid,
                "action": action, "before": before, "after": int(frame.at[index, "relation"]),
                "note": decision.get("review_note", ""),
            })
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        counts[split] = {
            "rows": len(frame),
            "positive": int((frame["relation"] == 1).sum()),
            "negative": int((frame["relation"] == 0).sum()),
        }
    pd.DataFrame(audit).to_csv(AUDIT, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_excel(BASE_DIR / "output" / "gts_review_round5_applied.xlsx", index=False)
    print(pd.DataFrame(counts).T.to_string())
    manifest_path = BASE_DIR / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gts"] = {"train": counts["train"]["rows"], "valid": counts["valid"]["rows"], "test": 168}
    manifest["gts_labels"] = {
        "train": {"positive": counts["train"]["positive"], "negative": counts["train"]["negative"]},
        "valid": {"positive": counts["valid"]["positive"], "negative": counts["valid"]["negative"]},
        "test": {"positive": 56, "negative": 112},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
