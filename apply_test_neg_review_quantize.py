"""应用 test 负例抽检，并将所有四元组分数量化到 0.2 间隔。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
GTS_DIR = BASE_DIR / "data" / "gts"
ANNOTATION_DIR = BASE_DIR / "data" / "annotations"
REVIEW = BASE_DIR.parent / "test_neg_review_25.xlsx"
OUTPUT = BASE_DIR / "output"

TO_POSITIVE = {
    "a_80000907_lex174": {"aspect_theme": "产品体验", "score": -0.8},
    "a_80000806_lex079": {"aspect_theme": "消费场景", "score": -0.6},
}


def quantize(value: Any) -> float:
    try:
        number = max(-1.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        number = 0.0
    scaled = number / 0.2
    if number >= 0:
        bucket = math.floor(scaled + 0.5)
    else:
        bucket = math.ceil(scaled - 0.5)
    return round(max(-5, min(5, bucket)) * 0.2, 1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # 1) Apply the reviewed test false negatives.
    test_path = GTS_DIR / "gts_test.csv"
    test = pd.read_csv(test_path, encoding="utf-8-sig", dtype=str).fillna("")
    test["relation"] = test["relation"].astype(int)
    review = pd.read_excel(REVIEW, dtype=str).fillna("")
    audit: List[Dict[str, Any]] = []
    for _, decision in review.iterrows():
        valid = str(decision.get("review_valid", "")).strip()
        if valid not in {"是", "否"}:
            continue
        pair_id = str(decision["pair_id"])
        matches = test.index[test["pair_id"] == pair_id].tolist()
        if not matches:
            continue
        index = matches[0]
        before = int(test.at[index, "relation"])
        action = "no_change"
        if valid == "否" and pair_id in TO_POSITIVE:
            for column, value in TO_POSITIVE[pair_id].items():
                test.at[index, column] = str(value)
            test.at[index, "relation"] = 1
            test.at[index, "review_applied"] = "否"
            test.at[index, "reclass_reason"] = "human_test_review"
            action = "to_positive"
        audit.append({"pair_id": pair_id, "review_valid": valid, "action": action, "before": before, "after": int(test.at[index, "relation"]), "note": decision.get("review_note", "")})
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(OUTPUT / "test_neg_review_25_applied.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_excel(OUTPUT / "test_neg_review_25_applied.xlsx", index=False)

    # 2) Quantize all GTS scores.
    quant_audit: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        path = GTS_DIR / f"gts_{split}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        for index, value in frame["score"].items():
            old = value
            new = quantize(value)
            if str(old) != str(new):
                quant_audit.append({"split": split, "pair_id": frame.at[index, "pair_id"], "old": old, "new": new})
            frame.at[index, "score"] = f"{new:.1f}"
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    # 3) Quantize annotation scores so rebuilding data stays consistent.
    for name in ("raw.jsonl", "generated.jsonl"):
        path = ANNOTATION_DIR / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            for sentence in row.get("sentences", []):
                for quad in sentence.get("quads", []):
                    quad["score"] = quantize(quad.get("score", 0))
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    pd.DataFrame(quant_audit).to_csv(OUTPUT / "score_quantization_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quant_audit).to_excel(OUTPUT / "score_quantization_report.xlsx", index=False)

    manifest_path = BASE_DIR / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gts_labels"]["test"] = {"positive": int((test["relation"] == 1).sum()), "negative": int((test["relation"] == 0).sum())}
    manifest["score_scheme"] = {"min": -1.0, "max": 1.0, "interval": 0.2, "values": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test rows", len(test), "positive", int((test["relation"] == 1).sum()), "negative", int((test["relation"] == 0).sum()))
    print("quantized rows", len(quant_audit))


if __name__ == "__main__":
    main()
