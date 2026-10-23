"""组装 raw/generated 训练数据、Filter 数据和 pair 级 GTS 数据。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from text_utils import split_article

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
SEP = "[SEP]"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_units() -> Dict[str, Dict[str, Any]]:
    units = {}
    for path in [DATA / "raw" / "units.jsonl", DATA / "generated" / "train.jsonl"]:
        for row in load_jsonl(path):
            units[row["article_id"]] = row
    return units


def load_annotations() -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(DATA / "annotations" / "raw.jsonl") + load_jsonl(DATA / "annotations" / "generated.jsonl")
    return {row["article_id"]: row for row in rows}


def unit_label(annotation: Dict[str, Any]) -> int:
    return int(annotation.get("label") or 0)


def build_filter_rows(units, annotations) -> Dict[str, List[Dict[str, Any]]]:
    result = {"train": [], "valid": [], "test": []}
    for article_id, annotation in annotations.items():
        unit = units.get(article_id)
        if not unit:
            continue
        split = annotation.get("split") or unit.get("split") or "train"
        if split not in result:
            continue
        sentences = split_article(unit["text"])
        flags = [0] * len(sentences)
        title_flags = [0] * len(sentences)
        for item in annotation.get("sentences", []):
            index = int(item["index"])
            if 0 <= index < len(sentences):
                flags[index] = int(item.get("comparison", 0))
                title_flags[index] = int(item.get("title_comparison", 0))
        run_id = 0
        chain_lengths = [0] * len(sentences)
        positions = [0] * len(sentences)
        i = 0
        while i < len(sentences):
            if flags[i] == 0:
                i += 1
                continue
            j = i
            while j < len(sentences) and flags[j] == 1:
                j += 1
            run_id += 1
            length = j - i
            for k in range(i, j):
                chain_lengths[k] = length
                positions[k] = k - i + 1
            i = j
        source = annotation.get("source", "raw")
        for index, sentence in enumerate(sentences):
            unit_id = f"{article_id}_cut{index + 1:02d}"
            startable = 1 if (flags[index] or title_flags[index]) else 0
            start_source = "u" if flags[index] else ("t" if title_flags[index] else "none")
            result[split].append({
                "article_id": article_id,
                "unit_id": unit_id,
                "source": source,
                "split": split,
                "task": "u",
                "text": sentence,
                "label": flags[index],
                "startable": startable,
                "start_source": start_source,
            })
            if unit.get("title"):
                result[split].append({
                    "article_id": article_id,
                    "unit_id": unit_id,
                    "source": source,
                    "split": split,
                    "task": "t",
                    "title": unit["title"],
                    "current": sentence,
                    "label": title_flags[index],
                    "startable": startable,
                    "start_source": "t" if title_flags[index] else start_source,
                })
            # L start: current positive, next exists.
            if flags[index] and index + 1 < len(sentences):
                result[split].append({
                    "article_id": article_id,
                    "unit_id": unit_id,
                    "source": source,
                    "split": split,
                    "task": "l",
                    "context": [sentence],
                    "next": sentences[index + 1],
                    "label": 1 if flags[index + 1] else 0,
                    "position": 1,
                    "chain_length": chain_lengths[index],
                    "decision": "continue" if flags[index + 1] else "stop",
                })
            # L continuation/stop.
            if index > 0 and flags[index - 1]:
                context = []
                k = index - 1
                while k >= 0 and flags[k] and len(context) < 5:
                    context.insert(0, sentences[k])
                    k -= 1
                if context:
                    result[split].append({
                        "article_id": article_id,
                        "unit_id": unit_id,
                        "source": source,
                        "split": split,
                        "task": "l",
                        "context": context,
                        "next": sentence,
                        "label": flags[index],
                        "position": len(context),
                        "chain_length": chain_lengths[index - 1],
                        "decision": "continue" if flags[index] else "stop",
                    })
    return result


def _sample(rows: Sequence[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:max(0, count)]


def balance_train(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    output = []
    for task in ("u", "t", "l"):
        raw = [r for r in rows if r["task"] == task and r["source"] == "raw"]
        gen = [r for r in rows if r["task"] == task and r["source"] == "generated"]
        if not raw or not gen:
            output.extend(raw + gen)
            continue
        if task in ("u", "t"):
            raw_pos = [r for r in raw if r["label"] == 1]
            raw_neg = [r for r in raw if r["label"] == 0]
            gen_pos = [r for r in gen if r["label"] == 1]
            gen_neg = [r for r in gen if r["label"] == 0]
            n_pos = min(len(raw_pos), len(gen_pos))
            n_neg = min(len(raw_neg), len(gen_neg))
            n_pos = min(n_pos, n_neg) if n_neg else n_pos
            n_neg = min(n_neg, n_pos) if n_pos else n_neg
            output += _sample(raw_pos, n_pos, rng) + _sample(gen_pos, n_pos, rng)
            output += _sample(raw_neg, n_neg, rng) + _sample(gen_neg, n_neg, rng)
        else:
            # L is the scarce head: keep all useful mid/boundary examples,
            # then control continue/stop ratio instead of forcing source equality.
            for position_class, ratio in (("mid", (2, 1)), ("boundary", (1, 1))):
                def pick(rows, label):
                    if position_class == "mid":
                        return [r for r in rows if r.get("position", 0) >= 2 and r["label"] == label]
                    return [r for r in rows if r.get("position", 0) == 1 and r["label"] == label]
                positives = pick(raw, 1) + pick(gen, 1)
                negatives = pick(raw, 0) + pick(gen, 0)
                if not positives or not negatives:
                    continue
                pos_rows = _sample(positives, len(positives), rng)
                neg_n = min(len(negatives), max(1, len(pos_rows) * ratio[1] // ratio[0]))
                output += pos_rows + _sample(negatives, neg_n, rng)
    # Keep raw/generated source balance for L as well, but retain mid-chain
    # examples first when trimming generated rows.
    non_l = [r for r in output if r["task"] != "l"]
    raw_l = [r for r in output if r["task"] == "l" and r["source"] == "raw"]
    gen_l = [r for r in output if r["task"] == "l" and r["source"] == "generated"]
    if raw_l and gen_l:
        target = min(len(raw_l), len(gen_l))
        mid = [r for r in gen_l if r.get("position", 0) >= 2]
        boundary = [r for r in gen_l if r.get("position", 0) == 1]
        chosen = _sample(mid, min(target, len(mid)), rng)
        chosen += _sample(boundary, target - len(chosen), rng)
        output = non_l + raw_l + chosen
    rng.shuffle(output)
    return output


def write_filter(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    counts = {}
    for split, rows in rows_by_split.items():
        if split == "train":
            rows = balance_train(rows)
        path = DATA / "filter" / f"filter_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def find_span(text: str, value: str) -> Tuple[int, int]:
    index = text.find(value)
    if index < 0:
        raise ValueError(value)
    return index, index + len(value)


def build_gts(units, annotations) -> Dict[str, int]:
    positives_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    negative_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for article_id, annotation in annotations.items():
        unit = units.get(article_id)
        if not unit:
            continue
        split = annotation.get("split") or unit.get("split") or "train"
        sentences = split_article(unit["text"])
        quads = []
        for item in annotation.get("sentences", []):
            index = int(item["index"])
            if 0 <= index < len(sentences):
                for quad in item.get("quads", []):
                    quads.append((index, quad))
        title = unit.get("title", "")
        for number, (index, quad) in enumerate(quads, start=1):
            sentence = sentences[index]
            text = f"{title} {SEP} {sentence}"
            try:
                ss, se = find_span(text, quad["subject"])
                os_, oe = find_span(text, quad["object"])
            except ValueError:
                continue
            positives_by_split[split].append({
                "pair_id": f"{article_id}_cut{index + 1:02d}_p{number:02d}",
                "mention_id": f"{article_id}_cut{index + 1:02d}_{number:02d}",
                "text_group_id": article_id,
                "text": text,
                "subject_raw": quad["subject"],
                "subject_start": ss,
                "subject_end": se,
                "object_raw": quad["object"],
                "object_start": os_,
                "object_end": oe,
                "relation": 1,
                "aspect_theme": quad["aspect_theme"],
                "score": quad["score"],
                "subject_source": quad.get("subject_source", "body"),
                "object_source": quad.get("object_source", "body"),
                "source": annotation.get("source", "raw"),
                "platform": annotation.get("platform", ""),
                "split": split,
                "augmented": 0,
            })
            # Directional augmentation: A better than B implies B worse than A.
            # Only train; test/valid remain untouched.
            if split == "train" and float(quad["score"]) > 0 and quad["subject"] != quad["object"]:
                positives_by_split[split].append({
                    "pair_id": f"{article_id}_cut{index + 1:02d}_pf{number:02d}",
                    "mention_id": f"{article_id}_cut{index + 1:02d}_f{number:02d}",
                    "text_group_id": article_id,
                    "text": text,
                    "subject_raw": quad["object"],
                    "subject_start": os_,
                    "subject_end": oe,
                    "object_raw": quad["subject"],
                    "object_start": ss,
                    "object_end": se,
                    "relation": 1,
                    "aspect_theme": quad["aspect_theme"],
                    "score": -float(quad["score"]),
                    "subject_source": quad.get("object_source", "body"),
                    "object_source": quad.get("subject_source", "body"),
                    "source": annotation.get("source", "raw"),
                    "platform": annotation.get("platform", ""),
                    "split": split,
                    "augmented": 1,
                })
        # Hard negatives from cross-quad entity pairs.
        entities = []
        for index, quad in quads:
            entities.append((index, quad["subject"]))
            entities.append((index, quad["object"]))
        seen = set()
        for a in range(len(entities)):
            for b in range(len(entities)):
                si, subj = entities[a]
                oi, obj = entities[b]
                if si == oi or subj == obj:
                    continue
                key = (subj, obj)
                if key in seen:
                    continue
                seen.add(key)
                text = f"{title} {SEP} {sentences[si]} {SEP} {sentences[oi]}" if si != oi else f"{title} {SEP} {sentences[si]}"
                try:
                    ss, se = find_span(text, subj)
                    os_, oe = find_span(text, obj)
                except ValueError:
                    continue
                negative_pool[split].append({
                    "pair_id": f"{article_id}_cut{si + 1:02d}_n{len(negative_pool[split]) + 1:03d}",
                    "mention_id": "",
                    "text_group_id": article_id,
                    "text": text,
                    "subject_raw": subj,
                    "subject_start": ss,
                    "subject_end": se,
                    "object_raw": obj,
                    "object_start": os_,
                    "object_end": oe,
                    "relation": 0,
                    "aspect_theme": "其他",
                    "score": 0.0,
                    "subject_source": "body",
                    "object_source": "body",
                    "source": annotation.get("source", "raw"),
                    "platform": annotation.get("platform", ""),
                    "split": split,
                })
    # Lexicon-based hard negatives for units without enough cross-quad pairs.
    lexicon_terms = set()
    for name in ("brand_alias.csv", "category_terms.csv"):
        path = BASE_DIR / "input" / "lexicons" / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        column = "alias" if "alias" in frame.columns else "term"
        lexicon_terms.update(str(value).strip() for value in frame[column] if str(value).strip())
    lexicon_terms = sorted({term for term in lexicon_terms if len(term) >= 2}, key=len, reverse=True)
    for article_id, annotation in annotations.items():
        unit = units.get(article_id)
        if not unit:
            continue
        split = annotation.get("split") or unit.get("split") or "train"
        text = f"{unit.get('title','')} {SEP} {unit['text']}"
        present = [term for term in lexicon_terms if term in text]
        made = 0
        for i in range(len(present)):
            for j in range(len(present)):
                if i == j or made >= 3:
                    continue
                subj, obj = present[i], present[j]
                if subj == obj:
                    continue
                try:
                    ss, se = find_span(text, subj)
                    os_, oe = find_span(text, obj)
                except ValueError:
                    continue
                if (subj, obj) in {(row["subject_raw"], row["object_raw"]) for row in positives_by_split.get(split, [])}:
                    continue
                negative_pool[split].append({
                    "pair_id": f"{article_id}_lex{len(negative_pool[split]) + 1:03d}",
                    "mention_id": "",
                    "text_group_id": article_id,
                    "text": text,
                    "subject_raw": subj,
                    "subject_start": ss,
                    "subject_end": se,
                    "object_raw": obj,
                    "object_start": os_,
                    "object_end": oe,
                    "relation": 0,
                    "aspect_theme": "其他",
                    "score": 0.0,
                    "subject_source": "body",
                    "object_source": "body",
                    "source": annotation.get("source", "raw"),
                    "platform": annotation.get("platform", ""),
                    "split": split,
                })
                made += 1

    counts = {}
    columns = ["pair_id", "mention_id", "text_group_id", "text", "subject_raw", "subject_start", "subject_end", "object_raw", "object_start", "object_end", "relation", "aspect_theme", "score", "subject_source", "object_source", "source", "platform", "split", "augmented"]
    for split in ("train", "valid", "test"):
        positives = positives_by_split.get(split, [])
        negatives = negative_pool.get(split, [])
        rng = random.Random(42)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        if split == "train":
            raw_pos = [r for r in positives if r["source"] == "raw"]
            gen_pos = [r for r in positives if r["source"] == "generated"]
            n_pos = min(len(raw_pos), len(gen_pos))
            raw_neg = [r for r in negatives if r["source"] == "raw"]
            gen_neg = [r for r in negatives if r["source"] == "generated"]
            n_neg = min(2 * n_pos, len(raw_neg), len(gen_neg))
            positives = raw_pos[:n_pos] + gen_pos[:n_pos]
            negatives = raw_neg[:n_neg] + gen_neg[:n_neg]
        else:
            negatives = negatives[:len(positives) * 2]
        for row in negatives:
            row.setdefault("augmented", 0)
        rows = positives + negatives
        rng.shuffle(rows)
        path = DATA / "gts" / f"gts_{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
        counts[split] = len(rows)
    return counts


def main() -> None:
    units = load_units()
    annotations = load_annotations()
    filter_rows = build_filter_rows(units, annotations)
    filter_counts = write_filter(filter_rows)
    gts_counts = build_gts(units, annotations)
    manifest = {
        "raw_units": len(load_jsonl(DATA / "raw" / "units.jsonl")),
        "generated_units": len(load_jsonl(DATA / "generated" / "train.jsonl")),
        "annotations": len(annotations),
        "filter": filter_counts,
        "gts": gts_counts,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
