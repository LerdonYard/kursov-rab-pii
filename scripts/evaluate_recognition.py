"""Оценка качества распознавания ПДн на подготовленном наборе данных."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attributes import ALL_ATTRS, DISPLAY, REQUIRED_ATTRS  # noqa: E402
from src.metrics import (  # noqa: E402
    entity_metrics,
    exact_span_metrics,
    spans_to_bio,
    token_metrics,
)
from src.recognize import PIIDetector  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_detector(items: list[dict], detector: PIIDetector) -> tuple[list, list, list]:
    refs: list[list[str]] = []
    preds: list[list[str]] = []
    ref_spans: list[list] = []
    pred_spans: list[list] = []
    for item in items:
        predicted = detector.extract(item["text"])
        tokens, ref_tags = spans_to_bio(item["text"], item["spans"])
        _, pred_tags = spans_to_bio(item["text"], predicted)
        refs.append(ref_tags)
        preds.append(pred_tags)
        ref_spans.append(item["spans"])
        pred_spans.append(predicted)
    return refs, preds, ref_spans, pred_spans


def error_examples(items: list[dict], detector: PIIDetector, limit: int = 15) -> list[dict]:
    examples = []
    for item in items:
        predicted = detector.extract(item["text"])
        ref = {(str(a), int(s), int(e)) for a, s, e in item["spans"]}
        pred = {(str(a), int(s), int(e)) for a, s, e in predicted}
        if ref != pred:
            examples.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "missing": sorted(ref - pred),
                    "extra": sorted(pred - ref),
                }
            )
            if len(examples) >= limit:
                break
    return examples


def evaluate() -> dict:
    detector = PIIDetector()
    test_items = load_jsonl(DATA / "test.jsonl")
    negative_items = load_jsonl(DATA / "negative.jsonl")

    refs, preds, ref_spans, pred_spans = run_detector(test_items, detector)
    entity = entity_metrics(refs, preds, ALL_ATTRS)
    token = token_metrics(refs, preds)
    exact = exact_span_metrics(ref_spans, pred_spans, ALL_ATTRS)

    per_attribute_files: dict[str, dict] = {}
    for attr in ALL_ATTRS:
        path = DATA / "test_by_attribute" / f"{attr.lower()}.jsonl"
        if not path.exists():
            continue
        items = load_jsonl(path)
        refs_attr, preds_attr, _, _ = run_detector(items, detector)
        metrics_attr = entity_metrics(refs_attr, preds_attr, [attr])
        per_attribute_files[attr] = metrics_attr["per_attribute"][attr]

    negative_predictions = []
    for item in negative_items:
        predicted = detector.extract(item["text"])
        if predicted:
            negative_predictions.append(
                {"id": item["id"], "text": item["text"], "predicted": predicted}
            )

    hard_path = DATA / "hard.jsonl"
    hard_block = None
    if hard_path.exists():
        hard_items = load_jsonl(hard_path)
        h_refs, h_preds, h_ref_spans, h_pred_spans = run_detector(hard_items, detector)
        hard_block = {
            "documents": len(hard_items),
            "tokens": int(sum(len(seq) for seq in h_refs)),
            "entity_level": entity_metrics(h_refs, h_preds, ALL_ATTRS),
            "exact_span_level": exact_span_metrics(h_ref_spans, h_pred_spans, ALL_ATTRS),
            "errors": error_examples(hard_items, detector, limit=10),
        }

    result = {
        "dataset": {
            "test_documents": len(test_items),
            "negative_documents": len(negative_items),
            "test_tokens": int(sum(len(seq) for seq in refs)),
            "attributes": list(ALL_ATTRS),
            "required_attributes": list(REQUIRED_ATTRS),
        },
        "entity_level": entity,
        "token_level": token,
        "exact_span_level": exact,
        "per_attribute_files": per_attribute_files,
        "hard_set": hard_block,
        "negative": {
            "documents": len(negative_items),
            "documents_with_false_positives": len(negative_predictions),
            "false_positive_spans": sum(
                len(item["predicted"]) for item in negative_predictions
            ),
            "examples": negative_predictions[:10],
        },
        "errors": error_examples(test_items, detector),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "recognition_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(result)
    write_markdown(result)
    return result


def write_csv(result: dict) -> None:
    path = RESULTS / "per_attribute_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "attribute",
                "display",
                "precision",
                "recall",
                "f1",
                "support",
                "tp",
                "fp",
                "fn",
                "per_attribute_file_f1",
            ]
        )
        for attr in ALL_ATTRS:
            row = result["entity_level"]["per_attribute"][attr]
            exact = result["exact_span_level"][attr]
            file_metrics = result["per_attribute_files"].get(attr, {})
            writer.writerow(
                [
                    attr,
                    DISPLAY.get(attr, attr),
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    row["support"],
                    exact["tp"],
                    exact["fp"],
                    exact["fn"],
                    f"{file_metrics.get('f1', 0.0):.4f}",
                ]
            )


def write_markdown(result: dict) -> None:
    lines = []
    lines.append("# Метрики распознавания ПДн\n")
    lines.append(
        f"Тестовый набор: {result['dataset']['test_documents']} документов, "
        f"{result['dataset']['test_tokens']} токенов, "
        f"{len(result['dataset']['attributes'])} атрибутов.\n"
    )
    lines.append("## Сущностные метрики (seqeval, BIO)\n")
    lines.append("| Атрибут | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for attr in ALL_ATTRS:
        row = result["entity_level"]["per_attribute"][attr]
        lines.append(
            f"| {DISPLAY.get(attr, attr)} (`{attr}`) | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )
    for name, label in (
        ("micro", "Micro"),
        ("macro", "Macro"),
        ("weighted", "Weighted"),
    ):
        row = result["entity_level"][name]
        lines.append(
            f"| **{label}** | **{row['precision']:.4f}** | **{row['recall']:.4f}** | "
            f"**{row['f1']:.4f}** | **{row['support']}** |"
        )

    lines.append("\n## Токенные метрики (BIO)\n")
    token = result["token_level"]
    lines.append("| Состав классов | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (
        ("macro_without_O", "Macro без O"),
        ("weighted_without_O", "Weighted без O"),
        ("macro_with_O", "Macro с O"),
        ("weighted_with_O", "Weighted с O"),
    ):
        row = token[key]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.append(f"\nТочность по токенам (accuracy): {token['accuracy']:.4f}.\n")

    lines.append("## Точное совпадение границ\n")
    overall = result["exact_span_level"]["overall"]
    lines.append(
        f"TP={overall['tp']}, FP={overall['fp']}, FN={overall['fn']}, "
        f"Precision={overall['precision']:.4f}, Recall={overall['recall']:.4f}, "
        f"F1={overall['f1']:.4f}.\n"
    )

    lines.append("## Ложные срабатывания на негативных примерах\n")
    negative = result["negative"]
    lines.append(
        f"Документов без ПДн: {negative['documents']}; "
        f"с ложными срабатываниями: {negative['documents_with_false_positives']}; "
        f"ложных сущностей: {negative['false_positive_spans']}.\n"
    )

    if result["errors"]:
        lines.append("## Примеры ошибок\n")
        for example in result["errors"][:10]:
            lines.append(f"- `{example['id']}`: {example['text']}")
            if example["missing"]:
                lines.append(f"  - пропущено: {example['missing']}")
            if example["extra"]:
                lines.append(f"  - лишнее: {example['extra']}")

    (RESULTS / "recognition_metrics.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(
        {
            "entity_micro": result["entity_level"]["micro"],
            "entity_macro": result["entity_level"]["macro"],
            "entity_weighted": result["entity_level"]["weighted"],
            "token_weighted_without_O": result["token_level"]["weighted_without_O"],
            "token_macro_without_O": result["token_level"]["macro_without_O"],
            "exact_overall": result["exact_span_level"]["overall"],
            "negative_false_positives": result["negative"]["false_positive_spans"],
        },
        ensure_ascii=False,
        indent=2,
    ))
