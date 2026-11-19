"""Токенизация, BIO-разметка и метрики качества распознавания."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report

from .attributes import ALL_ATTRS

# Токен — слово (включая дефисные формы) либо одиночный знак пунктуации.
TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:[-–][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]",
    re.UNICODE,
)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def spans_to_bio(
    text: str,
    spans: Iterable[Sequence[object]],
) -> tuple[list[str], list[str]]:
    """Преобразует список ``[attr, start, end]`` в последовательность BIO.

    Токен, с которого начинается сущность, получает ``B-ATTR``; остальные
    токены внутри сущности — ``I-ATTR``. Токены вне сущностей — ``O``.
    """
    tokens = tokenize_with_offsets(text)
    labels = ["O"] * len(tokens)

    normalized: list[tuple[str, int, int]] = []
    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        attr = str(item[0]).upper()
        try:
            start, end = int(item[1]), int(item[2])
        except (TypeError, ValueError):
            continue
        if start < 0 or end > len(text) or end <= start:
            continue
        normalized.append((attr, start, end))

    for attr, start, end in sorted(normalized, key=lambda x: (x[1], -x[2])):
        inside = False
        for index, (_, token_start, token_end) in enumerate(tokens):
            if token_end <= start or token_start >= end:
                continue
            if labels[index] != "O":
                # Пересечение не должно возникать после разрешения, но
                # защищаемся от двойной разметки.
                continue
            labels[index] = f"{'B' if not inside else 'I'}-{attr}"
            inside = True

    return [token for token, _, _ in tokens], labels


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def entity_metrics(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
    attrs: Sequence[str] = ALL_ATTRS,
) -> dict:
    """Сущностные метрики seqeval: P/R/F1 по атрибутам и в целом."""
    report = seq_classification_report(
        references, predictions, output_dict=True, zero_division=0
    )

    per_attr: dict[str, dict[str, float]] = {}
    for attr in attrs:
        row = report.get(attr, {})
        per_attr[attr] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }

    micro = report.get("micro avg", {})
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    return {
        "per_attribute": per_attr,
        "micro": {
            "precision": float(micro.get("precision", 0.0)),
            "recall": float(micro.get("recall", 0.0)),
            "f1": float(micro.get("f1-score", 0.0)),
            "support": int(micro.get("support", 0)),
        },
        "macro": {
            "precision": float(macro.get("precision", 0.0)),
            "recall": float(macro.get("recall", 0.0)),
            "f1": float(macro.get("f1-score", 0.0)),
            "support": int(macro.get("support", 0)),
        },
        "weighted": {
            "precision": float(weighted.get("precision", 0.0)),
            "recall": float(weighted.get("recall", 0.0)),
            "f1": float(weighted.get("f1-score", 0.0)),
            "support": int(weighted.get("support", 0)),
        },
    }


def token_metrics(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> dict:
    """Токенные метрики по BIO-меткам (как в задаче token classification)."""
    ref_flat = [tag for seq in references for tag in seq]
    pred_flat = [tag for seq in predictions for tag in seq]
    labels = sorted(set(ref_flat) | set(pred_flat))

    report = sklearn_classification_report(
        ref_flat,
        pred_flat,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    without_o = [label for label in labels if label != "O"]
    with_o = labels

    def aggregate(selected: Sequence[str]) -> dict[str, float]:
        if not selected:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        p = float(np.mean([report[label]["precision"] for label in selected]))
        r = float(np.mean([report[label]["recall"] for label in selected]))
        f = float(np.mean([report[label]["f1-score"] for label in selected]))
        return {"precision": p, "recall": r, "f1": f}

    def weighted(selected: Sequence[str]) -> dict[str, float]:
        support = sum(report[label]["support"] for label in selected)
        if not support:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        p = sum(report[label]["precision"] * report[label]["support"] for label in selected) / support
        r = sum(report[label]["recall"] * report[label]["support"] for label in selected) / support
        f = sum(report[label]["f1-score"] * report[label]["support"] for label in selected) / support
        return {"precision": p, "recall": r, "f1": f}

    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_without_O": aggregate(without_o),
        "weighted_without_O": weighted(without_o),
        "macro_with_O": aggregate(with_o),
        "weighted_with_O": weighted(with_o),
        "n_tokens": len(ref_flat),
    }


def exact_span_metrics(
    references: Sequence[Sequence[Sequence[object]]],
    predictions: Sequence[Sequence[Sequence[object]]],
    attrs: Sequence[str] = ALL_ATTRS,
) -> dict:
    """Метрики точного совпадения границ сущностей (span-level)."""
    per_attr = {attr: {"tp": 0, "fp": 0, "fn": 0} for attr in attrs}
    for ref_spans, pred_spans in zip(references, predictions):
        ref = {
            (str(item[0]).upper(), int(item[1]), int(item[2]))
            for item in ref_spans
            if isinstance(item, (list, tuple)) and len(item) >= 3
        }
        pred = {
            (str(item[0]).upper(), int(item[1]), int(item[2]))
            for item in pred_spans
            if isinstance(item, (list, tuple)) and len(item) >= 3
        }
        for attr, _, _ in ref & pred:
            if attr in per_attr:
                per_attr[attr]["tp"] += 1
        for attr, _, _ in pred - ref:
            if attr in per_attr:
                per_attr[attr]["fp"] += 1
        for attr, _, _ in ref - pred:
            if attr in per_attr:
                per_attr[attr]["fn"] += 1

    result = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for attr, stats in per_attr.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r)
        result[attr] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f,
        }
        for key in total:
            total[key] += stats[key]

    p = _safe_div(total["tp"], total["tp"] + total["fp"])
    r = _safe_div(total["tp"], total["tp"] + total["fn"])
    result["overall"] = {
        **total,
        "precision": p,
        "recall": r,
        "f1": _safe_div(2 * p * r, p + r),
    }
    return result


def merge_token_sequences(sequences: Sequence[Sequence[str]]) -> list[str]:
    return [tag for seq in sequences for tag in seq]


def confusion_counts(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> Counter:
    return Counter(zip(merge_token_sequences(references), merge_token_sequences(predictions)))
