"""Расчёт BERTScore для исходных, обезличенных и замаскированных текстов."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

# Совместимость bert-score 0.3.13 с transformers 5.x: у токенизатора
# model_max_length может быть очень большим числом, из-за чего truncation
# переполняет int. Ограничиваем длину явно.
import bert_score.utils as bert_utils  # noqa: E402


def _sent_encode(tokenizer, sent):
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    return tokenizer.encode(
        sent,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
    )


bert_utils.sent_encode = _sent_encode

from bert_score import score  # noqa: E402

RESULTS = ROOT / "results"
MODEL = "DeepPavlov/rubert-base-cased"
NUM_LAYERS = 9


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def calculate(candidates: list[str], references: list[str], label: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, recall, f1 = score(
        candidates,
        references,
        model_type=MODEL,
        num_layers=NUM_LAYERS,
        lang="ru",
        verbose=False,
        device=device,
        batch_size=16,
    )
    return {
        "label": label,
        "precision_mean": float(precision.mean()),
        "recall_mean": float(recall.mean()),
        "f1_mean": float(f1.mean()),
        "precision_std": float(precision.std()),
        "recall_std": float(recall.std()),
        "f1_std": float(f1.std()),
        "per_document": {
            "precision": [float(value) for value in precision],
            "recall": [float(value) for value in recall],
            "f1": [float(value) for value in f1],
        },
    }


def main() -> None:
    items = load_jsonl(RESULTS / "anonymized_test.jsonl")
    references = [item["text"] for item in items]
    anonymized = [item["anonymized"] for item in items]
    masked_pii = [item["masked_pii"] for item in items]
    masked_all = [item["masked_all"] for item in items]

    results = {
        "model": MODEL,
        "num_layers": NUM_LAYERS,
        "documents": len(items),
        "synonymic": calculate(anonymized, references, "синонимическое обезличивание"),
        "masked_pii": calculate(masked_pii, references, "маскирование только ПДн"),
        "masked_all": calculate(masked_all, references, "маскирование всего текста"),
    }

    (RESULTS / "bertscore_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Оценка сохранения смысла по BERTScore",
        "",
        f"Модель: `{MODEL}`, слой {NUM_LAYERS}, документов: {len(items)}.",
        "",
        "| Вариант | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("synonymic", "masked_pii", "masked_all"):
        row = results[key]
        lines.append(
            f"| {row['label']} | {row['precision_mean']:.4f} | "
            f"{row['recall_mean']:.4f} | {row['f1_mean']:.4f} |"
        )
    lines.append("")
    lines.append(
        "BERTScore сравнивает контекстные эмбеддинги токенов исходного и "
        "преобразованного текста. Чем выше F1, тем лучше сохранено смысловое "
        "содержание."
    )
    (RESULTS / "bertscore_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(
        {
            key: {
                "precision": results[key]["precision_mean"],
                "recall": results[key]["recall_mean"],
                "f1": results[key]["f1_mean"],
            }
            for key in ("synonymic", "masked_pii", "masked_all")
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
