#!/usr/bin/env bash
# Полный цикл воспроизведения экспериментов.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-../venv/bin/python}"

echo "[1/5] Генерация набора данных"
"$PYTHON" scripts/generate_dataset.py

echo "[2/5] Оценка распознавания"
"$PYTHON" scripts/evaluate_recognition.py

echo "[3/5] Оценка обезличивания"
"$PYTHON" scripts/evaluate_anonymization.py

echo "[4/5] BERTScore"
"$PYTHON" scripts/evaluate_bertscore.py

echo "[5/5] Юнит-тесты"
"$PYTHON" -m unittest discover -s tests -v

echo "Готово. Результаты: results/, отчёт: report/"
