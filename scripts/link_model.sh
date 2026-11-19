#!/usr/bin/env bash
# Привязывает обученную RuBERT-модель к проекту.
# Модель не входит в репозиторий (≈700 МБ). Укажите путь первым аргументом.
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-../../rubert_ner_factrueval_augmented_best}"

if [ ! -d "$SOURCE" ]; then
  echo "Каталог модели не найден: $SOURCE" >&2
  echo "Скачайте/распакуйте модель и передайте путь: scripts/link_model.sh /path/to/model" >&2
  exit 1
fi

mkdir -p models
ln -sfn "$(cd "$SOURCE" && pwd)" models/rubert_ner
echo "models/rubert_ner -> $(readlink -f models/rubert_ner)"
