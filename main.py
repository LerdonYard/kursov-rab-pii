#!/usr/bin/env python3
"""Командная строка курсовой работы.

Примеры:
    python main.py --text "Иван Петров, ИНН 7707083893"
    python main.py --file input.txt
    echo "Иван Петров" | python main.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import process  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Распознавание и синонимическое обезличивание ПДн",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", "-t", help="текст для обработки")
    source.add_argument("--file", "-f", type=Path, help="файл с текстом")
    parser.add_argument(
        "--json",
        action="store_true",
        help="вывести полный JSON (текст, замена, атрибуты, карта замен)",
    )
    parser.add_argument(
        "--attributes-only",
        action="store_true",
        help="вывести только список атрибутов",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="не загружать RuBERT (только правила; быстрый режим)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("укажите --text, --file или передайте текст через stdin")

    detector = None
    if args.no_model:
        from src.recognize import PIIDetector

        detector = PIIDetector(use_model=False)

    result = process(text, detector=detector)

    if args.attributes_only:
        print(json.dumps(result.attributes, ensure_ascii=False))
    elif args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.anonymized)
        print()
        print("Атрибуты:", json.dumps(result.attributes, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
