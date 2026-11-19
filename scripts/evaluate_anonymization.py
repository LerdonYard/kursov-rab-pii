"""Оценка алгоритмов обезличивания и подготовка данных для BERTScore."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anonymize import _cache_key, _morph_parse, anonymize_with_map  # noqa: E402
from src.attributes import ALL_ATTRS, DISPLAY, FIO_ATTRS  # noqa: E402
from src.validators import (  # noqa: E402
    digits,
    validate_birth_cert,
    validate_inn,
    validate_ip,
    validate_kladr,
    validate_luhn,
    validate_ogrnip,
    validate_passport,
    validate_snils,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mask_spans(text: str, spans: list[list], char: str = "*") -> str:
    pieces: list[str] = []
    cursor = 0
    for _, start, end in sorted(spans, key=lambda item: item[1]):
        pieces.append(text[cursor:start])
        pieces.append(char * (end - start))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _nondigits(value: str) -> str:
    return "".join(char for char in value if not char.isdigit())


def check_replacement(attr: str, original: str, replacement: str) -> list[str]:
    errors: list[str] = []
    if original == replacement:
        errors.append("значение не изменилось")

    if attr == "INN":
        if not validate_inn(replacement):
            errors.append("неверные контрольные разряды ИНН")
        if digits(original)[:4] != digits(replacement)[:4]:
            errors.append("не сохранён код налогового органа")
        if len(digits(original)) != len(digits(replacement)):
            errors.append("изменилась длина ИНН")
    elif attr == "PASSPORT":
        if not validate_passport(replacement):
            errors.append("неверный формат паспорта")
        if digits(original)[:4] != digits(replacement)[:4]:
            errors.append("не сохранена серия паспорта")
    elif attr == "BANK_CARD":
        if not validate_luhn(replacement):
            errors.append("неверная контрольная сумма карты")
        if digits(original)[:6] != digits(replacement)[:6]:
            errors.append("не сохранены первые 6 цифр карты")
        if digits(original)[-4:] != digits(replacement)[-4:]:
            errors.append("не сохранены последние 4 цифры карты")
    elif attr == "OGRNIP":
        if not validate_ogrnip(replacement):
            errors.append("неверный контрольный разряд ОГРНИП")
        if digits(original)[:4] != digits(replacement)[:4]:
            errors.append("не сохранены первые 4 цифры ОГРНИП")
    elif attr == "IP":
        if not validate_ip(replacement):
            errors.append("некорректный IP")
    elif attr == "EMAIL":
        if "@" not in replacement:
            errors.append("потерян @")
        elif original.rpartition("@")[2] != replacement.rpartition("@")[2]:
            errors.append("изменён домен e-mail")
    elif attr == "BIRTH_CERT":
        if not validate_birth_cert(replacement):
            errors.append("неверный формат свидетельства")
        original_series = re.sub(r"\d", "", original).strip()
        replacement_series = re.sub(r"\d", "", replacement).strip()
        if original_series != replacement_series:
            errors.append("не сохранена серия свидетельства")
    elif attr == "KLADR":
        if not validate_kladr(replacement):
            errors.append("неверный формат КЛАДР")
        if digits(original)[:2] != digits(replacement)[:2]:
            errors.append("не сохранён код региона КЛАДР")
    elif attr == "PHONE":
        if digits(original)[:4] != digits(replacement)[:4]:
            errors.append("не сохранены первые 4 цифры телефона")
        if _nondigits(original) != _nondigits(replacement):
            errors.append("изменены разделители телефона")
    elif attr == "SNILS":
        if not validate_snils(replacement):
            errors.append("неверное контрольное число СНИЛС")
        if _nondigits(original) != _nondigits(replacement):
            errors.append("изменён формат СНИЛС")
    return errors


def morphology_match(attr: str, original: str, replacement: str) -> bool:
    """Проверяет совпадение падежа/числа/рода по разборам pymorphy3."""
    def grams(word: str) -> dict[str, set[str]]:
        result = {"case": set(), "number": set(), "gender": set()}
        for part in re.split(r"[-–\s]+", word):
            for parse in _morph_parse(part):
                for field in result:
                    value = getattr(parse.tag, field)
                    if value:
                        result[field].add(value)
                # pymorphy помечает общие имена (Женя, Саша, Слава) как ms-f.
                if "ms-f" in parse.tag.grammemes:
                    result["gender"].update({"masc", "femn"})
        return result

    left = grams(original)
    right = grams(replacement)
    if not left["case"] or not right["case"]:
        return True
    if not (left["case"] & right["case"]):
        return False
    if left["number"] and right["number"] and not (left["number"] & right["number"]):
        return False
    if attr in {"SURNAME", "NAME"} and left["gender"] and right["gender"]:
        if not (left["gender"] & right["gender"]):
            return False
    return True


IDEMPOTENCY_CASES = (
    (
        "ИНН 7707083893 и снова 7707083893.",
        [("INN", "7707083893"), ("INN", "7707083893")],
    ),
    (
        "Паспорт 45 06 123456, повтор 4506 123456.",
        [("PASSPORT", "45 06 123456"), ("PASSPORT", "4506 123456")],
    ),
    (
        "Карта 4084520359695002 и 4084 5203 5969 5002.",
        [("BANK_CARD", "4084520359695002"), ("BANK_CARD", "4084 5203 5969 5002")],
    ),
    (
        "СНИЛС 112-233-445 95 и 112-233-445 95.",
        [("SNILS", "112-233-445 95"), ("SNILS", "112-233-445 95")],
    ),
    (
        "КЛАДР 7700000000000 и 7700000000000.",
        [("KLADR", "7700000000000"), ("KLADR", "7700000000000")],
    ),
    (
        "Свидетельство II-МЮ 123456 и II-МЮ 123456.",
        [("BIRTH_CERT", "II-МЮ 123456"), ("BIRTH_CERT", "II-МЮ 123456")],
    ),
    (
        "Иван Петров получил премию. Иван Петров вышел на сцену.",
        [("NAME", "Иван"), ("SURNAME", "Петров"), ("NAME", "Иван"), ("SURNAME", "Петров")],
    ),
    (
        "Алексей Долматов дал в рот Алексею Долматову.",
        [
            ("NAME", "Алексей"),
            ("SURNAME", "Долматов"),
            ("NAME", "Алексею"),
            ("SURNAME", "Долматову"),
        ],
    ),
)


def _find_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    positions = []
    start = 0
    while True:
        index = text.find(value, start)
        if index < 0:
            break
        positions.append((index, index + len(value)))
        start = index + len(value)
    return positions


def evaluate_idempotency() -> dict:
    checked = 0
    ok = 0
    failed = []
    for text, values in IDEMPOTENCY_CASES:
        spans = []
        used: Counter = Counter()
        for attr, value in values:
            occurrence = used[(attr, value)]
            positions = _find_occurrences(text, value)
            if occurrence >= len(positions):
                continue
            start, end = positions[occurrence]
            used[(attr, value)] += 1
            spans.append([attr, start, end])
        _, replacements = anonymize_with_map(text, spans)
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in replacements:
            groups[_cache_key(str(row["attr"]), str(row["original"]))].append(
                str(row["replacement"])
            )
        for key, values_out in groups.items():
            if len(values_out) < 2:
                continue
            checked += 1
            normalized = []
            for value in values_out:
                if key[0] in {"INN", "PASSPORT", "BANK_CARD", "SNILS", "KLADR", "BIRTH_CERT"}:
                    normalized.append("".join(str(digit) for digit in value if digit.isdigit()))
                else:
                    normalized.append(value)
            if len(set(normalized)) == 1:
                ok += 1
            else:
                failed.append({"text": text, "attr": key[0], "values": values_out})
    return {
        "checked": checked,
        "ok": ok,
        "failed": failed,
        "accuracy": ok / checked if checked else 0.0,
    }


def main() -> None:
    items = load_jsonl(DATA / "test.jsonl")
    RESULTS.mkdir(parents=True, exist_ok=True)

    stats = {
        "documents": len(items),
        "replacements": 0,
        "changed": 0,
        "unchanged": 0,
        "validation_errors": [],
        "morphology_checked": 0,
        "morphology_ok": 0,
        "morphology_failed": [],
        "idempotency_checked": 0,
        "idempotency_ok": 0,
        "idempotency_failed": [],
        "by_attribute": defaultdict(lambda: {"count": 0, "changed": 0, "errors": 0}),
        "second_run_different": 0,
    }

    anonymized_items: list[dict] = []
    for item in items:
        anonymized, replacements = anonymize_with_map(item["text"], item["spans"])
        masked_pii = mask_spans(item["text"], item["spans"])
        masked_all = "*" * len(item["text"])

        # Повторный запуск должен давать другие значения.
        second, _ = anonymize_with_map(item["text"], item["spans"])
        if second != anonymized:
            stats["second_run_different"] += 1

        # Проверка идемпотентности: одинаковые исходные значения в одном
        # документе должны получить одинаковые замены.
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in replacements:
            groups[(str(row["attr"]), str(row["original"]))].append(str(row["replacement"]))
        for key, values in groups.items():
            if len(values) < 2:
                continue
            stats["idempotency_checked"] += 1
            if len(set(values)) == 1:
                stats["idempotency_ok"] += 1
            else:
                stats["idempotency_failed"].append(
                    {"document": item["id"], "attr": key[0], "original": key[1], "values": values}
                )

        for row in replacements:
            attr = str(row["attr"])
            original = str(row["original"])
            replacement = str(row["replacement"])
            stats["replacements"] += 1
            stats["by_attribute"][attr]["count"] += 1
            if replacement != original:
                stats["changed"] += 1
                stats["by_attribute"][attr]["changed"] += 1
            else:
                stats["unchanged"] += 1

            errors = check_replacement(attr, original, replacement)
            if errors:
                stats["by_attribute"][attr]["errors"] += 1
                stats["validation_errors"].append(
                    {
                        "document": item["id"],
                        "attr": attr,
                        "original": original,
                        "replacement": replacement,
                        "errors": errors,
                    }
                )

            if attr in FIO_ATTRS:
                stats["morphology_checked"] += 1
                if morphology_match(attr, original, replacement):
                    stats["morphology_ok"] += 1
                else:
                    stats["morphology_failed"].append(
                        {
                            "document": item["id"],
                            "attr": attr,
                            "original": original,
                            "replacement": replacement,
                        }
                    )

        anonymized_items.append(
            {
                "id": item["id"],
                "text": item["text"],
                "anonymized": anonymized,
                "masked_pii": masked_pii,
                "masked_all": masked_all,
                "spans": item["spans"],
                "replacements": replacements,
            }
        )

    dedicated = evaluate_idempotency()
    stats["idempotency_checked"] = dedicated["checked"]
    stats["idempotency_ok"] = dedicated["ok"]
    stats["idempotency_failed"] = dedicated["failed"]
    stats["by_attribute"] = dict(stats["by_attribute"])
    stats["morphology_accuracy"] = (
        stats["morphology_ok"] / stats["morphology_checked"]
        if stats["morphology_checked"]
        else 0.0
    )
    stats["idempotency_accuracy"] = (
        stats["idempotency_ok"] / stats["idempotency_checked"]
        if stats["idempotency_checked"]
        else 0.0
    )
    stats["replacement_rate"] = (
        stats["changed"] / stats["replacements"] if stats["replacements"] else 0.0
    )

    with (RESULTS / "anonymized_test.jsonl").open("w", encoding="utf-8") as handle:
        for item in anonymized_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    (RESULTS / "anonymization_metrics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Оценка обезличивания",
        "",
        f"Документов: {stats['documents']}",
        f"Замен: {stats['replacements']}",
        f"Доля изменённых значений: {stats['replacement_rate']:.4f}",
        f"Ошибок проверки структуры/контрольных разрядов: {len(stats['validation_errors'])}",
        f"Сохранение морфологии: {stats['morphology_ok']}/{stats['morphology_checked']} "
        f"({stats['morphology_accuracy']:.4f})",
        f"Идемпотентность: {stats['idempotency_ok']}/{stats['idempotency_checked']} "
        f"({stats['idempotency_accuracy']:.4f})",
        f"Повторный запуск дал другие значения: {stats['second_run_different']}/{stats['documents']}",
        "",
        "| Атрибут | Замен | Изменено | Ошибок |",
        "|---|---:|---:|---:|",
    ]
    for attr in ALL_ATTRS:
        row = stats["by_attribute"].get(attr, {"count": 0, "changed": 0, "errors": 0})
        lines.append(
            f"| {DISPLAY.get(attr, attr)} | {row['count']} | {row['changed']} | {row['errors']} |"
        )
    (RESULTS / "anonymization_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(
        {
            "documents": stats["documents"],
            "replacements": stats["replacements"],
            "replacement_rate": stats["replacement_rate"],
            "validation_errors": len(stats["validation_errors"]),
            "morphology_accuracy": stats["morphology_accuracy"],
            "idempotency_accuracy": stats["idempotency_accuracy"],
            "second_run_different": stats["second_run_different"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
