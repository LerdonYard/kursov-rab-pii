"""Юнит-тесты курсовой работы."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anonymize import _morph_parse, anon, anonymize_with_map  # noqa: E402
from src.attributes import ALL_ATTRS  # noqa: E402
from src.recognize import PIIDetector  # noqa: E402
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


def span(text: str, attr: str, value: str) -> list[object]:
    start = text.find(value)
    assert start >= 0, (text, value)
    return [attr, start, start + len(value)]


class ValidatorTests(unittest.TestCase):
    def test_inn(self):
        self.assertTrue(validate_inn("7707083893"))
        self.assertTrue(validate_inn("500100732259"))
        self.assertFalse(validate_inn("1234567890"))

    def test_luhn(self):
        self.assertTrue(validate_luhn("4084520359695002"))
        self.assertFalse(validate_luhn("1234567890123456"))

    def test_snils(self):
        self.assertTrue(validate_snils("112-233-445 95"))
        self.assertFalse(validate_snils("112-233-445 94"))

    def test_ogrnip(self):
        self.assertTrue(validate_ogrnip("318669416793846"))
        self.assertFalse(validate_ogrnip("318669416793847"))

    def test_passport(self):
        self.assertTrue(validate_passport("45 06 123456"))
        self.assertFalse(validate_passport("00 00 123456"))

    def test_kladr(self):
        self.assertTrue(validate_kladr("7700000000000"))
        self.assertFalse(validate_kladr("0012345678901"))

    def test_birth_cert(self):
        self.assertTrue(validate_birth_cert("II-МЮ 123456"))
        self.assertFalse(validate_birth_cert("AA 123456"))

    def test_ip(self):
        self.assertTrue(validate_ip("83.187.170.157"))
        self.assertFalse(validate_ip("999.1.1.1"))


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PIIDetector(use_model=False)

    def test_all_structured_attributes(self):
        text = (
            "ИНН 7707083893, паспорт 45 06 123456, карта 4084520359695002, "
            "ОГРНИП 318669416793846, IP 83.187.170.157, "
            "почта user@example.com, свидетельство о рождении II-МЮ 123456, "
            "КЛАДР 7700000000000, телефон +7 (999) 123-45-67, "
            "СНИЛС 112-233-445 95."
        )
        found = {item[0] for item in self.detector.extract(text)}
        expected = {
            "INN", "PASSPORT", "BANK_CARD", "OGRNIP", "IP", "EMAIL",
            "BIRTH_CERT", "KLADR", "PHONE", "SNILS",
        }
        self.assertTrue(expected <= found, expected - found)

    def test_email_wins_over_names(self):
        text = "Напишите Ивану на ivanov@example.com."
        found = self.detector.extract(text)
        self.assertEqual([item[0] for item in found], ["EMAIL"])

    def test_negative_numbers(self):
        text = "ИНН 1234567890, карта 1234567890123456, КЛАДР 0012345678901."
        self.assertEqual(self.detector.extract(text), [])

    def test_spaced_and_dashed_formats(self):
        cases = {
            "ИНН 7707 083 893.": "INN",
            "ИНН 770-708-389-3.": "INN",
            "СНИЛС 112 233 445 95.": "SNILS",
            "КЛАДР: 77 00 0000 0000 0.": "KLADR",
            "Тел. 8-999-123-45-67.": "PHONE",
            "Паспорт 4506 123456 выдан ОВД.": "PASSPORT",
        }
        for text, expected in cases.items():
            found = [item[0] for item in self.detector.extract(text)]
            self.assertIn(expected, found, text)

    def test_unlabeled_digit_runs_are_ignored(self):
        for text in (
            "Число 1234567890123 не является КЛАДР.",
            "Код 7707 083 893 без метки.",
        ):
            self.assertEqual(self.detector.extract(text), [], text)

    def test_names_with_model(self):
        detector = PIIDetector()
        text = "Сотрудник Воронов Ефим Бенедиктович подписал указ."
        found = detector.extract(text)
        attrs = {item[0] for item in found}
        self.assertEqual(attrs, {"SURNAME", "NAME", "PATRONYMIC"})


class AnonymizationTests(unittest.TestCase):
    def test_contract_example(self):
        result = anon("Иван получил кредит", [["NAME", 0, 4]])
        self.assertNotEqual(result.split()[0], "Иван")
        self.assertTrue(result.endswith("получил кредит"))

    def test_randomness(self):
        text = "Иван получил кредит"
        first = anon(text, [["NAME", 0, 4]])
        second = anon(text, [["NAME", 0, 4]])
        self.assertNotEqual(first, second)

    def test_gender_and_case(self):
        text = "Иванов Иван Иванович передал документы Иванову Ивану Ивановичу."
        spans = [
            ["SURNAME", 0, 6],
            ["NAME", 7, 11],
            ["PATRONYMIC", 12, 21],
            ["SURNAME", text.rfind("Иванову"), text.rfind("Иванову") + 7],
            ["NAME", text.rfind("Ивану"), text.rfind("Ивану") + 5],
            ["PATRONYMIC", text.rfind("Ивановичу"), text.rfind("Ивановичу") + 9],
        ]
        result, replacements = anonymize_with_map(text, spans)
        self.assertNotEqual(result, text)
        # Формы одного человека должны быть согласованы: основы совпадают,
        # а падежные окончания различаются.
        by_attr = {}
        for row in replacements:
            by_attr.setdefault(str(row["attr"]), []).append(str(row["replacement"]))
        for attr in ("SURNAME", "NAME", "PATRONYMIC"):
            nominative, oblique = by_attr[attr]
            common = 0
            for left, right in zip(nominative, oblique):
                if left != right:
                    break
                common += 1
            self.assertGreaterEqual(common, 3, (attr, nominative, oblique))
            self.assertNotEqual(nominative, oblique)

    def test_ambiguous_name_context(self):
        male = anon("Женя подписал договор.", [["NAME", 0, 4]])
        female = anon("Женя подписала договор.", [["NAME", 0, 4]])
        male_name = male.split()[0]
        female_name = female.split()[0]
        male_parses = _morph_parse(male_name)
        female_parses = _morph_parse(female_name)
        self.assertTrue(any(p.tag.gender in {"masc", None} for p in male_parses))
        self.assertTrue(any(p.tag.gender == "femn" for p in female_parses))

    def test_double_surname(self):
        text = "Иванов-Петров Иван Иванович пришёл."
        result = anon(
            text,
            [
                span(text, "SURNAME", "Иванов-Петров"),
                span(text, "NAME", "Иван"),
                span(text, "PATRONYMIC", "Иванович"),
            ],
        )
        surname = result.split()[0]
        self.assertIn("-", surname)
        self.assertNotEqual(surname, "Иванов-Петров")
        self.assertEqual(len(surname.split("-")), 2)

    def test_structured_preservation(self):
        text = (
            "ИНН 7707083893, паспорт 45 06 123456, карта 4084520359695002, "
            "ОГРНИП 318669416793846, IP 83.187.170.157, "
            "почта user@example.com, свидетельство о рождении II-МЮ 123456, "
            "КЛАДР 7700000000000, телефон +7 (999) 123-45-67, "
            "СНИЛС 112-233-445 95."
        )
        attrs = [
            span(text, "INN", "7707083893"),
            span(text, "PASSPORT", "45 06 123456"),
            span(text, "BANK_CARD", "4084520359695002"),
            span(text, "OGRNIP", "318669416793846"),
            span(text, "IP", "83.187.170.157"),
            span(text, "EMAIL", "user@example.com"),
            span(text, "BIRTH_CERT", "II-МЮ 123456"),
            span(text, "KLADR", "7700000000000"),
            span(text, "PHONE", "+7 (999) 123-45-67"),
            span(text, "SNILS", "112-233-445 95"),
        ]
        result, replacements = anonymize_with_map(text, attrs)
        self.assertNotEqual(result, text)
        values = {str(row["attr"]): str(row["replacement"]) for row in replacements}
        self.assertTrue(validate_inn(values["INN"]))
        self.assertEqual(digits(values["INN"])[:4], [7, 7, 0, 7])
        self.assertTrue(validate_passport(values["PASSPORT"]))
        self.assertTrue(validate_luhn(values["BANK_CARD"]))
        self.assertEqual(digits(values["BANK_CARD"])[:6], [4, 0, 8, 4, 5, 2])
        self.assertEqual(digits(values["BANK_CARD"])[-4:], [5, 0, 0, 2])
        self.assertTrue(validate_ogrnip(values["OGRNIP"]))
        self.assertTrue(validate_ip(values["IP"]))
        self.assertTrue(values["EMAIL"].endswith("@example.com"))
        self.assertTrue(validate_birth_cert(values["BIRTH_CERT"]))
        self.assertTrue(validate_kladr(values["KLADR"]))
        self.assertEqual(digits(values["KLADR"])[:2], [7, 7])
        self.assertTrue(validate_snils(values["SNILS"]))
        self.assertEqual(values["PHONE"][:8], "+7 (999)")

    def test_idempotency_and_format_variants(self):
        text = "ИНН 7707083893 и снова 7707083893."
        first = text.find("7707083893")
        second = text.find("7707083893", first + 1)
        _, replacements = anonymize_with_map(
            text,
            [["INN", first, first + 10], ["INN", second, second + 10]],
        )
        values = [str(row["replacement"]) for row in replacements]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], values[1])

    def test_email_domain_preserved(self):
        text = "Почта ivanov@example.com."
        result = anon(text, [["EMAIL", 6, 24]])
        self.assertTrue(result.endswith("@example.com."))
        self.assertNotEqual(result.split()[1], "ivanov@example.com.")

    def test_mask(self):
        from scripts.evaluate_anonymization import mask_spans

        text = "Иван получил кредит"
        self.assertEqual(mask_spans(text, [["NAME", 0, 4]]), "**** получил кредит")


class DatasetTests(unittest.TestCase):
    def test_dataset_requirements(self):
        import json
        from collections import Counter

        path = ROOT / "data" / "test.jsonl"
        counts = Counter()
        documents = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                documents += 1
                self.assertEqual(len(item["tokens"]), len(item["ner_tags"]))
                for attr, start, end in item["spans"]:
                    self.assertLess(start, end)
                    counts[attr] += 1
        self.assertGreaterEqual(documents, 100)
        for attr in ALL_ATTRS:
            self.assertGreaterEqual(counts[attr], 50, attr)


if __name__ == "__main__":
    unittest.main()
