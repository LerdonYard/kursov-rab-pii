"""Комбинированный распознаватель ПДн: правила + валидаторы + RuBERT NER."""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .attributes import PRIORITY
from .name_lists import AMBIGUOUS_NAMES, NAMES_F, NAMES_M, PATRONYMICS_M, SURNAMES_M
from .validators import (
    digits,
    validate_birth_cert,
    validate_card,
    validate_inn,
    validate_ip,
    validate_kladr,
    validate_luhn,
    validate_ogrnip,
    validate_passport,
    validate_snils,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_model_dir() -> Path:
    env = os.environ.get("KURSOV_RAB_MODEL")
    if env:
        return Path(env)
    local = PROJECT_ROOT / "models" / "rubert_ner"
    if local.exists():
        return local
    # Запасной путь: модель, обученная в корне проекта для ЛР №1.
    return PROJECT_ROOT.parent / "rubert_ner_factrueval_augmented_best"


# ---------------------------------------------------------------------------
# Регулярные выражения
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+\-]*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+"
)

IPV4_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)(?!\.\d)"
)
IPV6_RE = re.compile(r"(?<![\w:])(?=[0-9A-Fa-f:]*:)[0-9A-Fa-f:]{2,39}(?![\w:])")

CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
OGRNIP_RE = re.compile(r"(?<!\d)\d{15}(?!\d)")
SNILS_RE = re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}\s+\d{2}(?!\d)")
PHONE_RE = re.compile(
    r"(?<![\d\-])(?:\+?7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
PASSPORT_RE = re.compile(
    r"(?<!\d)(\d{2})\s?(\d{2})\s*(?:№|N|номер)?\s*(\d{6})(?!\d)",
    re.IGNORECASE,
)
BIRTH_CERT_RE = re.compile(
    r"(?<![А-ЯЁA-Z])([IVX]{1,4})\s*[-–]?\s*([А-ЯЁ]{2})\s*(?:№|N|номер)?\s*(\d{6})(?!\d)",
    re.IGNORECASE,
)
KLADR_RE = re.compile(
    r"(?<!\d)(\d{2})\s?(\d{3})\s?(\d{3})\s?(\d{3})\s?(\d{2})(?!\d)"
)
INN_RE = re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)")

# Записи с разделителями: «ИНН 7707 083 893», «СНИЛС 112 233 445 95»,
# «КЛАДР 77 00 0000 0000 0». Требуют явной метки слева, иначе любая
# последовательность цифр могла бы стать реквизитом.
LABELED_INN_RE = re.compile(
    r"(?i)\bинн\b[^0-9]{0,15}(\d(?:[ \-]?\d){9,11})(?![\d\-])"
)
LABELED_SNILS_RE = re.compile(
    r"(?i)\bснилс\b[^0-9]{0,15}(\d(?:[ \-]?\d){10})(?![\d\-])"
)
LABELED_KLADR_RE = re.compile(
    r"(?i)\bкладр\b[^0-9]{0,15}(\d(?:[ \-]?\d){12})(?![\d\-])"
)

# Явные метки ФИО: «фамилия Толстая», «имя Паша», «отчество Валерьевна».
# Такие конструкции встречаются в анкетах и справках; метка надёжнее модели,
# особенно для общих имён (Женя, Саша, Слава, Паша) и редких фамилий.
LABELED_SURNAME_RE = re.compile(
    r"(?i)\bфамилия\b[^А-ЯЁа-яё]{0,3}([А-ЯЁ][а-яё]+(?:[-–][А-ЯЁ][а-яё]+)?)"
)
LABELED_NAME_RE = re.compile(
    r"(?i)\bимя\b[^А-ЯЁа-яё]{0,3}([А-ЯЁ][а-яё]+)"
)
LABELED_PATRONYMIC_RE = re.compile(
    r"(?i)\bотчество\b[^А-ЯЁа-яё]{0,3}([А-ЯЁ][а-яё]+)"
)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-–][A-Za-zА-Яа-яЁё]+)*")

# Слова, которые не должны становиться ФИО даже при ошибке модели.
NON_NAME_WORDS = frozenset(
    {
        "сотрудник",
        "сотрудница",
        "гражданин",
        "гражданка",
        "эксперт",
        "менеджер",
        "директор",
        "депутат",
        "глава",
        "вчера",
        "сегодня",
        "завтра",
        "москва",
        "россия",
        "номер",
        "телефон",
        "почта",
        "карта",
        "паспорт",
        "снилс",
        "инн",
        "огрнип",
        "кладр",
        "свидетельство",
        "рождение",
        "адрес",
        "код",
        "серия",
        "выдан",
        "дата",
        "город",
        "улица",
        "дом",
        "квартира",
        "организация",
        "компания",
        "банк",
        "договор",
        "заявление",
        "премия",
        "интервью",
        "прогулка",
        "кредит",
        "суд",
        "ведомство",
        "департамент",
        "указ",
        "встреча",
        "проект",
        "исследование",
        "переговоры",
        "партнер",
        "партнёры",
        "рынок",
        "тенденция",
        "подход",
        "доработка",
        "ситуация",
        "должность",
        "месяц",
        "день",
        "год",
        "время",
        "станция",
        "экспедиция",
        "командир",
        "бортинженер",
        "космонавт",
        "центр",
        "управление",
        "полет",
        "полёт",
        "вопрос",
        "ответ",
        "сон",
        "орбита",
        "работа",
        "аналитик",
        "журналист",
        "банковская",
        "банковской",
        "свидетельство",
        "свидетельства",
        "рождении",
        "сведения",
        "заявитель",
        "заявлении",
        "указаны",
        "предоставлены",
        "связаться",
        "почте",
        "электронная",
        "контактный",
        "корпоративная",
        "сервер",
        "доступен",
        "организация",
        "индивидуальный",
        "предприниматель",
        "оплата",
        "заказа",
        "служба",
        "поддержки",
        "страховой",
        "проверен",
        "документе",
        "фамилия",
        "отчество",
        "реквизиты",
        "контактный",
        "электронная",
        "ip",
        "e-mail",
        "e",
        "mail",
    }
)

NAME_GAZETTEER = frozenset(
    word.lower().replace("ё", "е")
    for pool in (SURNAMES_M, NAMES_M, NAMES_F, PATRONYMICS_M, AMBIGUOUS_NAMES)
    for word in pool
)

_MORPH = None


def _morph():
    global _MORPH
    if _MORPH is None:
        import pymorphy3

        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


def _is_name_like(word: str, attr: str) -> bool:
    """Отсекает прилагательные и служебные слова, ошибочно принятые за ФИО."""
    parts = [part for part in re.split(r"[-–]", word) if part]
    if not parts:
        return False
    kind = {"SURNAME": "Surn", "NAME": "Name", "PATRONYMIC": "Patr"}.get(attr)
    morph = _morph()
    for part in parts:
        lower = part.lower().replace("ё", "е")
        if lower in NAME_GAZETTEER:
            continue
        try:
            parses = morph.parse(part)
        except Exception:
            parses = []
        if kind and any(kind in parse.tag.grammemes for parse in parses):
            continue
        if any(parse.tag.POS == "NOUN" for parse in parses):
            continue
        return False
    return True


@dataclass(frozen=True)
class Candidate:
    attr: str
    start: int
    end: int
    source: str
    priority: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _context(text: str, start: int, end: int, window: int = 70) -> str:
    return text[max(0, start - window): min(len(text), end + window)].lower().replace("ё", "е")


def _has_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(needle in haystack for needle in needles)


def _candidate(
    attr: str,
    start: int,
    end: int,
    source: str,
    *,
    priority: int | None = None,
) -> Candidate:
    return Candidate(
        attr=attr,
        start=start,
        end=end,
        source=source,
        priority=PRIORITY.get(attr, 50) if priority is None else priority,
    )


# ---------------------------------------------------------------------------
# Распознаватель
# ---------------------------------------------------------------------------

class PIIDetector:
    """Комбинированный детектор.

    Параметр ``use_model=False`` позволяет работать только на правилах —
    это удобно для быстрых тестов и для окружений без GPU.
    """

    def __init__(self, model_dir: str | Path | None = None, use_model: bool = True) -> None:
        self.model_dir = Path(model_dir) if model_dir else _default_model_dir()
        self.use_model = use_model
        self._tokenizer = None
        self._model = None
        self._device = None

    # -- модель ---------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Не найдена NER-модель: {self.model_dir}. "
                "Скопируйте rubert_ner_factrueval_augmented_best в kursov_rab/models/rubert_ner "
                "или задайте переменную окружения KURSOV_RAB_MODEL."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    def _predict_words(self, words: Sequence[str]) -> list[str]:
        """Возвращает метки слов (B-NAME, B-SURNAME, B-PATRONYMIC, O)."""
        if not words:
            return []
        self._load_model()
        import torch

        assert self._tokenizer is not None and self._model is not None
        result: list[str] = []
        # Ограничиваем длину фрагмента, чтобы не выйти за 512 токенов.
        chunk_size = 350
        for chunk_start in range(0, len(words), chunk_size):
            chunk = list(words[chunk_start:chunk_start + chunk_size])
            encoding = self._tokenizer(
                chunk,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            word_ids = encoding.word_ids(batch_index=0)
            model_inputs = {
                key: value.to(self._device) for key, value in encoding.items()
            }
            with torch.no_grad():
                logits = self._model(**model_inputs).logits
            prediction = torch.argmax(logits, dim=2)[0].tolist()

            labels: list[str] = ["O"] * len(chunk)
            previous = None
            for index, word_index in enumerate(word_ids):
                if word_index is None or word_index == previous:
                    previous = word_index
                    continue
                label = self._model.config.id2label.get(prediction[index], "O")
                labels[word_index] = label
                previous = word_index
            result.extend(labels)
        return result

    # -- правила --------------------------------------------------------

    def _rule_candidates(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []

        def add(attr: str, start: int, end: int, source: str) -> None:
            start, end = _trim(text, start, end)
            if end > start:
                candidates.append(_candidate(attr, start, end, source))

        # E-mail
        for match in EMAIL_RE.finditer(text):
            add("EMAIL", match.start(), match.end(), "email-regex")

        # IP: IPv4 и IPv6 с проверкой парсером.
        for match in IPV4_RE.finditer(text):
            if validate_ip(match.group(0)):
                add("IP", match.start(), match.end(), "ipv4")
        for match in IPV6_RE.finditer(text):
            value = match.group(0)
            # Отбрасываем одиночные двоеточия и времена вида 12:30.
            if value.count(":") >= 2 and validate_ip(value):
                add("IP", match.start(), match.end(), "ipv6")

        # СНИЛС
        for match in SNILS_RE.finditer(text):
            if validate_snils(match.group(0)):
                add("SNILS", match.start(), match.end(), "snils-regex")

        # СНИЛС с пробелами: «СНИЛС 112 233 445 95».
        for match in LABELED_SNILS_RE.finditer(text):
            if validate_snils(match.group(1)):
                add("SNILS", match.start(1), match.end(1), "snils-labeled")

        # Телефон
        for match in PHONE_RE.finditer(text):
            add("PHONE", match.start(), match.end(), "phone-regex")

        # Свидетельство о рождении: серия + номер, нужен контекст слева.
        for match in BIRTH_CERT_RE.finditer(text):
            before = text[max(0, match.start() - 60):match.start()].lower().replace("ё", "е")
            if _has_any(before, ("свидетельств", "рожден", "серия")):
                add("BIRTH_CERT", match.start(), match.end(), "birth-cert-regex")

        # Паспорт: серия + номер. Для слитной записи (10 цифр) контекст
        # должен стоять именно слева и не быть меткой ИНН.
        for match in PASSPORT_RE.finditer(text):
            before = text[max(0, match.start() - 60):match.start()].lower().replace("ё", "е")
            raw = match.group(0)
            spaced = bool(re.search(r"\d{2}\s+\d{2}", raw))
            labeled_inn = bool(re.search(r"инн[^.!?]{0,12}$", before))
            if labeled_inn:
                continue
            if _has_any(before, ("паспорт", "серия", "выдан", "документ")) or spaced:
                if validate_passport(raw):
                    add("PASSPORT", match.start(), match.end(), "passport-regex")

        # КЛАДР: 13 цифр, обязателен контекст «КЛАДР» слева.
        for match in KLADR_RE.finditer(text):
            before = text[max(0, match.start() - 40):match.start()].lower().replace("ё", "е")
            if "кладр" in before and validate_kladr(match.group(0)):
                add("KLADR", match.start(), match.end(), "kladr-regex")

        # КЛАДР с пробелами: «КЛАДР 77 00 0000 0000 0».
        for match in LABELED_KLADR_RE.finditer(text):
            if validate_kladr(match.group(1)):
                add("KLADR", match.start(1), match.end(1), "kladr-labeled")

        # ОГРНИП: 15 цифр, начинается с 3, контрольный разряд.
        # 15-значное число может одновременно проходить проверку Луна
        # (карты American Express 34/37) — тогда без явной метки «ОГРНИП»
        # трактуем его как банковскую карту.
        for match in OGRNIP_RE.finditer(text):
            raw = match.group(0)
            if not validate_ogrnip(raw):
                continue
            before = text[max(0, match.start() - 40):match.start()].lower().replace("ё", "е")
            if validate_luhn(raw) and "огрнип" not in before:
                continue
            add("OGRNIP", match.start(), match.end(), "ogrnip-regex")

        # Банковская карта: 13..19 цифр, алгоритм Луна.
        for match in CARD_RE.finditer(text):
            raw = match.group(0)
            if validate_card(raw):
                add("BANK_CARD", match.start(), match.end(), "luhn-regex")

        # ИНН: 10/12 цифр с контрольными разрядами.
        for match in INN_RE.finditer(text):
            if validate_inn(match.group(0)):
                add("INN", match.start(), match.end(), "inn-regex")

        # ИНН с разделителями: «ИНН 7707 083 893».
        for match in LABELED_INN_RE.finditer(text):
            if validate_inn(match.group(1)):
                add("INN", match.start(1), match.end(1), "inn-labeled")

        # ФИО с явными метками: «фамилия Толстая, имя Паша».
        for regex, attr in (
            (LABELED_SURNAME_RE, "SURNAME"),
            (LABELED_NAME_RE, "NAME"),
            (LABELED_PATRONYMIC_RE, "PATRONYMIC"),
        ):
            for match in regex.finditer(text):
                value = match.group(1)
                if value.lower().replace("ё", "е") in NON_NAME_WORDS:
                    continue
                add(attr, match.start(1), match.end(1), f"{attr.lower()}-labeled")

        return candidates

    # -- NER ------------------------------------------------------------

    def _name_candidates(self, text: str, reserved: Sequence[Candidate]) -> list[Candidate]:
        if not self.use_model:
            return []

        # В модель передаются ВСЕ слова предложения, включая строчные:
        # контекст («Сотрудник Пров успешно…») важен для определения типа
        # сущности. Фильтрация заглавных слов выполняется после предсказания.
        words: list[str] = []
        spans: list[tuple[int, int]] = []
        for match in WORD_RE.finditer(text):
            word = match.group(0)
            start, end = match.start(), match.end()
            if len(word) < 2:
                continue
            if any(start < item.end and end > item.start for item in reserved):
                continue
            words.append(word)
            spans.append((start, end))

        labels = self._predict_words(words)
        candidates: list[Candidate] = []
        for (start, end), label in zip(spans, labels):
            if label == "O":
                continue
            word = text[start:end]
            if not word[0].isupper():
                continue
            if word.lower().replace("ё", "е") in NON_NAME_WORDS:
                continue
            attr = label.split("-")[-1]
            if attr not in {"SURNAME", "NAME", "PATRONYMIC"}:
                continue
            if not _is_name_like(word, attr):
                continue
            candidates.append(_candidate(attr, start, end, "rubert"))
        return candidates

    # -- объединение ----------------------------------------------------

    @staticmethod
    def _resolve(candidates: Sequence[Candidate]) -> list[Candidate]:
        """Жадно выбирает непересекающиеся кандидаты по приоритету."""
        accepted: list[Candidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.priority, -item.length, item.start),
        ):
            if all(
                candidate.end <= other.start or candidate.start >= other.end
                for other in accepted
            ):
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: item.start)

    def extract_detailed(self, text: str) -> list[Candidate]:
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой")
        if not text:
            return []
        rule_candidates = self._rule_candidates(text)
        name_candidates = self._name_candidates(text, rule_candidates)
        return self._resolve(rule_candidates + name_candidates)

    def extract(self, text: str) -> list[list[object]]:
        """Возвращает список ``[attr, start, end]`` в формате ЛР №1."""
        return [
            [candidate.attr, candidate.start, candidate.end]
            for candidate in self.extract_detailed(text)
        ]


# Удобная функция-обёртка для совместимости с ЛР №1.
_DEFAULT_DETECTOR: PIIDetector | None = None


def extract_pii(text: str, *, use_model: bool = True) -> list[list[object]]:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None or _DEFAULT_DETECTOR.use_model != use_model:
        _DEFAULT_DETECTOR = PIIDetector(use_model=use_model)
    return _DEFAULT_DETECTOR.extract(text)
