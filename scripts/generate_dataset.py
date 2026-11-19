"""Генерация синтетического набора данных для курсовой работы.

Набор состоит из:
  * train.jsonl  — factRuEval2016 train + синтетические ФИО (для NER);
  * val.jsonl    — factRuEval2016 test (валидация NER);
  * test.jsonl   — 180 комплексных документов, все 13 атрибутов;
  * test_by_attribute/*.jsonl — по 60 документов на каждый атрибут;
  * negative.jsonl — 60 документов без ПДн (проверка ложных срабатываний).

Разметка — BIO. Для каждого документа хранятся текст, список сущностей
``[attr, start, end]``, токены и BIO-метки.
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attributes import ALL_ATTRS  # noqa: E402
from src.metrics import spans_to_bio  # noqa: E402
from src.name_lists import (  # noqa: E402
    AMBIGUOUS_NAMES,
    NAMES_F,
    NAMES_M,
    PATRONYMICS_M,
    SURNAMES_M,
)
from src.validators import (  # noqa: E402
    _inn_check_digit,
    _luhn_sum,
    _snils_check,
)

SEED = 20261119
RNG = random.Random(SEED)

REGIONS = ("77", "50", "78", "66", "23", "16", "61", "59", "47", "52")
CARD_PREFIXES = ("4084", "4276", "5536", "2200", "5469", "4279", "4890", "2202")
EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "mail.ru", "corp.ru")
BIRTH_SERIES = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
CYRILLIC = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"


def rand_digits(length: int) -> str:
    return "".join(RNG.choice(string.digits) for _ in range(length))


def female_surname(male: str) -> str:
    if male.endswith("ский"):
        return male[:-4] + "ская"
    if male.endswith("цкий"):
        return male[:-4] + "цкая"
    if male.endswith("ой"):
        return male[:-2] + "ая"
    if male.endswith("ий"):
        return male[:-2] + "ая"
    return male + "а"


def female_patronymic(male: str) -> str:
    if male == "Ильич":
        return "Ильинична"
    if male.endswith("ович"):
        return male[:-4] + "овна"
    if male.endswith("евич"):
        return male[:-4] + "евна"
    if male.endswith("ич"):
        return male[:-2] + "ична"
    return male + "на"


def make_inn(length: int = 12) -> str:
    region = RNG.choice(REGIONS)
    if length == 10:
        body = region + rand_digits(7)
        values = [int(char) for char in body]
        check = _inn_check_digit(values, (2, 4, 10, 3, 5, 9, 4, 6, 8))
        return body + str(check)
    body = region + rand_digits(8)
    values = [int(char) for char in body]
    check11 = _inn_check_digit(values, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    values11 = values + [check11]
    check12 = _inn_check_digit(values11, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    return body + str(check11) + str(check12)


def make_card(length: int = 16) -> str:
    prefix = RNG.choice(CARD_PREFIXES)
    body = prefix + rand_digits(length - len(prefix) - 1)
    for check in range(10):
        candidate = body + str(check)
        if _luhn_sum([int(char) for char in candidate]) % 10 == 0:
            return candidate
    raise RuntimeError("не удалось подобрать контрольную цифру карты")


def make_ogrnip() -> str:
    first14 = "3" + str(RNG.randint(15, 25)).zfill(2) + RNG.choice(REGIONS) + rand_digits(9)
    first14 = first14[:14]
    check = (int(first14) % 13) % 10
    return first14 + str(check)


def make_snils() -> str:
    body = [int(char) for char in rand_digits(9)]
    check = _snils_check(body)
    return f"{''.join(map(str, body[:3]))}-{''.join(map(str, body[3:6]))}-{''.join(map(str, body[6:9]))} {check:02d}"


def make_passport() -> str:
    series = RNG.choice(REGIONS) + rand_digits(2)
    number = rand_digits(6)
    return f"{series[:2]} {series[2:]} {number}"


def make_birth_cert() -> str:
    series = RNG.choice(BIRTH_SERIES)
    letters = "".join(RNG.choice(CYRILLIC) for _ in range(2))
    return f"{series}-{letters} {rand_digits(6)}"


def make_kladr() -> str:
    return RNG.choice(REGIONS) + rand_digits(11)


def make_ip() -> str:
    first = RNG.choice(
        [5, 31, 37, 46, 62, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
         124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
         139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
         153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
         167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
         181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
         195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
         209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222,
         223]
    )
    return f"{first}.{RNG.randint(1, 254)}.{RNG.randint(1, 254)}.{RNG.randint(1, 254)}"


def make_email() -> str:
    local = RNG.choice(["user", "contact", "info", "office", "mail"]) + rand_digits(4)
    return f"{local}@{RNG.choice(EMAIL_DOMAINS)}"


def make_phone() -> str:
    return f"+7 ({RNG.randint(900, 999)}) {RNG.randint(100, 999)}-{RNG.randint(10, 99)}-{RNG.randint(10, 99)}"


def make_person(double_surname: bool = False, ambiguous: bool = False) -> dict[str, str]:
    gender = RNG.choice(("m", "f"))
    if gender == "m":
        surname = RNG.choice(SURNAMES_M)
        if double_surname:
            second = RNG.choice([s for s in SURNAMES_M if s != surname])
            surname = f"{surname}-{second}"
        name = RNG.choice(AMBIGUOUS_NAMES if ambiguous else NAMES_M)
        patronymic = RNG.choice(PATRONYMICS_M)
    else:
        surname = female_surname(RNG.choice(SURNAMES_M))
        if double_surname:
            second = female_surname(RNG.choice([s for s in SURNAMES_M if female_surname(s) != surname]))
            surname = f"{surname}-{second}"
        name = RNG.choice(AMBIGUOUS_NAMES if ambiguous else NAMES_F)
        patronymic = female_patronymic(RNG.choice(PATRONYMICS_M))
    return {
        "gender": gender,
        "surname": surname,
        "name": name,
        "patronymic": patronymic,
    }


TEMPLATES = (
    "Сведения о сотруднике: {surname} {name} {patronymic}. "
    "ИНН: {inn}. Паспорт: {passport}. Банковская карта: {card}. "
    "ОГРНИП: {ogrnip}. IP-адрес: {ip}. E-mail: {email}. "
    "Свидетельство о рождении: {birth_cert}. Код КЛАДР: {kladr}. "
    "Телефон: {phone}. СНИЛС: {snils}.",

    "В документе указаны фамилия {surname}, имя {name}, отчество {patronymic}. "
    "Контактный телефон {phone}, электронная почта {email}. "
    "Реквизиты: ИНН {inn}, ОГРНИП {ogrnip}, карта {card}, СНИЛС {snils}. "
    "Паспорт {passport}. Свидетельство о рождении {birth_cert}. "
    "IP {ip}. КЛАДР {kladr}.",

    "Заявитель: {surname} {name} {patronymic}. Предоставлены сведения: "
    "паспорт {passport}, ИНН {inn}, ОГРНИП {ogrnip}. "
    "Банковская карта {card}, телефон {phone}, e-mail {email}. "
    "IP-адрес {ip}, код КЛАДР {kladr}, СНИЛС {snils}, "
    "свидетельство о рождении {birth_cert}.",

    "В заявлении указаны: {surname} {name} {patronymic}. "
    "ИНН — {inn}, ОГРНИП — {ogrnip}. Паспорт: {passport}. "
    "Карта: {card}. Связаться можно по телефону {phone} или почте {email}. "
    "IP: {ip}. КЛАДР: {kladr}. СНИЛС: {snils}. "
    "Свидетельство о рождении: {birth_cert}.",
)


def make_values(double_surname: bool = False, ambiguous: bool = False) -> dict[str, str]:
    person = make_person(double_surname, ambiguous)
    values = {
        **person,
        "inn": make_inn(12 if RNG.random() < 0.5 else 10),
        "passport": make_passport(),
        "card": make_card(16 if RNG.random() < 0.7 else 19),
        "ogrnip": make_ogrnip(),
        "ip": make_ip(),
        "email": make_email(),
        "birth_cert": make_birth_cert(),
        "kladr": make_kladr(),
        "phone": make_phone(),
        "snils": make_snils(),
    }
    return values


PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def render_template(template: str, values: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Подставляет значения и запоминает точные границы каждого плейсхолдера."""
    pieces: list[str] = []
    positions: dict[str, tuple[int, int]] = {}
    cursor = 0
    length = 0
    for match in PLACEHOLDER_RE.finditer(template):
        prefix = template[cursor:match.start()]
        pieces.append(prefix)
        length += len(prefix)
        key = match.group(1)
        value = values[key]
        positions[key] = (length, length + len(value))
        pieces.append(value)
        length += len(value)
        cursor = match.end()
    pieces.append(template[cursor:])
    return "".join(pieces), positions


def make_document(identifier: str, *, double_surname: bool = False, ambiguous: bool = False) -> dict:
    values = make_values(double_surname, ambiguous)
    template = RNG.choice(TEMPLATES)
    text, positions = render_template(template, values)
    attr_keys = (
        ("SURNAME", "surname"),
        ("NAME", "name"),
        ("PATRONYMIC", "patronymic"),
        ("INN", "inn"),
        ("PASSPORT", "passport"),
        ("BANK_CARD", "card"),
        ("OGRNIP", "ogrnip"),
        ("IP", "ip"),
        ("EMAIL", "email"),
        ("BIRTH_CERT", "birth_cert"),
        ("KLADR", "kladr"),
        ("PHONE", "phone"),
        ("SNILS", "snils"),
    )
    spans = [[attr, positions[key][0], positions[key][1]] for attr, key in attr_keys]
    spans.sort(key=lambda item: item[1])
    tokens, tags = spans_to_bio(text, spans)
    return {
        "id": identifier,
        "text": text,
        "spans": spans,
        "tokens": tokens,
        "ner_tags": tags,
    }


PER_ATTR_TEMPLATES = {
    "SURNAME": ("Гражданин {value} подал официальное заявление.", "surname"),
    "NAME": ("Сотрудник {value} успешно выполнил поставленную задачу.", "name"),
    "PATRONYMIC": ("Менеджер {value} подписал трудовой договор.", "patronymic"),
    "INN": ("Организация имеет ИНН {value} в реестре.", "inn"),
    "PASSPORT": ("Паспорт: {value}.", "passport"),
    "BANK_CARD": ("Оплата заказа прошла с карты {value}.", "card"),
    "OGRNIP": ("Индивидуальный предприниматель имеет ОГРНИП {value}.", "ogrnip"),
    "IP": ("Сервер доступен по IP-адресу {value}.", "ip"),
    "EMAIL": ("Корпоративная почта: {value}.", "email"),
    "BIRTH_CERT": ("Свидетельство о рождении: {value}.", "birth_cert"),
    "KLADR": ("Код КЛАДР: {value}.", "kladr"),
    "PHONE": ("Служба поддержки доступна по телефону {value}.", "phone"),
    "SNILS": ("Страховой номер СНИЛС {value} проверен.", "snils"),
}


def make_per_attribute_document(attr: str, identifier: str) -> dict:
    template, key = PER_ATTR_TEMPLATES[attr]
    values = make_values()
    text, positions = render_template(template, {"value": values[key]})
    start, end = positions["value"]
    spans = [[attr, start, end]]
    tokens, tags = spans_to_bio(text, spans)
    return {
        "id": identifier,
        "text": text,
        "spans": spans,
        "tokens": tokens,
        "ner_tags": tags,
    }


NEGATIVE_TEXTS = (
    "В отчёте указано число 1234567890, но это не ИНН.",
    "Номер 0000 0000 0000 0000 не является банковской картой.",
    "Код 0012345678901 не является кодом КЛАДР.",
    "Время 12:30:45 и дата 12.03.2024 не относятся к персональным данным.",
    "Цена товара 999 рублей, скидка 10 процентов.",
    "Телефонный номер 123-45-67 не является полным номером.",
    "ИНН 1234567890 содержит неверные контрольные разряды.",
    "Карта 1234567890123456 не проходит проверку Луна.",
    "Паспорт 00 00 000000 имеет недопустимую серию.",
    "Свидетельство о рождении с латинскими буквами AA 000000 не является корректным примером.",
    "IP-адрес 999.999.999.999 не существует.",
    "Код 0012345678901 не является корректным КЛАДР.",
    "ОГРНИП 300000000000001 не проходит проверку контрольного разряда.",
    "В тексте нет персональных данных, только общие сведения.",
    "Компания опубликовала финансовый отчёт за квартал.",
)


# ---------------------------------------------------------------------------
# «Сложный» тестовый набор: другие формулировки и варианты записи реквизитов.
# В нём нет явных меток «фамилия/имя/отчество», поэтому ФИО должен находить
# RuBERT, а реквизиты записаны в разных форматах.
# ---------------------------------------------------------------------------

HARD_TEMPLATES = (
    "Заявление подал {surname} {name} {patronymic}. "
    "Приложены копии: паспорт {passport}, ИНН {inn}, "
    "банковская карта {card}. Контакты: {phone}, {email}.",

    "Сегодня на встрече {surname} {name} {patronymic} обсуждал проект. "
    "Реквизиты для оплаты: ИНН {inn}, ОГРНИП {ogrnip}, карта {card}. "
    "Код КЛАДР {kladr}, СНИЛС {snils}, свидетельство {birth_cert}.",

    "Клиент {surname} {name} {patronymic} зарегистрирован по адресу с кодом "
    "КЛАДР {kladr}. Паспорт {passport}, телефон {phone}, почта {email}. "
    "IP-адрес {ip}. ИНН {inn}.",

    "{surname} {name} {patronymic} направил заявку. "
    "Свидетельство о рождении {birth_cert}, СНИЛС {snils}, "
    "банковская карта {card}, ОГРНИП {ogrnip}, паспорт {passport}.",

    "Справка: {surname} {name} {patronymic} работает в организации. "
    "ИНН организации {inn}, ОГРНИП {ogrnip}. Личная почта {email}, "
    "телефон {phone}, карта {card}.",

    "Интервью с {surname} {name} {patronymic}. "
    "Для связи: {email}, {phone}. Документы: паспорт {passport}, "
    "ИНН {inn}, СНИЛС {snils}, свидетельство {birth_cert}. "
    "КЛАДР {kladr}, IP {ip}.",

    "Анкета: {surname} {name} {patronymic}, паспорт {passport}. "
    "Банковская карта {card}. ОГРНИП {ogrnip}, ИНН {inn}. "
    "СНИЛС {snils}, свидетельство {birth_cert}, КЛАДР {kladr}. "
    "Почта {email}, телефон {phone}, IP {ip}.",

    "Договор подписан {surname} {name} {patronymic}. "
    "Реквизиты сторон: ИНН {inn}, карта {card}, ОГРНИП {ogrnip}. "
    "Свидетельство о рождении {birth_cert}. "
    "Связаться: {phone}, {email}, IP {ip}, КЛАДР {kladr}, СНИЛС {snils}.",
)


def spaced(value: str, group: int = 3, sep: str = " ") -> str:
    """Расставляет разделители внутри цифровой строки."""
    digits_only = "".join(char for char in value if char.isdigit())
    chunks = [digits_only[i:i + group] for i in range(0, len(digits_only), group)]
    return sep.join(chunks)


def make_hard_document(identifier: str, index: int) -> dict:
    """Документ с другими формулировками и вариантами записи реквизитов."""
    values = make_values(double_surname=index % 6 == 0, ambiguous=index % 5 == 0)
    # Меняем оформление реквизитов: часть без меток, часть с разделителями.
    style = index % 4
    if style == 0:
        # Паспорт без пробелов между серией и номером, карта с пробелами.
        digits_only = "".join(char for char in values["passport"] if char.isdigit())
        values["passport"] = digits_only
        values["card"] = spaced(values["card"], 4, " ")
    elif style == 1:
        values["card"] = spaced(values["card"], 4, "-")
        values["inn"] = spaced(values["inn"], 4)
    elif style == 2:
        values["snils"] = spaced(values["snils"], 3)
        values["kladr"] = spaced(values["kladr"], 4)
    else:
        values["phone"] = values["phone"].replace("+7 ", "8-").replace(" (", "-").replace(") ", "-")

    template = HARD_TEMPLATES[index % len(HARD_TEMPLATES)]
    text, positions = render_template(template, values)
    attr_keys = (
        ("SURNAME", "surname"),
        ("NAME", "name"),
        ("PATRONYMIC", "patronymic"),
        ("INN", "inn"),
        ("PASSPORT", "passport"),
        ("BANK_CARD", "card"),
        ("OGRNIP", "ogrnip"),
        ("IP", "ip"),
        ("EMAIL", "email"),
        ("BIRTH_CERT", "birth_cert"),
        ("KLADR", "kladr"),
        ("PHONE", "phone"),
        ("SNILS", "snils"),
    )
    spans = [
        [attr, positions[key][0], positions[key][1]]
        for attr, key in attr_keys
        if key in positions
    ]
    spans.sort(key=lambda item: item[1])
    tokens, tags = spans_to_bio(text, spans)
    return {
        "id": identifier,
        "text": text,
        "spans": spans,
        "tokens": tokens,
        "ner_tags": tags,
    }


def make_negative_document(identifier: str, index: int) -> dict:
    text = NEGATIVE_TEXTS[index % len(NEGATIVE_TEXTS)]
    tokens, tags = spans_to_bio(text, [])
    return {
        "id": identifier,
        "text": text,
        "spans": [],
        "tokens": tokens,
        "ner_tags": tags,
    }


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- train / val из factRuEval2016 и синтетики ЛР №1 -----------------
    parent = ROOT.parent
    train_items: list[dict] = []
    for filename in ("factrueval_train.jsonl", "factrueval_synthetic.jsonl"):
        source = parent / filename
        if source.exists():
            for index, item in enumerate(load_jsonl(source)):
                train_items.append(
                    {
                        "id": f"train-{filename}-{index:06d}",
                        "tokens": item["tokens"],
                        "ner_tags": item["ner_tags"],
                    }
                )
    val_items: list[dict] = []
    source = parent / "factrueval_test.jsonl"
    if source.exists():
        for index, item in enumerate(load_jsonl(source)):
            val_items.append(
                {
                    "id": f"val-{index:06d}",
                    "tokens": item["tokens"],
                    "ner_tags": item["ner_tags"],
                }
            )
    write_jsonl(data_dir / "train.jsonl", train_items)
    write_jsonl(data_dir / "val.jsonl", val_items)

    # --- основной тестовый набор ---------------------------------------
    test_items: list[dict] = []
    for index in range(180):
        double_surname = index % 5 == 0
        ambiguous = index % 7 == 0
        test_items.append(
            make_document(
                f"test-{index:04d}",
                double_surname=double_surname,
                ambiguous=ambiguous,
            )
        )
    write_jsonl(data_dir / "test.jsonl", test_items)

    # --- тесты по отдельным атрибутам ----------------------------------
    for attr in ALL_ATTRS:
        items = [
            make_per_attribute_document(attr, f"{attr.lower()}-{index:04d}")
            for index in range(60)
        ]
        write_jsonl(data_dir / "test_by_attribute" / f"{attr.lower()}.jsonl", items)

    # --- негативные примеры --------------------------------------------
    negative_items = [
        make_negative_document(f"negative-{index:04d}", index) for index in range(60)
    ]
    write_jsonl(data_dir / "negative.jsonl", negative_items)

    # --- «сложный» набор: другие формулировки и форматы записи ----------
    hard_items = [
        make_hard_document(f"hard-{index:04d}", index) for index in range(90)
    ]
    write_jsonl(data_dir / "hard.jsonl", hard_items)

    # --- статистика -----------------------------------------------------
    def count_spans(items: list[dict]) -> Counter:
        counter: Counter = Counter()
        for item in items:
            for attr, _, _ in item["spans"]:
                counter[attr] += 1
        return counter

    stats = {
        "seed": SEED,
        "train_documents": len(train_items),
        "val_documents": len(val_items),
        "test_documents": len(test_items),
        "negative_documents": len(negative_items),
        "hard_documents": len(hard_items),
        "per_attribute_documents": 60,
        "test_spans": dict(count_spans(test_items)),
        "test_tokens": sum(len(item["tokens"]) for item in test_items),
        "attributes": list(ALL_ATTRS),
    }
    (data_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
