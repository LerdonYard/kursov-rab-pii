"""Синонимическое обезличивание персональных данных (курсовая работа).

Публичный API:
    anon(text, attributes) -> str

`attributes` — список найденных сущностей в формате ЛР №1:
    [["SURNAME", 0, 6], ["NAME", 7, 10], ...]

Поддерживаются атрибуты:
    SURNAME, NAME, PATRONYMIC, INN, PASSPORT, BANK_CARD, OGRNIP, IP,
    EMAIL, BIRTH_CERT, KLADR, PHONE, SNILS.

Ключевые свойства:
    * ФИО       — случайные ФИО того же рода/числа/падежа;
    * E-mail    — случайный local-part, домен сохраняется;
    * ИНН       — первые 4 цифры (код налогового органа) и контрольные
                  разряды сохраняются/пересчитываются;
    * Паспорт   — серия сохраняется, номер заменяется;
    * Карта     — первые 6 и последние 4 цифры сохраняются, LUHN валиден;
    * ОГРНИП    — структура и контрольный разряд сохраняются;
    * IP        — генерируется другой корректный IPv4/IPv6;
    * Свидетельство о рождении — серия сохраняется, номер заменяется;
    * КЛАДР     — код региона сохраняется, остальные цифры заменяются;
    * Телефон   — первые 4 цифры и все нецифровые символы сохраняются;
    * СНИЛС     — формат и контрольное число сохраняются.

Все случайные значения генерируются через `secrets.SystemRandom`, поэтому
повторный вызов `anon` с теми же аргументами даёт разные результаты.
В пределах одного вызова одинаковые нормализованные значения получают
одинаковую замену (свойство идемпотентности/согласованности).
"""

from __future__ import annotations

import ipaddress
import re
import secrets
import string
from numbers import Integral
from typing import Iterable, NamedTuple, Sequence

try:  # pymorphy3 — основной морфологический анализатор
    import pymorphy3
except ImportError:  # pragma: no cover - на случай отсутствия зависимости
    pymorphy3 = None  # type: ignore[assignment]


__all__ = [
    "anon",
    "anonymize",
    "anonymize_with_map",
    "validate_inn",
    "validate_luhn",
    "validate_snils",
]


# ---------------------------------------------------------------------------
# Инициализация зависимостей
# ---------------------------------------------------------------------------

_RNG = secrets.SystemRandom()

if pymorphy3 is not None:
    _MORPH = pymorphy3.MorphAnalyzer()
else:  # pragma: no cover
    _MORPH = None


# ---------------------------------------------------------------------------
# Атрибуты и их синонимы
# ---------------------------------------------------------------------------

FIO_ATTRS = frozenset({"SURNAME", "NAME", "PATRONYMIC"})

_ATTR_ALIASES = {
    # фамилия
    "SURNAME": "SURNAME",
    "LASTNAME": "SURNAME",
    "LAST_NAME": "SURNAME",
    "ФАМИЛИЯ": "SURNAME",
    # имя
    "NAME": "NAME",
    "FIRSTNAME": "NAME",
    "FIRST_NAME": "NAME",
    "ИМЯ": "NAME",
    # отчество
    "PATRONYMIC": "PATRONYMIC",
    "MIDDLENAME": "PATRONYMIC",
    "MIDDLE_NAME": "PATRONYMIC",
    "ОТЧЕСТВО": "PATRONYMIC",
    # e-mail
    "EMAIL": "EMAIL",
    "E_MAIL": "EMAIL",
    "MAIL": "EMAIL",
    "ПОЧТА": "EMAIL",
    "ЭЛЕКТРОННАЯ_ПОЧТА": "EMAIL",
    # ИНН
    "INN": "INN",
    "ИНН": "INN",
    # телефон
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "ТЕЛЕФОН": "PHONE",
    "НОМЕР_ТЕЛЕФОНА": "PHONE",
    # банковская карта
    "BANK_CARD": "BANK_CARD",
    "BANKCARD": "BANK_CARD",
    "CARD": "BANK_CARD",
    "CREDIT_CARD": "BANK_CARD",
    "БАНКОВСКАЯ_КАРТА": "BANK_CARD",
    "КАРТА": "BANK_CARD",
    # СНИЛС (бонус: есть в ЛР №1)
    "SNILS": "SNILS",
    "СНИЛС": "SNILS",
    # паспорт
    "PASSPORT": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "ПАСПОРТ": "PASSPORT",
    # ОГРНИП
    "OGRNIP": "OGRNIP",
    "ОГРНИП": "OGRNIP",
    # IP
    "IP": "IP",
    "IP_ADDRESS": "IP",
    "IP_ADDR": "IP",
    "IP_АДРЕС": "IP",
    # свидетельство о рождении
    "BIRTH_CERT": "BIRTH_CERT",
    "BIRTH_CERTIFICATE": "BIRTH_CERT",
    "СВИДЕТЕЛЬСТВО_О_РОЖДЕНИИ": "BIRTH_CERT",
    # КЛАДР
    "KLADR": "KLADR",
    "КЛАДР": "KLADR",
}

_KIND = {
    "SURNAME": "Surn",
    "NAME": "Name",
    "PATRONYMIC": "Patr",
}


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------

class _Entity(NamedTuple):
    attr: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class _Gram(NamedTuple):
    case: str
    gender: str
    number: str


class _Info(NamedTuple):
    """Разобранная информация об одном ФИО-токене."""

    attr: str
    word: str
    score: float
    cases: frozenset[str]
    genders: frozenset[str]
    numbers: frozenset[str]
    case_guess: str
    gender_guess: str
    number_guess: str
    lemma: str


class _Person:
    """Обезличенный «человек»: одна замена на все его упоминания."""

    __slots__ = ("gender", "attrs", "bases")

    def __init__(self, gender: str, attrs: dict[str, str]) -> None:
        self.gender = gender
        self.attrs = dict(attrs)
        self.bases: dict[tuple[str, int], str] = {}

    def base(self, attr: str, shape: int = 1) -> str:
        """Возвращает замену для атрибута с учётом числа дефисных частей.

        shape=1 — обычная фамилия/имя, shape=2 — «Иванов-Петров».
        Одна и та же основа переиспользуется во всех падежах.
        """
        shape = max(1, int(shape))
        key = (attr, shape)
        if key in self.bases:
            return self.bases[key]

        if shape == 1:
            # Если человек уже встречался с дефисной формой, берём её
            # первую часть, чтобы «Иванов-Петров» и «Иванов» согласовались.
            for (other_attr, other_shape), value in self.bases.items():
                if other_attr == attr and other_shape > 1:
                    return value.split("-")[0]
            value = _replacement_base(attr, self.gender, self.attrs.get(attr, ""))
        else:
            parts = [self.base(attr, 1)]
            attempts = 0
            while len(parts) < shape and attempts < 100:
                attempts += 1
                candidate = _pick_base(attr, self.gender, "sing")
                if candidate not in parts:
                    parts.append(candidate)
            while len(parts) < shape:
                parts.append(_pick_base(attr, self.gender, "sing"))
            value = "-".join(parts)

        self.bases[key] = value
        return value


def _lemma_compatible(old: str, new: str) -> bool:
    """Совместимы ли леммы: полное имя и инициал «И.» — да."""
    if old == new:
        return True
    if len(old) == 1 and new.startswith(old):
        return True
    if len(new) == 1 and old.startswith(new):
        return True
    # «Иванов-Петров» и краткое «Иванов» — тоже один человек.
    old_parts = old.split("-")
    new_parts = new.split("-")
    if len(old_parts) > 1 and new in old_parts:
        return True
    if len(new_parts) > 1 and old in new_parts:
        return True
    return False


class _PersonRegistry:
    """Сопоставляет повторные упоминания одного человека одной замене."""

    def __init__(self) -> None:
        self.people: list[_Person] = []

    def resolve(self, attrs: dict[str, str], gender: str) -> _Person:
        for person in self.people:
            if person.gender != gender:
                continue
            shared = person.attrs.keys() & attrs.keys()
            if not shared:
                continue
            # Все общие атрибуты должны совпадать. Разные фамилии при
            # одинаковом имени — это разные люди.
            if all(
                _lemma_compatible(person.attrs[key], attrs[key])
                for key in shared
            ):
                for key, value in attrs.items():
                    old = person.attrs.get(key)
                    # Полное имя информативнее инициала.
                    if old is None or (len(old) == 1 and len(value) > 1):
                        person.attrs[key] = value
                return person
        person = _Person(gender, attrs)
        self.people.append(person)
        return person


class _FioPlan(NamedTuple):
    gram: _Gram
    base: str


_DEFAULT_CASE = "nomn"
_DEFAULT_GENDER = "masc"
_DEFAULT_NUMBER = "sing"


# ---------------------------------------------------------------------------
# Нормализация входа
# ---------------------------------------------------------------------------

def _normalize_attr(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    key = re.sub(r"[\s\-]+", "_", raw.strip().upper())
    key = key.replace("Ё", "Е")
    return _ATTR_ALIASES.get(key)


def _as_index(value: object) -> int | None:
    """Приводит индекс к int, отбрасывая bool и откровенный мусор."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def _coerce_entities(text: str, attributes: object) -> list[_Entity]:
    """Аккуратно приводит пользовательский список к списку _Entity.

    Всё, что не похоже на корректный кортеж (атрибут, start, end), молча
    пропускается: обезличивание не должно падать из-за «мусора» на входе.
    """
    if attributes is None or isinstance(attributes, (str, bytes)):
        return []

    # Разрешаем передать одну сущность без внешнего списка: ["NAME", 0, 4]
    if (
        isinstance(attributes, (list, tuple))
        and len(attributes) == 3
        and isinstance(attributes[0], str)
        and _as_index(attributes[1]) is not None
        and _as_index(attributes[2]) is not None
    ):
        attributes = [attributes]

    try:
        iterator = iter(attributes)  # type: ignore[arg-type]
    except TypeError:
        return []

    result: list[_Entity] = []
    for item in iterator:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        attr = _normalize_attr(item[0])
        if attr is None:
            continue
        start = _as_index(item[1])
        end = _as_index(item[2])
        if start is None or end is None:
            continue
        if start < 0 or end > len(text) or end <= start:
            continue
        result.append(_Entity(attr, start, end))
    return result


def _select_non_overlapping(entities: Iterable[_Entity]) -> list[_Entity]:
    """Сортирует сущности и убирает пересечения.

    При конфликте выигрывает более длинная сущность (она «специфичнее»),
    при равной длине — та, что идёт раньше.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -e.length))
    selected: list[_Entity] = []
    cursor = -1
    for entity in ordered:
        if entity.start < cursor:
            continue
        selected.append(entity)
        cursor = entity.end
    return selected


# ---------------------------------------------------------------------------
# Морфология ФИО
# ---------------------------------------------------------------------------

def _morph_parse(word: str):
    if _MORPH is None:
        return []
    try:
        return _MORPH.parse(word)
    except Exception:  # pragma: no cover - защита от неожиданных входов
        return []


def _candidates(attr: str, word: str):
    """Возвращает разборы, относящиеся к нужному типу слова."""
    parses = _morph_parse(word)
    if not parses:
        return []
    kind = _KIND.get(attr)
    if kind:
        tagged = [p for p in parses if kind in p.tag.grammemes]
        if tagged:
            return tagged
    nouns = [p for p in parses if p.tag.POS == "NOUN"]
    return nouns or parses


def _guess_gender(word: str, attr: str, parses) -> str:
    """Эвристика рода, когда pymorphy не даёт однозначного ответа."""
    lower = word.lower().replace("ё", "е")

    if attr == "SURNAME":
        fem_nom = (
            "ова", "ева", "ина", "ына", "ская", "цкая", "ая", "яя",
        )
        mas = (
            "ов", "ев", "ин", "ын", "ский", "цкий", "ий", "ый",
            "ко", "ук", "юк", "ич", "ец",
        )
        if lower.endswith(fem_nom) and any(p.tag.gender == "femn" for p in parses):
            return "femn"
        if lower.endswith("ой"):
            # Толстой (м.) vs Ивановой (ж. р., косвенный падеж)
            if not any(
                p.tag.gender == "masc" and p.tag.case == "nomn" for p in parses
            ):
                return "femn"
            return "masc"
        if lower.endswith(mas):
            return "masc"
        if lower.endswith(("а", "я")):
            return "femn"
        genders = {p.tag.gender for p in parses}
        if "femn" in genders and "masc" not in genders:
            return "femn"
        return "masc"

    if attr == "PATRONYMIC":
        return "femn" if lower.endswith("на") else "masc"

    # NAME
    male_exceptions = {
        "никита", "илья", "фома", "лука", "кузьма", "савва", "данила",
        "добрыня", "гаврила", "михайла", "иона", "савватий",
    }
    if lower in male_exceptions:
        return "masc"
    if lower.endswith(("а", "я")):
        return "femn"
    genders = {p.tag.gender for p in parses}
    if "femn" in genders and "masc" not in genders:
        return "femn"
    return "masc"


def _info_for_entity(text: str, entity: _Entity) -> _Info:
    raw_word = text[entity.start:entity.end]
    # Распознаватель из ЛР №1 иногда включает в спан запятую или точку.
    # Для анализа берём только буквенное ядро, а при замене пунктуация
    # всё равно сохраняется.
    _, core, _ = _strip_nonletters(raw_word)
    word = core or raw_word
    parses = _candidates(entity.attr, word)
    score = max((p.score for p in parses), default=0.0)

    cases = frozenset(p.tag.case for p in parses if p.tag.case)
    genders = frozenset(
        p.tag.gender for p in parses if p.tag.gender in ("masc", "femn")
    )
    numbers = frozenset(p.tag.number for p in parses if p.tag.number)

    gender_guess = _guess_gender(word, entity.attr, parses)

    # Уточняем падеж/число по разборам, совпадающим с выбранным родом:
    # это важно для «Иванова», где первый разбор — мужской родительный,
    # а нам нужен женский именительный.
    same_gender = [p for p in parses if p.tag.gender == gender_guess]
    pool = same_gender or parses

    def top(field: str, default: str) -> str:
        values = [getattr(p.tag, field) for p in pool if getattr(p.tag, field)]
        if values:
            return values[0]
        all_values = [getattr(p.tag, field) for p in parses if getattr(p.tag, field)]
        return all_values[0] if all_values else default

    # Лемма нужна, чтобы узнать одно и то же ФИО в разных падежах:
    # «Алексей» и «Алексею» должны дать «алексей».
    lemma_source = pool or parses
    lemma = ""
    for parse in lemma_source:
        if parse.tag.case == "nomn" and parse.tag.number == "sing":
            lemma = parse.normal_form
            break
    if not lemma and lemma_source:
        lemma = lemma_source[0].normal_form
    if not lemma:
        lemma = word
    lemma = lemma.lower().replace("ё", "е")

    return _Info(
        attr=entity.attr,
        word=raw_word,
        score=score,
        cases=cases,
        genders=genders,
        numbers=numbers,
        case_guess=top("case", _DEFAULT_CASE),
        gender_guess=gender_guess,
        number_guess=top("number", _DEFAULT_NUMBER),
        lemma=lemma,
    )


def _intersect(sets: Sequence[frozenset[str]]) -> frozenset[str]:
    if not sets:
        return frozenset()
    result = set(sets[0])
    for value in sets[1:]:
        result &= value
    return frozenset(result)


def _choose(infos: Sequence[_Info], field: str) -> str:
    """Выбирает общий род/падеж/число для группы ФИО.

    Сначала пробуем пересечение множеств допустимых значений. Если оно
    неоднозначно, доверяем более информативному атрибуту: NAME → PATRONYMIC
    → SURNAME. Это позволяет корректно обрабатывать «Иванова Мария».
    """
    sets = [getattr(info, "cases" if field == "case" else field + "s") for info in infos]
    common = _intersect(sets)
    if len(common) == 1:
        return next(iter(common))

    for preferred in ("NAME", "PATRONYMIC", "SURNAME"):
        for info in infos:
            if info.attr == preferred:
                value = (
                    info.gender_guess
                    if field == "gender"
                    else getattr(info, field + "_guess")
                )
                if value:
                    return value

    best = max(infos, key=lambda i: i.score)
    value = (
        best.gender_guess if field == "gender" else getattr(best, field + "_guess")
    )
    defaults = {
        "case": _DEFAULT_CASE,
        "gender": _DEFAULT_GENDER,
        "number": _DEFAULT_NUMBER,
    }
    return value or defaults[field]


def _can_join(previous: _Entity, current: _Entity, text: str) -> bool:
    gap = text[previous.end:current.start]
    return len(gap) <= 6 and re.search(r"[A-Za-zА-Яа-яЁё0-9]", gap) is None


def _duplicate_role(group: Sequence[_Entity], current: _Entity) -> bool:
    # В одном ФИО не может быть двух имён/фамилий/отчеств. Это защищает
    # от склейки двух соседних людей в одну группу.
    return any(entity.attr == current.attr for entity in group)


def _group_fio(text: str, entities: Sequence[_Entity]) -> list[list[_Entity]]:
    groups: list[list[_Entity]] = []
    current: list[_Entity] = []
    for entity in entities:
        if (
            current
            and _can_join(current[-1], entity, text)
            and len(current) < 3
            and not _duplicate_role(current, entity)
        ):
            current.append(entity)
        else:
            if current:
                groups.append(current)
            current = [entity]
    if current:
        groups.append(current)
    return groups


def _word_shape(raw: str) -> int:
    """Сколько дефисных частей в слове: Иванов = 1, Иванов-Петров = 2."""
    _, core, _ = _strip_nonletters(raw)
    if not core:
        return 1
    return max(1, len([part for part in core.split("-") if part]))


_AMBIGUOUS_NAMES = frozenset(
    {"женя", "саша", "валя", "слава", "паша", "шура", "сима"}
)

_MALE_ROLE_WORDS = frozenset(
    {
        "сотрудник", "гражданин", "мужчина", "юноша", "мальчик", "отец",
        "сын", "брат", "дедушка", "директор", "заявитель", "работник",
        "клиент", "пациент", "студент", "аспирант", "военнослужащий",
    }
)
_FEMALE_ROLE_WORDS = frozenset(
    {
        "сотрудница", "гражданка", "женщина", "девушка", "девочка",
        "мать", "дочь", "сестра", "бабушка", "директриса", "заявительница",
        "работница", "клиентка", "пациентка", "студентка", "аспирантка",
    }
)


def _context_tokens(text: str, group: Sequence[_Entity]) -> list[str]:
    """Токены ближайшего предложения: сначала слева (от центра наружу),
    затем справа. Это не даёт захватить глагол из соседнего предложения."""
    left = text[max(0, group[0].start - 140):group[0].start]
    right = text[group[-1].end:min(len(text), group[-1].end + 140)]
    left_clause = re.split(r"[.!?;]", left)[-1]
    right_clause = re.split(r"[.!?;]", right)[0]
    left_tokens = re.findall(r"[А-Яа-яЁё]+", left_clause.lower())
    right_tokens = re.findall(r"[А-Яа-яЁё]+", right_clause.lower())
    return list(reversed(left_tokens[-8:])) + right_tokens[:8]


def _context_gender(text: str, group: Sequence[_Entity]) -> str | None:
    """Определяет род по ближайшим словам: «Женя подписал/подписала»,
    «сотрудник/сотрудница Иванова»."""
    tokens = _context_tokens(text, group)
    for token in tokens:
        if token in _MALE_ROLE_WORDS:
            return "masc"
        if token in _FEMALE_ROLE_WORDS:
            return "femn"
        if token in {"он", "его", "ему", "им", "нем"}:
            return "masc"
        if token in {"она", "ее", "ей", "ею", "ней"}:
            return "femn"

    # Глаголы прошедшего времени и прилагательные дают род надёжнее,
    # чем окончание самого имени.
    for token in tokens:
        for parse in _morph_parse(token):
            tag = parse.tag
            if tag.POS in ("VERB", "INFN", "PRTF", "PRTS") and tag.tense == "past":
                if tag.gender in ("masc", "femn"):
                    return tag.gender
            if tag.POS == "ADJF" and tag.gender in ("masc", "femn"):
                return tag.gender
    return None


def _context_number(text: str, group: Sequence[_Entity]) -> str | None:
    tokens = _context_tokens(text, group)
    for token in tokens:
        for parse in _morph_parse(token):
            if parse.tag.POS == "VERB" and parse.tag.tense == "past":
                if parse.tag.number == "plur":
                    return "plur"
    return None


def _resolve_fio(text: str, entities: Sequence[_Entity]) -> dict[_Entity, _FioPlan]:
    result: dict[_Entity, _FioPlan] = {}
    registry = _PersonRegistry()
    fio = [e for e in entities if e.attr in FIO_ATTRS]
    for group in _group_fio(text, fio):
        infos = [_info_for_entity(text, entity) for entity in group]
        context_gender = _context_gender(text, group)
        context_number = _context_number(text, group)

        name_info = next((info for info in infos if info.attr == "NAME"), None)
        ambiguous_name = (
            name_info is not None
            and name_info.word.lower().replace("ё", "е") in _AMBIGUOUS_NAMES
        )
        gender = _choose(infos, "gender")
        if context_gender and (ambiguous_name or name_info is None):
            gender = context_gender
        number = _choose(infos, "number")
        if context_number and (ambiguous_name or name_info is None):
            number = context_number

        gram = _Gram(
            case=_choose(infos, "case"),
            gender=gender,
            number=number,
        )
        attrs = {info.attr: info.lemma for info in infos}
        person = registry.resolve(attrs, gram.gender)
        for entity in group:
            shape = _word_shape(text[entity.start:entity.end])
            result[entity] = _FioPlan(
                gram=gram,
                base=person.base(entity.attr, shape),
            )
    return result


# ---------------------------------------------------------------------------
# Генерация ФИО
# ---------------------------------------------------------------------------

# Списки подобраны так, чтобы pymorphy3 уверенно разбирал и склонял все
# слова. Faker тоже подошёл бы, но он иногда выдаёт редкие имена и слова,
# которые pymorphy трактует как обычные существительные (например,
# «Мясник»), из-за чего падежные формы получаются неверными.

_SURNAMES_M = (
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров",
    "Соколов", "Михайлов", "Новиков", "Фёдоров", "Морозов", "Волков",
    "Алексеев", "Лебедев", "Семёнов", "Егоров", "Павлов", "Козлов",
    "Степанов", "Николаев", "Орлов", "Андреев", "Макаров", "Никитин",
    "Захаров", "Зайцев", "Соловьёв", "Борисов", "Яковлев", "Григорьев",
    "Романов", "Воробьёв", "Сергеев", "Кузьмин", "Фролов", "Александров",
    "Дмитриев", "Королёв", "Гусев", "Киселёв", "Ильин", "Максимов",
    "Поляков", "Сорокин", "Виноградов", "Ковалёв", "Белов", "Медведев",
    "Антонов", "Тарасов", "Жуков", "Баранов", "Филиппов", "Комаров",
    "Давыдов", "Беляев", "Герасимов", "Богданов", "Осипов", "Сидоров",
    "Матвеев", "Титов", "Марков", "Миронов", "Крылов", "Куликов",
    "Карпов", "Власов", "Мельников", "Денисов", "Гаврилов", "Тихонов",
    "Казаков", "Афанасьев", "Данилов", "Савельев", "Тимофеев", "Фомин",
    "Чернов", "Абрамов", "Мартынов", "Ефимов", "Федотов", "Щербаков",
    "Назаров", "Калинин", "Исаев", "Чернышёв", "Быков", "Маслов",
    "Родионов", "Коновалов", "Лазарев", "Воронин", "Климов", "Филатов",
    "Пономарёв", "Голубев", "Кудрявцев", "Прохоров", "Наумов", "Потапов",
    "Журавлёв", "Овчинников", "Трофимов", "Леонов", "Соболев", "Ермаков",
    "Колесников", "Гончаров", "Емельянов", "Никифоров", "Грачёв", "Котов",
    "Гришин", "Ефремов", "Архипов", "Громов", "Малышев",
    "Панов", "Моисеев", "Румянцев", "Акимов", "Кондратьев", "Бирюков",
    "Горбунов", "Анисимов", "Ерёмин", "Тихомиров", "Галкин", "Лукьянов",
    "Михеев", "Скворцов", "Юдин", "Белоусов", "Нестеров", "Симонов",
    "Прокофьев", "Харитонов", "Князев", "Цветков", "Левин", "Митрофанов",
    "Воронов", "Аксёнов", "Мальцев", "Логинов", "Горшков",
    "Савин", "Краснов", "Майоров", "Демидов", "Елисеев", "Рыбаков",
    "Сафонов", "Платонов", "Филимонов", "Гущин", "Ширяев", "Кулагин",
    "Толстой", "Достоевский",
)

_NAMES_M = (
    "Александр", "Алексей", "Андрей", "Антон", "Аркадий", "Арсений",
    "Артём", "Богдан", "Борис", "Вадим", "Валентин", "Валерий",
    "Василий", "Виктор", "Виталий", "Владимир", "Владислав",
    "Всеволод", "Вячеслав", "Геннадий", "Георгий", "Герман", "Глеб",
    "Григорий", "Даниил", "Денис", "Дмитрий", "Евгений", "Егор",
    "Захар", "Иван", "Игорь", "Илья", "Кирилл", "Константин", "Лев",
    "Леонид", "Максим", "Марат", "Марк", "Матвей", "Михаил", "Никита",
    "Николай", "Олег", "Павел", "Пётр", "Платон", "Роман", "Ростислав",
    "Руслан", "Савелий", "Семён", "Сергей", "Станислав", "Степан",
    "Тарас", "Тимур", "Трофим", "Фёдор", "Филипп", "Эдуард", "Юрий",
    "Ярослав",
)

_NAMES_F = (
    "Александра", "Алина", "Алла", "Анастасия", "Ангелина", "Анна",
    "Антонина", "Валентина", "Валерия", "Варвара", "Вера", "Вероника",
    "Виктория", "Галина", "Дарья", "Диана", "Евгения", "Екатерина",
    "Елена", "Елизавета", "Жанна", "Зинаида", "Зоя", "Инна", "Ирина",
    "Карина", "Кира", "Клавдия", "Кристина", "Ксения", "Лариса",
    "Лидия", "Любовь", "Людмила", "Маргарита", "Марина", "Мария",
    "Милана", "Надежда", "Наталья", "Нина", "Оксана", "Ольга",
    "Полина", "Раиса", "Регина", "Светлана", "Серафима", "Снежана",
    "Софья", "Тамара", "Татьяна", "Ульяна", "Юлия", "Яна",
)

_PATRONYMICS_M = (
    "Александрович", "Алексеевич", "Андреевич", "Антонович",
    "Аркадьевич", "Арсеньевич", "Артёмович", "Богданович",
    "Борисович", "Вадимович", "Валентинович", "Валерьевич",
    "Васильевич", "Викторович", "Витальевич", "Владимирович",
    "Владиславович", "Всеволодович", "Вячеславович", "Геннадьевич",
    "Георгиевич", "Германович", "Глебович", "Григорьевич",
    "Данилович", "Денисович", "Дмитриевич", "Евгеньевич",
    "Егорович", "Захарович", "Иванович", "Игоревич", "Ильич",
    "Кириллович", "Константинович", "Львович", "Леонидович",
    "Максимович", "Маркович", "Матвеевич", "Михайлович", "Никитич",
    "Николаевич", "Олегович", "Павлович", "Петрович", "Платонович",
    "Романович", "Ростиславович", "Русланович", "Савельевич",
    "Семёнович", "Сергеевич", "Станиславович", "Степанович",
    "Тарасович", "Тимурович", "Трофимович", "Фёдорович",
    "Филиппович", "Эдуардович", "Юрьевич", "Ярославович",
)

_LATIN_NAMES_M = (
    "Alexander", "Alexey", "Andrew", "Anton", "Boris", "Daniel",
    "Dmitry", "Eugene", "Ivan", "Maxim", "Michael", "Nikolay",
    "Pavel", "Roman", "Sergey", "Victor", "Vladimir", "Yuri",
)
_LATIN_NAMES_F = (
    "Alexandra", "Alina", "Anastasia", "Anna", "Daria", "Elena",
    "Elizabeth", "Irina", "Ksenia", "Maria", "Marina", "Natalia",
    "Olga", "Polina", "Svetlana", "Tatiana", "Valentina", "Vera",
)
_LATIN_SURNAMES = (
    "Ivanov", "Smirnov", "Kuznetsov", "Popov", "Vasiliev", "Petrov",
    "Sokolov", "Mikhailov", "Novikov", "Fedorov", "Morozov", "Volkov",
    "Alekseev", "Lebedev", "Semenov", "Egorov", "Pavlov", "Kozlov",
)


def _female_surname(male: str) -> str:
    if male.endswith(("ов", "ев", "ёв", "ин", "ын")):
        return male + "а"
    if male.endswith(("ский", "цкий")):
        return male[:-2] + "ая"
    if male.endswith(("ой", "ый")):
        return male[:-2] + "ая"
    if male.endswith("ко"):
        return male  # несклоняемая фамилия
    return male + "а"


def _female_patronymic(male: str) -> str:
    if male == "Ильич":
        return "Ильинична"
    if male.endswith(("ович", "евич")):
        return male[:-2] + "на"
    if male.endswith("ич"):
        return male[:-2] + "ична"
    return male + "на"


def _simple_lemma(word: str) -> str:
    parses = _morph_parse(word)
    if parses:
        return parses[0].normal_form.lower().replace("ё", "е")
    return word.lower().replace("ё", "е")


def _base_conflicts(candidate: str, forbidden: str) -> bool:
    """Проверяет, не совпадает ли замена с исходным словом по лемме."""
    if not forbidden:
        return False
    forbidden_parts = forbidden.split("-")
    candidate_parts = candidate.split("-")
    if len(forbidden_parts) == len(candidate_parts) and len(candidate_parts) > 1:
        return all(
            _simple_lemma(candidate_part) == forbidden_part
            for candidate_part, forbidden_part in zip(candidate_parts, forbidden_parts)
        )
    if len(forbidden) == 1:
        # Инициал: сравниваем только первую букву.
        return _simple_lemma(candidate)[:1] == forbidden
    return _simple_lemma(candidate) == forbidden


def _replacement_base(attr: str, gender: str, forbidden: str = "") -> str:
    """Случайная основа, по возможности не совпадающая с исходной леммой."""
    candidate = _pick_base(attr, gender, "sing")
    for _ in range(80):
        if not _base_conflicts(candidate, forbidden):
            return candidate
        candidate = _pick_base(attr, gender, "sing")
    return candidate


def _pick_base(attr: str, gender: str, number: str) -> str:
    female = gender == "femn"
    if attr == "SURNAME":
        male = _RNG.choice(_SURNAMES_M)
        return _female_surname(male) if female else male
    if attr == "PATRONYMIC":
        male = _RNG.choice(_PATRONYMICS_M)
        return _female_patronymic(male) if female else male
    return _RNG.choice(_NAMES_F if female else _NAMES_M)


def _inflect(base: str, gram: _Gram, attr: str) -> str:
    if _MORPH is None:
        return base

    kind = _KIND.get(attr)
    parses = _morph_parse(base)
    if not parses:
        return base

    candidates = list(parses)
    if kind:
        tagged = [p for p in candidates if kind in p.tag.grammemes]
        if tagged:
            candidates = tagged
    nouns = [p for p in candidates if p.tag.POS == "NOUN"]
    if nouns:
        candidates = nouns

    if gram.number == "sing":
        same_gender = [p for p in candidates if p.tag.gender == gram.gender]
        if same_gender:
            candidates = same_gender

    grams = {gram.case, gram.number}
    if gram.number == "sing" and gram.gender in ("masc", "femn"):
        grams.add(gram.gender)

    for parse in candidates:
        try:
            form = parse.inflect(grams)
        except Exception:  # pragma: no cover
            form = None
        if form is not None:
            return form.word

    # Мягкий откат: без рода (иногда мешает для несклоняемых слов)
    for parse in candidates:
        try:
            form = parse.inflect({gram.case, gram.number})
        except Exception:  # pragma: no cover
            form = None
        if form is not None:
            return form.word

    return base


def _match_case(original: str, generated: str) -> str:
    letters = [char for char in original if char.isalpha()]
    if not letters:
        return generated
    if original.isupper():
        return generated.upper()
    if letters[0].isupper():
        return generated[:1].upper() + generated[1:]
    return generated.lower()


def _strip_nonletters(token: str) -> tuple[str, str, str]:
    match = re.search(r"[A-Za-zА-Яа-яЁё]", token)
    if not match:
        return token, "", ""
    start = match.start()
    end = max(i for i, char in enumerate(token) if char.isalpha()) + 1
    return token[:start], token[start:end], token[end:]


def _random_letters(length: int, *, cyrillic: bool) -> str:
    alphabet = "абвгдежзиклмнопрстуфхцчшщэюя" if cyrillic else "abcdefghijklmnopqrstuvwxyz"
    return "".join(_RNG.choice(alphabet) for _ in range(max(1, length)))


def _gen_person_core(
    core: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    # Инициалы: «И.» -> «А.», сохраняем длину и регистр. Если замена уже
    # известна, берём первую букву из неё — тогда «И.» и «Ивану» одного
    # человека дадут согласованные «А.» и «Александру».
    if len(core) == 1:
        if base:
            letter = base[:1].lower()
            if not re.fullmatch(r"[а-яё]", letter):
                letter = _RNG.choice("абвгдежзиклмнопрстуфхцчшщэюя")
        else:
            letter = _RNG.choice("абвгдежзиклмнопрстуфхцчшщэюя")
        return _match_case(core, letter)

    provided_base = base is not None
    if base is None:
        base = _pick_base(attr, gram.gender, gram.number)
    form = _inflect(base, gram, attr)

    # Совсем неизвестное слово (аббревиатура, опечатка) без готовой
    # замены — сохраняем длину и алфавит, чтобы не сломать структуру.
    if not provided_base and not _morph_parse(core):
        cyrillic = bool(re.search(r"[А-Яа-яЁё]", core))
        return _match_case(core, _random_letters(len(core), cyrillic=cyrillic))

    return _match_case(core, form)


def _gen_person_token(
    token: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    prefix, core, suffix = _strip_nonletters(token)
    if not core:
        return token
    for _ in range(6):
        generated = _gen_person_core(core, attr, gram, base)
        if generated.lower() != core.lower():
            return prefix + generated + suffix
    return prefix + generated + suffix


def _gen_latin_token(token: str) -> str:
    """Запасной вариант для ФИО, записанных латиницей."""
    prefix, core, suffix = _strip_nonletters(token)
    if not core:
        return token
    if len(core) == 1:
        return prefix + _match_case(core, _RNG.choice(string.ascii_uppercase)) + suffix
    pool = _LATIN_SURNAMES + _LATIN_NAMES_M + _LATIN_NAMES_F
    for _ in range(8):
        candidate = _match_case(core, _RNG.choice(pool))
        if candidate.lower() != core.lower():
            return prefix + candidate + suffix
    return prefix + _match_case(core, _RNG.choice(pool)) + suffix


def _anonymize_fio(
    raw: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    # Для дефисных форм («Иванов-Петров») основа тоже дефисная:
    # «Смирнов-Кузнецов». Каждая часть склоняется отдельно.
    base_parts = None
    if base and "-" in base and "-" in raw:
        base_parts = base.split("-")

    parts = re.split(r"([\s\-]+)", raw)
    result: list[str] = []
    part_index = 0
    for part in parts:
        if not part or re.fullmatch(r"[\s\-]+", part):
            result.append(part)
            continue
        if re.search(r"[А-Яа-яЁё]", part):
            if base_parts and part_index < len(base_parts):
                part_base = base_parts[part_index]
            else:
                part_base = base
            result.append(_gen_person_token(part, attr, gram, part_base))
            part_index += 1
        elif re.search(r"[A-Za-z]", part):
            result.append(_gen_latin_token(part))
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# Контрольные разряды
# ---------------------------------------------------------------------------

_INN_COEFFS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_check_digit(digits: Sequence[int], coeffs: Sequence[int]) -> int:
    return sum(digit * coeff for digit, coeff in zip(digits, coeffs)) % 11 % 10


def validate_inn(value: str) -> bool:
    """Проверяет контрольные разряды ИНН (10 или 12 цифр)."""
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) == 10:
        return digits[-1] == _inn_check_digit(digits[:9], _INN_COEFFS_10)
    if len(digits) == 12:
        if digits[-2] != _inn_check_digit(digits[:10], _INN_COEFFS_11):
            return False
        return digits[-1] == _inn_check_digit(digits[:11], _INN_COEFFS_12)
    return False


def _luhn_sum(digits: Sequence[int]) -> int:
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total


def validate_luhn(value: str) -> bool:
    """Проверяет номер банковской карты по алгоритму Луна."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    return _luhn_sum(digits) % 10 == 0


def _snils_check(digits: Sequence[int]) -> int:
    total = sum(digit * (9 - index) for index, digit in enumerate(digits[:9]))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    check = total % 101
    return 0 if check == 100 else check


def validate_snils(value: str) -> bool:
    """Проверяет контрольное число СНИЛС."""
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) != 11:
        return False
    return digits[9] * 10 + digits[10] == _snils_check(digits)


# ---------------------------------------------------------------------------
# Генерация числовых атрибутов
# ---------------------------------------------------------------------------

def _rand_digit() -> str:
    return _RNG.choice(string.digits)


def _substitute_digits(raw: str, new_digits: Sequence[str]) -> str:
    """Подставляет новые цифры в исходную строку, сохраняя все разделители."""
    result: list[str] = []
    index = 0
    for char in raw:
        if char.isdigit():
            if index < len(new_digits):
                result.append(new_digits[index])
                index += 1
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def _digits_of(raw: str) -> list[str]:
    return [char for char in raw if char.isdigit()]


def _anonymize_inn(raw: str) -> str:
    digits = _digits_of(raw)
    length = len(digits)

    if length < 4:
        new_digits = [_rand_digit() for _ in range(length)]
    elif length == 10:
        body = digits[:4] + [_rand_digit() for _ in range(5)]
        new_digits = body + [str(_inn_check_digit([int(d) for d in body], _INN_COEFFS_10))]
    elif length == 12:
        body = digits[:4] + [_rand_digit() for _ in range(6)]  # 10 цифр
        check11 = str(_inn_check_digit([int(d) for d in body], _INN_COEFFS_11))
        body11 = body + [check11]
        check12 = str(_inn_check_digit([int(d) for d in body11], _INN_COEFFS_12))
        new_digits = body11 + [check12]
    else:
        # Нестандартная длина: сохраняем первые 4 цифры и обезличиваем остальные.
        new_digits = digits[:4] + [_rand_digit() for _ in range(length - 4)]

    if new_digits == digits and length > 4:
        return _anonymize_inn(raw) if length in (10, 12) else raw
    return _substitute_digits(raw, new_digits)


def _luhn_fix(digits: list[str], middle: Sequence[int]) -> bool:
    """Подбирает одну цифру в середине так, чтобы номер стал валиден по Луну."""
    numeric = [int(d) for d in digits]
    for index in middle:
        original = numeric[index]
        for candidate in range(10):
            numeric[index] = candidate
            if _luhn_sum(numeric) % 10 == 0:
                digits[index] = str(candidate)
                return True
        numeric[index] = original
    return False


def _anonymize_card(raw: str) -> str:
    digits = _digits_of(raw)
    length = len(digits)

    if length < 10:
        # Невозможно одновременно сохранить 6+4 цифры — просто обезличиваем.
        new_digits = [_rand_digit() for _ in range(length)]
        return _substitute_digits(raw, new_digits)

    middle = list(range(6, length - 4))
    if not middle:
        # 10 цифр: первые 6 и последние 4 занимают всё число, менять нечего.
        return raw

    for _ in range(8):
        new_digits = list(digits)
        for index in middle:
            new_digits[index] = _rand_digit()
        if not _luhn_fix(new_digits, middle):
            break
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)

    # Крайне маловероятный откат: меняем один средний разряд вручную.
    new_digits = list(digits)
    for index in middle:
        for candidate in range(10):
            new_digits[index] = str(candidate)
            if _luhn_sum([int(d) for d in new_digits]) % 10 == 0:
                return _substitute_digits(raw, new_digits)
    return raw


def _anonymize_phone(raw: str) -> str:
    digits = _digits_of(raw)
    if len(digits) <= 4:
        # Меньше 5 цифр — сохранять и обезличивать нечего.
        return raw

    for _ in range(8):
        new_digits = digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)])


def _anonymize_snils(raw: str) -> str:
    digits = _digits_of(raw)
    if len(digits) != 11:
        return _substitute_digits(raw, [_rand_digit() for _ in range(len(digits))])

    for _ in range(8):
        base = [_rand_digit() for _ in range(9)]
        check = _snils_check([int(d) for d in base])
        new_digits = base + [str(check // 10), str(check % 10)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_passport(raw: str) -> str:
    """Паспорт: серия (первые 4 цифры) сохраняется, номер заменяется."""
    digits = _digits_of(raw)
    if len(digits) <= 4:
        return _substitute_digits(raw, [_rand_digit() for _ in range(len(digits))])
    for _ in range(8):
        new_digits = digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_ogrnip(raw: str) -> str:
    """ОГРНИП: первые 4 цифры сохраняются, контрольный разряд пересчитан."""
    digits = _digits_of(raw)
    if len(digits) != 15:
        if len(digits) <= 4:
            return _substitute_digits(raw, [_rand_digit() for _ in range(len(digits))])
        return _substitute_digits(
            raw, digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)]
        )
    for _ in range(8):
        body = digits[:4] + [_rand_digit() for _ in range(10)]
        check = (int("".join(body)) % 13) % 10
        new_digits = body + [str(check)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_ip(raw: str) -> str:
    """Заменяет IP на другой корректный адрес того же семейства."""
    stripped = raw.strip()
    try:
        address = ipaddress.ip_address(stripped)
    except ValueError:
        return raw
    if isinstance(address, ipaddress.IPv4Address):
        for _ in range(16):
            first = _RNG.randint(1, 223)
            if first in (10, 127):
                continue
            candidate = ipaddress.IPv4Address(
                f"{first}.{_RNG.randint(1, 254)}."
                f"{_RNG.randint(1, 254)}.{_RNG.randint(1, 254)}"
            )
            if candidate != address:
                return str(candidate)
        return str(ipaddress.IPv4Address("203.0.113.1"))
    for _ in range(16):
        candidate = ipaddress.IPv6Address(_RNG.getrandbits(128))
        if candidate != address:
            text = str(candidate)
            if "::" not in stripped:
                text = candidate.exploded
            if stripped.isupper():
                text = text.upper()
            return text
    return str(ipaddress.IPv6Address("2001:db8::1"))


def _anonymize_birth_cert(raw: str) -> str:
    """Свидетельство о рождении: серия сохраняется, номер заменяется."""
    match = re.search(
        r"([IVX]{1,4}\s*[-–]?\s*[А-ЯЁ]{2}\s*(?:№|N|номер)?\s*)(\d{6})",
        raw,
        re.IGNORECASE,
    )
    if match:
        for _ in range(8):
            number = "".join(_rand_digit() for _ in range(6))
            if number != match.group(2):
                return raw[:match.start(2)] + number + raw[match.end(2):]
    return _substitute_digits(raw, [_rand_digit() for _ in range(len(_digits_of(raw)))])


def _anonymize_kladr(raw: str) -> str:
    """КЛАДР: код региона (первые 2 цифры) сохраняется."""
    digits = _digits_of(raw)
    if len(digits) <= 2:
        return _substitute_digits(raw, [_rand_digit() for _ in range(len(digits))])
    for _ in range(8):
        new_digits = digits[:2] + [_rand_digit() for _ in range(len(digits) - 2)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


# ---------------------------------------------------------------------------
# E-mail
# ---------------------------------------------------------------------------

def _random_local_part(local: str) -> str:
    result: list[str] = []
    for char in local:
        if char.isalpha():
            alphabet = string.ascii_uppercase if char.isupper() else string.ascii_lowercase
            result.append(_RNG.choice(alphabet))
        elif char.isdigit():
            result.append(_rand_digit())
        else:
            result.append(char)
    return "".join(result)


def _anonymize_email(raw: str) -> str:
    if "@" not in raw:
        # Не e-mail, но обезличить всё равно нужно: сохраняем пунктуацию.
        for _ in range(8):
            candidate = _random_local_part(raw)
            if candidate != raw:
                return candidate
        return _random_local_part(raw)

    local, _, domain = raw.rpartition("@")
    if not local:
        return raw

    for _ in range(8):
        candidate = _random_local_part(local)
        if candidate != local:
            return candidate + "@" + domain
    return _random_local_part(local) + "@" + domain


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def _anonymize_value(
    raw: str,
    entity: _Entity,
    plan: _FioPlan | None,
) -> str:
    if entity.attr in FIO_ATTRS:
        if plan is None:
            gram = _Gram("nomn", "masc", "sing")
            plan = _FioPlan(
                gram=gram,
                base=_pick_base(entity.attr, gram.gender, gram.number),
            )
        return _anonymize_fio(raw, entity.attr, plan.gram, plan.base)
    if entity.attr == "EMAIL":
        return _anonymize_email(raw)
    if entity.attr == "INN":
        return _anonymize_inn(raw)
    if entity.attr == "PASSPORT":
        return _anonymize_passport(raw)
    if entity.attr == "PHONE":
        return _anonymize_phone(raw)
    if entity.attr == "BANK_CARD":
        return _anonymize_card(raw)
    if entity.attr == "OGRNIP":
        return _anonymize_ogrnip(raw)
    if entity.attr == "IP":
        return _anonymize_ip(raw)
    if entity.attr == "BIRTH_CERT":
        return _anonymize_birth_cert(raw)
    if entity.attr == "KLADR":
        return _anonymize_kladr(raw)
    if entity.attr == "SNILS":
        return _anonymize_snils(raw)
    return raw


_NUMERIC_ATTRS = frozenset(
    {"INN", "PASSPORT", "BANK_CARD", "OGRNIP", "PHONE", "SNILS", "KLADR"}
)


def _cache_key(attr: str, raw: str) -> tuple[str, str]:
    """Нормализует значение для словаря подстановок.

    Одинаковые реквизиты в разном оформлении (пробелы, дефисы, регистр)
    должны получить одну замену.
    """
    if attr in _NUMERIC_ATTRS:
        return attr, "".join(_digits_of(raw))
    if attr == "BIRTH_CERT":
        return attr, re.sub(r"\s+", " ", raw.upper()).strip()
    if attr == "EMAIL":
        return attr, raw.lower()
    if attr == "IP":
        try:
            return attr, str(ipaddress.ip_address(raw.strip()))
        except ValueError:
            return attr, raw
    return attr, raw


def anonymize_with_map(
    text: str, attributes: object
) -> tuple[str, list[dict[str, object]]]:
    """Обезличивает текст и возвращает карту замен.

    Возвращает ``(новый_текст, replacements)``, где каждый элемент
    replacements содержит атрибут, исходное значение, замену, границы и
    признак повторного использования словаря подстановок.
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")

    entities = _select_non_overlapping(_coerce_entities(text, attributes))
    if not entities:
        return text, []

    fio_plans = _resolve_fio(text, entities)
    # Повторяющиеся не-ФИО значения (один и тот же телефон, e-mail и т.п.)
    # тоже должны заменяться одинаково в пределах одного текста.
    exact_cache: dict[tuple[str, str], str] = {}
    replacements: list[dict[str, object]] = []

    pieces: list[str] = []
    cursor = 0
    for entity in entities:
        pieces.append(text[cursor:entity.start])
        raw = text[entity.start:entity.end]
        reused = False
        if entity.attr in FIO_ATTRS:
            replacement = _anonymize_value(raw, entity, fio_plans.get(entity))
        else:
            key = _cache_key(entity.attr, raw)
            cached = exact_cache.get(key)
            if cached is None:
                replacement = _anonymize_value(raw, entity, None)
                exact_cache[key] = replacement
            elif entity.attr in _NUMERIC_ATTRS:
                # То же значение в другом оформлении: берём те же цифры,
                # но расставляем их по разделителям текущего написания.
                replacement = _substitute_digits(raw, _digits_of(cached))
                reused = True
            else:
                replacement = cached
                reused = True
        replacements.append(
            {
                "attr": entity.attr,
                "start": entity.start,
                "end": entity.end,
                "original": raw,
                "replacement": replacement,
                "reused": reused,
            }
        )
        pieces.append(replacement)
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces), replacements


def anon(text: str, attributes: object) -> str:
    """Обезличивает переданные сущности в тексте.

    Параметры
    ---------
    text:
        Исходный текст.
    attributes:
        Список сущностей в формате ЛР №1: ``[["NAME", 0, 4], ...]``.

    Возвращает
    ----------
    str
        Текст, в котором каждая корректная сущность заменена на случайное
        значение с сохранением требуемых свойств. Некорректные элементы
        списка игнорируются, исходный текст в этих местах не меняется.
    """
    return anonymize_with_map(text, attributes)[0]


# Синоним на случай, если удобнее называть функцию полным словом.
anonymize = anon


if __name__ == "__main__":
    sample = (
        "Иванов Иван Иванович, ИНН 7707083893, телефон +7 (999) 123-45-67, "
        "e-mail ivanov@example.com, карта 4084520359695002."
    )

    def span(attr: str, value: str) -> list[object]:
        start = sample.find(value)
        return [attr, start, start + len(value)]

    found = [
        span("SURNAME", "Иванов"),
        span("NAME", "Иван"),
        span("PATRONYMIC", "Иванович"),
        span("INN", "7707083893"),
        span("PHONE", "+7 (999) 123-45-67"),
        span("EMAIL", "ivanov@example.com"),
        span("BANK_CARD", "4084520359695002"),
    ]
    print("До:   ", sample)
    print("После:", anon(sample, found))
