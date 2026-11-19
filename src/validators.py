"""Проверки структуры и контрольных разрядов реквизитов ПДн."""

from __future__ import annotations

import ipaddress
import re
from typing import Sequence

__all__ = [
    "digits",
    "validate_inn",
    "validate_luhn",
    "validate_card",
    "validate_snils",
    "validate_ogrnip",
    "validate_ip",
    "validate_passport",
    "validate_kladr",
    "validate_birth_cert",
]


def digits(value: str) -> list[int]:
    return [int(char) for char in value if char.isdigit()]


# --- ИНН -----------------------------------------------------------------

_INN_COEFFS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_check_digit(values: Sequence[int], coeffs: Sequence[int]) -> int:
    return sum(value * coeff for value, coeff in zip(values, coeffs)) % 11 % 10


def validate_inn(value: str) -> bool:
    """ИНН: 10 цифр (организация) или 12 цифр (ИП/физлицо)."""
    values = digits(value)
    if len(values) == 10:
        return values[-1] == _inn_check_digit(values[:9], _INN_COEFFS_10)
    if len(values) == 12:
        if values[-2] != _inn_check_digit(values[:10], _INN_COEFFS_11):
            return False
        return values[-1] == _inn_check_digit(values[:11], _INN_COEFFS_12)
    return False


# --- Банковская карта -----------------------------------------------------

def _luhn_sum(values: Sequence[int]) -> int:
    total = 0
    for index, value in enumerate(reversed(values)):
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total


def validate_luhn(value: str) -> bool:
    values = digits(value)
    if not 13 <= len(values) <= 19:
        return False
    return _luhn_sum(values) % 10 == 0


def validate_card(value: str) -> bool:
    """Карта: Луна + отсечение заведомо некарточных номеров."""
    values = digits(value)
    if not 13 <= len(values) <= 19:
        return False
    if values[0] == 0 or len(set(values)) == 1:
        return False
    return _luhn_sum(values) % 10 == 0


# --- СНИЛС ----------------------------------------------------------------

def _snils_check(values: Sequence[int]) -> int:
    total = sum(value * (9 - index) for index, value in enumerate(values[:9]))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    check = total % 101
    return 0 if check == 100 else check


def validate_snils(value: str) -> bool:
    values = digits(value)
    if len(values) != 11:
        return False
    return values[9] * 10 + values[10] == _snils_check(values)


# --- ОГРНИП ---------------------------------------------------------------

def validate_ogrnip(value: str) -> bool:
    """ОГРНИП: 15 цифр, первая 3, контрольный разряд — остаток от деления
    первых 14 цифр на 13, взятый по модулю 10."""
    values = digits(value)
    if len(values) != 15 or values[0] != 3:
        return False
    body = int("".join(str(v) for v in values[:14]))
    return values[14] == (body % 13) % 10


# --- IP -------------------------------------------------------------------

def validate_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


# --- Паспорт --------------------------------------------------------------

def validate_passport(value: str) -> bool:
    """Паспорт РФ: серия 4 цифры + номер 6 цифр.

    Серия содержит код региона (01..99) и код подразделения; номер
    шестизначный. Контрольной суммы у паспорта нет.
    """
    values = digits(value)
    if len(values) != 10:
        return False
    series = values[:4]
    number = values[4:]
    if not 1 <= series[0] * 10 + series[1] <= 99:
        return False
    return number[0] != 0 or any(number)


# --- КЛАДР ----------------------------------------------------------------

def validate_kladr(value: str) -> bool:
    """Код КЛАДР — 13 цифр; первые две — код региона 01..99."""
    values = digits(value)
    if len(values) != 13:
        return False
    return 1 <= values[0] * 10 + values[1] <= 99


# --- Свидетельство о рождении --------------------------------------------

_BIRTH_CERT_RE = re.compile(
    r"(?<![А-ЯЁA-Z])([IVX]{1,4})\s*[-–]?\s*([А-ЯЁ]{2})\s*(?:№|N|номер)?\s*(\d{6})(?!\d)",
    re.IGNORECASE,
)


def validate_birth_cert(value: str) -> bool:
    return _BIRTH_CERT_RE.search(value) is not None
