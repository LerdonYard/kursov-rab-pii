"""Справочник атрибутов ПДн курсовой работы (Вариант 20, чётный набор).

Вариант 20 относится к чётным номерам, поэтому обязательный набор совпадает
с Вариантом 6 из задания:

    SURNAME, NAME, INN, PASSPORT, BANK_CARD, OGRNIP, IP, EMAIL,
    BIRTH_CERT, KLADR.

Дополнительно сохранены атрибуты, уже реализованные в ЛР №1/№2 и полезные
для демонстрации морфологии и контрольных разрядов:

    PATRONYMIC, PHONE, SNILS.
"""

from __future__ import annotations

# Обязательные атрибуты чётного варианта.
REQUIRED_ATTRS = (
    "SURNAME",
    "NAME",
    "INN",
    "PASSPORT",
    "BANK_CARD",
    "OGRNIP",
    "IP",
    "EMAIL",
    "BIRTH_CERT",
    "KLADR",
)

# Дополнительные атрибуты, которые поддерживает программа.
EXTRA_ATTRS = (
    "PATRONYMIC",
    "PHONE",
    "SNILS",
)

ALL_ATTRS = REQUIRED_ATTRS + EXTRA_ATTRS

FIO_ATTRS = frozenset({"SURNAME", "NAME", "PATRONYMIC"})

# Атрибуты, у которых при обезличивании сохраняются косвенные признаки.
STRUCTURED_ATTRS = frozenset(
    {
        "INN",
        "PASSPORT",
        "BANK_CARD",
        "OGRNIP",
        "IP",
        "EMAIL",
        "BIRTH_CERT",
        "KLADR",
        "PHONE",
        "SNILS",
    }
)

DISPLAY = {
    "SURNAME": "Фамилия",
    "NAME": "Имя",
    "PATRONYMIC": "Отчество",
    "INN": "ИНН",
    "PASSPORT": "Номер паспорта",
    "BANK_CARD": "Номер банковской карты",
    "OGRNIP": "ОГРНИП",
    "IP": "IP-адрес",
    "EMAIL": "E-mail",
    "BIRTH_CERT": "Свидетельство о рождении",
    "KLADR": "Код КЛАДР",
    "PHONE": "Номер телефона",
    "SNILS": "СНИЛС",
}

# Синонимы для устойчивого чтения ручной разметки и старых файлов ЛР №1.
ALIASES = {
    "SURNAME": "SURNAME",
    "LASTNAME": "SURNAME",
    "LAST_NAME": "SURNAME",
    "ФАМИЛИЯ": "SURNAME",
    "NAME": "NAME",
    "FIRSTNAME": "NAME",
    "FIRST_NAME": "NAME",
    "ИМЯ": "NAME",
    "PATRONYMIC": "PATRONYMIC",
    "MIDDLENAME": "PATRONYMIC",
    "MIDDLE_NAME": "PATRONYMIC",
    "ОТЧЕСТВО": "PATRONYMIC",
    "INN": "INN",
    "ИНН": "INN",
    "PASSPORT": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "ПАСПОРТ": "PASSPORT",
    "BANK_CARD": "BANK_CARD",
    "BANKCARD": "BANK_CARD",
    "CARD": "BANK_CARD",
    "CREDIT_CARD": "BANK_CARD",
    "БАНКОВСКАЯ_КАРТА": "BANK_CARD",
    "КАРТА": "BANK_CARD",
    "OGRNIP": "OGRNIP",
    "ОГРНИП": "OGRNIP",
    "IP": "IP",
    "IP_ADDRESS": "IP",
    "IP_ADDR": "IP",
    "IP-АДРЕС": "IP",
    "IP_АДРЕС": "IP",
    "EMAIL": "EMAIL",
    "E_MAIL": "EMAIL",
    "MAIL": "EMAIL",
    "ПОЧТА": "EMAIL",
    "ЭЛЕКТРОННАЯ_ПОЧТА": "EMAIL",
    "BIRTH_CERT": "BIRTH_CERT",
    "BIRTH_CERTIFICATE": "BIRTH_CERT",
    "СВИДЕТЕЛЬСТВО_О_РОЖДЕНИИ": "BIRTH_CERT",
    "KLADR": "KLADR",
    "КЛАДР": "KLADR",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "ТЕЛЕФОН": "PHONE",
    "НОМЕР_ТЕЛЕФОНА": "PHONE",
    "SNILS": "SNILS",
    "СНИЛС": "SNILS",
}

# Приоритеты разрешения пересечений: чем больше число, тем специфичнее
# правило. Структурированные реквизиты всегда выигрывают у NER-имён.
PRIORITY = {
    "EMAIL": 100,
    "IP": 96,
    "SNILS": 95,
    "PHONE": 94,
    "BIRTH_CERT": 93,
    "PASSPORT": 92,
    "KLADR": 91,
    "OGRNIP": 90,
    "BANK_CARD": 89,
    "INN": 88,
    "SURNAME": 50,
    "NAME": 50,
    "PATRONYMIC": 50,
}
