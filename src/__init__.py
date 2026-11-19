"""Курсовая работа: распознавание и синонимическое обезличивание ПДн."""

from .attributes import ALL_ATTRS, DISPLAY, REQUIRED_ATTRS
from .recognize import PIIDetector, extract_pii

__all__ = ["ALL_ATTRS", "DISPLAY", "REQUIRED_ATTRS", "PIIDetector", "extract_pii"]
