"""Полный конвейер: распознавание ПДн и синонимическое обезличивание."""

from __future__ import annotations

from dataclasses import dataclass, field

from .anonymize import anonymize_with_map
from .recognize import PIIDetector, extract_pii


@dataclass
class PipelineResult:
    """Результат обработки одного текста."""

    text: str
    anonymized: str
    attributes: list[list] = field(default_factory=list)
    replacements: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "anonymized": self.anonymized,
            "attributes": self.attributes,
            "replacements": self.replacements,
        }


_DEFAULT_DETECTOR: PIIDetector | None = None


def get_detector() -> PIIDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = PIIDetector()
    return _DEFAULT_DETECTOR


def process(text: str, detector: PIIDetector | None = None) -> PipelineResult:
    """Распознаёт ПДн и возвращает обезличенный текст и список атрибутов.

    Формат атрибутов совпадает с ЛР №1: ``[["NAME", 0, 4], ...]``.
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    detector = detector or get_detector()
    attributes = detector.extract(text)
    anonymized, replacements = anonymize_with_map(text, attributes)
    return PipelineResult(
        text=text,
        anonymized=anonymized,
        attributes=[list(item) for item in attributes],
        replacements=replacements,
    )


__all__ = ["PipelineResult", "extract_pii", "get_detector", "process"]
