"""Протоколы движков — в том же стиле, что Ptt и Responder.

Аудио везде одно и то же: list[float] в [-1, 1] на рабочей частоте (16 кГц), как
отдают FileSource/MicSource. Благодаря этому RVC позже встанет декоратором поверх
TtsEngine, не меняя ничего вокруг.
"""
from __future__ import annotations

from typing import Dict, List, Protocol


class SttEngine(Protocol):
    def transcribe(self, audio: List[float]) -> str:
        """Распознать фразу. Пустая строка — «не разобрал», агент промолчит."""
        ...


class LlmEngine(Protocol):
    def reply(self, messages: List[Dict[str, str]]) -> str:
        """messages — в формате OpenAI chat (role/content). Вернуть текст ответа."""
        ...


class TtsEngine(Protocol):
    def synth(self, text: str) -> List[float]:
        """Синтезировать речь на рабочей частоте, уровень уже нормализован."""
        ...
