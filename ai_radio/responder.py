"""Responder — то, что превращает принятую фразу в ответную.

Это главный шов для будущего LLM: на этапе 1 ParrotResponder просто возвращает
принятое (репитер-«попугай»); позже LLMResponder = tts(llm(stt(utterance))),
и больше в конвейере ничего менять не нужно.
"""
from __future__ import annotations

from typing import List, Optional, Protocol


class Responder(Protocol):
    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        """Вернуть аудио-ответ (list[float] в [-1,1]) или None, если не отвечаем."""
        ...


class ParrotResponder:
    """Возвращает принятое без изменений — «попугай»."""

    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        return utterance
