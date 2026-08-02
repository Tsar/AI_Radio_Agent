"""Движки этапа 2: STT, LLM, TTS.

Каждый — отдельный модуль с локальными («ленивыми») импортами тяжёлых библиотек,
поэтому `ai_radio.engines` можно импортировать на машине без faster-whisper и piper.
Протоколы — в base.py; фабрики собирают их по конфигу.
"""
from __future__ import annotations

from .base import LlmEngine, SttEngine, TtsEngine

__all__ = ["SttEngine", "LlmEngine", "TtsEngine", "make_stt", "make_llm", "make_tts"]


_HINT = "не установлен: .venv/bin/pip install -r requirements-ai.txt"


def make_stt(cfg, sample_rate: int = 16000) -> "SttEngine":
    from .stt_whisper import FasterWhisperStt
    try:
        # сам faster_whisper импортируется в конструкторе, ловить надо здесь
        return FasterWhisperStt(cfg, sample_rate=sample_rate)
    except ImportError as exc:
        raise RuntimeError(f"faster-whisper {_HINT}") from exc


def make_llm(cfg) -> "LlmEngine":
    from .llm_llamacpp import LlamaServerLlm   # только stdlib, ставить нечего
    return LlamaServerLlm(cfg)


def make_tts(cfg, sample_rate: int = 16000) -> "TtsEngine":
    from .tts_piper import PiperTts
    try:
        return PiperTts(cfg, sample_rate=sample_rate)
    except ImportError as exc:
        raise RuntimeError(f"piper-tts {_HINT} (плюс системный espeak-ng)") from exc
