"""TTS: Piper (VITS в ONNX) на CPU.

На CPU он остаётся сознательно: конвейер строго последовательный, и Piper работает
ровно тогда, когда GPU уже отдал ответ LLM. Перенос на карту дал бы выигрыш только
на бумаге, зато столкнул бы onnxruntime и CTranslate2 на общем cuDNN.

Piper синтезирует на своей частоте (обычно 22050 Гц) — приводим к рабочей 16 кГц и
нормализуем уровень: глубина модуляции рации зависит от амплитуды на её входе.

Фонемизацию русского делает espeak-ng, но системный пакет не нужен: piper-tts >= 1.6
несёт espeakbridge.so и espeak-ng-data внутри колеса.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from ..audio_io import normalize_peak, pcm16_to_floats, resample_linear
from ..config import TtsConfig

# Корень проекта: .../ai_radio/engines/tts_piper.py -> ../../..
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_voice(path: str) -> str:
    """Дефолт `models/piper/…` — путь относительный, и запуск из другого каталога
    ронял синтез на «голос не найден». Ищем ещё и относительно корня проекта:
    голос лежит там, а не там, откуда позвали."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_ROOT, path)
    return candidate if os.path.exists(candidate) else path


class PiperTts:
    def __init__(self, cfg: TtsConfig, sample_rate: int = 16000) -> None:
        from piper import PiperVoice  # тяжёлый импорт — только здесь
        self.cfg = cfg
        self.sample_rate = sample_rate
        voice_path = _resolve_voice(cfg.voice)
        try:
            self.voice = PiperVoice.load(voice_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"голос Piper не найден: {cfg.voice} (нужны и .onnx, и .onnx.json). "
                f"Скачайте его — команда есть в README — или укажите путь флагом --voice"
            ) from exc

    def _voice_rate(self, default: int = 22050) -> int:
        config = getattr(self.voice, "config", None)
        return int(getattr(config, "sample_rate", default) or default)

    def _synthesize_raw(self, text: str) -> Tuple[bytes, int]:
        """API piper-tts менялся между версиями, поддерживаем оба варианта."""
        voice = self.voice
        length_scale = self.cfg.length_scale

        if hasattr(voice, "synthesize_stream_raw"):        # piper-tts 1.2.x
            kwargs = {} if length_scale is None else {"length_scale": length_scale}
            return b"".join(voice.synthesize_stream_raw(text, **kwargs)), self._voice_rate()

        chunks = None                                       # piper-tts 1.3+
        if length_scale is not None:
            try:
                from piper import SynthesisConfig
                chunks = list(voice.synthesize(
                    text, syn_config=SynthesisConfig(length_scale=length_scale)))
            except (ImportError, TypeError):
                chunks = None
        if chunks is None:
            chunks = list(voice.synthesize(text))
        if not chunks:
            return b"", self._voice_rate()

        rate = int(getattr(chunks[0], "sample_rate", 0) or self._voice_rate())
        pcm = b"".join(getattr(c, "audio_int16_bytes", b"") for c in chunks)
        return pcm, rate

    def _cap_duration(self, samples: List[float]) -> List[float]:
        """Аварийный потолок длительности (cfg.max_seconds).

        Стоит именно здесь, а не в репитере: RvcVoice оборачивает этот движок, и
        обрезать надо **до** отправки в RVC. Пик VRAM у RVC растёт с длиной фразы,
        и на 5 ГБ длинная передача роняет весь стек в OOM — а заодно надолго
        занимает канал, пока агент глух. До этого рубежа доходить не должно:
        выше по конвейеру текст уже подрезан max_sentences и max_chars.
        """
        limit = int(self.cfg.max_seconds * self.sample_rate)
        if limit <= 0 or len(samples) <= limit:
            return samples
        was = len(samples) / self.sample_rate
        samples = samples[:limit]
        fade = min(int(0.05 * self.sample_rate), limit)   # 50 мс, чтобы не щёлкнуло
        for i in range(fade):
            samples[limit - fade + i] *= (fade - 1 - i) / fade
        print(f"[TTS] ответ обрезан: {was:.2f} с → {self.cfg.max_seconds:.0f} с "
              f"(потолок эфира)")
        return samples

    def synth(self, text: str) -> List[float]:
        if not text.strip():
            return []
        pcm, rate = self._synthesize_raw(text)
        samples = pcm16_to_floats(pcm)
        if not samples:
            return []
        if rate != self.sample_rate:
            samples = resample_linear(samples, rate, self.sample_rate)
        return self._cap_duration(normalize_peak(samples, self.cfg.peak_dbfs))
