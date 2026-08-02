"""TTS: Piper (VITS в ONNX) на CPU.

На CPU он остаётся сознательно: конвейер строго последовательный, и Piper работает
ровно тогда, когда GPU уже отдал ответ LLM. Перенос на карту дал бы выигрыш только
на бумаге, зато столкнул бы onnxruntime и CTranslate2 на общем cuDNN.

Piper синтезирует на своей частоте (обычно 22050 Гц) — приводим к рабочей 16 кГц и
нормализуем уровень: глубина модуляции рации зависит от амплитуды на её входе.

Требует системный espeak-ng — он делает фонемизацию русского.
"""
from __future__ import annotations

from typing import List, Tuple

from ..audio_io import normalize_peak, pcm16_to_floats, resample_linear
from ..config import TtsConfig


class PiperTts:
    def __init__(self, cfg: TtsConfig, sample_rate: int = 16000) -> None:
        from piper import PiperVoice  # тяжёлый импорт — только здесь
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.voice = PiperVoice.load(cfg.voice)

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

    def synth(self, text: str) -> List[float]:
        if not text.strip():
            return []
        pcm, rate = self._synthesize_raw(text)
        samples = pcm16_to_floats(pcm)
        if not samples:
            return []
        if rate != self.sample_rate:
            samples = resample_linear(samples, rate, self.sample_rate)
        return normalize_peak(samples, self.cfg.peak_dbfs)
