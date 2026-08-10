"""Преобразование голоса (RVC) поверх готового TTS.

Декоратор над `TtsEngine`: Piper синтезирует фразу, RVC переозвучивает её целевым
голосом. Шов `Responder` и весь конвейер этого не замечают — меняется только то,
какой объект подставлен в `LLMResponder`.

RVC живёт **отдельным процессом со своим venv**: ему нужны numpy 1.23 и fairseq,
а у нас numpy 2.x ради faster-whisper и piper. Тот же приём, что с llama-server.
Сервис поднимается из форка RVC:

    .venv/bin/python infer-http-service.py --port 8081

Если сервис недоступен, фраза уходит в эфир **голосом Piper**, а не пропадает:
молчать из-за отказа косметического звена нельзя.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
import wave
from typing import List, Optional

from ..audio_io import _floats_to_pcm16, normalize_peak, pcm16_to_floats, resample_linear
from ..config import RvcConfig
from .base import TtsEngine


class RvcVoice:
    """Оборачивает любой TtsEngine и переозвучивает его выход через RVC-сервис."""

    def __init__(self, inner: TtsEngine, cfg: RvcConfig, sample_rate: int = 16000) -> None:
        self.inner = inner
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._warned = False        # об отказе сервиса сообщаем один раз, не на каждую фразу

    def _url(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        params = [
            f"voice={self.cfg.voice}",
            f"input_voice={self.cfg.input_voice}",
            f"f0_method={self.cfg.f0_method}",
            f"resample_sr={self.sample_rate}",   # просим сразу рабочую частоту
        ]
        if self.cfg.pitch is not None:
            params.append(f"pitch={self.cfg.pitch}")
        if self.cfg.formant_shift is not None:
            params.append(f"formant_shift={self.cfg.formant_shift}")
        return f"{base}/convert?" + "&".join(params)

    def _to_wav(self, samples: List[float]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(_floats_to_pcm16(samples))
        return buf.getvalue()

    def _from_wav(self, data: bytes) -> List[float]:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            samples = pcm16_to_floats(w.readframes(w.getnframes()))
        if rate != self.sample_rate:
            samples = resample_linear(samples, rate, self.sample_rate)
        return samples

    def convert(self, samples: List[float]) -> List[float]:
        """Переозвучить готовое аудио. Бросает URLError/OSError, если сервис лежит."""
        req = urllib.request.Request(
            self._url(),
            data=self._to_wav(samples),
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            return self._from_wav(resp.read())

    def synth(self, text: str) -> List[float]:
        samples = self.inner.synth(text)
        if not samples:
            return samples
        try:
            converted = self.convert(samples)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, wave.Error) as exc:
            if not self._warned:
                print(f"[warn] RVC недоступен ({exc}) — передаём голосом Piper")
                self._warned = True
            return samples
        self._warned = False
        if not converted:
            return samples
        # RVC отдаёт свой уровень — приводим обратно к целевому пику для эфира
        return normalize_peak(converted, self.cfg.peak_dbfs)

    def ping(self) -> bool:
        url = self.cfg.base_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False
