"""Клиент к «мозгу» на другой машине: Responder, который шлёт фразу по HTTP.

Зачем: машина с видеокартой стоит там, где ей удобно, а рация — там, где антенна и
розетка. Тонкому клиенту у рации нужны только звук, VAD и PTT (numpy, sounddevice,
pyserial); весь тяжёлый стек — Whisper, LLM, Piper, RVC — живёт на сервере
(`main.py serve`, ai_radio/server.py). Шов тот же, что между репитером и
LLMResponder: фраза целиком туда, готовое аудио обратно, None — молчим.

Протокол нарочно повторяет RVC-сервис — WAV в обе стороны, отлаживается curl'ом:

    POST /respond   тело audio/wav (16 бит, моно, рабочая частота)
        200 audio/wav   ответ — его надо передать в эфир
        204             сервер решил молчать: не разобрал, не позвали, галлюцинация
        4xx/5xx JSON    {"error": "…"} — клиент пропускает фразу, как при отказе LLM
    GET  /health    JSON: позывной, модель STT, RVC, частота, занят ли сервер

В ответах на /respond есть заголовок X-AI-Radio-Meta — JSON с двумя полями:
    log      строки, которые сервер напечатал в свой журнал по этой фразе
             («[STT] …», «[--] не нам — молчим», «[LLM] …»). Клиент печатает их у
             себя: у рации должно быть видно, что услышал Whisper и почему агент
             молчит, без хождения в журнал сервера.
    timings  секунды по звеньям (stt / llm / tts) плюс server — всё время на сервере.
JSON в заголовке — чистый ASCII (кириллица как \\uXXXX): HTTP не обещает в
заголовках ничего, кроме ASCII, а тело отдано под аудио, чтобы ответ можно было
сохранить как есть (`curl … -o reply.wav`) и послушать.

Только stdlib (urllib), как и клиенты llama-server и RVC.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import wave
from typing import Callable, Dict, List, Optional

from .audio_io import floats_to_wav, wav_to_floats
from .config import RemoteConfig

RESPOND_PATH = "/respond"
HEALTH_PATH = "/health"
META_HEADER = "X-AI-Radio-Meta"
# http.client не примет строку заголовка длиннее 64 КиБ. Реальный журнал фразы —
# единицы килобайт даже в \\uXXXX, но предел лучше держать с запасом.
META_MAX_BYTES = 32000


class RemoteUnavailable(RuntimeError):
    """Сервер не отвечает или отказал — понятная ошибка вместо стека urllib."""


def _error_text(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        return str(json.loads(body).get("error", body))[:500]
    except (ValueError, AttributeError):
        return body[:500]


def _parse_meta(raw: Optional[str]) -> Dict[str, object]:
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except ValueError:
        return {}
    return meta if isinstance(meta, dict) else {}


class RemoteResponder:
    """Responder, который отдаёт фразу серверу и возвращает его аудио."""

    def __init__(self, cfg: RemoteConfig, sample_rate: int = 16000,
                 log: Callable[[str], None] = print) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.log = log
        base = cfg.base_url.rstrip("/")
        self.url = base + RESPOND_PATH
        self.health_url = base + HEALTH_PATH
        self.timings: Dict[str, float] = {}   # как у LLMResponder — для bench

    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        self.timings = {}
        req = urllib.request.Request(
            self.url,
            data=floats_to_wav(utterance, self.sample_rate),
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                status = resp.status
                meta = _parse_meta(resp.headers.get(META_HEADER))
                body = resp.read()
        except urllib.error.HTTPError as exc:
            # 4xx/5xx: журнал фразы сервер всё равно прислал — покажем, что он
            # успел сделать до отказа (например, «[STT] …» перед падением LLM)
            for line in _parse_meta(exc.headers.get(META_HEADER)).get("log", []) or []:
                self.log(str(line))
            raise RemoteUnavailable(f"сервер ответил {exc.code}: {_error_text(exc)}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RemoteUnavailable(
                f"сервер недоступен по {self.cfg.base_url} ({exc}). "
                f"Запущен ли там `main.py serve`?") from exc
        elapsed = time.monotonic() - t0

        for line in meta.get("log", []) or []:
            self.log(str(line))
        timings: Dict[str, float] = {}
        for key, value in (meta.get("timings", {}) or {}).items():
            if isinstance(value, (int, float)):
                timings[str(key)] = float(value)
        # сеть и упаковка WAV — всё, что не вошло в замер самого сервера
        timings["net"] = max(0.0, elapsed - timings.get("server", 0.0))
        self.timings = timings
        self.log(f"[NET] сервер: {elapsed:.2f} с (из них сеть {timings['net']:.2f} с)")

        if status == 204 or not body:
            return None
        try:
            return wav_to_floats(body, self.sample_rate)
        except (wave.Error, EOFError) as exc:
            raise RemoteUnavailable(f"сервер прислал не WAV: {exc}") from exc

    def health(self) -> Optional[Dict[str, object]]:
        """Что за мозг на том конце: позывной, модель STT, RVC. None — не отвечает."""
        try:
            with urllib.request.urlopen(self.health_url, timeout=5.0) as resp:
                if not 200 <= resp.status < 300:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
        return data if isinstance(data, dict) else {}

    def ping(self) -> bool:
        return self.health() is not None
