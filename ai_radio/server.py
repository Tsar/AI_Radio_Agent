"""Сервер «мозга»: LLMResponder за HTTP для тонких клиентов у рации.

Протокол описан в remote.py, здесь его серверная половина на http.server из stdlib —
тянуть веб-фреймворк ради двух путей незачем, а у прода интернет только на время
настройки. Запуск: `main.py serve`, обычно на машине с видеокартой.

Одна фраза за раз. Конвейер строго последовательный (GPU занят Whisper, потом
llama-server, потом RVC), а DialogState — один разговор: один позывной, один канал.
Второй клиент, если он есть, ждёт в очереди на замке, а /health отвечает всегда —
для этого сервер многопоточный.

Ошибка внутри respond() — это ответ 500 с текстом, а не смерть сервера: клиент
пропустит фразу и останется в эфире (та же логика, что у репитера при OOM).
"""
from __future__ import annotations

import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

from .audio_io import floats_to_wav, wav_to_floats
from .config import Config
from .remote import HEALTH_PATH, META_HEADER, META_MAX_BYTES, RESPOND_PATH


class LogTap:
    """print плюс копия строк для клиента. Строки копятся за одну фразу — под
    замком сервера, — и забираются take() перед отправкой ответа."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, line: str) -> None:
        print(line)
        self.lines.append(line)

    def take(self) -> List[str]:
        lines, self.lines = self.lines, []
        return lines


def _meta_header(meta: Dict[str, object]) -> str:
    text = json.dumps(meta, ensure_ascii=True, separators=(",", ":"))
    if len(text) > META_MAX_BYTES:
        # Тайминги важнее журнала: без них сломается bench, без журнала — нет
        meta = dict(meta, log=[f"[NET] журнал фразы не влез в заголовок "
                               f"({len(text)} байт) — смотрите его на сервере"])
        text = json.dumps(meta, ensure_ascii=True, separators=(",", ":"))
    return text


class BrainServer(ThreadingHTTPServer):
    daemon_threads = True       # Ctrl+C не ждёт застрявший обработчик

    def __init__(self, address: "tuple[str, int]", cfg: Config, responder, tap: LogTap,
                 kind: str) -> None:
        super().__init__(address, _Handler)
        self.cfg = cfg
        self.responder = responder
        self.tap = tap
        self.kind = kind
        self.lock = threading.Lock()
        self.n_phrases = 0

    def health(self) -> Dict[str, object]:
        cfg = self.cfg
        return {
            "status": "ok",
            "busy": self.lock.locked(),
            "responder": self.kind,
            "sample_rate": cfg.audio.sample_rate,
            "callsign": cfg.dialog.callsign,
            "stt": cfg.stt.model,
            "rvc": cfg.rvc.enabled,
            "reply_length": cfg.llm.reply_length,
            "phrases": self.n_phrases,
        }


class _Handler(BaseHTTPRequestHandler):
    server: BrainServer
    timeout = 30.0              # клиент оборвался посреди загрузки — не висеть вечно

    def log_message(self, fmt: str, *args: object) -> None:
        # свои строки в журнале; штатная печатает на каждую фразу IP, путь и код
        pass

    def _send(self, code: int, body: bytes = b"", content_type: Optional[str] = None,
              meta: Optional[str] = None) -> None:
        self.send_response(code)
        if meta is not None:
            self.send_header(META_HEADER, meta)
        if body:
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, code: int, obj: Dict[str, object], meta: Optional[str] = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8", meta)

    def do_GET(self) -> None:
        if self.path == HEALTH_PATH:
            self._send_json(200, self.server.health())
        else:
            self._send_json(404, {"error": f"нет такого пути: {self.path}"})

    def do_POST(self) -> None:
        if self.path != RESPOND_PATH:
            self._send_json(404, {"error": f"нет такого пути: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"error": "пустое тело — ожидается WAV (16 бит, моно)"})
            return
        rate = self.server.cfg.audio.sample_rate
        try:
            utterance = wav_to_floats(self.rfile.read(length), rate)
        except (wave.Error, EOFError) as exc:
            self._send_json(400, {"error": f"тело — не WAV: {exc}"})
            return
        if not utterance:
            self._send_json(400, {"error": "в WAV нет отсчётов"})
            return

        srv = self.server
        with srv.lock:
            srv.n_phrases += 1
            print(f"[NET] фраза #{srv.n_phrases}: {len(utterance) / rate:.2f} с "
                  f"от {self.client_address[0]}")
            t0 = time.monotonic()
            try:
                audio = srv.responder.respond(utterance)
            except Exception as exc:          # noqa: BLE001 — одна фраза не роняет сервер
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[ERR] фраза не обработана: {msg}")
                if "out of memory" in msg.lower():
                    print("[ERR] это нехватка VRAM: кто держит память — nvidia-smi; "
                          "чаще всего помогает `systemctl --user restart ai-radio-rvc`")
                meta = _meta_header({"log": srv.tap.take(), "timings": {}})
                self._send_json(500, {"error": msg}, meta)
                return
            timings = dict(getattr(srv.responder, "timings", None) or {})
            timings["server"] = time.monotonic() - t0
            meta = _meta_header({"log": srv.tap.take(), "timings": timings})

        if audio:
            self._send(200, floats_to_wav(audio, rate), "audio/wav", meta)
        else:
            self._send(204, meta=meta)


def serve(cfg: Config, host: str, port: int, kind: str = "llm") -> int:
    from .responder import ParrotResponder, build_llm_responder

    tap = LogTap()
    if kind == "llm":
        responder = build_llm_responder(cfg, log=tap)
    elif kind == "parrot":
        responder = ParrotResponder()
        print("[init] ответчик: parrot — эхо, чтобы проверить связь и PTT без моделей")
    else:
        raise ValueError(f"неизвестный ответчик: {kind!r}")

    try:
        server = BrainServer((host, port), cfg, responder, tap, kind)
    except OSError as exc:
        raise RuntimeError(f"не удалось занять {host}:{port}: {exc}") from exc

    print(f"Сервер слушает {host}:{port}. Клиент у рации: "
          f"main.py run --live --responder remote --server http://<адрес этой машины>:{port}. "
          f"Ctrl+C — стоп.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        server.server_close()
    print(f"итого фраз: {server.n_phrases}")
    return 0
