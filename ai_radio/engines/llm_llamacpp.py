"""LLM: llama-server (llama.cpp) по OpenAI-совместимому HTTP.

Отдельный процесс, а не библиотека в нашем: модель всегда в VRAM (агент может
молчать часами, повторная загрузка стоила бы 10+ с на первый ответ), CUDA-зависимости
не смешиваются с CTranslate2, а перезапуск llama-server не трогает аудио-конвейер.

Запросы через urllib из stdlib — тянуть requests ради одного POST незачем.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List

from ..config import LlmConfig
from ..textnorm import strip_think


class LlmUnavailable(RuntimeError):
    """Сервер не отвечает — понятная ошибка вместо стека urllib."""


class LlamaServerLlm:
    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    def reply(self, messages: List[Dict[str, str]]) -> str:
        cfg = self.cfg
        payload: Dict[str, object] = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": False,
        }
        if cfg.no_think:
            # Qwen3 иначе выдаёт <think>…</think> — это минуты рассуждений в эфир
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LlmUnavailable(f"llama-server ответил {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LlmUnavailable(
                f"llama-server недоступен по {cfg.base_url} ({exc}). "
                f"Запущен ли он? Пример: llama-server -m модель.gguf -ngl 99 -c 4096"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmUnavailable(f"неожиданный ответ llama-server: {data!r}") from exc

        # подстраховка: no_think может не сработать на чужом chat-шаблоне
        return strip_think(content).strip()

    def ping(self) -> bool:
        """Проверка при старте, чтобы не выяснять это на первой же фразе в эфире."""
        url = self.cfg.base_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False
