"""Responder — то, что превращает принятую фразу в ответную.

Главный шов конвейера: ParrotResponder (этап 1) просто возвращает принятое,
LLMResponder (этап 2) — это tts(llm(stt(utterance))). Всё остальное — VAD, PTT,
тайминги, state machine репитера — не знает, какой из них подключён.

Возврат None означает «не отвечаем»: репитер тогда не жмёт PTT. Молчание — штатный
и частый исход, потому что агент реагирует только на свой позывной.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Protocol

from .config import Config
from .dialog import DialogState
from .engines.base import LlmEngine, SttEngine, TtsEngine
from .engines.llm_llamacpp import LlmUnavailable
from .textnorm import clean_llm_reply


class Responder(Protocol):
    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        """Вернуть аудио-ответ (list[float] в [-1,1]) или None, если не отвечаем."""
        ...


class ParrotResponder:
    """Возвращает принятое без изменений — «попугай»."""

    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        return utterance


class LLMResponder:
    """Цепочка STT → триггер/диалог → LLM → нормализация текста → TTS."""

    def __init__(self, cfg: Config, stt: SttEngine, llm: LlmEngine, tts: TtsEngine,
                 dialog: Optional[DialogState] = None) -> None:
        self.cfg = cfg
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.dialog = dialog if dialog is not None else DialogState(cfg.dialog)
        self.timings: Dict[str, float] = {}   # длительности звеньев последнего ответа, с

    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        self.timings = {}

        t0 = time.monotonic()
        text = self.stt.transcribe(utterance)
        self.timings["stt"] = time.monotonic() - t0
        if not text:
            print("[STT] не разобрал — молчим")
            return None
        print(f"[STT] {text}")

        if self.dialog.is_end_phrase(text):
            self.dialog.reset()
            print("[--] отбой — диалог закрыт")
            return None

        answer, reason = self.dialog.should_answer(text)
        if not answer:
            print(f"[--] {reason} — молчим")
            return None
        print(f"[..] отвечаем ({reason})")

        self.dialog.add_user(text)
        t0 = time.monotonic()
        try:
            raw = self.llm.reply(self.dialog.messages(self.cfg.llm.system_prompt))
        except LlmUnavailable as exc:
            self.dialog.history.pop()      # неудачную реплику в контексте не копим
            print(f"[ERR] {exc}")
            return None
        finally:
            self.timings["llm"] = time.monotonic() - t0

        reply = clean_llm_reply(raw, max_sentences=self.cfg.llm.max_sentences)
        if not reply:
            self.dialog.history.pop()
            print("[LLM] пустой ответ — молчим")
            return None
        print(f"[LLM] {reply}")
        self.dialog.add_assistant(reply)

        t0 = time.monotonic()
        audio = self.tts.synth(reply)
        self.timings["tts"] = time.monotonic() - t0
        if not audio:
            print("[TTS] нечего передавать")
            return None
        return audio


def build_llm_responder(cfg: Config, check_llm: bool = True) -> LLMResponder:
    """Собрать движки по конфигу. Всё грузится один раз здесь, на старте."""
    from .engines import make_llm, make_stt, make_tts

    print(f"[init] STT: {cfg.stt.model} ({cfg.stt.device}, {cfg.stt.compute_type})…")
    stt = make_stt(cfg.stt, sample_rate=cfg.audio.sample_rate)
    print(f"[init] TTS: {cfg.tts.voice}…")
    tts = make_tts(cfg.tts, sample_rate=cfg.audio.sample_rate)
    llm = make_llm(cfg.llm)
    ping = getattr(llm, "ping", None)
    if check_llm and callable(ping) and not ping():
        print(f"[warn] llama-server не отвечает по {cfg.llm.base_url} — "
              f"ответы работать не будут, пока он не поднят")
    else:
        print(f"[init] LLM: {cfg.llm.base_url}")
    print(f"[init] позывной: {cfg.dialog.callsign}, окно диалога {cfg.dialog.window_s:.0f} с, "
          f"ответы {cfg.llm.reply_length} (≤{cfg.llm.max_sentences} предл.)")
    return LLMResponder(cfg, stt, llm, tts)
