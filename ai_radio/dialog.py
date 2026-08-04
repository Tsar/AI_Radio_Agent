"""Кому отвечать и что помнить: триггер по позывному + состояние диалога.

Канал общий, поэтому агент молчит, пока его не позвали по позывному. После своего
ответа открывается окно продолжения (window_s), внутри которого позывной повторять
не нужно — иначе живой разговор превращается в перекличку.

Только stdlib: сравнение строк через difflib, чтобы `trigger-test` работал без моделей.
"""
from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .config import DialogConfig

_PUNCT = re.compile(r"[^а-яё0-9 ]")


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, без пунктуации — общая форма для сравнения."""
    text = text.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _windows(words: Sequence[str], max_join: int = 3) -> "list[str]":
    """Склейки 1..max_join соседних слов.

    STT регулярно разбивает позывной на части («фе ечка», «фея чка»), поэтому
    сравнивать пословно недостаточно — нужны и склейки соседей.
    """
    out: List[str] = []
    for i in range(len(words)):
        for n in range(1, max_join + 1):
            if i + n <= len(words):
                out.append("".join(words[i:i + n]))
    return out


class CallsignTrigger:
    """Нечёткий поиск позывного: точного вхождения от STT ждать не приходится."""

    def __init__(self, cfg: DialogConfig) -> None:
        self.threshold = cfg.match_threshold
        self.variants = [normalize(v) for v in cfg.callsign_variants if v.strip()]
        if not self.variants:
            self.variants = [normalize(cfg.callsign)]

    def match(self, text: str) -> Tuple[bool, float, str]:
        """→ (сработал ли, лучшая схожесть, что именно совпало). Схожесть нужна
        для `trigger-test`: по ней подбирается порог."""
        best_score = 0.0
        best_token = ""
        for token in _windows(normalize(text).split()):
            for variant in self.variants:
                # заведомо разные длины сравнивать незачем
                if abs(len(token) - len(variant)) > 3:
                    continue
                score = SequenceMatcher(None, token, variant).ratio()
                if score > best_score:
                    best_score, best_token = score, token
        return best_score >= self.threshold, best_score, best_token

    def found(self, text: str) -> bool:
        return self.match(text)[0]


class DialogState:
    """История реплик + окно продолжения разговора."""

    def __init__(self, cfg: DialogConfig) -> None:
        self.cfg = cfg
        self.trigger = CallsignTrigger(cfg)
        self.history: Deque[Dict[str, str]] = deque(maxlen=cfg.max_history)
        self.last_reply_at: Optional[float] = None
        self._end_phrases = [normalize(p) for p in cfg.end_phrases]

    def _now(self) -> float:
        return time.monotonic()

    def window_open(self) -> bool:
        if self.last_reply_at is None:
            return False
        return (self._now() - self.last_reply_at) <= self.cfg.window_s

    def is_end_phrase(self, text: str) -> bool:
        norm = normalize(text)
        return any(p and p in norm for p in self._end_phrases)

    def should_answer(self, text: str) -> Tuple[bool, str]:
        """→ (отвечать ли, причина для лога)."""
        if self.trigger.found(text):
            return True, "позывной"
        if self.window_open():
            return True, "окно диалога"
        return False, "не нам"

    def reset(self) -> None:
        self.history.clear()
        self.last_reply_at = None

    def expire_if_stale(self) -> bool:
        """Окно закрылось — считаем прошлый разговор оконченным и чистим историю.

        Иначе его реплики протекут в контекст следующего обращения: скажешь
        позывной через час, а модель ответит так, будто разговор не прерывался.
        Таймаут и явное «отбой» должны значить одно и то же.
        """
        if self.history and not self.window_open():
            self.reset()
            return True
        return False

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self.last_reply_at = self._now()

    def messages(self, system_prompt: str) -> List[Dict[str, str]]:
        return [{"role": "system", "content": system_prompt}] + list(self.history)
