"""Repeater — state machine: приём (VAD) → ответ (Responder) → передача (PTT+Sink).

Строго half-duplex: во время передачи вход не слушаем. Логика preroll и hangtime
живёт здесь; VAD отвечает только на вопрос «этот кадр — речь?».
"""
from __future__ import annotations

import time
from collections import deque
from typing import Iterable, List

from .config import Config
from .ptt import Ptt
from .responder import Responder
from .vad import EnergyVad


class Repeater:
    def __init__(self, cfg: Config, vad: EnergyVad, responder: Responder,
                 ptt: Ptt, sink, realtime: bool = False) -> None:
        self.cfg = cfg
        self.vad = vad
        self.responder = responder
        self.ptt = ptt
        self.sink = sink
        self.realtime = realtime  # реально ли выдерживать паузы warmup/tail/cooldown
        self.source = None

        fs = cfg.audio.frame_samples
        self.frame_samples = fs
        self.start_frames = cfg.vad.start_frames
        self.hangtime_frames = max(1, cfg.vad.hangtime_ms // cfg.audio.frame_ms)
        self.preroll_frames = max(0, cfg.vad.preroll_ms // cfg.audio.frame_ms)

        self.n_transmissions = 0

    def run(self, source) -> None:
        """source — объект с методом frames() -> Iterable[list[float]]."""
        self.source = source
        self._run_frames(source.frames())

    def _run_frames(self, frames: Iterable[List[float]]) -> None:
        state = "IDLE"
        speech_run = 0
        silence_run = 0
        utterance: List[float] = []
        preroll: "deque[List[float]]" = deque(maxlen=self.preroll_frames)

        for frame in frames:
            speech = self.vad.is_speech(frame)

            if state == "IDLE":
                preroll.append(frame)
                if speech:
                    speech_run += 1
                    if speech_run >= self.start_frames:
                        state = "RECEIVING"
                        utterance = []
                        for pf in preroll:      # preroll уже содержит атаку фразы
                            utterance.extend(pf)
                        silence_run = 0
                else:
                    speech_run = 0

            elif state == "RECEIVING":
                utterance.extend(frame)
                if speech:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= self.hangtime_frames:
                        self._end_utterance(utterance, silence_run)
                        state = "IDLE"
                        speech_run = silence_run = 0
                        utterance = []
                        preroll.clear()

        # поток кончился, а мы ещё принимали — до-передать
        if state == "RECEIVING":
            self._end_utterance(utterance, silence_run)

    def _end_utterance(self, utterance: List[float], trailing_silence_frames: int) -> None:
        # срезать хвостовую тишину (hangtime), чтобы не гнать её в эфир
        trim = trailing_silence_frames * self.frame_samples
        if 0 < trim < len(utterance):
            utterance = utterance[:-trim]
        dur = len(utterance) / self.cfg.audio.sample_rate
        print(f"[RX] принята фраза: {dur:.2f} с")

        response = self.responder.respond(utterance)
        if not response:
            print("[--] ответа нет, не передаём")
            return
        self._transmit(response)
        self._drain_source()

    def _transmit(self, audio: List[float]) -> None:
        dur = len(audio) / self.cfg.audio.sample_rate
        self.n_transmissions += 1
        print(f"[TX] передача #{self.n_transmissions}: {dur:.2f} с")
        self.ptt.key()
        self._sleep(self.cfg.tx.warmup_ms)
        self.sink.play(audio)
        self._sleep(self.cfg.tx.tail_ms)
        self.ptt.unkey()
        self._sleep(self.cfg.tx.cooldown_ms)

    def _drain_source(self) -> None:
        """Сбросить вход, накопленный за время передачи (актуально для микрофона)."""
        drain = getattr(self.source, "drain", None)
        if callable(drain):
            drain()

    def _sleep(self, ms: int) -> None:
        if self.realtime and ms > 0:
            time.sleep(ms / 1000.0)
