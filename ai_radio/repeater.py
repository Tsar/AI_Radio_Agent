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
        self.max_utterance_samples = max(fs, cfg.vad.max_utterance_ms * cfg.audio.sample_rate // 1000)

        self.n_transmissions = 0
        self.n_failures = 0          # неудачных фраз подряд (см. _report_failure)

    def run(self, source) -> None:
        """source — объект с методом frames() -> Iterable[list[float]]."""
        self.source = source
        self._run_frames(source.frames())

    def _run_frames(self, frames: Iterable[List[float]]) -> None:
        state = "IDLE"
        speech_run = 0
        silence_run = 0
        capped = False               # буфер достиг лимита — дальше не копим
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
                        capped = False
                else:
                    speech_run = 0

            elif state == "RECEIVING":
                if not capped:
                    room = self.max_utterance_samples - len(utterance)
                    if room >= len(frame):
                        utterance.extend(frame)
                    else:
                        if room > 0:
                            utterance.extend(frame[:room])  # добить ровно до лимита
                        capped = True
                        limit_s = self.cfg.vad.max_utterance_ms / 1000.0
                        print(f"[RX] буфер достиг лимита {limit_s:.0f} с — "
                              f"дальше не буферизуем, ждём конца передачи")
                if speech:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= self.hangtime_frames:
                        # при cap хвостовой тишины в буфере нет — срезать нечего
                        self._end_utterance(utterance, 0 if capped else silence_run)
                        state = "IDLE"
                        speech_run = silence_run = 0
                        capped = False
                        utterance = []
                        preroll.clear()

        # поток кончился, а мы ещё принимали — до-передать
        if state == "RECEIVING":
            self._end_utterance(utterance, 0 if capped else silence_run)

    def _end_utterance(self, utterance: List[float], trailing_silence_frames: int) -> None:
        # срезать хвостовую тишину (hangtime), чтобы не гнать её в эфир
        trim = trailing_silence_frames * self.frame_samples
        if 0 < trim < len(utterance):
            utterance = utterance[:-trim]
        dur = len(utterance) / self.cfg.audio.sample_rate
        print(f"[RX] принята фраза: {dur:.2f} с")

        # Обрывок короче min_utterance_ms — щелчок PTT или скрип, а не речь. Гнать
        # его в Whisper незачем: на таких кусках он и выдаёт свои субтитры (245 раз
        # «Продолжение следует» на фразах короче 0.4 с в журнале прода), а секунда
        # работы STT тратится всерьёз.
        min_dur = self.cfg.hallucination.min_utterance_ms / 1000.0
        if self.cfg.hallucination.enabled and dur < min_dur:
            # Текста в журнале здесь не будет: Whisper мы для такого куска не зовём.
            # Чтобы увидеть, что он на них выдумывает, поставьте min_utterance_ms=0.
            print(f"[--] короче {min_dur:.2f} с — щелчок, а не речь: в STT не отдаём")
            return

        # Вход на паузе на всё время «думаем + передаём» (строгий half-duplex).
        # respond() у LLM работает секунды: незачитанный поток успел бы переполниться,
        # а после возврата мы бы прогнали через VAD звук, накопившийся за время раздумий.
        self._pause_source()
        try:
            response = self.responder.respond(utterance)
            self.n_failures = 0       # конвейер отработал — серия неудач кончилась
            if not response:
                print("[--] ответа нет, не передаём")
                return
            self._transmit(response)
        except Exception as exc:      # noqa: BLE001 — см. ниже, падать тут нельзя
            self._report_failure(exc)
        finally:
            self._resume_source()     # входной буфер сброшен рестартом стрима

    def _report_failure(self, exc: BaseException) -> None:
        """Одна неудачная фраза не должна ронять агента.

        Так было до 11.08.2026: `CUDA failed with error out of memory` из Whisper
        поднимался до main(), процесс выходил с кодом 1, systemd поднимал его заново —
        и снова, потому что память держал сосед (RVC после своего OOM не отдаёт пул).
        19 рестартов подряд, вылеченных перезагрузкой машины. Пропустить фразу и
        остаться в эфире — единственное осмысленное поведение: соседние сервисы
        могут прийти в себя сами, а агент к тому моменту должен быть жив.
        """
        self.n_failures += 1
        print(f"[ERR] фраза не обработана ({self.n_failures}-я подряд): "
              f"{type(exc).__name__}: {exc}")
        if "out of memory" in str(exc).lower():
            print("[ERR] это нехватка VRAM, а не сбой самой фразы. Кто держит память —"
                  " смотреть в nvidia-smi; чаще всего помогает "
                  "`systemctl --user restart ai-radio-rvc`")

    def _transmit(self, audio: List[float]) -> None:
        dur = len(audio) / self.cfg.audio.sample_rate
        self.n_transmissions += 1
        # Порядок строк в журнале повторяет порядок событий на линии: сначала
        # нажали PTT, выждали warmup, и только теперь пошёл звук. Раньше «[TX]»
        # печаталось до нажатия, и при разборе тракта по логу это путало.
        self.ptt.key()
        self._sleep(self.cfg.tx.warmup_ms)
        print(f"[TX] передача #{self.n_transmissions}: {dur:.2f} с")
        self.sink.play(audio)
        self._sleep(self.cfg.tx.tail_ms)
        self.ptt.unkey()
        self._sleep(self.cfg.tx.cooldown_ms)

    def _pause_source(self) -> None:
        fn = getattr(self.source, "pause", None)
        if callable(fn):
            fn()

    def _resume_source(self) -> None:
        fn = getattr(self.source, "resume", None)
        if callable(fn):
            fn()

    def _sleep(self, ms: int) -> None:
        if self.realtime and ms > 0:
            time.sleep(ms / 1000.0)
