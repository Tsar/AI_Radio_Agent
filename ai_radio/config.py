"""Конфигурация конвейера. Простые dataclass'ы, без внешних зависимостей."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 16000      # рабочая частота дискретизации, Гц
    frame_ms: int = 20            # длина кадра, мс (10/20/30 — типовые для VAD)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class VadConfig:
    threshold: float = 0.004      # порог RMS [0..1]; ~ -48 dBFS по калибровке Optim-778
                                  # (шумовой пол ~-60 dBFS + 12 дБ). Калибруйте под свою установку.
    start_frames: int = 3         # кадров подряд выше порога, чтобы начать приём
    hangtime_ms: int = 1000       # тишины до конца передачи; >800 мс склеивает паузы
                                  # внутри одной реплики (проверено на записи Optim-778)
    preroll_ms: int = 300         # сколько удерживать до срабатывания (не резать атаку)
    max_utterance_ms: int = 60000 # предел буфера приёма; сверх — не буферизуем, но приём
                                  # закрываем всё равно по тишине (уходит первая минута)


@dataclass
class TxConfig:
    warmup_ms: int = 200          # пауза после нажатия PTT до подачи аудио (передатчик «поднимается»)
    tail_ms: int = 150            # пауза после аудио до отпускания PTT (не резать хвост)
    cooldown_ms: int = 300        # защитный интервал перед возвратом к приёму


@dataclass
class PttConfig:
    backend: str = "dummy"        # dummy | txdbreak
    port: str = "/dev/ttyUSB0"
    invert: bool = True           # у нашего USB-UART сигнал инвертирован


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    tx: TxConfig = field(default_factory=TxConfig)
    ptt: PttConfig = field(default_factory=PttConfig)
    input_device: "int | str | None" = None
    output_device: "int | str | None" = None
