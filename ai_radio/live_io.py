"""Живой звук: захват с line-in и воспроизведение в line-out через sounddevice.

Интерфейсы совпадают с файловыми (FileSource.frames() / *Sink.play()), поэтому
репитер не различает файл и реальную звуковую карту.

numpy и sounddevice импортируются ЛОКАЛЬНО внутри методов — чтобы файловая
калибровка (stdlib + ffmpeg) работала на машине без этих пакетов.

Частоты: работаем на 16 кГц. Если звуковая карта не умеет 16 кГц, укажите её
родную частоту (device_rate, напр. 48000) — включится целочисленная
ре-дискретизация (децимация на входе, линейная интерполяция на выходе).
"""
from __future__ import annotations

from typing import Iterator, List, Optional


def _resolve_device(dev: "str | int | None") -> "str | int | None":
    if dev is None:
        return None
    try:
        return int(dev)
    except (TypeError, ValueError):
        return dev


class MicSource:
    """Захват с line-in. Отдаёт кадры list[float] в [-1,1] длиной frame_samples."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 320,
                 device: "str | int | None" = None, device_rate: Optional[int] = None) -> None:
        self.work_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = _resolve_device(device)

        cap_rate = device_rate or sample_rate
        if cap_rate % sample_rate != 0:
            raise ValueError(
                f"device_rate={cap_rate} должен быть кратен рабочей частоте {sample_rate}")
        self.decimate = cap_rate // sample_rate
        self.cap_rate = cap_rate
        self.cap_frame = frame_samples * self.decimate
        self._stream = None
        self._reopen_on_resume = False   # см. resume(): бэкенд не умеет start() после stop()

    def _ensure(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.cap_rate, channels=1, dtype="float32",
            blocksize=self.cap_frame, device=self.device)
        self._stream.start()

    def frames(self) -> Iterator[List[float]]:
        import numpy as np
        self._ensure()
        assert self._stream is not None
        while True:
            data, _overflowed = self._stream.read(self.cap_frame)
            mono = data[:, 0]
            if self.decimate > 1:
                n = (len(mono) // self.decimate) * self.decimate
                mono = mono[:n].reshape(-1, self.decimate).mean(axis=1)
            yield mono.tolist()

    def pause(self) -> None:
        """Остановить захват на время передачи. При resume PortAudio сбрасывает
        буфер, поэтому сказанное во время воспроизведения не накопится и не будет
        воспроизведено повторно. Заодно нет input-overrun на чистой ALSA."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def resume(self) -> None:
        if self._stream is None or self._stream.active:
            return
        if self._reopen_on_resume:
            self._reopen()
            return
        try:
            self._stream.start()
        except Exception as exc:        # noqa: BLE001 — sounddevice.PortAudioError и что угодно ещё
            # Хост-API PulseAudio в PortAudio 19.7 (устройства «alsa_input.… PulseAudio»
            # на любом десктопе с PipeWire) после stop() отвечает на start()
            # «PortAudio not initialized», и агент падал на первой же фразе. Открыть
            # поток заново стоит те же десятки миллисекунд, а буфер после этого пуст —
            # ровно то, ради чего пауза и делалась. Дальше сразу переоткрываем, не
            # пробуя start(): об этом бэкенде одной строки в журнале достаточно.
            print(f"[MIC] поток не возобновился ({exc}) — дальше открываю заново на каждой паузе")
            self._reopen_on_resume = True
            self._reopen()

    def _reopen(self) -> None:
        try:
            self.close()
        except Exception:               # noqa: BLE001 — поток уже неисправен, закрывать нечего
            self._stream = None
        self._ensure()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "MicSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SpeakerSink:
    """Воспроизведение ответа в line-out (блокирующе, на время передачи)."""

    def __init__(self, sample_rate: int = 16000, device: "str | int | None" = None,
                 device_rate: Optional[int] = None) -> None:
        self.work_rate = sample_rate
        self.play_rate = device_rate or sample_rate
        self.device = _resolve_device(device)
        self.total_samples = 0

    def play(self, samples: List[float]) -> None:
        import numpy as np
        import sounddevice as sd
        from .audio_io import resample_linear
        self.total_samples += len(samples)
        if self.play_rate != self.work_rate:
            samples = resample_linear(samples, self.work_rate, self.play_rate)
        audio = np.asarray(samples, dtype=np.float32)
        # latency="high" сглаживает output-underrun на чистой ALSA (ценой большей задержки)
        sd.play(audio, samplerate=self.play_rate, device=self.device, latency="high")
        sd.wait()

    def close(self) -> None:
        pass

    def __enter__(self) -> "SpeakerSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
