"""Источники аудио. На этапе 1 — FileSource (декод через ffmpeg), на stdlib.

Кадр — это list[float] со значениями в диапазоне [-1.0, 1.0], длиной frame_samples.
Такой же тип позже будет отдавать MicSource (там уже с numpy/sounddevice),
поэтому VAD и репитер не зависят от источника.
"""
from __future__ import annotations

import array
import shutil
import subprocess
import time
from typing import Iterator, List, Optional, Sequence

from .vad import dbfs_to_rms


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg не найден в PATH — установите пакет ffmpeg")
    return exe


class FileSource:
    """Декодирует аудиофайл (mp3/wav/...) через ffmpeg в 16 кГц моно и отдаёт кадры.

    ffmpeg берёт на себя декодирование любых форматов и ресемплинг/микс в моно,
    поэтому мы не зависим от версии libsndfile (mp3 на старых системах она не читает).
    """

    def __init__(
        self,
        path: str,
        sample_rate: int = 16000,
        frame_samples: int = 320,
        realtime: bool = False,
    ) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.realtime = realtime  # True — имитировать реальный темп эфира (спать между кадрами)
        self._proc: Optional[subprocess.Popen] = None

    def frames(self) -> Iterator[List[float]]:
        cmd = [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-i", self.path,
            "-ac", "1", "-ar", str(self.sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._proc = proc
        bytes_per_frame = self.frame_samples * 2  # s16 = 2 байта на отсчёт
        frame_dur = self.frame_samples / self.sample_rate
        produced_any = False
        assert proc.stdout is not None and proc.stderr is not None
        try:
            while True:
                buf = proc.stdout.read(bytes_per_frame)
                if not buf:
                    break
                produced_any = True
                if len(buf) < bytes_per_frame:
                    buf = buf + b"\x00" * (bytes_per_frame - len(buf))  # добить хвост нулями
                yield pcm16_to_floats(buf)
                if self.realtime:
                    time.sleep(frame_dur)
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
            proc.terminate()
            err = b""
            try:
                err = proc.stderr.read()
            except OSError:
                pass
            finally:
                proc.stderr.close()
            proc.wait()
            if not produced_any:
                msg = err.decode(errors="replace").strip() or "неизвестная ошибка"
                raise RuntimeError(f"ffmpeg не выдал аудио из {self.path!r}: {msg}")


def resample_linear(samples: Sequence[float], src_rate: int, dst_rate: int) -> List[float]:
    """Линейная ре-дискретизация. Одна реализация на всех: выход TTS (22050 → 16000)
    и вывод в карту, не умеющую рабочую частоту (16000 → 48000).

    numpy используется, если доступен (быстрее), иначе честный цикл — модуль обязан
    оставаться импортируемым на голом stdlib, как и весь файловый режим.
    """
    n_in = len(samples)
    if src_rate == dst_rate or n_in == 0:
        return list(samples)
    n_out = max(1, int(round(n_in * dst_rate / src_rate)))
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, np.asarray(samples, dtype=np.float64)).tolist()

    step = n_in / n_out
    out: List[float] = []
    for i in range(n_out):
        pos = i * step
        j = int(pos)
        a = samples[j]
        b = samples[j + 1] if j + 1 < n_in else a
        out.append(a + (b - a) * (pos - j))
    return out


def normalize_peak(samples: Sequence[float], target_dbfs: float = -3.0) -> List[float]:
    """Привести пик к target_dbfs. Уровень TTS иначе гуляет от фразы к фразе, а
    глубина модуляции рации прямо зависит от амплитуды на её микрофонном входе."""
    peak = max((abs(s) for s in samples), default=0.0)
    if peak <= 0.0:
        return list(samples)
    gain = dbfs_to_rms(target_dbfs) / peak      # dBFS → линейная амплитуда
    return [max(-1.0, min(1.0, s * gain)) for s in samples]


def _floats_to_pcm16(samples: List[float]) -> bytes:
    pcm = array.array("h", (max(-32768, min(32767, int(s * 32767))) for s in samples))
    return pcm.tobytes()


def pcm16_to_floats(data: bytes) -> List[float]:
    """PCM s16le → list[float] в [-1, 1] — общий формат кадра во всём проекте."""
    pcm = array.array("h")
    pcm.frombytes(data)
    return [s / 32768.0 for s in pcm]


class NullSink:
    """Приёмник-заглушка: считает длительность, ничего не пишет."""

    def __init__(self) -> None:
        self.total_samples = 0

    def play(self, samples: List[float]) -> None:
        self.total_samples += len(samples)

    def close(self) -> None:
        pass


class WavFileSink:
    """Пишет «переданное» аудио в WAV (16 бит, моно) — проверка симуляции без радио."""

    def __init__(self, path: str, sample_rate: int = 16000) -> None:
        import wave  # stdlib, локальный импорт
        self._wf = wave.open(path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(sample_rate)
        self.total_samples = 0

    def play(self, samples: List[float]) -> None:
        self._wf.writeframes(_floats_to_pcm16(samples))
        self.total_samples += len(samples)

    def close(self) -> None:
        self._wf.close()
