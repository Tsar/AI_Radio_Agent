"""Калибровка порога VAD по аудио — из файла или вживую с микрофона.

Прогоняет аудио, собирает RMS по кадрам, оценивает шумовой пол (фон / закрытый
шумоподавитель) и уровень сигнала, рекомендует порог и рисует ASCII-гистограмму.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .audio_io import FileSource
from .config import Config
from .vad import dbfs_to_rms, frame_rms, rms_to_dbfs


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


@dataclass
class CalibrationResult:
    n_frames: int
    frame_ms: int
    duration_s: float
    floor_db: float          # шумовой пол (тишина / фон)
    peak_db: float           # уровень сигнала
    recommended_db: float    # рекомендованный порог, dBFS
    recommended_rms: float   # он же в долях полной шкалы
    speech_seconds: float    # сколько «речи» при этом пороге
    db_values: List[float]   # RMS всех кадров в dBFS (для гистограммы)


def _analyze(db_values: List[float], cfg: Config) -> CalibrationResult:
    if not db_values:
        raise RuntimeError("нет кадров для анализа")
    ordered = sorted(db_values)
    floor_db = percentile(ordered, 15)   # нижняя часть — фон
    peak_db = percentile(ordered, 92)    # верхняя часть — сигнал

    # Порог: на 12 дБ выше фона, но не выше середины между фоном и сигналом.
    midpoint_db = (floor_db + peak_db) / 2.0
    recommended_db = min(floor_db + 12.0, midpoint_db)
    recommended_rms = dbfs_to_rms(recommended_db)

    n_speech = sum(1 for d in db_values if d > recommended_db)
    speech_seconds = n_speech * cfg.audio.frame_ms / 1000.0

    return CalibrationResult(
        n_frames=len(db_values),
        frame_ms=cfg.audio.frame_ms,
        duration_s=len(db_values) * cfg.audio.frame_ms / 1000.0,
        floor_db=floor_db,
        peak_db=peak_db,
        recommended_db=recommended_db,
        recommended_rms=recommended_rms,
        speech_seconds=speech_seconds,
        db_values=db_values,
    )


def calibrate_file(path: str, cfg: Config) -> CalibrationResult:
    src = FileSource(path, sample_rate=cfg.audio.sample_rate,
                     frame_samples=cfg.audio.frame_samples, realtime=False)
    db_values = [rms_to_dbfs(frame_rms(f)) for f in src.frames()]
    return _analyze(db_values, cfg)


def calibrate_live(cfg: Config, device: "str | int | None" = None,
                   device_rate: Optional[int] = None, seconds: float = 12.0) -> CalibrationResult:
    from .live_io import MicSource  # локально: тянет numpy/sounddevice
    src = MicSource(sample_rate=cfg.audio.sample_rate,
                    frame_samples=cfg.audio.frame_samples,
                    device=device, device_rate=device_rate)
    need = int(seconds * 1000 / cfg.audio.frame_ms)
    per_half_sec = max(1, 500 // cfg.audio.frame_ms)
    db_values: List[float] = []
    print(f"Слушаю {seconds:.0f} с. Совет: первые ~4 с помолчите (фон), потом говорите.")
    try:
        for frame in src.frames():
            db_values.append(rms_to_dbfs(frame_rms(frame)))
            if len(db_values) % per_half_sec == 0:
                t = len(db_values) * cfg.audio.frame_ms / 1000.0
                print(f"  {t:4.1f}s  уровень {db_values[-1]:6.1f} dBFS")
            if len(db_values) >= need:
                break
    finally:
        src.close()
    return _analyze(db_values, cfg)


def _histogram(db_values: List[float], lo: float = -90.0, hi: float = 0.0,
               step: float = 5.0, width: int = 50) -> str:
    n_bins = int((hi - lo) / step)
    counts = [0] * n_bins
    for d in db_values:
        idx = int((d - lo) / step)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i in range(n_bins - 1, -1, -1):  # сверху громкие, снизу тихие
        band_lo = lo + i * step
        band_hi = band_lo + step
        bar = "#" * round(counts[i] / peak * width)
        lines.append(f"{band_lo:6.0f}..{band_hi:<4.0f} dB | {bar} {counts[i]}")
    return "\n".join(lines)


def print_report(res: CalibrationResult) -> None:
    print("=" * 60)
    print("КАЛИБРОВКА VAD")
    print("=" * 60)
    print(f"кадров:            {res.n_frames} × {res.frame_ms} мс  "
          f"(~{res.duration_s:.1f} с)")
    print(f"шумовой пол:       {res.floor_db:6.1f} dBFS  "
          f"(RMS {dbfs_to_rms(res.floor_db):.5f})")
    print(f"уровень сигнала:   {res.peak_db:6.1f} dBFS  "
          f"(RMS {dbfs_to_rms(res.peak_db):.5f})")
    print(f"разнос сигнал/шум: {res.peak_db - res.floor_db:6.1f} dB")
    print("-" * 60)
    print(f">>> РЕКОМЕНДОВАННЫЙ ПОРОГ: {res.recommended_db:.1f} dBFS  "
          f"(threshold = {res.recommended_rms:.5f})")
    print(f"    при нём «речью» считается ~{res.speech_seconds:.1f} с из "
          f"{res.duration_s:.1f} с")
    print("-" * 60)
    print("распределение громкости по кадрам:")
    print(_histogram(res.db_values))
    print("=" * 60)
