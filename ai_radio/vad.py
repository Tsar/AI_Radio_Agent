"""Energy VAD: детекция голоса по громкости (RMS). Без внешних зависимостей.

Логика hangtime/preroll/буферизации живёт в репитере (state machine); здесь —
только «сырое» решение speech/silence по кадру и вспомогательные метрики.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

FLOOR = 1e-6  # нижняя отсечка, чтобы не брать log10(0)


def frame_rms(frame: Sequence[float]) -> float:
    """RMS кадра. Значения кадра — в [-1, 1], результат — в [0, 1]."""
    n = len(frame)
    if n == 0:
        return 0.0
    return math.sqrt(sum(s * s for s in frame) / n)


def rms_to_dbfs(rms: float) -> float:
    """RMS [0..1] → dBFS (0 dB = полная шкала)."""
    return 20.0 * math.log10(max(rms, FLOOR))


def dbfs_to_rms(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


@dataclass
class EnergyVad:
    threshold: float  # порог RMS в долях полной шкалы

    def is_speech(self, frame: Sequence[float]) -> bool:
        return frame_rms(frame) > self.threshold
