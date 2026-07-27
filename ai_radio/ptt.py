"""Управление PTT (нажатие «передача»).

Backend'ы:
  DummyPtt    — только пишет в лог KEY/UNKEY (для симуляции без железа).
  TxdBreakPtt — держит линию TX USB-UART в состоянии break на время передачи
                (наш проверенный способ; сигнал инвертирован, см. флаг invert).

Общий интерфейс: key(), unkey(), close(). Поддерживает контекст-менеджер.
"""
from __future__ import annotations

from typing import Protocol


class Ptt(Protocol):
    def key(self) -> None: ...
    def unkey(self) -> None: ...
    def close(self) -> None: ...


class DummyPtt:
    """Ничего не коммутирует — только логирует. Для прогонов на файле."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.keyed = False

    def key(self) -> None:
        self.keyed = True
        if self.verbose:
            print("[PTT] KEY  (передача)")

    def unkey(self) -> None:
        self.keyed = False
        if self.verbose:
            print("[PTT] UNKEY (приём)")

    def close(self) -> None:
        if self.keyed:
            self.unkey()

    def __enter__(self) -> "DummyPtt":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TxdBreakPtt:
    """PTT через break-состояние линии TX USB-UART.

    invert=True: рация переходит в передачу, когда break СНЯТ (инвертированный
    сигнал, подтверждённый осциллографом). invert=False — наоборот.
    Безопасное состояние (приём) выставляется при открытии и при close().
    """

    def __init__(self, port: str = "/dev/ttyUSB0", invert: bool = True,
                 baudrate: int = 9600) -> None:
        import serial  # локальный импорт: нужен только для этого backend'а
        self.invert = invert
        self._ser = serial.Serial(port, baudrate)
        self.keyed = False
        self._set_break(False)  # старт из состояния «приём»

    def _set_break(self, active: bool) -> None:
        # active=True — хотим «передача». При инверсии break снимается для передачи.
        self._ser.break_condition = (not active) if self.invert else active

    def key(self) -> None:
        self._set_break(True)
        self.keyed = True

    def unkey(self) -> None:
        self._set_break(False)
        self.keyed = False

    def close(self) -> None:
        try:
            self._set_break(False)  # гарантированно вернуть в приём
        finally:
            self._ser.close()

    def __enter__(self) -> "TxdBreakPtt":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_ptt(backend: str, port: str = "/dev/ttyUSB0", invert: bool = True) -> Ptt:
    if backend == "dummy":
        return DummyPtt()
    if backend == "txdbreak":
        return TxdBreakPtt(port=port, invert=invert)
    raise ValueError(f"неизвестный PTT backend: {backend!r}")
