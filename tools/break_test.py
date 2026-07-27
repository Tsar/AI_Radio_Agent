#!/usr/bin/env python3
"""Тест PTT через break-состояние линии TX (USB-UART).

Периодически включает и выключает break_condition, пока не остановишь (Ctrl+C).
Смотри осциллографом на пин TX относительно GND:
  break OFF (idle / mark)  -> высокий уровень (~3.3 или 5 В)
  break ON  (space)        -> низкий уровень (~0 В)

Запуск:
  python3 break_test.py --port /dev/ttyUSB0 --on 0.5 --off 0.5
"""
import argparse
import sys
import time

import serial


def main() -> int:
    p = argparse.ArgumentParser(
        description="Мигаем break на линии TX (для проверки осциллографом)."
    )
    p.add_argument("--port", default="/dev/ttyUSB0",
                   help="serial-порт (по умолчанию /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=9600,
                   help="скорость (не важна, break данные не шлёт)")
    p.add_argument("--on", type=float, default=0.5,
                   help="сколько секунд держать break включённым")
    p.add_argument("--off", type=float, default=0.5,
                   help="сколько секунд держать break выключенным")
    args = p.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud)
    except serial.SerialException as e:
        print(f"Не удалось открыть порт {args.port}: {e}", file=sys.stderr)
        return 1

    # стартуем из безопасного состояния (break снят)
    ser.break_condition = False
    print(f"Порт {args.port} открыт. TX мигает break: "
          f"{args.on}s ON / {args.off}s OFF. Ctrl+C — стоп.")

    try:
        while True:
            ser.break_condition = True
            print("break ON   (TX -> low)")
            time.sleep(args.on)
            ser.break_condition = False
            print("break OFF  (TX -> high)")
            time.sleep(args.off)
    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        ser.break_condition = False
        ser.close()
        print("break снят, порт закрыт.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
