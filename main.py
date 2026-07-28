#!/usr/bin/env python3
"""AI Radio Agent — точка входа (CLI).

Режимы:
    calibrate  — подобрать порог VAD по аудиофайлу
    run        — репитер: на файле (parrot → out.wav) или живьём (--live)
    devices    — список аудиоустройств (для выбора --in-device/--out-device)

Примеры:
    python3 main.py calibrate --in-file запись.mp3
    python3 main.py run --in-file запись.mp3 --out-file out.wav
    python3 main.py devices
    python3 main.py run --live --ptt txdbreak --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import sys

from ai_radio.calibrate import calibrate_file, calibrate_live, print_report
from ai_radio.config import Config
from ai_radio.ptt import make_ptt
from ai_radio.repeater import Repeater
from ai_radio.responder import ParrotResponder
from ai_radio.vad import EnergyVad


def _apply_common(cfg: Config, args: argparse.Namespace) -> None:
    cfg.audio.frame_ms = args.frame_ms
    if getattr(args, "threshold", None) is not None:
        cfg.vad.threshold = args.threshold
    if getattr(args, "hangtime_ms", None) is not None:
        cfg.vad.hangtime_ms = args.hangtime_ms


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.audio.frame_ms = args.frame_ms
    if args.live:
        res = calibrate_live(cfg, device=args.in_device,
                             device_rate=args.device_rate, seconds=args.seconds)
    else:
        if not args.in_file:
            print("нужен --in-file (или --live для микрофона)", file=sys.stderr)
            return 2
        res = calibrate_file(args.in_file, cfg)
    print_report(res)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.live:
        return _run_live(args)
    return _run_file(args)


def _run_file(args: argparse.Namespace) -> int:
    from ai_radio.audio_io import FileSource, NullSink, WavFileSink
    if not args.in_file:
        print("для файлового режима нужен --in-file (или используйте --live)", file=sys.stderr)
        return 2
    cfg = Config()
    _apply_common(cfg, args)

    source = FileSource(args.in_file, sample_rate=cfg.audio.sample_rate,
                        frame_samples=cfg.audio.frame_samples, realtime=args.realtime)
    sink = WavFileSink(args.out_file, cfg.audio.sample_rate) if args.out_file else NullSink()
    ptt = make_ptt("dummy")
    repeater = Repeater(cfg, EnergyVad(cfg.vad.threshold), ParrotResponder(),
                        ptt, sink, realtime=args.realtime)
    try:
        repeater.run(source)
    finally:
        sink.close()
        ptt.close()

    print(f"\nитого передач: {repeater.n_transmissions}")
    if args.out_file:
        secs = sink.total_samples / cfg.audio.sample_rate
        print(f"записано в {args.out_file}: {secs:.2f} с аудио")
    return 0


def _run_live(args: argparse.Namespace) -> int:
    from ai_radio.live_io import MicSource, SpeakerSink
    cfg = Config()
    _apply_common(cfg, args)

    source = MicSource(sample_rate=cfg.audio.sample_rate, frame_samples=cfg.audio.frame_samples,
                       device=args.in_device, device_rate=args.device_rate)
    sink = SpeakerSink(sample_rate=cfg.audio.sample_rate, device=args.out_device,
                       device_rate=args.device_rate)
    ptt = make_ptt(args.ptt, port=args.port, invert=not args.no_invert)
    repeater = Repeater(cfg, EnergyVad(cfg.vad.threshold), ParrotResponder(),
                        ptt, sink, realtime=True)

    print(f"Живой режим. PTT={args.ptt}. Ctrl+C — стоп.")
    try:
        repeater.run(source)
    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        source.close()
        sink.close()
        ptt.close()
    print(f"итого передач: {repeater.n_transmissions}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    import sounddevice as sd
    print(sd.query_devices())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai_radio", description="AI Radio Agent")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="подобрать порог VAD по файлу или с микрофона")
    c.add_argument("--in-file", help="путь к аудиофайлу (mp3/wav/...)")
    c.add_argument("--live", action="store_true", help="калибровать с живого микрофона")
    c.add_argument("--seconds", type=float, default=12.0, help="сколько слушать в live, с")
    c.add_argument("--in-device", help="устройство записи для live (индекс или часть имени)")
    c.add_argument("--device-rate", type=int, help="родная частота карты, если не умеет 16000")
    c.add_argument("--frame-ms", type=int, default=20, help="длина кадра, мс (по умолчанию 20)")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="репитер: файл (parrot → wav) или живьём (--live)")
    r.add_argument("--live", action="store_true", help="живой режим: микрофон/динамик + PTT")
    r.add_argument("--in-file", help="входной аудиофайл (файловый режим)")
    r.add_argument("--out-file", help="куда записать переданное WAV (файловый режим)")
    r.add_argument("--threshold", type=float, help="порог VAD (переопределить дефолт)")
    r.add_argument("--hangtime-ms", type=int, help="тишины до конца передачи (склейка пауз)")
    r.add_argument("--frame-ms", type=int, default=20, help="длина кадра, мс")
    r.add_argument("--realtime", action="store_true",
                   help="файловый режим в реальном темпе (в live всегда включено)")
    # живой режим:
    r.add_argument("--in-device", help="устройство записи (индекс или часть имени)")
    r.add_argument("--out-device", help="устройство воспроизведения (индекс или часть имени)")
    r.add_argument("--device-rate", type=int,
                   help="родная частота карты, если не умеет 16000 (напр. 48000)")
    r.add_argument("--ptt", default="dummy", choices=["dummy", "txdbreak"],
                   help="backend PTT (по умолчанию dummy — безопасно)")
    r.add_argument("--port", default="/dev/ttyUSB0", help="serial-порт для txdbreak")
    r.add_argument("--no-invert", action="store_true",
                   help="отключить инверсию PTT (по умолчанию инвертировано)")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("devices", help="список аудиоустройств")
    d.set_defaults(func=cmd_devices)

    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
