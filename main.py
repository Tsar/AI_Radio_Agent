#!/usr/bin/env python3
"""AI Radio Agent — точка входа (CLI).

Режимы:
    calibrate     — подобрать порог VAD по аудиофайлу или с живого микрофона
    run           — репитер: на файле (→ out.wav) или живьём (--live);
                    ответчик выбирается флагом --responder (parrot | llm)
    bench         — прогнать запись через STT→LLM→TTS и показать задержку по звеньям
    trigger-test  — проверить срабатывание позывного на тексте (без моделей)
    devices       — список аудиоустройств (для выбора --in-device/--out-device)

Примеры:
    python3 main.py calibrate --in-file запись.mp3
    python3 main.py run --in-file запись.mp3 --out-file out.wav
    python3 main.py bench --in-file запись.mp3
    python3 main.py trigger-test "феечка как слышно" "привет всем"
    python3 main.py run --live --responder llm --ptt txdbreak --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import sys

from ai_radio.calibrate import calibrate_file, calibrate_live, print_report
from ai_radio.config import (DEFAULT_PROFILE, PROFILES, REPLY_LENGTHS, Config, apply_profile,
                             apply_reply_length)
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
    if getattr(args, "max_utterance_ms", None) is not None:
        cfg.vad.max_utterance_ms = args.max_utterance_ms


def _apply_ai(cfg: Config, args: argparse.Namespace) -> None:
    """Профиль и точечные переопределения стека этапа 2."""
    # Профиль применяем всегда: без флага берём DEFAULT_PROFILE, а не дефолт
    # SttConfig.model. Точечные --stt-model/--stt-device ниже всё равно перекрывают.
    apply_profile(cfg, getattr(args, "profile", None) or DEFAULT_PROFILE)
    if getattr(args, "reply_length", None):
        apply_reply_length(cfg, args.reply_length)
    if getattr(args, "stt_model", None):
        cfg.stt.model = args.stt_model
    if getattr(args, "stt_device", None):
        cfg.stt.device = args.stt_device
    if getattr(args, "llm_url", None):
        cfg.llm.base_url = args.llm_url
    if getattr(args, "voice", None):
        cfg.tts.voice = args.voice
    if getattr(args, "rvc", False):
        cfg.rvc.enabled = True
    if getattr(args, "rvc_url", None):
        cfg.rvc.base_url = args.rvc_url
    if getattr(args, "rvc_voice", None):
        cfg.rvc.voice = args.rvc_voice
    if getattr(args, "rvc_formant", None) is not None:
        cfg.rvc.formant_shift = args.rvc_formant
    if getattr(args, "callsign", None):
        # свой позывной — свои варианты написания: дефолтные подобраны под «Феечку»
        cfg.dialog.callsign = args.callsign
        cfg.dialog.callsign_variants = [args.callsign]


def _make_responder(cfg: Config, args: argparse.Namespace):
    if getattr(args, "responder", "parrot") == "llm":
        from ai_radio.responder import build_llm_responder
        return build_llm_responder(cfg)
    return ParrotResponder()


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
    _apply_ai(cfg, args)

    source = FileSource(args.in_file, sample_rate=cfg.audio.sample_rate,
                        frame_samples=cfg.audio.frame_samples, realtime=args.realtime)
    sink = WavFileSink(args.out_file, cfg.audio.sample_rate) if args.out_file else NullSink()
    ptt = make_ptt("dummy")
    repeater = Repeater(cfg, EnergyVad(cfg.vad.threshold), _make_responder(cfg, args),
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
    _apply_ai(cfg, args)

    source = MicSource(sample_rate=cfg.audio.sample_rate, frame_samples=cfg.audio.frame_samples,
                       device=args.in_device, device_rate=args.device_rate)
    sink = SpeakerSink(sample_rate=cfg.audio.sample_rate, device=args.out_device,
                       device_rate=args.device_rate)
    ptt = make_ptt(args.ptt, port=args.port, invert=not args.no_invert)
    repeater = Repeater(cfg, EnergyVad(cfg.vad.threshold), _make_responder(cfg, args),
                        ptt, sink, realtime=True)

    print(f"Живой режим. Ответчик={args.responder}, PTT={args.ptt}. Ctrl+C — стоп.")
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


def cmd_bench(args: argparse.Namespace) -> int:
    from ai_radio.bench import run_bench
    if not args.in_file:
        print("нужен --in-file с записью из эфира", file=sys.stderr)
        return 2
    cfg = Config()
    _apply_common(cfg, args)
    _apply_ai(cfg, args)
    return run_bench(cfg, args.in_file, budget_s=args.budget, realtime=args.realtime)


def cmd_trigger_test(args: argparse.Namespace) -> int:
    from ai_radio.dialog import CallsignTrigger
    cfg = Config()
    _apply_ai(cfg, args)
    trigger = CallsignTrigger(cfg.dialog)
    print(f"позывной: {cfg.dialog.callsign} (порог {cfg.dialog.match_threshold:.2f}), "
          f"варианты: {', '.join(trigger.variants)}\n")
    for phrase in args.text:
        ok, score, token = trigger.match(phrase)
        mark = "СРАБОТАЛ" if ok else "молчим  "
        matched = f"«{token}»" if token else "—"
        print(f"  {mark}  score={score:.2f}  {matched:<12} {phrase!r}")
    return 0


def cmd_hallucination_test(args: argparse.Namespace) -> int:
    from ai_radio.hallucinations import verdict
    cfg = Config()
    _apply_ai(cfg, args)
    h = cfg.hallucination
    print(f"длительность фразы: {args.duration} с "
          f"(минимум {h.min_utterance_ms / 1000:.2f} с)\n")
    for phrase in args.text:
        if args.duration and args.duration < h.min_utterance_ms / 1000.0:
            reason = "короче минимума — в STT такое вообще не попадает"
        else:
            reason = verdict(phrase, args.duration,
                             initial_prompt=cfg.stt.initial_prompt,
                             min_chars_per_s=h.min_chars_per_s,
                             max_chars_per_s=h.max_chars_per_s,
                             long_dur_s=h.long_dur_s,
                             long_min_chars=h.long_min_chars)
        mark = "ОТСЕЯНА" if reason else "пропущена"
        print(f"  {mark}  {phrase!r}\n            {reason or 'похоже на живую речь'}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    import sounddevice as sd
    print(sd.query_devices())
    return 0


def _add_ai_args(p: argparse.ArgumentParser) -> None:
    # Подсказку собираем из самих PROFILES — переименование профилей её не рассинхронит
    profiles = " | ".join(f"{n} ({p['model']} на {p['device']})"
                          for n, p in sorted(PROFILES.items()))
    p.add_argument("--profile", choices=sorted(PROFILES),
                   help=f"профиль STT: {profiles}. По умолчанию {DEFAULT_PROFILE}")
    lengths = ", ".join(f"{n} ({p.air_s})" for n, p in REPLY_LENGTHS.items())
    p.add_argument("--reply-length", choices=list(REPLY_LENGTHS),
                   help=f"длина ответа в эфире: {lengths}. По умолчанию short")
    p.add_argument("--stt-model", help="модель faster-whisper (переопределяет профиль)")
    p.add_argument("--stt-device", choices=["cuda", "cpu"], help="где считать STT")
    p.add_argument("--llm-url", help="адрес llama-server (по умолчанию http://127.0.0.1:8080)")
    p.add_argument("--voice", help="путь к голосу Piper (.onnx)")
    p.add_argument("--callsign", help="позывной агента (по умолчанию Феечка)")
    p.add_argument("--rvc", action="store_true",
                   help="переозвучивать ответ через RVC (нужен infer-http-service.py)")
    p.add_argument("--rvc-voice", help="целевой голос RVC (по умолчанию voicevox_speaker_43)")
    p.add_argument("--rvc-url", help="адрес RVC-сервиса (по умолчанию http://127.0.0.1:8081)")
    p.add_argument("--rvc-formant", type=float,
                   help="сдвиг формант (по умолчанию пресет сервиса)")


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

    r = sub.add_parser("run", help="репитер: файл (→ wav) или живьём (--live)")
    r.add_argument("--live", action="store_true", help="живой режим: микрофон/динамик + PTT")
    r.add_argument("--responder", default="parrot", choices=["parrot", "llm"],
                   help="кто формирует ответ (по умолчанию parrot — этап 1)")
    r.add_argument("--in-file", help="входной аудиофайл (файловый режим)")
    r.add_argument("--out-file", help="куда записать переданное WAV (файловый режим)")
    r.add_argument("--threshold", type=float, help="порог VAD (переопределить дефолт)")
    r.add_argument("--hangtime-ms", type=int, help="тишины до конца передачи (склейка пауз)")
    r.add_argument("--max-utterance-ms", type=int, help="предел буфера приёма, мс (по умолчанию 60000)")
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
    _add_ai_args(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", help="замер задержки STT/LLM/TTS на записи из эфира")
    b.add_argument("--in-file", help="запись из эфира (mp3/wav/...)")
    b.add_argument("--budget", type=float, default=10.0, help="допустимая задержка ответа, с")
    b.add_argument("--threshold", type=float, help="порог VAD")
    b.add_argument("--hangtime-ms", type=int, help="тишины до конца передачи")
    b.add_argument("--max-utterance-ms", type=int, help="предел буфера приёма, мс")
    b.add_argument("--frame-ms", type=int, default=20, help="длина кадра, мс")
    b.add_argument("--realtime", action="store_true", help="прогонять запись в реальном темпе")
    _add_ai_args(b)
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("trigger-test", help="проверить срабатывание позывного (без моделей)")
    t.add_argument("text", nargs="+", help="фразы для проверки")
    _add_ai_args(t)
    t.set_defaults(func=cmd_trigger_test)

    hl = sub.add_parser("hallucination-test",
                        help="проверить, отсеется ли фраза как галлюцинация STT")
    hl.add_argument("text", nargs="+", help="распознанные фразы для проверки")
    hl.add_argument("--duration", type=float, default=4.0,
                    help="длительность фразы в секундах (часть правил зависит от неё)")
    _add_ai_args(hl)
    hl.set_defaults(func=cmd_hallucination_test)

    d = sub.add_parser("devices", help="список аудиоустройств")
    d.set_defaults(func=cmd_devices)

    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        # отсутствующая зависимость или сломанный вход — сообщение, а не стек
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
