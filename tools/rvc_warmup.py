#!/usr/bin/env python3
"""Прогрев RVC-сервиса сразу после старта.

Первая конвертация после запуска стоит примерно на 7 секунд дороже остальных:
на ней инициализируется CUDA-контекст и подбираются свёрточные алгоритмы. Замер
на Quadro P2000 (fp32, f0_method=pm):

    #1  2.4 с аудио -> 8.90 с   RTF 3.77   <- холодный
    #2  2.7 с аудио -> 1.92 с   RTF 0.70
    #5  7.4 с аудио -> 2.86 с   RTF 0.38

В эфире этот штраф достаётся первому же вызвавшему: конвейер укладывается в бюджет
10 с только прогретым. Поэтому греем при старте сервиса, а не за счёт корреспондента.

Фразы разной длины: короткие и длинные идут разными путями внутри пайплайна, и
одной мало. Синтезируем их тем же Piper'ом и шлём тем же клиентом, что и в бою, —
прогревается ровно тот код, который потом работает.

Запускается из юнита ai-radio-rvc как ExecStartPost. Штатно молчит про мелочи и
никогда не роняет сервис: не прогрелись — просто первая боевая фраза будет долгой.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_radio.config import Config                       # noqa: E402
from ai_radio.engines.tts_piper import PiperTts          # noqa: E402
from ai_radio.engines.tts_rvc import RvcVoice            # noqa: E402

# Короткая, средняя и длинная — перекрывают диапазон реальных ответов (short = ≤2
# предложения, medium = ≤4). Содержание неважно, важна длительность.
PHRASES = [
    "Приём.",
    "Понял вас, продолжайте передачу.",
    "Связь устойчивая, помех не наблюдаю, качество сигнала хорошее. "
    "Продолжаю работу на этой частоте, до связи.",
]


def wait_health(url: str, timeout_s: float) -> bool:
    """Сервис поднимается ~15 с (грузит hubert и net_g), а ExecStartPost стартует
    сразу после fork'а — без ожидания мы бы стучались в закрытый порт."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="прогрев RVC-сервиса")
    p.add_argument("--url", default=None, help="адрес сервиса (по умолчанию из конфига)")
    p.add_argument("--wait", type=float, default=120.0, help="сколько ждать /health, с")
    args = p.parse_args()

    cfg = Config()
    cfg.rvc.enabled = True
    if args.url:
        cfg.rvc.base_url = args.url

    if not wait_health(cfg.rvc.base_url, args.wait):
        print(f"[warmup] сервис не ответил за {args.wait:.0f} с — пропускаем прогрев")
        return 1

    tts = RvcVoice(PiperTts(cfg.tts, cfg.audio.sample_rate), cfg.rvc, cfg.audio.sample_rate)
    started = time.perf_counter()
    for i, phrase in enumerate(PHRASES, 1):
        t0 = time.perf_counter()
        out = tts.synth(phrase)
        dt = time.perf_counter() - t0
        # RvcVoice при отказе сервиса молча отдаёт голос Piper — на прогреве это
        # означает, что греть нечего, и дальше идти незачем
        if not out:
            print(f"[warmup] фраза {i}: пусто, прекращаем")
            return 1
        print(f"[warmup] фраза {i}/{len(PHRASES)}: {dt:.2f} с")
    print(f"[warmup] готово за {time.perf_counter() - started:.1f} с")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                      # прогрев не должен ронять сервис
        print(f"[warmup] не удался: {exc}")
        sys.exit(1)
