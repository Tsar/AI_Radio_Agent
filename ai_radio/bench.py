"""Замер задержки по звеньям на реальной записи из эфира.

Главный инструмент переноса на прод: dev-машина (Ryzen + RTX 2070) и прод
(i7-2600 без AVX2 + Pascal) расходятся в разы — особенно на Piper, который считает
на CPU. Качество моделей проверяется где угодно, а вот время знает только то железо,
на котором bench запущен.

Прогон идёт через настоящий конвейер (FileSource → VAD → Repeater → LLMResponder),
поэтому меряется ровно то, что будет в эфире, включая фразы, на которые агент молчит.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .audio_io import FileSource, NullSink
from .config import Config
from .ptt import DummyPtt
from .repeater import Repeater
from .responder import LLMResponder
from .vad import EnergyVad


class BenchResponder:
    """Обёртка над LLMResponder: считает время каждой фразы, не меняя поведения."""

    def __init__(self, inner: LLMResponder, sample_rate: int) -> None:
        self.inner = inner
        self.sample_rate = sample_rate
        self.records: List[Dict[str, float]] = []

    def respond(self, utterance: List[float]) -> Optional[List[float]]:
        t0 = time.monotonic()
        out = self.inner.respond(utterance)
        total = time.monotonic() - t0
        rec = dict(self.inner.timings)
        rec["rx_s"] = len(utterance) / self.sample_rate
        rec["total"] = total
        rec["tx_s"] = (len(out) / self.sample_rate) if out else 0.0
        rec["answered"] = 1.0 if out else 0.0
        self.records.append(rec)
        return out


def _fmt(value: Optional[float]) -> str:
    return f"{value:6.2f}" if value is not None else "     —"


def print_report(records: List[Dict[str, float]], budget_s: float) -> None:
    if not records:
        print("\nФраз не обнаружено — проверьте порог VAD (--threshold).")
        return

    print("\n  #   приём   STT     LLM     TTS    итого   эфир  ответ")
    print("  ─────────────────────────────────────────────────────────")
    for i, rec in enumerate(records, 1):
        print(f"  {i:<3}{_fmt(rec.get('rx_s'))}  {_fmt(rec.get('stt'))}  "
              f"{_fmt(rec.get('llm'))}  {_fmt(rec.get('tts'))}  "
              f"{_fmt(rec.get('total'))}  {_fmt(rec.get('tx_s'))}   "
              f"{'да' if rec.get('answered') else 'нет'}")

    answered = [r for r in records if r.get("answered")]
    print(f"\n  фраз: {len(records)}, отвечено: {len(answered)}")
    if not answered:
        print("  Ни одного ответа: либо не было позывного, либо не поднят llama-server.")
        return

    for key, label in (("stt", "STT"), ("llm", "LLM"), ("tts", "TTS"), ("total", "итого")):
        values = [r[key] for r in answered if key in r]
        if values:
            print(f"  {label:<6} среднее {sum(values) / len(values):5.2f} с, "
                  f"максимум {max(values):5.2f} с")

    worst = max(r["total"] for r in answered)
    verdict = "укладываемся" if worst <= budget_s else "НЕ УКЛАДЫВАЕМСЯ"
    print(f"  бюджет {budget_s:.0f} с — {verdict} (худшая фраза {worst:.2f} с)")


def run_bench(cfg: Config, in_file: str, budget_s: float = 10.0,
              realtime: bool = False) -> int:
    from .responder import build_llm_responder

    responder = build_llm_responder(cfg)
    bench = BenchResponder(responder, cfg.audio.sample_rate)

    source = FileSource(in_file, sample_rate=cfg.audio.sample_rate,
                        frame_samples=cfg.audio.frame_samples, realtime=realtime)
    sink = NullSink()
    ptt = DummyPtt(verbose=False)
    repeater = Repeater(cfg, EnergyVad(cfg.vad.threshold), bench, ptt, sink,
                        realtime=realtime)
    try:
        repeater.run(source)
    finally:
        sink.close()
        ptt.close()

    print_report(bench.records, budget_s)
    return 0
