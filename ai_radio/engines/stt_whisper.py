"""STT: faster-whisper (CTranslate2).

Модель грузится один раз при старте и живёт в процессе — загрузка весов занимает
секунды, на каждую фразу её позволить нельзя.

Про compute_type: на Pascal (P102-100, Quadro P2000 — compute capability 6.1) FP16
идёт в 1/64 скорости, а CTranslate2 требует для него cc >= 7.0. Рабочий режим —
int8, и держать его надо и на dev-карте: квантование слегка меняет выход, иначе
качество на тестах и в проде разойдётся.
"""
from __future__ import annotations

from typing import List

from ..audio_io import resample_linear
from ..config import SttConfig

WHISPER_RATE = 16000


def preload_cuda_libs() -> None:
    """Подгрузить libcublas/libcudnn из pip-пакетов nvidia-*.

    CTranslate2 берёт их через dlopen по SONAME, а каталогов pip-пакетов в путях
    загрузчика нет; LD_LIBRARY_PATH после старта процесса менять уже поздно.
    Загружаем сами по полному пути — dlopen по SONAME потом отдаёт готовый объект.
    Иначе всё работает до первой транскрипции и падает на «libcublas.so.12 is not found».

    Два прохода: у cuDNN есть зависимости от cuBLAS и от собственных модулей.
    """
    import ctypes
    import glob
    import os

    try:
        import nvidia
    except ImportError:
        return  # системный CUDA — грузить нечего

    libs: List[str] = []
    for root in nvidia.__path__:
        libs.extend(sorted(glob.glob(os.path.join(root, "*", "lib", "lib*.so*"))))
    for _ in range(2):
        for so in libs:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class FasterWhisperStt:
    def __init__(self, cfg: SttConfig, sample_rate: int = 16000) -> None:
        from faster_whisper import WhisperModel  # тяжёлый импорт — только здесь
        self.cfg = cfg
        self.sample_rate = sample_rate
        if cfg.device != "cpu":
            preload_cuda_libs()
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe(self, audio: List[float]) -> str:
        import numpy as np

        if not audio:
            return ""
        if self.sample_rate != WHISPER_RATE:
            audio = resample_linear(audio, self.sample_rate, WHISPER_RATE)

        cfg = self.cfg
        segments, _info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=cfg.language,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            initial_prompt=cfg.initial_prompt or None,
            # ключевое против галлюцинаций: без этого Whisper на шуме и тишине
            # дописывает «продолжение следует» и прочие титры из обучающих данных
            condition_on_previous_text=False,
        )

        parts: List[str] = []
        for seg in segments:
            no_speech = getattr(seg, "no_speech_prob", None)
            if no_speech is not None and no_speech > cfg.no_speech_threshold:
                continue
            logprob = getattr(seg, "avg_logprob", None)
            if logprob is not None and logprob < cfg.log_prob_threshold:
                continue
            text = (seg.text or "").strip()
            if text:
                parts.append(text)

        result = " ".join(parts).strip()
        # огрызок в пару символов — это шум щелчка PTT, а не речь
        return result if len(result) >= cfg.min_chars else ""
