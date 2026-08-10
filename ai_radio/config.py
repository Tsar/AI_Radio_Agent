"""Конфигурация конвейера. Простые dataclass'ы, без внешних зависимостей."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AudioConfig:
    sample_rate: int = 16000      # рабочая частота дискретизации, Гц
    frame_ms: int = 20            # длина кадра, мс (10/20/30 — типовые для VAD)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class VadConfig:
    threshold: float = 0.004      # порог RMS [0..1]; ~ -48 dBFS по калибровке Optim-778
                                  # (шумовой пол ~-60 dBFS + 12 дБ). Калибруйте под свою установку.
    start_frames: int = 3         # кадров подряд выше порога, чтобы начать приём
    hangtime_ms: int = 1000       # тишины до конца передачи; >800 мс склеивает паузы
                                  # внутри одной реплики (проверено на записи Optim-778)
    preroll_ms: int = 300         # сколько удерживать до срабатывания (не резать атаку)
    max_utterance_ms: int = 60000 # предел буфера приёма; сверх — не буферизуем, но приём
                                  # закрываем всё равно по тишине (уходит первая минута)


@dataclass
class TxConfig:
    warmup_ms: int = 200          # пауза после нажатия PTT до подачи аудио (передатчик «поднимается»)
    tail_ms: int = 150            # пауза после аудио до отпускания PTT (не резать хвост)
    cooldown_ms: int = 300        # защитный интервал перед возвратом к приёму


@dataclass
class PttConfig:
    backend: str = "dummy"        # dummy | txdbreak
    port: str = "/dev/ttyUSB0"
    invert: bool = True           # у нашего USB-UART сигнал инвертирован


CALLSIGN = "феечка"

# Whisper коверкает имена собственные, особенно на узкой полосе рации: точное
# вхождение не годится, поэтому держим список вариантов + нечёткое сравнение.
# «фея» сюда сознательно не входит: склейка соседних слов и так ловит «фея чка»
# (0.91), а отдельное слово «фея» в чужой речи давало бы ложные срабатывания.
CALLSIGN_VARIANTS = ["феечка", "феичка", "фечка", "фиечка", "феюшка"]

SYSTEM_PROMPT = (
    "Ты — Феечка, автоматическая радиостанция в эфире. "
    "{length} "
    "Твой ответ читает вслух синтезатор речи, поэтому пиши только по-русски, "
    "без списков, разметки, эмодзи, латиницы и ссылок. "
    "Говори простыми фразами, как в радиосвязи."
)


@dataclass
class ReplyLength:
    """Пресет длины ответа. Крутить эти три ручки порознь бессмысленно: промпт
    определяет, сколько модель напишет, max_sentences режет лишнее, а max_tokens —
    только предохранитель (он рубит по счётчику и может оборвать на полуслове)."""
    max_tokens: int
    max_sentences: int
    hint: str          # подставляется в SYSTEM_PROMPT вместо {length}
    air_s: str         # замер на Qwen3-4B: сколько такой ответ занимает эфира


# Во время своей передачи агент глух, поэтому длина — это не только вопрос вкуса.
REPLY_LENGTHS: Dict[str, ReplyLength] = {
    "short": ReplyLength(
        80, 2, "Отвечай кратко: одно-два коротких предложения.", "~6 с"),
    "medium": ReplyLength(
        200, 4, "Отвечай тремя-четырьмя предложениями.", "~11 с"),
    "long": ReplyLength(
        400, 8, "Отвечай развёрнуто, шестью-восемью предложениями.", "~35 с"),
}
DEFAULT_REPLY_LENGTH = "short"


@dataclass
class SttConfig:
    model: str = "large-v3-turbo"  # имя модели faster-whisper или путь к CT2-каталогу;
                                   # держать в согласии с PROFILES[DEFAULT_PROFILE]
    device: str = "cuda"
    compute_type: str = "int8"     # на Pascal FP16 идёт 1/64 скорости; int8 держим и на dev,
                                   # чтобы качество совпадало с продом (квантование меняет выход)
    language: str = "ru"
    beam_size: int = 5
    vad_filter: bool = False       # встроенный Silero VAD ВРЕДЕН на нашем сигнале: на узкой
                                   # полосе рации он вырезает почти всю речь, и Whisper
                                   # галлюцинирует на остатке (вплоть до пересказа
                                   # initial_prompt). Фразу нам и так вырезал energy VAD.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0  # сегменты ниже — отбрасываем (типовой источник галлюцинаций)
    min_chars: int = 3             # короче — считаем мусором и молчим
    initial_prompt: str = "Радиосвязь. Позывной Феечка. Как слышно, приём."


@dataclass
class LlmConfig:
    base_url: str = "http://127.0.0.1:8080"   # llama-server (OpenAI-совместимый /v1)
    model: str = "local"
    # длина ответа задаётся пресетом (--reply-length), а не тремя ручками порознь
    reply_length: str = DEFAULT_REPLY_LENGTH
    system_prompt: str = SYSTEM_PROMPT.format(
        length=REPLY_LENGTHS[DEFAULT_REPLY_LENGTH].hint)
    max_tokens: int = REPLY_LENGTHS[DEFAULT_REPLY_LENGTH].max_tokens
    max_sentences: int = REPLY_LENGTHS[DEFAULT_REPLY_LENGTH].max_sentences
    temperature: float = 0.7
    timeout_s: float = 60.0
    no_think: bool = True          # Qwen3 без этого генерит <think>…</think> — время и токены впустую


@dataclass
class TtsConfig:
    # путь к модели Piper (.onnx, рядом должен лежать одноимённый .onnx.json) —
    # по умолчанию туда, куда README велит её скачать
    voice: str = "models/piper/ru_RU-irina-medium.onnx"
    peak_dbfs: float = -3.0                  # уровень отдаваемого в эфир аудио
    length_scale: "float | None" = None      # None — как в модели; >1 медленнее, <1 быстрее


@dataclass
class RvcConfig:
    """Преобразование голоса поверх Piper. Отдельный процесс: RVC нужен numpy 1.23
    и fairseq, у нас numpy 2.x — в один venv не ставится."""
    enabled: bool = False          # включается флагом --rvc
    base_url: str = "http://127.0.0.1:8081"
    voice: str = "voicevox_speaker_43"
    input_voice: str = "irina"     # поправка питча на входной голос; irina ≈ shimmer (0)
    pitch: "int | None" = None     # None — пресет сервиса (для voicevox_speaker_43 это +8)
    formant_shift: "float | None" = None   # None — пресет сервиса (1.0)
    # pm вместо rmvpe: на входе у RVC чистый синтез Piper, а не шумный эфир, поэтому
    # нейросетевой экстрактор F0 не нужен — на слух не отличить, а 335 МБ VRAM
    # освобождаются. На P2000 (5 ГБ) без этого не хватает места под turbo + 4B.
    f0_method: str = "pm"
    peak_dbfs: float = -3.0        # RVC меняет уровень, нормализуем перед эфиром
    timeout_s: float = 60.0


@dataclass
class DialogConfig:
    callsign: str = CALLSIGN
    callsign_variants: List[str] = field(default_factory=lambda: list(CALLSIGN_VARIANTS))
    match_threshold: float = 0.75  # схожесть слова с позывным (difflib), 1.0 — точное совпадение
    window_s: float = 60.0         # после ответа столько отвечаем без позывного. Чем окно
                                   # шире, тем выше шанс ответить на чужую передачу в общем
                                   # канале; паузы внутри живого диалога — единицы секунд
    max_history: int = 8           # реплик в контексте (считая свои)
    end_phrases: List[str] = field(default_factory=lambda: ["отбой", "конец связи", "до связи"])


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    tx: TxConfig = field(default_factory=TxConfig)
    ptt: PttConfig = field(default_factory=PttConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    rvc: RvcConfig = field(default_factory=RvcConfig)
    dialog: DialogConfig = field(default_factory=DialogConfig)
    input_device: "int | str | None" = None
    output_device: "int | str | None" = None


# Профиль подбирается под видеопамять машины. Замеры (пик, int8, на RTX 2070):
#   small 465 МБ | large-v3-turbo 1231 МБ | medium 1074 МБ | large-v3 2027 МБ
# medium не используем: он тяжелее turbo и на нашем материале распознаёт хуже small.
# Имена — по видеопамяти карты, а не по машине: обе прод-машины одинаково боевые,
# различаются только объёмом VRAM.
PROFILES: Dict[str, Dict[str, str]] = {
    # 5 ГБ (Quadro P2000): turbo + Qwen3-4B + RVC = 3.9 ГБ, запас 1.2 ГБ
    "vram5": {"model": "large-v3-turbo", "device": "cuda", "compute_type": "int8"},
    # 10 ГБ (P102-100): хватает на полную large-v3
    "vram10": {"model": "large-v3", "device": "cuda", "compute_type": "int8"},
    # для отладки, когда качество распознавания вторично
    "small": {"model": "small", "device": "cuda", "compute_type": "int8"},
    "cpu": {"model": "small", "device": "cpu", "compute_type": "int8"},
}
# Профиль по умолчанию применяется и без флага --profile: раньше он был объявлен,
# но не использовался, и команда без флага молча брала дефолт SttConfig.model.
# На 5 ГБ это разница между turbo (1231 МБ) и large-v3 (2027 МБ) — то есть между
# рабочим стеком и «CUDA out of memory» на первой же транскрипции.
DEFAULT_PROFILE = "vram5"


def apply_profile(cfg: Config, name: str) -> None:
    """Профиль задаёт только параметры STT — модель LLM выбирается при запуске
    llama-server, голос Piper одинаков везде."""
    try:
        preset = PROFILES[name]
    except KeyError:
        raise ValueError(
            f"неизвестный профиль {name!r}; доступны: {', '.join(sorted(PROFILES))}") from None
    cfg.stt.model = preset["model"]
    cfg.stt.device = preset["device"]
    cfg.stt.compute_type = preset["compute_type"]


def apply_reply_length(cfg: Config, name: str) -> None:
    """Переключить длину ответа одним пресетом: промпт, max_sentences и max_tokens
    меняются согласованно. Перезаписывает system_prompt — если он правился руками,
    правьте шаблон SYSTEM_PROMPT, а не поле конфига."""
    try:
        preset = REPLY_LENGTHS[name]
    except KeyError:
        raise ValueError(
            f"неизвестная длина ответа {name!r}; доступны: "
            f"{', '.join(REPLY_LENGTHS)}") from None
    cfg.llm.reply_length = name
    cfg.llm.max_tokens = preset.max_tokens
    cfg.llm.max_sentences = preset.max_sentences
    cfg.llm.system_prompt = SYSTEM_PROMPT.format(length=preset.hint)
