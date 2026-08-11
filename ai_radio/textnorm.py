"""Нормализация текста перед синтезом речи.

Ни один русский TTS не делает этого сам: цифры он либо читает по одной, либо молчит,
латиницу произносит по-английски, а разметку и эмодзи озвучивает буквально. Всё это
идёт в эфир, поэтому чистим до подачи в Piper.

Модуль на stdlib; num2words импортируется лениво — без него числа просто останутся
цифрами, а весь остальной конвейер (и `trigger-test`) продолжит работать.
"""
from __future__ import annotations

import re
from typing import Optional

# Что оставляем после чистки: кириллица, цифры, пробел и минимальная пунктуация.
# Всё прочее (эмодзи, стрелки, псевдографика) — выкидываем.
_ALLOWED = re.compile(r"[^а-яёА-ЯЁ0-9 .,!?;:%-]")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)  # обрыв по max_tokens
_TAG = re.compile(r"<[^>]+>")

_SYMBOLS = [
    ("%", " процентов "),
    ("№", " номер "),
    ("&", " и "),
    ("+", " плюс "),
    ("=", " равно "),
    ("°", " градусов "),
    ("€", " евро "),
    ("$", " долларов "),
    ("₽", " рублей "),
]

# Латиница фонетически — иначе espeak-ng прочтёт её по-английски посреди русской фразы.
_TRANSLIT_DIGRAPHS = [
    ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"),
    ("yu", "ю"), ("ya", "я"), ("yo", "ё"), ("ee", "и"), ("oo", "у"), ("th", "т"),
    ("ph", "ф"), ("ck", "к"), ("qu", "кв"),
]
_TRANSLIT_CHARS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "й", "z": "з",
}

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RANGE = re.compile(r"(?<=\d)-(?=\d)")

_num2words = None
_num2words_tried = False


def _get_num2words():
    """Ленивый импорт: модуль должен работать и без установленного num2words."""
    global _num2words, _num2words_tried
    if not _num2words_tried:
        _num2words_tried = True
        try:
            from num2words import num2words as fn
        except ImportError:
            fn = None
        _num2words = fn
    return _num2words


def strip_think(text: str) -> str:
    """Убрать рассуждения Qwen3. Второй паттерн ловит случай, когда ответ обрезан
    по max_tokens и закрывающего тега просто нет."""
    text = _THINK.sub(" ", text)
    return _THINK_OPEN.sub(" ", text)


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)     # блоки кода
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"_([^_]*)_", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)        # ссылки → текст ссылки
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)   # маркеры списка
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)  # нумерация: иначе «1.»
                                                                    # читается как конец фразы
    return _TAG.sub(" ", text)


def numbers_to_words(text: str, lang: str = "ru") -> str:
    """Числа прописью. Без num2words оставляем как есть — Piper прочитает по цифрам."""
    fn = _get_num2words()
    if fn is None:
        return text

    def repl(m: "re.Match[str]") -> str:
        raw = m.group(0).replace(",", ".")
        try:
            value = float(raw) if "." in raw else int(raw)
            return " " + fn(value, lang=lang) + " "
        except (ValueError, NotImplementedError, OverflowError):
            return m.group(0)

    return _NUMBER.sub(repl, text)


def translit_latin(text: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        word = m.group(0).lower()
        for src, dst in _TRANSLIT_DIGRAPHS:
            word = word.replace(src, dst)
        return "".join(_TRANSLIT_CHARS.get(ch, ch) for ch in word)

    return re.sub(r"[A-Za-z]+", repl, text)


def normalize_for_tts(text: str, lang: str = "ru") -> str:
    """Полная чистка: разметка и рассуждения → символы словами → числа прописью →
    латиница кириллицей → выброс всего неречевого."""
    text = strip_think(text)
    text = strip_markdown(text)
    for src, dst in _SYMBOLS:
        text = text.replace(src, dst)
    text = _TIME.sub(r"\1 \2", text)          # 12:30 → «двенадцать тридцать»
    text = _RANGE.sub(" ", text)              # 5-10 → «пять десять»
    text = numbers_to_words(text, lang=lang)
    text = translit_latin(text)
    text = _ALLOWED.sub(" ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_words(text: str, max_chars: int) -> str:
    """Обрезать по границе слова, не длиннее max_chars.

    Нужен потому, что max_sentences беззащитен перед текстом без знаков препинания:
    на просьбу прочитать стихи Qwen3 выдал 80 токенов сплошным потоком, `split`
    вернул один «предложение» на всю строку, и в эфир ушло 14.76 с вместо ~6.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars // 2:      # у слова-переростка режем прямо по лимиту
        cut = cut[:space]
    # точка в конце: без неё синтезатор тянет незавершённую интонацию
    return cut.rstrip(" ,;:-—") + "."


def clean_llm_reply(text: str, max_sentences: Optional[int] = None,
                    max_chars: Optional[int] = None) -> str:
    """Ответ LLM → готовая к синтезу строка. max_sentences подрезает многословие
    модели, max_chars страхует его на тексте без точек: занимать эфир длинной
    передачей нельзя, а во время неё агент ещё и глух."""
    text = normalize_for_tts(text)
    if max_sentences is not None and max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(parts[:max_sentences]).strip()
    if max_chars is not None:
        text = truncate_words(text, max_chars)
    return text
