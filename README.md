# AI Radio Agent

Автономная радиостанция: принимает голосовую передачу из эфира, формирует ответ и
передаёт его обратно. Рация (Optim-Sprint / Optim-778) подключается к компьютеру
через звуковую карту (line-in/line-out), передача (PTT) коммутируется через USB-UART.

Так как аппаратного сигнала «идёт приём» (COS) от рации нет, начало и конец передачи
определяются по звуку — **VAD** (детектор голоса по громкости). Шумоподавитель рации
держим закрытым, поэтому на вход приходит звук только во время реальной передачи.

## Статус

**Этап 1 — готов:** parrot-репитер («попугай») — принятое ретранслируется обратно.
Работает и на записи из файла (симуляция), и вживую на реальном железе.

**Этап 2 — готов:** голосовой ответчик `STT → LLM → TTS`, целиком локальный (машина
поедет на дачу без интернета). Агент отвечает **не на каждую передачу**, а по позывному
(**«Феечка»**) плюс окно продолжения диалога. Включается флагом `--responder llm`,
parrot остаётся ответчиком по умолчанию.

**Этап 3 — готов:** преобразование голоса (**RVC**) поверх Piper — флаг `--rvc`.
Синтезированный ответ переозвучивается целевым голосом целиком, одним проходом.

## Как это работает

```
                 RX audio (выход динамика рации)
   ┌─────────┐  ─────────────────────────────▶  ┌──────────────┐
   │  РАЦИЯ   │                                    │  КОМПЬЮТЕР   │
   │ Optim-  │  ◀─────────────────────────────    │ звуковая     │
   │ 778 /   │   TX audio (в микрофонный вход)     │ карта        │
   │ Sprint  │                                    │              │
   │         │  ◀── PTT (break на TX) ──USB-UART──┘ (pyserial)   │
   └─────────┘                                    └──────────────┘
```

Конвейер (строго half-duplex — во время передачи вход не слушаем):

```
источник → EnergyVAD → [preroll + hangtime] → Responder → PTT + приёмник
 (файл/мик)  порог RMS    склейка фразы     parrot | LLM    (динамик/рация)
```

`Responder` — единственный шов, за которым живёт весь этап 2:

```
utterance → Whisper → позывной? → LLM → нормализация → Piper → нормализация
  (аудио)    (STT)    окно диалога  (llama-server)  текста   (TTS)   уровня
```

Состояния репитера: `IDLE → RECEIVING (буфер) → RESPONDING → TRANSMIT → IDLE`.
`preroll` не даёт срезать атаку фразы, `hangtime` склеивает паузы между словами в одну
реплику. Вход приостанавливается на **всё время «думаем + передаём»**: STT→LLM→TTS
занимает секунды, и незачитанный поток иначе переполнится, а агент отреагировал бы на
звук, накопившийся за время раздумий.

Если агент не разобрал фразу, не услышал позывного или LLM недоступен — он **молчит**
(`respond()` возвращает `None`, PTT не жмётся). Молчание здесь штатный исход.

Готовые команды запуска (три процесса) — в [RUN.md](RUN.md). Для установки на
прод-машину, ту, что уедет на дачу, есть отдельный чеклист — [DEPLOY.md](DEPLOY.md):
автозапуск, проверка офлайн-готовности и диагностика.

## Требования

- **Python 3.10+**
- **ffmpeg** — декодирование/ресемплинг аудиофайлов (калибровка и файловый режим)
- Для живого режима: **numpy**, **sounddevice**, **pyserial** и системный **libportaudio2**
- Для этапа 2 (`--responder llm`): **faster-whisper**, **piper-tts**, **num2words**
  и запущенный **llama-server**. Системный `espeak-ng` не нужен — piper-tts ≥ 1.6
  несёт его внутри пакета

Калибровка и прогон на файле работают только на stdlib + ffmpeg — numpy/sounddevice не нужны.
Команда `trigger-test` тоже обходится stdlib: проверять позывной можно без моделей.

### Модели этапа 2

| Звено | Что используем | Почему |
|---|---|---|
| STT | faster-whisper `large-v3`, **`compute_type=int8`** | устойчив к шуму и узкой полосе рации; на Pascal FP16 идёт в 1/64 скорости, а CTranslate2 требует для него cc ≥ 7.0 |
| LLM | llama-server + Qwen3-4B `Q4_K_M` | отдельный процесс, модель постоянно в VRAM |
| TTS | Piper `ru_RU-irina-medium` (CPU) | быстрый; разницу с тяжёлыми TTS всё равно срезает полоса 300–3400 Гц |
| Голос | RVC `voicevox_speaker_43` (опционально) | тембр различим и в полосе рации, в отличие от «натуральности» TTS |

`int8` держим **и на dev-машине**: квантование слегка меняет выход, иначе качество на
тестах и в проде разойдётся. Профиль укладывается в ~4.6 ГБ VRAM.

Две настройки, проверенные на реальной записи с Optim-778 и на имитации тракта
(Piper → полоса 300–3400 Гц → шум), обе неочевидные:

- **`vad_filter` выключен.** Silero VAD внутри faster-whisper на узкополосном сигнале
  вырезает почти всю речь, после чего Whisper галлюцинирует на остатке — вплоть до
  пересказа собственного `initial_prompt`. Фразу нам уже вырезал energy VAD.
- **`initial_prompt` длинный, целой фразой.** На тесте с позывным он дал 4/4
  срабатывания против 3/4 без промпта. Короткий промпт (`"Феечка."`) оказался хуже
  всех — 1/4: Whisper считает позывной уже прозвучавшим и **опускает** его в выводе.
  Плата — изредка промпт протекает в результат, когда сказанное на него похоже.

## Установка

```bash
# системное
sudo apt install ffmpeg libportaudio2
# для venv на Debian/Ubuntu бывает нужен пакет venv (в нём ensurepip):
sudo apt install python3-venv          # или python3.X-venv под вашу версию

# python-зависимости (для живого режима) в изолированном окружении:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Дальше запускайте через `.venv/bin/python main.py …` (или `source .venv/bin/activate`).
Калибровка и прогон на файле работают и на системном `python3` без venv.

### Этап 2: модели и llama-server

**Версия Python важна:** целимся в **3.10** (системный на Ubuntu 22.04, где стоит прод).
Под свежие 3.13/3.14 ML-колёс может не быть — проще всего `uv` (скачает нужный
интерпретатор сам, системный не тронет):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
uv pip install -r requirements.txt -r requirements-ai.txt
```

`uv venv` создаёт окружение **без pip** — ставьте через `uv pip install`. И не забудьте
`requirements.txt`: пересоздание venv стирает зависимости этапа 1.

CUDA-библиотеки ставятся pip-пакетами (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), сам
CUDA toolkit в системе не нужен. Их каталоги не лежат в путях загрузчика, поэтому
`preload_cuda_libs()` в `engines/stt_whisper.py` подгружает их вручную — без этого всё
работает до первой транскрипции и падает на `libcublas.so.12 is not found`.

Голос Piper (~63 МБ) — кладём в `models/piper/`, это путь по умолчанию в конфиге:

```bash
mkdir -p models/piper
voice=https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx
curl -L -o models/piper/ru_RU-irina-medium.onnx      "$voice"
curl -L -o models/piper/ru_RU-irina-medium.onnx.json "$voice.json"
```

`llama.cpp` собирается один раз под **обе** карты (Pascal `61` + Turing `75`) —
тогда бинарь одинаково идёт и на dev-машине, и в проде. Обязательно с **CUDA 12**:
в CUDA 13 поддержка Pascal вырезана. В репозитории Ubuntu лежит подходящая 12.4:

Собирается **рядом с проектом**, не внутри него:

```bash
sudo apt install nvidia-cuda-toolkit cmake g++-13
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="61;75" \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build -j --target llama-server
```

`g++-13` здесь не прихоть: nvcc 12.4 отказывается работать с gcc новее 13-го, а на
свежих Ubuntu системный уже 15.x. Системный компилятор при этом не трогаем.

Проверить, что бинарь умеет обе карты (должны быть и `sm_61`, и `sm_75`):
```bash
cuobjdump --list-elf ../llama.cpp/build/bin/libggml-cuda.so | grep -oE "sm_[0-9]+" | sort -u
```

Модель (~2.5 ГБ):
```bash
mkdir -p models/llm && curl -L -o models/llm/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

Запуск из корня проекта (бинарь **не в PATH** — путь указываем целиком). Модель должна
жить в VRAM постоянно: агент молчит часами, а повторная загрузка стоила бы 10+ с на
первый ответ:

```bash
../llama.cpp/build/bin/llama-server -m models/llm/Qwen3-4B-Q4_K_M.gguf \
    -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

На карте с 4 ГБ добавьте `-c 2048 --cache-type-k q8_0 --cache-type-v q8_0`.

### Перенос на машину без интернета

Голос Piper и `.gguf` — просто файлы в `models/`, их достаточно скопировать. А вот
**модель Whisper faster-whisper скачивает сам** при первом запуске (`large-v3` ≈ 2.9 ГБ,
`small` ≈ 464 МБ) и кладёт в кэш HuggingFace. На даче интернета нет, поэтому либо
копируем кэш целиком:

```bash
rsync -a ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3 \
         прод:~/.cache/huggingface/hub/
```

либо — надёжнее — забираем каталог со снапшотом и указываем путь явно:

```bash
ls -d ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/*/
# скопировать этот каталог (config.json, model.bin, tokenizer.json, vocabulary.txt)
python3 main.py run --live --responder llm --stt-model /путь/к/снапшоту
```

Во втором случае `--stt-model` принимает каталог, и в сеть никто не ходит.

## Использование

### Калибровка порога VAD

По записи из файла:
```bash
python3 main.py calibrate --in-file запись.mp3
```
Или вживую с микрофона (первые ~4 с молчите — замерится фон, потом говорите):
```bash
.venv/bin/python main.py calibrate --live --seconds 12
```
Печатает шумовой пол, уровень сигнала, рекомендованный порог и ASCII-гистограмму
громкости. Примеры замеров:
```
# рация Optim-778 (line-in):      фон -60.6 → порог -48.6 dBFS (threshold 0.00369)
# живой микрофон (тихая комната):  фон -70.6 → порог -58.6 dBFS (threshold 0.00118)
```
Полученный порог подставляйте в `--threshold`.

### Репитер на файле (симуляция)

```bash
python3 main.py run --in-file запись.mp3 --out-file out.wav
# покрутить параметры:
python3 main.py run --in-file запись.mp3 --threshold 0.004 --hangtime-ms 1000
```
Прогоняет запись через VAD и parrot, «переданное» пишет в `out.wav`. PTT — заглушка
(логирует KEY/UNKEY).

### Живой режим (реальная рация)

```bash
python3 main.py devices                    # индексы звуковых карт
# сначала безопасно (PTT-заглушка, эфир не жмётся, слышно себя в динамике ПК):
python3 main.py run --live --in-device 2 --out-device 2
# боевой PTT на рацию:
python3 main.py run --live --ptt txdbreak --port /dev/ttyUSB0 --in-device 2 --out-device 2
# если карта не умеет 16 кГц — указать её родную частоту:
python3 main.py run --live --device-rate 48000 ...
```

### Тест без рации (микрофон + наушники)

Живой конвейер можно проверить без радио: микрофон играет роль «приёма», наушники —
«передачи», PTT — заглушка. Петли/эха нет, если использовать **наушники, а не колонки**.

```bash
# 1) откалибровать порог под микрофон
.venv/bin/python main.py calibrate --live --seconds 12
# 2) запустить parrot с полученным порогом
.venv/bin/python main.py run --live --threshold 0.0012
```
Говорите → замолкаете на ~секунду (hangtime) → слышите свой голос обратно.

Вход/выход выбираются в настройках звука (или флагами `--in-device`/`--out-device`).
Проще всего оставить системный дефолт через PipeWire/PulseAudio — он сам ресемплит в 16 кГц.
**Bluetooth-наушники** для вывода годятся, но: (1) добавляют задержку ~0.1–0.3 с (parrot
вернётся с паузой — это норма); (2) держите вход на отдельном (не BT) микрофоне, иначе
гарнитура уйдёт в режим HFP и звук выхода схлопнется до «телефонного».

### Голосовой ответчик (этап 2)

Сначала в отдельном терминале поднимите `llama-server` (см. установку) — без него агент
распознает фразу, но ответить не сможет и промолчит. Затем:

```bash
# на записи из эфира — «переданное» уходит в out.wav
.venv/bin/python main.py run --in-file запись.mp3 --out-file out.wav --responder llm
# вживую с микрофоном/наушниками, без рации
.venv/bin/python main.py run --live --responder llm --threshold 0.0012
# боевой режим
.venv/bin/python main.py run --live --responder llm --ptt txdbreak --port /dev/ttyUSB0
```

Скажите «**Феечка**, как слышно?» → пауза на обдумывание → голосовой ответ. Фраза без
позывного вне окна диалога остаётся без ответа — так и задумано.

### Преобразование голоса (RVC, этап 3)

Ответ синтезируется Piper'ом, а затем целиком переозвучивается целевым голосом. RVC
живёт **отдельным процессом со своим venv**: ему нужны `numpy 1.23` и `fairseq`, а у нас
`numpy 2.x` ради faster-whisper и piper — в одно окружение это не ставится. Схема та же,
что с llama-server.

#### Установка RVC-сервиса

Сервис живёт в форке RVC — https://github.com/Tsar/Retrieval-based-Voice-Conversion-WebUI,
ветка **`offline-http-inference`** (скрипт `infer-http-service.py`). Ставится рядом с
проектом, **своим venv на Python 3.10**:

```bash
cd .. && git clone -b offline-http-inference \
    https://github.com/Tsar/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
# torch с PyPI приезжает под CUDA 13, где нет Pascal — берём сборку под CUDA 12:
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# setuptools 81+ выпилил pkg_resources, без него не импортируются librosa и pyworld:
uv pip install "setuptools<81"
```

Затем базовые модели (hubert и rmvpe) — их качает скрипт из репозитория:

```bash
.venv/bin/python tools/download_models.py
```

**Голосовые модели в git не хранятся** (`assets/weights/*` в `.gitignore`) — положите
нужные `.pth` в `assets/weights/` вручную. Сервису нужны те, что перечислены в его
`TARGET_VOICES`: `voicevox_speaker_43.pth`, `xiangling_eng_30_epochs_with_pitch.pth`,
`citlali_jap.pth`.

Запуск:

```bash
.venv/bin/python infer-http-service.py --port 8081
```

Затем агент запускается с `--rvc`:

```bash
.venv/bin/python main.py run --live --responder llm --rvc --threshold 0.0012
```

Голоса и их пресеты (`--rvc-voice`):

| Голос | Транспонирование | Сдвиг формант |
|---|---|---|
| `voicevox_speaker_43` | +8 | +1.0 (по умолчанию) |
| `xiangling_eng` | +12 | +1.0 |
| `citlali_jap` | +6 | 0 |

Транспонирование считается как `pitch - INPUT_VOICES_PITCH[input_voice]`, где для нашего
Piper-голоса `irina` поправка нулевая (он примерно так же низок, как shimmer у OpenAI).

**Сдвиг формант укорачивает фразу.** В реалтайм-версии RVC его дают бесплатно, попросив
`net_g` сгенерировать больше фреймов, чем будет проиграно. Офлайн такой ручки нет, поэтому
сдвиг делается ресемплингом готового аудио — и фраза становится короче на `1 - 1/2^(shift/12)`:
около 6% при `+1.0`, 11% при `+2`, 16% при `+3`. Для эфира это скорее плюс — передача короче.

Если сервис не поднят, агент **передаёт голосом Piper**, а не молчит: отказ косметического
звена не должен стоить ответа. Предупреждение печатается один раз, а не на каждую фразу.

### Замер задержки (`bench`)

Скорость известна только на том железе, где померена: dev-машина и прод расходятся
в разы, особенно на Piper (он считает на CPU). Прогон идёт через настоящий конвейер:

```bash
.venv/bin/python main.py bench --in-file запись.mp3 --threshold 0.004
.venv/bin/python main.py bench --in-file запись.mp3 --profile fast --budget 10
```

Печатает разбивку по звеньям (STT / LLM / TTS), время на фразу и вердикт по бюджету.
Порог берите из `calibrate` для этой же записи — иначе VAD может не найти в ней фраз.

Замер на dev-машине (RTX 2070 + Ryzen 9950X3D), профиль `prod`:

```
  #   приём   STT     LLM     TTS    итого   эфир  ответ
  1    2.50    0.46    0.11    0.05    0.62    1.49   да
  бюджет 10 с — укладываемся (худшая фраза 0.62 с)
```

### Проверка позывного (`trigger-test`)

Whisper коверкает имена собственные, поэтому позывной ищется **нечётко** — по склейкам
соседних слов. Проверить подбор порога можно без моделей и без GPU:

```bash
python3 main.py trigger-test "феечка как слышно" "фея чка ответь" "привет всем"
```
```
  СРАБОТАЛ  score=1.00  «феечка»     'феечка как слышно'
  СРАБОТАЛ  score=0.91  «феячка»     'фея чка ответь'
  молчим    score=0.36  «всемна»     'привет всем'
```

### Тест PTT-линии (осциллографом)

```bash
python3 tools/break_test.py --port /dev/ttyUSB0 --on 0.5 --off 0.5
```
Периодически включает/снимает break на линии TX — проверить уровнем/осциллографом,
что PTT коммутируется.

## Конфигурация

Значения по умолчанию — в `ai_radio/config.py` (откалибровано под запись Optim-778):

| Параметр | Default | Смысл |
|---|---|---|
| `sample_rate` | 16000 | рабочая частота, Гц |
| `frame_ms` | 20 | длина кадра VAD |
| `vad.threshold` | 0.004 | порог RMS (~-48 dBFS); **калибруйте под свою установку** |
| `vad.start_frames` | 3 | кадров подряд выше порога для старта приёма |
| `vad.hangtime_ms` | 1000 | тишины до конца передачи (склейка пауз) |
| `vad.preroll_ms` | 300 | сколько удерживать до срабатывания (не резать атаку) |
| `vad.max_utterance_ms` | 60000 | предел буфера приёма; сверх — не буферизуем, но закрываем по тишине |
| `tx.warmup_ms` | 200 | пауза после нажатия PTT до подачи аудио |
| `tx.tail_ms` | 150 | пауза после аудио до отпускания PTT |
| `tx.cooldown_ms` | 300 | защитный интервал перед возвратом к приёму |
| `ptt.invert` | True | инверсия PTT (наш USB-UART инвертирует сигнал) |

Порог, hangtime и предел буфера переопределяются флагами `--threshold` / `--hangtime-ms` /
`--max-utterance-ms`.

Этап 2 (там же, `config.py`):

| Параметр | Default | Смысл |
|---|---|---|
| `stt.model` | `large-v3` | модель faster-whisper (или путь к CT2-каталогу) |
| `stt.compute_type` | `int8` | **не менять на float16** — Pascal его не тянет, а выход отличается |
| `stt.vad_filter` | False | **не включать:** встроенный Silero VAD на узкой полосе рации вырезает почти всю речь, и Whisper галлюцинирует на остатке |
| `stt.min_chars` | 3 | короче — считаем шумом щелчка PTT и молчим |
| `stt.initial_prompt` | «…Позывной Феечка…» | подсказка Whisper: заметно улучшает распознавание позывного |
| `llm.base_url` | `http://127.0.0.1:8080` | адрес llama-server |
| `llm.reply_length` | `short` | пресет длины ответа, см. ниже (`--reply-length`) |
| `llm.max_tokens` | 80 | предохранитель генерации; задаётся пресетом |
| `llm.max_sentences` | 2 | фактический ограничитель длины; задаётся пресетом |
| `llm.no_think` | True | гасит `<think>…</think>` у Qwen3 |
| `tts.voice` | `models/piper/ru_RU-irina-medium.onnx` | голос Piper (.onnx + одноимённый .onnx.json) |
| `tts.peak_dbfs` | -3.0 | уровень в эфир: от него зависит глубина модуляции |
| `dialog.callsign` | `феечка` | позывной |
| `dialog.match_threshold` | 0.75 | схожесть для срабатывания (см. `trigger-test`) |
| `rvc.enabled` | False | включается флагом `--rvc` |
| `rvc.base_url` | `http://127.0.0.1:8081` | адрес RVC-сервиса |
| `rvc.voice` | `voicevox_speaker_43` | целевой голос (`--rvc-voice`) |
| `rvc.input_voice` | `irina` | поправка питча на входной голос; для irina она 0 |
| `rvc.pitch` / `rvc.formant_shift` | None | None — пресет сервиса (+8 и +1.0) |
| `dialog.window_s` | 60 | сколько отвечаем без позывного после своего ответа; по истечении история диалога очищается |
| `dialog.max_history` | 8 | реплик в контексте |

Профили STT: `--profile prod` (large-v3), `fast` (small), `cpu` (small на CPU).
Точечно: `--stt-model`, `--stt-device`, `--llm-url`, `--voice`, `--callsign`.

### Длина ответа

Одним флагом `--reply-length`, потому что крутить три ручки порознь бессмысленно:
системный промпт определяет, сколько модель напишет, `max_sentences` режет лишнее,
а `max_tokens` — лишь предохранитель (он рубит по счётчику и может оборвать на
полуслове). Пресет меняет все три согласованно.

| Пресет | Предложений | Эфир | |
|---|---|---|---|
| `short` | ≤ 2 | ~4–6 с | по умолчанию |
| `medium` | ≤ 4 | ~11 с | полноценный ответ, канал ещё не блокирует |
| `long` | ≤ 8 | ~35–40 с | **осторожно**, см. ниже |

```bash
.venv/bin/python main.py run --live --responder llm --reply-length medium
```

Длина — это не только вопрос вкуса: **во время своей передачи агент глух**. На `long`
он полминуты не услышит ни вызова, ни попытки его перебить, да и в общем канале такой
монолог робота — помеха. Плюс у многих раций есть таймаут передачи (TOT, обычно
30–60 с), который просто оборвёт её на середине фразы. Практический потолок — `medium`.

Замеры сделаны на Qwen3-4B; с другой моделью длительности сдвинутся, сам пресет
продолжит работать.

## Железо

- **RX:** выход динамика (ext-spk) рации → line-in компьютера.
- **TX:** line-out компьютера → микрофонный вход рации. Линейный уровень (~1 В) в
  микрофонный вход (~10 мВ) требует **аттенюатора ~40–50 дБ** (резистивный делитель).
- **PTT:** линия **TX** USB-UART в состоянии *break* коммутирует PTT рации. Сигнал
  **инвертирован** (передача при снятом break) — учтено флагом `invert`. Микроконтроллер
  не нужен. На разъёме USB-UART достаточно GND и TX.
- Рекомендуется **гальваническая развязка**: аудио — трансформаторы 600:600 Ом, PTT —
  оптрон. Это убирает фон 50 Гц и ВЧ-наводки при передаче.
- **Шумоподавитель рации держать закрытым** — тогда на вход приходит только реальный
  сигнал, и energy VAD работает надёжно.

## Структура проекта

```
ai_radio/
  config.py      # пороги, тайминги, устройства, модели, профили
  audio_io.py    # FileSource (ffmpeg) + Wav/Null sink + resample/normalize — stdlib
  live_io.py     # MicSource + SpeakerSink (sounddevice)  — numpy импортируется локально
  vad.py         # energy VAD + метрики (RMS/dBFS)
  calibrate.py   # подбор порога по файлу/микрофону + гистограмма
  ptt.py         # DummyPtt + TxdBreakPtt (break на TX, инверсия)
  responder.py   # Responder (шов) + ParrotResponder + LLMResponder
  repeater.py    # state machine (half-duplex, preroll/hangtime, пауза входа на ответ+TX)
  dialog.py      # нечёткий триггер по позывному + история и окно диалога — stdlib
  textnorm.py    # чистка текста перед TTS (числа, латиница, разметка, <think>)
  bench.py       # замер задержки по звеньям на реальной записи
  engines/
    base.py         # протоколы SttEngine / LlmEngine / TtsEngine
    stt_whisper.py  # faster-whisper (int8)
    llm_llamacpp.py # llama-server по HTTP (urllib, stdlib)
    tts_piper.py    # Piper + ресемплинг и нормализация уровня
    tts_rvc.py      # RVC-декоратор над TtsEngine (HTTP, stdlib)
main.py          # CLI: calibrate | run [--live] | bench | trigger-test | devices
tools/
  break_test.py  # проверка PTT-линии осциллографом
```

Тяжёлые библиотеки импортируются локально внутри движков, поэтому этап 1 (калибровка,
parrot, PTT) продолжает работать на машине без faster-whisper и piper.

## Дорожная карта

- Перенос на прод: там Pascal, и RVC пойдёт в **fp32** (fp16 на нём в 1/64 скорости).
  Переключение автоматическое — `device_config` в RVC ловит P102-100 по имени карты.
- Квитанция приёма («принял» в эфир до обдумывания) — если замеры на железе покажут,
  что пауза ощущается как потеря связи.
- Аппаратный сигнал приёма (COS) от рации через микроконтроллер — если понадобится
  отказаться от VAD в пользу точного детектора несущей.
