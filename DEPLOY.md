# Развёртывание на прод-машине

Чеклист для установки на машину, которая уедет на дачу. Объяснения — в README,
здесь только последовательность и то, что специфично для прода.

Машин две, и различаются они существенно:

| | **Дачная** (Dell OptiPlex 790) | **Домашняя** |
|---|---|---|
| Имя хоста | `u22-i7` | |
| CPU | i7-2600, 4/8, **без AVX2** | Xeon W-1250, 6/12, AVX2 есть |
| RAM | 8 ГБ | 31 ГБ |
| GPU | **Quadro P2000, 5 ГБ** | **P102-100, 10 ГБ** |
| compute capability | 6.1 (Pascal) | 6.1 (Pascal) |
| Драйвер | 580 (последняя ветка с Pascal) | 535.86.05 |
| ОС | Ubuntu 22.04, Python 3.10 | **Ubuntu 20.04, Python 3.8** |
| Диск | SSD 224 ГБ: `/` на 120 ГБ, свободно было 46 ГБ | 59 ГБ на `/`, **1.1 ТБ на `/mnt/2TB`** |
| Звук | Sound Blaster Live! 5.1 | — |
| Профиль STT | **`--profile vram5`** (turbo) | **`--profile vram10`** (large-v3) |
| RVC fp32 | **только флагом `--fp32`** | ловится автоматически |

Обе на Pascal, поэтому CUDA 12 и fp32 в RVC актуальны для обеих.

**Особенности дачной машины:**

- **RVC не догадается про fp32 сам.** `configs/config.py:device_config()` ищет в имени
  карты подстроки `16` / `P40` / `P10` / `1060` / `1070` / `1080`. `P102-100` попадает
  под «P10», а `Quadro P2000` — ни под одну, и `is_half` останется `True`. Запускать
  сервис с `--fp32`, проверять по `/health`, что там `"is_half": false`.
- **i7-2600 — Sandy Bridge: есть AVX, нет AVX2/FMA/F16C/BMI2.** Бинарь llama.cpp,
  собранный на dev-машине с дефолтным `GGML_NATIVE=ON` (то есть `-march=native`),
  здесь падает с SIGILL.
- **Ubuntu 22.04 — это glibc 2.35 и GLIBCXX 3.4.30**, заметно старше, чем на
  dev-машине. Бинарь, собранный там даже с правильными флагами, не запустится.
  Оба ограничения снимаются одной сборкой в контейнере — шаг 1.2.
- Встроенная Intel HD 2000 в BIOS отключена (в `lspci` её нет), так что монитор
  висит на P2000 и десктоп забирает VRAM. Отсюда headless: `systemctl set-default
  multi-user.target`, агент под `systemd --user` с `loginctl enable-linger`.

**Особенности домашней машины:**

- **Python 3.8 слишком стар** — нашему коду нужен 3.10+. Ставить venv через `uv`
  (`uv venv --python 3.10`), как на dev-машине; системный интерпретатор не трогать.
- **Драйвер 535 против CUDA 12.4**, с которой собран llama.cpp. По правилу
  minor-version compatibility этого достаточно (нужен ≥525), но проверить надо первым
  делом: `llama-server` должен подняться и ответить на `/health`.
- P102-100 — майнинговая карта: убедиться, что `nvidia-smi` её видит и что хватает
  питания и обдува.
- **Место ставить не на `/`**: корень занят наполовину (59 ГБ свободно из 123), а
  на `/mnt/2TB` больше терабайта. Туда же имеет смысл увести и кэш HuggingFace —
  `export HF_HOME=/mnt/2TB/…`, иначе модели Whisper осядут в `~/.cache` на корне.

**Ключевое отличие дачной машины от домашней:** интернет там есть **только во время
настройки**. Потом его не будет, поэтому всё, что скачивается «на лету», должно быть
скачано и проверено заранее — см. «Проверка офлайн-готовности».

---

## 1. Установка

Интернет на этом шаге нужен. Порядок — как в README, повторять здесь не буду:

1. **Агент** — venv на системном Python 3.10 (`uv` не нужен, 22.04 даёт 3.10 сам):
   ```bash
   sudo apt install ffmpeg libportaudio2 python3-venv
   git clone git@github.com:Tsar/AI_Radio_Agent.git && cd AI_Radio_Agent
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt -r requirements-ai.txt
   ```
2. **llama.cpp** — не собирать на i7-2600 (займёт часы), но и не копировать рабочий
   dev-билд: он собран под `-march=native` dev-процессора (на Sandy Bridge это SIGILL)
   и против её же libc. Оба ограничения снимает сборка **в контейнере с целевой
   Ubuntu**: считает по-прежнему dev-машина, то есть быстро, а ABI и набор инструкций
   получаются продовые.
   ```bash
   # Образ 20.04 даёт бинарь, который пойдёт и на домашней (20.04), и на дачной
   # (22.04): вниз по версиям glibc совместимость работает, вверх — нет.
   # Собирать можно на любой машине с docker — проверено на домашнем сервере.
   docker run --rm -e DEBIAN_FRONTEND=noninteractive -e TZ=Etc/UTC \
       -v $HOME/llama.cpp:/src -w /src \
       nvidia/cuda:12.4.1-devel-ubuntu20.04 bash -c '
     set -e
     apt-get update -qq && apt-get install -y -qq --no-install-recommends \
         curl ca-certificates git
     # cmake из 20.04 — 3.16, а ggml-cuda требует 3.18 (ради CMAKE_CUDA_ARCHITECTURES)
     curl -sL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
       | tar xz -C /opt
     export PATH=/opt/cmake-3.31.6-linux-x86_64/bin:$PATH
     cmake -S . -B build-prod -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
           -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF \
           -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF \
           -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
     cmake --build build-prod -j --target llama-server
     chown -R '"$(id -u):$(id -g)"' build-prod'
   rsync -a $HOME/llama.cpp/build-prod/bin/ прод:~/llama.cpp/build/bin/
   ```
   Четыре неочевидных места, каждое ломает сборку или бинарь:

   - **`DEBIAN_FRONTEND=noninteractive`** — без него `apt-get install cmake` тянет
     `tzdata`, тот через debconf спрашивает часовой пояс и ждёт ввода. Контейнер
     стоит намертво, снаружи выглядит как «сборка идёт», хотя нагрузки нет.
   - **cmake ставится бинарём, а не из apt.** В 20.04 он 3.16, а
     `ggml/src/ggml-cuda/CMakeLists.txt` требует 3.18 — ровно ради
     `CMAKE_CUDA_ARCHITECTURES`, которым мы задаём `sm_61`.
   - **`-DGGML_CUDA_NCCL=OFF`** — опция по умолчанию `ON`, и в образе NCCL есть,
     так что бинарь прилинкуется к `libnccl.so.2` (на dev-машине её нет, поэтому
     там проблема не проявляется). На проде такой библиотеки не будет, а нужна она
     только для multi-GPU.
   - **`-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`** — иначе в RUNPATH попадёт путь
     *внутри контейнера* (`/src/build-prod/bin`), и рядом лежащие `libllama*.so`
     не найдутся. С `$ORIGIN` они ищутся рядом с бинарём, и `LD_LIBRARY_PATH`
     нужен только для CUDA.

   `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` в контейнере не нужен: там gcc 9,
   а упирается nvcc 12.4 только в gcc новее 13-го.

   **Правильных флагов мало — важна ещё libc.** Сборка теми же флагами, но прямо на
   dev-машине, даёт бинарь, который на проде не стартует вовсе:
   ```
   version `GLIBC_2.38' not found       # на Ubuntu 22.04 стоит 2.35
   version `GLIBCXX_3.4.32' not found   # есть только 3.4.30
   version `CXXABI_1.3.15' not found    # есть только 1.3.13
   ```
   Флагами компилятора это не лечится, и glibc на 22.04 не обновить — отсюда контейнер.
   Симметрично и обратное: собранное под 20.04 идёт на 22.04, но не наоборот.

   **`-DGGML_BMI2=OFF` обязателен**, хотя его нет в списке «AVX2/FMA/F16C»: cmake
   включает `-mbmi2` независимо от них, а BMI2 появился только в Haswell — на Sandy
   Bridge был бы тот же SIGILL, просто на других инструкциях. Правильная строка в
   выводе cmake выглядит так (без `bmi2`):
   ```
   -- Adding CPU backend variant ggml-cpu: -msse4.2;-mavx GGML_SSE42;GGML_AVX
   ```
   Проверить готовый бинарь, не доверяя флагам, — все три счётчика должны быть по нулю:
   ```bash
   objdump -d build-prod/bin/libggml-cpu.so | grep -cE 'vfmadd|vfnmadd'          # FMA
   objdump -d build-prod/bin/libggml-cpu.so | grep -cE 'vpbroadcast|vperm2i128'  # AVX2
   objdump -d build-prod/bin/libggml-cpu.so | grep -cE '\b(pdep|pext|mulx|bzhi)\b'  # BMI2
   ```
   А заодно ABI — на **прод-машине**, после копирования; строк быть не должно:
   ```bash
   ldd ~/llama.cpp/build/bin/llama-server | grep -E "not found|GLIBC|GLIBCXX|CXXABI"
   ```
   У сборки в контейнере 20.04 максимальные требуемые версии — `GLIBC_2.14`
   и `GLIBCXX_3.4.21`, то есть с запасом ниже, чем на обеих прод-машинах.
   Финальная проверка — что карта видна:
   ```bash
   LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cuda_runtime/lib \
     ~/llama.cpp/build/bin/llama-server --list-devices
   # CUDA0: NVIDIA P102-100 (10150 MiB, 8915 MiB free)
   ```
   `61` вместо `61;75`: обе прод-машины на Pascal, Turing нужен только самой dev-машине.
   Готовых linux+CUDA бинарей у llama.cpp нет — CUDA-сборки публикуют только под Windows.

   **RUNPATH уезжает вместе с бинарём.** По умолчанию cmake прописывает туда
   абсолютный путь сборочного каталога — при сборке в контейнере это вообще
   `/src/build-prod/bin`, — и лежащие рядом `libllama*.so` / `libggml*.so` не
   находятся, хотя они в том же каталоге. Флаг `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`
   выше меняет это на `$ORIGIN`, проверять так:
   ```bash
   objdump -x build/bin/llama-server | grep RUNPATH   # ожидается: $ORIGIN
   ```
   Если собирали без него, каталог с бинарём должен идти в `LD_LIBRARY_PATH` юнита
   **первым**, до CUDA-путей (см. раздел 6).

   CUDA линкуется **динамически**, а `nvidia-cuda-toolkit` на прод ставить незачем —
   всё нужное приезжает pip-пакетами в venv агента. Нужны три библиотеки, и `ldd`
   показывает именно их:

   | Библиотека | Откуда |
   |---|---|
   | `libcublas.so.12` | `nvidia-cublas-cu12` (уже стоит ради faster-whisper) |
   | `libcublasLt.so.12` | оттуда же, тем же пакетом |
   | `libcudart.so.12` | `pip install nvidia-cuda-runtime-cu12` (~9 МБ) |

   Каталоги подставляются переменной окружения — проверено, что с ней все три
   резолвятся из venv, а не из системы:
   ```bash
   NV=~/AI_Radio_Agent/.venv/lib/python3.10/site-packages/nvidia
   LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cuda_runtime/lib" ldd build/bin/libggml-cuda.so \
     | grep -E 'cublas|cudart'
   ```
   В юните это `Environment=LD_LIBRARY_PATH=…`. Альтернатива — собрать с `-DGGML_STATIC=ON`.
3. **RVC-сервис** — форк, ветка `offline-http-inference`, свой venv на Python 3.10.
   Команды в README (не забыть `torch` из индекса `cu121` и `setuptools<81`). Если
   ставите обычным `pip`, а не `uv`, — сначала `pip install "pip<24.1"`, иначе
   `fairseq` не разрешится (подробности в README).
4. **Веса**: `Qwen3-4B-Q4_K_M.gguf`, голос Piper, модель Whisper, три `.pth` RVC,
   `hubert_base.pt` + `rmvpe.pt`. Голосовые `.pth` в git не лежат — перенести с dev.

### Сколько это займёт места

Замерено на dev-машине; на дачной машине на `/` было 46 ГБ свободных, то есть влезает
с запасом примерно в 29 ГБ. Учтите ещё pip-кэш: он разрастается до ~4 ГБ и после
установки его можно снести (`pip cache purge`), но на даче он же — единственный способ
переставить пакет без интернета.

| Компонент | Размер |
|---|---|
| **Агент — итого** | **6.8 ГБ** |
| код + `.git` | 3 МБ |
| venv (из них CUDA-библиотеки 2.3 ГБ) | 2.7 ГБ |
| `Qwen3-4B-Q4_K_M.gguf` | 2.4 ГБ |
| голос Piper | 61 МБ |
| Whisper `large-v3-turbo` в HF-кэше (дача) | 1.6 ГБ |
| Whisper `large-v3` (домашняя машина) | 2.9 ГБ |
| **RVC — итого** | **8.9 ГБ** |
| код + `.git` | 53 МБ |
| venv (torch 1.6 + CUDA 4.7 + triton 0.55) | 8.3 ГБ |
| `hubert_base.pt` + `rmvpe.pt` | 354 МБ |
| три голоса `.pth` | 159 МБ |
| **llama.cpp** (только `build/bin`) | **282 МБ** |
| **ВСЕГО** | **~16 ГБ** (дача) / ~17 ГБ (дом) |

**Форк RVC не копировать целиком!** В нём ~20 ГБ того, что на проде не нужно, и
наивный `rsync` забьёт диск:

| Каталог | Размер | Зачем он там |
|---|---|---|
| `logs/` | 15 ГБ | тренировочные данные |
| `export_onnx_new/` | 1.8 ГБ | ONNX-экспорты, мы их не используем |
| `assets/pretrained_v2/` | 1.3 ГБ | базовые модели для обучения |
| `assets/pretrained/` | 1.1 ГБ | то же |
| `assets/uvr5_weights/` | 595 МБ | разделение вокала |

Правильно: `git clone` ветки `offline-http-inference` + точечно `hubert_base.pt`,
`rmvpe.pt` и три `.pth`.

Из 17 ГБ семь приходятся на **две копии CUDA-библиотек** — по одной в каждом venv.
Это плата за изоляцию окружений (numpy 1.23 против 2.x), и платить её стоит: попытка
расшарить библиотеки между venv через `LD_LIBRARY_PATH` — ровно та хрупкость, из-за
которой окружения и разведены.

Если места не хватит, сокращать в таком порядке: `pip cache purge` (~4 ГБ) → `triton`
из venv RVC (555 МБ, нужен только для `torch.compile`) → задействовать `/dev/sda1`:
на дачном SSD рядом с корнем лежит несмонтированный ext4-раздел на 94 ГБ.

---

## 2. Проверка офлайн-готовности — только для дачной машины

Самая вероятная причина «на даче ничего не работает»: **faster-whisper скачивает
модель при первом запуске, а не при установке**. Машина, где всё поставили, но ни разу
не прогнали, на даче полезет в сеть и упадёт.

На домашнем сервере интернет остаётся, так что этот раздел для него необязателен —
но прогнать полный конвейер после установки всё равно стоит.

```bash
# 1) прогнать полный конвейер, чтобы всё догрузилось в кэш
.venv/bin/python main.py bench --in-file запись.mp3 --profile vram5 --threshold ПОРОГ

# 2) убедиться, что модель Whisper осела на диске (~1.6 ГБ).
#    Каталог НЕ «Systran»: large-v3-turbo резолвится в mobiuslabsgmbh — спросите
#    саму библиотеку, а не угадывайте имя
du -sh "$(.venv/bin/python -c 'from faster_whisper.utils import download_model as d; print(d("large-v3-turbo"))')"

# 3) отключить сеть и прогнать ещё раз — это и есть проверка
sudo ip link set <интерфейс> down
.venv/bin/python main.py bench --in-file запись.mp3 --profile vram5 --threshold ПОРОГ
sudo ip link set <интерфейс> up
```

Второй прогон должен отработать без единой ошибки. Если Whisper полезет в сеть —
скопировать каталог снапшота и указать путь явно через `--stt-model` (см. README,
раздел «Перенос на машину без интернета»).

Проверить offline и остальные два сервиса: `llama-server` и `infer-http-service.py`
ничего не качают, но убедиться, что они стартуют с выключенной сетью, стоит.

---

## 3. Железо

> **TODO:** осталось то, что требует тракта. Софтовая часть проверена целиком —
> `tools/headless_check.sh` даёт «Всё сошлось», в том числе после перезагрузки.

- [x] Видеокарта: `nvidia-smi` видит, `compute_cap` = 6.1, драйвер 580.173.02
      (CUDA 12; в 13 поддержки Pascal нет)
- [x] `llama-server` поднимается и отвечает на `/health` — проверка того, что
      бинарь, собранный с CUDA 12.4, дружит с установленным драйвером
- [x] ~~Монитор на встроенную Intel HD 2000~~ — **не выйдет**: в BIOS OptiPlex 790
      она отключается при установленной дискретной карте, в `lspci` её нет. Вместо
      этого машина уходит в `multi-user.target`, что освобождает те же 270–400 МиБ
- [ ] Звук виден агенту: `main.py devices` показывает железо, а не только
      `pulse`/`default`. Если нет — пользователь не в группе `audio`, см. раздел 6
- [ ] RX: выход динамика рации → line-in Sound Blaster
- [ ] TX: line-out → микрофонный вход рации через **аттенюатор 40–50 дБ**
- [ ] PTT: линия TX USB-UART, проверить осциллографом (`tools/break_test.py`)
- [ ] Гальваническая развязка: трансформаторы 600:600 на аудио, оптрон на PTT
- [ ] Шумоподавитель рации закрыт
- [ ] Питание: Quadro P2000 берёт 75 Вт от слота, доп. питание не нужно — штатному
      БП OptiPlex этого хватает (P102-100 с её 250 Вт потребовала бы замены БП, и
      именно на этой плате она вообще не завелась)

Sound Blaster Live! работает на 48 кГц, поэтому всем командам нужен `--device-rate 48000`.
Индексы устройств — `main.py devices`.

---

## 4. Калибровка

> **TODO:** заполнить измеренными значениями.

```bash
.venv/bin/python main.py calibrate --live --in-device N --device-rate 48000 --seconds 20
```

Молчать первые ~4 с (замер фона), затем дать реальную передачу из эфира.

| Параметр | Значение | Примечание |
|---|---|---|
| шумовой пол | _TODO_ dBFS | при закрытом шумодаве |
| уровень сигнала | _TODO_ dBFS | |
| **порог** | _TODO_ | подставлять в `--threshold` |

Ориентир из README (запись Optim-778 через line-in): фон −60.6 → порог −48.6 dBFS
(`threshold 0.00369`). Пороги, снятые с микрофона на dev-машине, здесь не годятся.

---

## 5. Проверка по нарастающей

Всё, что можно проверить без рации, собрано в один скрипт — он же годится как
регрессия после любой правки конфигурации:

```bash
tools/headless_check.sh
# на домашней машине: PROFILE=vram10 tools/headless_check.sh
```

Пять блоков: свободна ли видеопамять, поднимаются ли оба сервиса (с проверкой
`is_half: false` и того, что прогрев отработал), раскладка VRAM по процессам,
полный `bench` — и отдельно, **работал ли RVC или ответ втихую ушёл голосом
Piper**, — и наконец Whisper с `HF_HUB_OFFLINE=1`. Выход 0 или 1, провалы
перечислены в конце. Вывод `bench` печатается целиком: он единственный источник
причины, когда что-то падает.

Живой режим можно обкатать **до сборки тракта**, на любом USB-микрофоне с 48 кГц:
частота та же, что у Sound Blaster, значит `--device-rate 48000` и логика выбора
устройств проверяются заранее, а остаётся только сам аналоговый тракт. PTT при этом
держим заглушкой, слушаем ответ в наушниках (не в колонках — иначе агент услышит сам
себя).

Дальше — то, что требует железа. Каждый шаг только после успешного предыдущего.

1. `main.py trigger-test "феечка приём" "привет всем"` — без моделей, мгновенно
2. `main.py bench --in-file запись.mp3 --profile vram5 --threshold ПОРОГ` — реальные
   времена. Замер на дачной машине, headless, ответ ~7 с эфира:

   | Звено | Время |
   |---|---|
   | STT (`large-v3-turbo`, int8, GPU) | 1.26 с |
   | LLM (Qwen3-4B Q4_K_M, ctx 2048) | 1.54 с |
   | TTS: Piper | ~0.8 с |
   | TTS: RVC **прогретый** | ~3.2 с |
   | **итого** | **~6.8 с** при бюджете 10 |

   **Гнать `bench` только после прогрева RVC** — иначе он померит холодный вызов и
   покажет 10.9 с, то есть провал бюджета на ровном месте. Юнит греет сам
   (`ExecStartPost`), а при ручном запуске сервиса прогрейте руками:
   ```bash
   .venv/bin/python tools/rvc_warmup.py
   ```
   Piper из оценки можно вычеркнуть: на i7-2600 он даёт 0.41–0.43 с на фразу
   (RTF 0.10–0.17), отсутствие AVX2 ему не мешает. Медленное звено — RVC в fp32.
3. `main.py run --live --responder llm --profile vram5 --device-rate 48000
   --threshold ПОРОГ` — **PTT остаётся заглушкой**, эфир не жмётся, ответ слышен
   в динамике
4. То же с `--rvc` (сначала поднять RVC-сервис — **с `--fp32`**, см. выше)
5. Боевой PTT: добавить `--ptt txdbreak --port /dev/ttyUSB0`

---

## 6. Автозапуск (systemd)

Три сервиса. Агенту нужен доступ к звуковой карте, поэтому — **user-юниты**, а не
системные: так он работает со своим пользовательским аудио-стеком (PipeWire), а не
с ALSA напрямую. Чтобы они поднимались без входа в систему:

```bash
sudo loginctl enable-linger $USER
sudo usermod -aG dialout $USER      # /dev/ttyUSB0 для PTT
sudo usermod -aG audio $USER        # /dev/snd/* — без этого звука не будет вовсе
mkdir -p ~/.config/systemd/user
```

**Группа `audio` не формальность.** Доступ к звуковым картам раздаёт logind, и
раздаёт его ACL'ом пользователю *активного места*:

```
$ getfacl -p /dev/snd/controlC2
user:кто-сидит-за-монитором:rw-
other::---
```

Агент за монитором не сидит: он приходит по ssh или поднимается через `linger`, а
таким сессиям места не назначается и ACL не выписывается. Остаётся только группа.
Симптом при этом обманчивый — `pipewire` и его менеджер сессии **работают**, просто
не видят ни одного устройства:

```bash
pw-cli list-objects Device | grep -c device.name   # 0 при четырёх картах в системе
arecord -l                                         # «не найдено ни одной звуковой карты»
main.py devices                                    # только pulse и default, без железа
```

Изменение группы подхватывается новым входом — проще перелогиниться или
перезагрузиться, `systemctl --user restart pipewire` тут не поможет.

**И наоборот: PulseAudio с PipeWire у этого пользователя надо погасить.** Агент ходит
в карту напрямую через ALSA (`--in-device` по имени), а звуковой сервер, если он
запущен, открывает карту первым и держит её. Признак — `arecord -l` показывает
`Подустройства: 0/1` вместо `1/1`, и `main.py devices` просто не перечисляет карту:
PortAudio пропускает устройства, которые не может открыть. На Ubuntu 22.04 у
пользователя по умолчанию включены **оба** сервера сразу, и они ещё и мешают друг
другу — PipeWire остаётся с нулём устройств, а карты забирает PulseAudio.

```bash
systemctl --user mask --now pulseaudio.socket pulseaudio.service \
    pipewire.socket pipewire.service pipewire-media-session.service
printf 'autospawn = no\ndaemon-binary = /bin/true\n' > ~/.config/pulse/client.conf
```

`mask`, а не `disable`: сокет иначе поднимет сервис по первому обращению, а
`autospawn = no` закрывает последнюю лазейку — libpulse запускает демон сам.
Обратно — `systemctl --user unmask` тем же списком.

Отсюда же и `After=`/`Wants=` в юните агента: там **нет** `pipewire.service`, и это
намеренно — ссылка на замаскированный юнит только вводила бы в заблуждение.

Юниты ставятся **тому пользователю, под кем крутится агент**: `systemctl --user`
смотрит на свой `$XDG_RUNTIME_DIR`, и юниты, положенные другому, просто не увидятся.
Графическая сессия при этом не нужна и вредна — лишний PipeWire делил бы ту же
звуковую карту, а десктоп съедает VRAM. Отсюда `sudo systemctl set-default
multi-user.target`.

`~/.config/systemd/user/ai-radio-llm.service`:

```ini
[Unit]
Description=AI Radio — LLM (llama-server)
After=network.target

[Service]
WorkingDirectory=%h/AI_Radio_Agent
# Только CUDA: системного toolkit на проде нет, библиотеки берём из venv агента.
# Соседние libllama*.so / libggml*.so сюда не нужны — RUNPATH собран с $ORIGIN.
Environment=LD_LIBRARY_PATH=%h/AI_Radio_Agent/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:%h/AI_Radio_Agent/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
# -c 2048 + KV q8_0, а не -c 4096: в 5 ГБ иначе не влезает (см. раздел 8) —
# 3100 МБ на llama-server + 1231 на turbo + 592 на RVC это уже 4.9 ГБ до
# накладных расходов драйвера. На домашней машине можно и -c 4096.
ExecStart=%h/llama.cpp/build/bin/llama-server -m models/llm/Qwen3-4B-Q4_K_M.gguf \
          -ngl 99 -c 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
          --host 127.0.0.1 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/ai-radio-rvc.service`:

```ini
[Unit]
Description=AI Radio — voice conversion (RVC)
After=network.target

[Service]
WorkingDirectory=%h/Retrieval-based-Voice-Conversion-WebUI
# Стек идёт впритык в 5 ГБ: без этого пик RVC 942-972 МиБ вместо 720-744 (раздел 8)
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --fp32 обязателен на Quadro P2000: авто-детект Pascal её не ловит (см. выше),
# и сервис ушёл бы в fp16, то есть в 1/64 скорости. На P102-100 флаг не нужен.
ExecStart=%h/Retrieval-based-Voice-Conversion-WebUI/.venv/bin/python \
          infer-http-service.py --port 8081 --fp32
# Прогрев: первая конвертация после старта на ~7 с дороже остальных, и без него
# этот штраф достаётся первому вызвавшему в эфире. «-» — отказ прогрева не должен
# ронять сервис. Юнит агента ждёт конца ExecStartPost, то есть видит RVC прогретым.
ExecStartPost=-%h/AI_Radio_Agent/.venv/bin/python %h/AI_Radio_Agent/tools/rvc_warmup.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/ai-radio-agent.service` — ждёт, пока поднимутся оба.
Порог, профиль и остальные флаги вынесены в файл окружения: калибровка и переход
на боевой PTT не должны требовать правки юнита.

```ini
[Unit]
Description=AI Radio — agent
# Звуковых серверов в зависимостях нет намеренно: карта берётся напрямую
# через ALSA, а PulseAudio и PipeWire у этого пользователя замаскированы
After=ai-radio-llm.service ai-radio-rvc.service
Wants=ai-radio-llm.service ai-radio-rvc.service

[Service]
WorkingDirectory=%h/AI_Radio_Agent
EnvironmentFile=%h/.config/ai-radio/agent.env
ExecStartPre=/usr/bin/curl -s --retry 60 --retry-delay 2 --retry-connrefused \
             --retry-all-errors -o /dev/null http://127.0.0.1:8080/health
ExecStart=%h/AI_Radio_Agent/.venv/bin/python -u main.py run --live --responder llm --rvc \
          --profile ${PROFILE} --threshold ${THRESHOLD} --reply-length ${REPLY_LENGTH} \
          $EXTRA_ARGS
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

`~/.config/ai-radio/agent.env`:

```ini
THRESHOLD=0.0012        # снять своим calibrate, значение здесь — только ориентир
PROFILE=vram5
REPLY_LENGTH=short
EXTRA_ARGS=             # пусто — PTT заглушка и звук через PipeWire по умолчанию
```

Боевой режим — дописать в `EXTRA_ARGS`: `--ptt txdbreak --port /dev/ttyUSB0`, плюс
`--device-rate 48000`, если работаете с картой напрямую, мимо PipeWire.

**`$EXTRA_ARGS` systemd режет по пробелам, и кавычки этому не мешают.** Имя
устройства из двух слов развалится на два аргумента, поэтому в `--in-device`
писать одно слово: подстроку имени (`Rx`, `Live`) или индекс из `main.py devices`.
Разница именно в записи: `${PROFILE}` подставляется одним аргументом, `$EXTRA_ARGS`
разбивается — на то и рассчитано, иначе туда нельзя было бы положить несколько
флагов сразу.

```bash
systemctl --user daemon-reload
systemctl --user enable --now ai-radio-llm ai-radio-rvc ai-radio-agent
systemctl --user status ai-radio-agent
journalctl --user -u ai-radio-agent -f
```

`-u` у python обязателен: без него вывод буферизуется и в журнале ничего не видно
до самого конца.

Агент переживает отказ соседей: без llama-server он молчит, без RVC — передаёт голосом
Piper. Так что при перезапуске одного из сервисов эфир не ломается.

---

## 7. Диагностика

| Симптом | Причина и что делать |
|---|---|
| `libcublas.so.12 is not found` | CUDA-библиотеки не подгрузились. Проверить, что стоят `nvidia-cublas-cu12` и `nvidia-cudnn-cu12`; их подхватывает `preload_cuda_libs()` в `engines/stt_whisper.py` |
| torch не видит карту, `sm_61 is not compatible` | приехала сборка под CUDA 13. Переставить с `--index-url https://download.pytorch.org/whl/cu121` |
| `ModuleNotFoundError: pkg_resources` | setuptools 81+. `pip install "setuptools<81"` |
| Whisper молчит или несёт чушь на реальном эфире | не включён ли `vad_filter` — на узкой полосе рации он вырезает речь. Должен быть `False` |
| Агент не реагирует на позывной | посмотреть строку `[STT]` в логе, скормить распознанный текст в `trigger-test` и глянуть score |
| Фразы не ловятся вообще | порог VAD велик, перекалибровать; проверить `--device-rate 48000` |
| Ответ звучит голосом Piper вместо RVC | RVC-сервис лежит, в логе есть `[warn] RVC недоступен` |
| Первый ответ после долгой тишины идёт 10+ с | модель выгрузилась из VRAM — проверить, что `llama-server` не перезапускался |
| Речь в эфире тихая или перемодулированная | аттенюатор, `tts.peak_dbfs`, уровень line-out в микшере |
| `CUDA failed with error out of memory` на дачной машине | первым делом — не поднялась ли графика: GNOME на P2000 берёт 270–400 МиБ, и без headless стек не влезает. Дальше: `--profile vram5` (не `vram10`), `rvc.f0_method=pm` (не `rmvpe`), `llama-server` с `-c 2048` и KV `q8_0` (не `-c 4096`), `PYTORCH_CUDA_ALLOC_CONF` в юните RVC |
| OOM выпадает то на RVC, то на STT | это одна и та же нехватка, просто падает тот, кто выделяет память последним. С прогревом RVC занимает свои ~726 МиБ сразу, и упирается уже агент с Whisper. Не ищите причину в том звене, чьё имя в трейсе |
| `llama-server` падает с `Illegal instruction` (SIGILL) | бинарь собран под чужой процессор. i7-2600 — Sandy Bridge: нет AVX2, FMA, F16C **и BMI2**. Пересобрать, как в шаге 1.2, и проверить `objdump`-ом |
| `llama-server` не стартует: `libcudart.so.12: cannot open shared object file` | не установлен `nvidia-cuda-runtime-cu12` либо не задан `LD_LIBRARY_PATH` на каталоги `nvidia/*/lib` в venv агента |
| `llama-server` не стартует: `libllama-server-impl.so: cannot open shared object file` | файл лежит рядом с бинарём, но RUNPATH ведёт на каталог сборочной машины. Добавить `%h/llama.cpp/build/bin` первым в `LD_LIBRARY_PATH` |
| `version 'GLIBC_2.38' not found` / `GLIBCXX_3.4.32` / `CXXABI_1.3.15` | бинарь собран против более новой libc, чем на проде. Флагами компилятора не лечится — пересобрать в контейнере с целевой Ubuntu, см. шаг 1.2 |
| RVC отвечает, но конвертация идёт десятки секунд | сервис поднялся в fp16. На `Quadro P2000` авто-детект Pascal не срабатывает — нужен `--fp32`. Проверить `curl -s 127.0.0.1:8081/health`: должно быть `"is_half": false` |
| Первая фраза после старта RVC идёт ~8 с, дальше ~3 | не отработал прогрев. Смотреть `journalctl --user -u ai-radio-rvc \| grep warmup`; вручную — `tools/rvc_warmup.py` |
| `ошибка: голос Piper не найден: models/piper/…` | путь в конфиге относительный. Движок ищет голос ещё и от корня проекта, так что это значит, что файла нет — скачайте его (команда в README) или укажите `--voice` |
| `ResolutionImpossible` / `omegaconf` при установке RVC | pip 24.1+ отбрасывает колёса omegaconf 2.0.x из-за битых метаданных. `pip install "pip<24.1"` и повторить |
| Whisper лезет в сеть на даче, хотя кэш «на месте» | каталог кэша для turbo — `models--mobiuslabsgmbh--…`, а не `Systran`. Проверять путём, который отдаёт сама библиотека (см. раздел 2) |

---

## 8. Видеопамять: что во что влезает

Замерено на RTX 2070 (пик, `int8`), на Pascal значения те же:

| Компонент | VRAM |
|---|---|
| Whisper `large-v3-turbo` (`--profile vram5`) | 1231 МБ |
| Whisper `large-v3` (`--profile vram10`) | 2027 МБ |
| Whisper `small` (`--profile small`) | 465 МБ |
| Qwen3-4B Q4_K_M, ctx 2048, KV `q8_0` | 2674 МБ |
| RVC fp32 с `f0_method=pm` | 592 МБ |
| RVC fp32 с `f0_method=rmvpe` | 926 МБ |

**Замер на самой дачной машине (P2000, fp32, всё поднято одновременно):**

| Процесс | VRAM |
|---|---|
| `llama-server`, ctx 2048, KV `q8_0` | 2674 МиБ |
| Whisper `large-v3-turbo` в агенте | 1126 МиБ |
| RVC, `f0_method=pm`, **fp32** | **942 МиБ** |
| GNOME на той же карте | 277 МиБ |
| **итого** | **5019 МиБ из 5120** |

Так стек **не работает**: RVC падает с `torch.OutOfMemoryError` на первой же
конвертации, не сумев выделить 42 МиБ. Агент при этом не ломается — передаёт голосом
Piper, — но RVC не работает вовсе.

Два уточнения к цифрам выше, обе не в нашу пользу. **RVC в fp32 берёт 942 МиБ, а не
592**: 592 — это замер fp16 с dev-машины, а на Pascal fp16 неприменим (см. `--fp32`).
И **десктоп надо считать**: 277 МиБ уходят на GNOME, потому что монитор висит на
P2000 — встроенная Intel HD 2000 в BIOS OptiPlex отключена. Отсюда headless как
обязательный шаг, а не как гигиена: без него не хватает.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` для RVC отыгрывает ~230 МиБ** и
нужен в любом случае: с ним пик RVC 720–744 МиБ вместо 942–972. Стоит прямо в юните
(раздел 6). Но **одного его мало**: с живым десктопом стек всё равно падает, не сумев
выделить 68 МиБ. Нужны оба рычага — и переменная, и headless.

Так и вышло на деле. Замер после выключения графики, всё поднято:

| | |
|---|---|
| в простое, до запуска сервисов | 11 МиБ |
| `llama-server` | 2670 МиБ |
| RVC (в простое 180, под нагрузкой) | ~740 МиБ |
| Whisper turbo в агенте | 1126 МиБ |
| **на пике** | **~4540 из 5120** |

Запас около 580 МиБ, `torch.OutOfMemoryError` больше не воспроизводится.

Полезное свойство: **пик RVC не растёт с длиной фразы** — замер на 3.5, 6.9, 13.8 и
20.9 с дал одинаковые 680–720 МиБ, сервис режет вход на куски. То есть длинные ответы
(`--reply-length medium`) память не ломают, они стоят только времени: конвертация идёт
примерно 0.3–0.45 от длительности фразы.

Если и этого не хватит — RVC целиком на процессор, `Environment=CUDA_VISIBLE_DEVICES=`
в том же drop-in: освобождает всю его долю ценой заметно более долгой конвертации.

Полная `large-v3` вместо turbo не влезает тем более, `rmvpe` вместо `pm` — тоже.

**Домашняя (10 ГБ):** влезает `--profile vram10` с `large-v3` (4.7 ГБ), остаётся место
и под модель LLM покрупнее — например Qwen3-8B Q4_K_M вместо 4B.

Два решения, которые сделали дачный вариант возможным: **turbo вместо large-v3**
(качество на записи с рации то же, памяти вдвое меньше) и **`pm` вместо `rmvpe`**
(на вход RVC идёт чистый синтез Piper, а не шумный эфир, так что нейросетевой
экстрактор F0 не нужен — на слух не отличается, а 335 МБ освобождаются).

## 9. Перед отъездом

- [x] Конвейер без сети проходит без ошибок: `tools/headless_check.sh` — «Всё
      сошлось». Осталось повторить с рацией, когда будет собран тракт
- [x] `llama-server` и RVC стартуют после `reboot` сами, без входа в систему
      (`linger`). Агент добавится, когда появится порог для его юнита
- [ ] Боевой PTT проверен, эфир слышен на второй рации
- [ ] Порог VAD снят на реальной установке и прописан в юните
- [x] Свободное место: нужно ~17 ГБ (разбивка выше), на дачной машине было 46 ГБ
- [ ] `journalctl --user -u ai-radio-agent` пишется и читается

**Отдельно про выключение графики.** Машина живёт в `multi-user.target`, и это не
косметика: без этого не хватает видеопамяти (раздел 8). Переключать её стоит
последним шагом настройки — сначала всё, что требует сети, потом headless. Иначе
за каждой недокачанной мелочью придётся возвращать графику и перезагружаться.
