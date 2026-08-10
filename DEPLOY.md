# Развёртывание на прод-машине

Чеклист для установки на машину, которая уедет на дачу. Объяснения — в README,
здесь только последовательность и то, что специфично для прода.

Машин две, и различаются они существенно:

| | **Дачная** (Dell OptiPlex 790) | **Домашняя** |
|---|---|---|
| CPU | i7-2600, 4/8, **без AVX2** | Xeon W-1250, 6/12, AVX2 есть |
| RAM | 8 ГБ | 31 ГБ |
| GPU | **Quadro P2000, 5 ГБ** | **P102-100, 10 ГБ** |
| compute capability | 6.1 (Pascal) | 6.1 (Pascal) |
| Драйвер | ставится при настройке | 535.86.05 |
| ОС | Ubuntu 22.04, Python 3.10 | **Ubuntu 20.04, Python 3.8** |
| Диск свободно | ~30 ГБ на SSD (+ три HDD) | 59 ГБ на `/`, **1.1 ТБ на `/mnt/2TB`** |
| Звук | Sound Blaster Live! 5.1 | — |
| Профиль STT | **`--profile vram5`** (turbo) | **`--profile vram10`** (large-v3) |

Обе на Pascal, поэтому CUDA 12 и fp32 в RVC актуальны для обеих.

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
2. **llama.cpp** — не собирать на i7-2600 (займёт часы). Скопировать с dev-машины
   каталог `build/bin/` целиком: бинарь собран с `sm_61;sm_75` и пойдёт на обеих картах.
   Если всё же собирать — команды в README, нужен `nvidia-cuda-toolkit` и `g++-13`.
3. **RVC-сервис** — форк, ветка `offline-http-inference`, свой venv на Python 3.10.
   Команды в README (не забыть `torch` из индекса `cu121` и `setuptools<81`).
4. **Веса**: `Qwen3-4B-Q4_K_M.gguf`, голос Piper, модель Whisper, три `.pth` RVC,
   `hubert_base.pt` + `rmvpe.pt`. Голосовые `.pth` в git не лежат — перенести с dev.

### Сколько это займёт места

Замерено на dev-машине; на проде было ~30 ГиБ свободных, то есть влезает с запасом
примерно в 13 ГБ.

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

Если места не хватит, сокращать в таком порядке: `triton` из venv RVC (555 МБ, нужен
только для `torch.compile`) → перенести веса и venv'ы на HDD, их в OptiPlex три штуки.

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

# 2) убедиться, что модель Whisper осела на диске
du -sh ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3-turbo

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

> **TODO:** заполнить по факту, когда придёт видеокарта и будет собран тракт.

- [ ] Видеокарта: `nvidia-smi` видит, `compute_cap` = 6.1, драйвер ветки ≤580
      (CUDA 12; в 13 поддержки Pascal нет)
- [ ] `llama-server` поднимается и отвечает на `/health` — проверка того, что
      бинарь, собранный с CUDA 12.4, дружит с установленным драйвером
- [ ] Монитор — на встроенную Intel HD 2000, чтобы вся VRAM осталась под модели
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

Каждый шаг — только после успешного предыдущего.

1. `main.py trigger-test "феечка приём" "привет всем"` — без моделей, мгновенно
2. `main.py bench --in-file запись.mp3 --profile vram5 --threshold ПОРОГ` — реальные
   времена на этом железе. Ожидание для дачи: **~2 с** на фразу (на dev-машине с
   тем же профилем вышло 0.93 с, P2000 примерно вдвое медленнее 2070, плюс Piper
   на i7-2600 без AVX2). Бюджет 10 с.
3. `main.py run --live --responder llm --device-rate 48000 --threshold ПОРОГ` —
   **PTT остаётся заглушкой**, эфир не жмётся, ответ слышен в динамике
4. То же с `--rvc` (сначала поднять RVC-сервис)
5. Боевой PTT: добавить `--ptt txdbreak --port /dev/ttyUSB0`

---

## 6. Автозапуск (systemd)

Три сервиса. Агенту нужен доступ к звуковой карте, поэтому — **user-юниты**, а не
системные: так он работает с тем же аудио-стеком, что и залогиненный пользователь.
Чтобы они поднимались без входа в систему:

```bash
sudo loginctl enable-linger $USER
mkdir -p ~/.config/systemd/user
```

`~/.config/systemd/user/ai-radio-llm.service`:

```ini
[Unit]
Description=AI Radio — LLM (llama-server)
After=network.target

[Service]
WorkingDirectory=%h/AI_Radio_Agent
ExecStart=%h/llama.cpp/build/bin/llama-server -m models/llm/Qwen3-4B-Q4_K_M.gguf \
          -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
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
ExecStart=%h/Retrieval-based-Voice-Conversion-WebUI/.venv/bin/python \
          infer-http-service.py --port 8081
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/ai-radio-agent.service` — ждёт, пока поднимутся оба:

```ini
[Unit]
Description=AI Radio — agent
After=ai-radio-llm.service ai-radio-rvc.service sound.target
Wants=ai-radio-llm.service ai-radio-rvc.service

[Service]
WorkingDirectory=%h/AI_Radio_Agent
ExecStartPre=/usr/bin/curl -s --retry 60 --retry-delay 2 --retry-connrefused \
             --retry-all-errors -o /dev/null http://127.0.0.1:8080/health
ExecStart=%h/AI_Radio_Agent/.venv/bin/python -u main.py run --live --responder llm --rvc \
          --profile vram5 --device-rate 48000 --threshold ПОРОГ \
          --ptt txdbreak --port /dev/ttyUSB0
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

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
| `CUDA failed with error out of memory` на дачной машине | в 5 ГБ P2000 стек влезает только с `--profile vram5` (turbo) и `rvc.f0_method=pm`; проверить, что не включён `vram10` или `rmvpe` |

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

**Дачная машина (5 ГБ):** turbo + 4B + RVC(pm) = **3.9 ГБ**, запас ~1.2 ГБ. Полная
`large-v3` вместо turbo уже не влезает (5.3 ГБ), `rmvpe` вместо `pm` — впритык.

**Домашняя (10 ГБ):** влезает `--profile vram10` с `large-v3` (4.7 ГБ), остаётся место
и под модель LLM покрупнее — например Qwen3-8B Q4_K_M вместо 4B.

Два решения, которые сделали дачный вариант возможным: **turbo вместо large-v3**
(качество на записи с рации то же, памяти вдвое меньше) и **`pm` вместо `rmvpe`**
(на вход RVC идёт чистый синтез Piper, а не шумный эфир, так что нейросетевой
экстрактор F0 не нужен — на слух не отличается, а 335 МБ освобождаются).

## 9. Перед отъездом

- [ ] Полный прогон **с выключенной сетью** проходит без ошибок
- [ ] Все три сервиса стартуют после `reboot` сами (проверить перезагрузкой)
- [ ] Боевой PTT проверен, эфир слышен на второй рации
- [ ] Порог VAD снят на реальной установке и прописан в юните
- [ ] Свободное место: нужно ~17 ГБ (разбивка выше), на прод-машине было ~30 ГиБ
- [ ] `journalctl --user -u ai-radio-agent` пишется и читается
