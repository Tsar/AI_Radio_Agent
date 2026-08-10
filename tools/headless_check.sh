#!/usr/bin/env bash
# Автономная проверка стека на прод-машине: сеть не нужна, вопросов не задаёт.
#
#     tools/headless_check.sh
#
# Гоняет то, что нельзя проверить с dev-машины: влезает ли стек в видеопамять,
# работает ли RVC (а не молчаливый откат на Piper), укладывается ли конвейер
# в бюджет и не полезет ли Whisper в сеть. Выход 0 — всё сошлось, 1 — есть
# провалы, они перечислены в конце.
#
# Профиль и порог можно переопределить через окружение:
#     PROFILE=vram10 THRESHOLD=0.004 tools/headless_check.sh
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Путь к голосу Piper в конфиге относительный, да и bench ищет модели от корня
cd "$ROOT" || exit 1

PROFILE=${PROFILE:-vram5}
THRESHOLD=${THRESHOLD:-0.00369}   # из calibrate на прилагаемой записи с Optim-778
BUDGET=${BUDGET:-10}
# Позывного в тестовой записи нет — подменяем словом оттуда, иначе LLM и TTS не
# отработают и в замер попадёт один STT. «окончание» слышат одинаково и turbo,
# и small (turbo даёт ещё «записываем», small — нет).
CALLSIGN=${CALLSIGN:-окончание}

FAILS=()
ok()    { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()   { printf '  \033[31m✗\033[0m %s\n' "$1"; FAILS+=("$1"); }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

vram()  { nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits; }
total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits; }

TOTAL=$(total) || { echo "nvidia-smi не отвечает — драйвер не установлен?"; exit 1; }

head_ "1. Графика погашена, видеопамять свободна"
if pgrep -x gnome-shell >/dev/null; then
    bad "gnome-shell жив — десктоп займёт ~270 МиБ, которых стеку не хватит"
else
    ok "gnome-shell не запущен"
fi
# Гасим свои сервисы, иначе замер «карта пуста» соврёт, когда они на автостарте
systemctl --user stop ai-radio-llm ai-radio-rvc 2>/dev/null
sleep 3
BASE=$(vram)
echo "     в простое: ${BASE} МиБ из ${TOTAL}"
[ "$BASE" -lt 100 ] && ok "карта свободна" || bad "в простое занято ${BASE} МиБ"

head_ "2. Поднимаем сервисы"
# RVC стартует дольше: ExecStartPost прогревает его тремя фразами
systemctl --user restart ai-radio-llm ai-radio-rvc
for _ in $(seq 1 60); do
    curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"' && break
    sleep 5
done
curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"' \
    && ok "llama-server отвечает" \
    || bad "llama-server не поднялся (journalctl --user -u ai-radio-llm)"

for _ in $(seq 1 40); do
    curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
H=$(curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null)
echo "     $H"
echo "$H" | grep -q '"status":"ok"' \
    && ok "RVC отвечает" || bad "RVC не поднялся (journalctl --user -u ai-radio-rvc)"
echo "$H" | grep -q '"is_half": *false' \
    && ok "RVC в fp32" || bad "RVC в fp16 — нужен --fp32, на Pascal это 1/64 скорости"
journalctl --user -u ai-radio-rvc --since "5 minutes ago" --no-pager 2>/dev/null \
    | grep -q "warmup] готово" \
    && ok "RVC прогрет" \
    || bad "прогрев не отработал — первая фраза будет на ~7 с дольше (tools/rvc_warmup.py)"

head_ "3. Видеопамять с поднятыми сервисами"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/     /'
USED=$(vram); echo "     занято ${USED} из ${TOTAL} МиБ, свободно $((TOTAL-USED))"

head_ "4. Полный конвейер (bench)"
REC=$(ls "$ROOT"/*.mp3 2>/dev/null | head -1)
if [ -z "$REC" ]; then
    bad "не нашёл тестовую запись *.mp3 в $ROOT"
else
    echo "     запись: $(basename "$REC")"
    echo "     профиль: $PROFILE, порог: $THRESHOLD, позывной: $CALLSIGN"
    OUT=$(.venv/bin/python main.py bench --in-file "$REC" --profile "$PROFILE" \
            --threshold "$THRESHOLD" --rvc --callsign "$CALLSIGN" --budget "$BUDGET" 2>&1 \
          | grep -viE "onnxruntime")
    # Печатаем всё: фильтр по строкам прятал бы сообщение об ошибке, и провал
    # выглядел бы как пустая таблица без объяснения
    echo "$OUT" | sed 's/^/     /'
    echo
    # Сначала убеждаемся, что прогон вообще дошёл до таблицы: если bench упал
    # раньше (например, по видеопамяти), то отсутствие строки «RVC недоступен»
    # означает не успех, а что до RVC дело не дошло
    if ! echo "$OUT" | grep -q "бюджет ${BUDGET} с"; then
        bad "bench не отработал — причина в его выводе выше"
    else
        echo "$OUT" | grep -q "RVC недоступен" \
            && bad "RVC отвалился, ответ ушёл голосом Piper (journalctl --user -u ai-radio-rvc)" \
            || ok "RVC отработал, переозвучка учтена в строке TTS"
        echo "$OUT" | grep -q "укладываемся" \
            && ok "уложились в бюджет ${BUDGET} с" || bad "не уложились в бюджет ${BUDGET} с"
    fi
fi
echo "     после прогона занято $(vram) МиБ"

head_ "5. Офлайн: сеть не нужна"
# Whisper качает модель при первом запуске, а не при установке. HF_HUB_OFFLINE
# заставит библиотеку упасть, если она всё же соберётся в сеть.
if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c "
from faster_whisper.utils import download_model
import sys; print('   снапшот:', download_model(sys.argv[1]))
" "$(.venv/bin/python -c "
from ai_radio.config import PROFILES; print(PROFILES['$PROFILE']['model'])")" 2>&1 \
     | grep -viE "onnxruntime|warning"; then
    ok "модель Whisper берётся из кэша"
else
    bad "Whisper полез бы в сеть — модели нет в кэше"
fi

head_ "ИТОГ"
if [ ${#FAILS[@]} -eq 0 ]; then
    printf '  \033[32mВсё сошлось.\033[0m\n'
    exit 0
fi
printf '  \033[31mПровалов: %s\033[0m\n' "${#FAILS[@]}"
for f in "${FAILS[@]}"; do echo "   - $f"; done
exit 1
