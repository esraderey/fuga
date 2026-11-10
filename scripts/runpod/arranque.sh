#!/usr/bin/env bash
# Arranque del entrenamiento final de FUGA en un pod de RunPod (pensado para una H100 de 80 GB).
#
# Dentro del pod, con la variable HF_TOKEN definida al crearlo (token de Hugging Face con escritura):
#   cd /workspace && git clone -b runpod-h100 https://github.com/esraderey/fuga && cd fuga && bash scripts/runpod/arranque.sh
#
# Variables opcionales:
#   FUGA_PLAN=final|completo   final (por defecto): corrida larga + candidatos cortos + selección. completo: solo la larga.
#   FUGA_REPO_HF=usuario/repo  dónde publicar el modelo (por defecto esraderey/fuga-s-eco-rel). Vacío = no publicar.
#   FUGA_APAGAR=1|0            detener el pod al terminar o si algo falla (por defecto 1).
#   FUGA_PRECIO_HORA=3.0       precio por hora del pod, solo para la proyección de coste.
#   FUGA_PRUEBA=1              ensayo en miniatura (nano, pocos pasos); no publica ni apaga y no exige GPU ni HF_TOKEN.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
PY="$(command -v python || command -v python3)"

PLAN="${FUGA_PLAN:-final}"
REPO_HF="${FUGA_REPO_HF-esraderey/fuga-s-eco-rel}"
APAGAR="${FUGA_APAGAR:-1}"
PRECIO="${FUGA_PRECIO_HORA:-3.0}"
PRUEBA="${FUGA_PRUEBA:-0}"
RAIZ_RUN="runs/final_h100"
echo "== FUGA · arranque · plan=$PLAN · publicar en=${REPO_HF:-nada} · apagar=$APAGAR · prueba=$PRUEBA =="

if [ "$PRUEBA" != "1" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: no hay GPU visible"; exit 1; }
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: falta HF_TOKEN. Añádela como variable de entorno del pod (token de Hugging Face con permiso de escritura)."
    exit 1
  fi
fi

"$PY" -m pip install -q -e ".[zspace,hf,dev]" "numpy>=2"
"$PY" - <<'PY'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert v >= (2, 4), f"FUGA necesita torch >= 2.4 y hay {torch.__version__}"
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU")
PY

"$PY" scripts/runpod/datos.py bajar
"$PY" -m pytest -q -x

if [ "$PRUEBA" = "1" ]; then
  "$PY" scripts/cadena_final.py --prueba --plan "$PLAN" --raiz runs/_prueba_arranque --device auto
  exit $?
fi

mkdir -p runs
ARGS=(--plan "$PLAN" --raiz "$RAIZ_RUN" --device cuda --precio_hora "$PRECIO" --registro_hf esraderey/fuga-datos)
[ -n "$REPO_HF" ] && ARGS+=(--subir "$REPO_HF")
[ "$APAGAR" = "1" ] && ARGS+=(--apagar)
nohup "$PY" scripts/cadena_final.py "${ARGS[@]}" > "$RAIZ_RUN.salida.log" 2>&1 &
echo "$!" > "$RAIZ_RUN.pid"
echo
echo "Entrenamiento lanzado en segundo plano (PID $(cat "$RAIZ_RUN.pid")). Puedes cerrar esta terminal."
echo "  Hitos y proyección de tiempo:   tail -f $RAIZ_RUN.salida.log"
echo "  Registro completo:              tail -f $RAIZ_RUN/cadena.log"
echo "La proyección de tiempo y coste aparece en el primer minuto, tras la sonda de velocidad."
