#!/usr/bin/env bash
# Arranque del entrenamiento final de FUGA en un pod de RunPod (pensado para una H100 de 80 GB).
#
# Lo ejecuta el contenedor al crearse el pod con `python scripts/runpod/lanzar.py crear --confirmar`. A mano, dentro del
# pod y con HF_TOKEN definido (token de Hugging Face con escritura):
#   cd /workspace && git clone -b runpod-h100 https://github.com/esraderey/fuga && cd fuga && bash scripts/runpod/arranque.sh
#
# Variables opcionales:
#   FUGA_PLAN=final|completo   final (por defecto): corrida larga + candidatos cortos + selección. completo: solo la larga.
#   FUGA_REPO_HF=usuario/repo  dónde publicar el modelo (por defecto esraderey/fuga-s-eco-rel). Vacío = no publicar.
#   FUGA_APAGAR=1|0            detener el pod al terminar o si algo falla (por defecto 1).
#   FUGA_MAX_HORAS=5           tope de vida del pod: pasado ese tiempo se detiene aunque no haya terminado.
#   FUGA_PRECIO_HORA=3.0       precio por hora del pod, solo para la proyección de coste.
#   FUGA_PRUEBA=1              ensayo en miniatura (nano, pocos pasos); no publica ni apaga y no exige GPU ni HF_TOKEN.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
PY="$(command -v python || command -v python3)"

PLAN="${FUGA_PLAN:-final}"
REPO_HF="${FUGA_REPO_HF-esraderey/fuga-s-eco-rel}"
APAGAR="${FUGA_APAGAR:-1}"
MAX_HORAS="${FUGA_MAX_HORAS:-5}"
PRECIO="${FUGA_PRECIO_HORA:-3.0}"
PRUEBA="${FUGA_PRUEBA:-0}"
RAIZ_RUN="runs/final_h100"
mkdir -p runs
LOG="runs/arranque.log"
exec > >(tee -a "$LOG") 2>&1
hito() { echo "### $(date '+%Y-%m-%d %H:%M:%S') $*"; }

apagar_pod() {
  if [ -z "${RUNPOD_POD_ID:-}" ]; then echo "RUNPOD_POD_ID no definido: no se apaga nada"; return 0; fi
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || echo "NO SE PUDO DETENER EL POD: DETENLO DESDE LA CONSOLA"
}
subir_log() {   # copia este registro al dataset privado para poder leerlo sin entrar al pod (si falla, no pasa nada)
  sleep 1
  "$PY" - <<'PY' || true
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="runs/arranque.log", path_in_repo="corridas/final_h100/arranque.log",
                    repo_id="esraderey/fuga-datos", repo_type="dataset", commit_message="registro del arranque")
PY
}
al_salir() {
  rc=$?
  if [ "$rc" != "0" ] && [ "$PRUEBA" != "1" ]; then
    hito "ARRANQUE FALLÓ con código $rc"
    subir_log
    if [ "$APAGAR" = "1" ]; then apagar_pod; fi
  fi
}
trap al_salir EXIT

hito "arranque de FUGA: plan=$PLAN publicar en=${REPO_HF:-nada} apagar=$APAGAR tope=${MAX_HORAS}h prueba=$PRUEBA commit=$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"

if [ "$PRUEBA" != "1" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: no hay GPU visible"; exit 1; }
  case "${HF_TOKEN:-}" in
    "") echo "ERROR: falta HF_TOKEN. Defínela al crear el pod (secreto de RunPod o variable de entorno)."; exit 1 ;;
    *RUNPOD_SECRET*|"{{"*) echo "ERROR: el secreto de RunPod no se resolvió: comprueba que existe y que su nombre coincide."; exit 1 ;;
  esac
  if [ "$APAGAR" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then   # temporizador de seguridad
    nohup bash -c "sleep $((MAX_HORAS * 3600)); runpodctl pod stop $RUNPOD_POD_ID || runpodctl stop pod $RUNPOD_POD_ID" >/dev/null 2>&1 &
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
hito "datos verificados; pasando los tests"
"$PY" -m pytest -q -x

if [ "$PRUEBA" = "1" ]; then
  "$PY" scripts/cadena_final.py --prueba --plan "$PLAN" --raiz runs/_prueba_arranque --device auto
  exit $?
fi

ARGS=(--plan "$PLAN" --raiz "$RAIZ_RUN" --device cuda --precio_hora "$PRECIO" --registro_hf esraderey/fuga-datos)
if [ -n "$REPO_HF" ]; then ARGS+=(--subir "$REPO_HF"); fi
if [ "$APAGAR" = "1" ]; then ARGS+=(--apagar); fi
nohup "$PY" scripts/cadena_final.py "${ARGS[@]}" > "$RAIZ_RUN.salida.log" 2>&1 &
echo "$!" > "$RAIZ_RUN.pid"
hito "ARRANQUE OK: cadena lanzada en segundo plano (PID $(cat "$RAIZ_RUN.pid"))"
subir_log
echo "Puedes cerrar esta terminal."
echo "  Hitos y proyección de tiempo:   tail -f $RAIZ_RUN.salida.log"
echo "  Registro completo:              tail -f $RAIZ_RUN/cadena.log"
echo "La proyección de tiempo y coste aparece en el primer minuto, tras la sonda de velocidad."
