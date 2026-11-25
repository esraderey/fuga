# Entrenamiento final de FUGA en RunPod (H100)

Plan y criterios: [`ensayo/prerregistro_final.md`](../../ensayo/prerregistro_final.md). Todo lo que corre en el pod está en
este repositorio: `scripts/runpod/arranque.sh` prepara el entorno y lanza `scripts/cadena_final.py`.

## Qué hace la cadena

1. Sonda de 100 pasos: mide la velocidad real y escribe la proyección de tiempo y coste.
2. Corrida `completo`: 18 380 pasos (las 6 h de la RTX 4060 traducidas a pasos), con un checkpoint Z-Space cada 500.
3. Candidatos de presupuesto corto con decaimiento completo: 4 500, 6 000 y 8 000 pasos.
4. Evalúa todos los checkpoints en validación y test completos y elige por validación.
5. Genera 12 piezas con el elegido y mide copia, escala y repetición.
6. Exporta a safetensors con ficha y publica en Hugging Face, con los registros en `corrida/`.
7. Detiene el pod, también si algo falla, para que no siga facturando.

Mientras corre, cada 5 min copia sus registros a `esraderey/fuga-datos`, carpeta `corridas/final_h100/`: así se sigue el
progreso desde fuera del pod.

## Lanzamiento desde tu equipo (`lanzar.py`)

Requisitos, una sola vez y hechos por ti (las credenciales no pasan por ningún archivo del proyecto):

1. **Saldo** en RunPod. El tope de esta corrida son 3 h de pod.
2. **API key de RunPod** con permiso de lectura y escritura (consola → Settings → API Keys), guardada con entrada oculta:

   ```bash
   python scripts/runpod/lanzar.py clave
   ```

3. **Secreto de RunPod** llamado `hf_token` (consola → Secrets → Create Secret) con tu token de Hugging Face de escritura.
   El pod lo recibe como `HF_TOKEN`; el lanzador solo envía la referencia `{{ RUNPOD_SECRET_hf_token }}`.

Después:

```bash
python scripts/runpod/lanzar.py simulacro --precio_max 4.5           # precios y stock, pod que se crearía, coste; no crea nada
python scripts/runpod/lanzar.py crear --confirmar --precio_max 4.5   # crea el pod: una H100 bajo demanda, tope de precio y de 3 h
python scripts/runpod/lanzar.py vigilar              # sigue los hitos y detiene el pod al terminar o al pasar el tope
python scripts/runpod/lanzar.py registros            # últimas líneas del contenedor, para diagnosticar
python scripts/runpod/lanzar.py estado               # estado y coste por hora
python scripts/runpod/lanzar.py eliminar --confirmar # borra el pod y su volumen cuando el modelo ya esté publicado
```

El pod clona el commit exacto desde el que se lanzó (`FUGA_COMMIT`), así que hay que lanzar con todo subido a GitHub.

El lanzador usa la API REST v2 de RunPod (`https://api.runpod.io/v2`); la v1 (`rest.runpod.io`) se retiró el
2026-11-15. En la v2 cada pod lleva un solo tipo de GPU, así que se prueban una a una, de la más barata a la más cara.
Antes de crear nada, `simulacro` y `crear` comprueban que la clave funciona, que el secreto existe (solo por su nombre),
y los precios y la disponibilidad en vivo del catálogo.

`--gpu a100` alquila una A100 en vez de una H100 (misma imagen). `--precio_max` (3 USD/h por defecto) se elige al lanzar:
entre las GPU de la familia con disponibilidad y precio bajo el tope se prueba primero la más barata, en la nube segura.

Seguros contra el gasto: la cadena detiene el pod al terminar o fallar; si la proyección medida tras la sonda no cabe
en el tope (menos 20 min), se aborta antes de entrenar; el contenedor lleva un temporizador que lo detiene a las 3 h;
`vigilar` lo detiene si no da señales en 35 min, si termina y no se apaga solo, o si pasa el tope; y el precio se
comprueba antes y después de crear (si sale por encima del tope, el pod se elimina).

## Lanzamiento a mano (alternativa)

1. En https://console.runpod.io → Pods → Deploy: una H100 de 80 GB bajo demanda, imagen
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 20 GB de contenedor, 40 GB de volumen en `/workspace` y la
   variable de entorno `HF_TOKEN`.
2. En la terminal web del pod:

   ```bash
   cd /workspace && git clone -b runpod-h100 https://github.com/esraderey/fuga && cd fuga && bash scripts/runpod/arranque.sh
   ```

## Opciones del arranque (variables de entorno del pod)

| Variable | Por defecto | Efecto |
|---|---|---|
| `FUGA_PLAN` | `final` | `completo` entrena solo la corrida larga y publica su mejor checkpoint |
| `FUGA_REPO_HF` | `esraderey/fuga-s-eco-rel` | repositorio donde publicar; vacío para no publicar |
| `FUGA_APAGAR` | `1` | `0` deja el pod encendido al terminar |
| `FUGA_MAX_HORAS` | `3` | tope de vida del pod; si la proyección no cabe en él, no se entrena |
| `FUGA_PRECIO_HORA` | `3.0` | precio del pod, solo para la proyección de coste |
| `FUGA_PRUEBA` | `0` | `1` ensaya la cadena en miniatura sin publicar ni apagar |

## Si algo falla

- La cadena es reanudable: volver a arrancar el pod salta lo terminado (cada corrida con `log.json`).
- Todo queda en `/workspace/fuga/runs/final_h100/` (el volumen sobrevive a detener el pod) y los registros, incluido el
  del arranque si falla, en `esraderey/fuga-datos`, carpeta `corridas/final_h100/`.
- Datos que no coinciden con el manifiesto: `python scripts/runpod/datos.py estado`.

## Tiempo y coste previstos

El paso está limitado por la CPU, no por la GPU: cada paso despacha unas 19 500 operaciones de PyTorch, que en la RTX
4060 local cuestan 0,18 s aunque el lote sea diminuto (medido el 2026-11-25), mientras que el cálculo útil, 7,3 TFLOP
por paso, le lleva a una H100 unos 30 ms. En un servidor Linux el despacho debería bajar a 0,05–0,10 s; por eso una
H100 SXM apenas gana a una H100 NVL o a una A100. La sonda sustituye estas cifras por una medida.

| Plan | Pasos | Entrenamiento (0,06–0,12 s/paso) | Total con arranque, evaluaciones, muestras y subida |
|---|---|---|---|
| `final` | 36 880 | 37–74 min | 1 h 10 – 1 h 55 |
| `completo` | 18 380 | 18–37 min | 50 min – 1 h 15 |

Con los precios vistos en la consola el 2026-11-25 (H100 PCIe 2,89 USD/h sin capacidad; H100 NVL 3,19; H100 SXM 4,39;
A100 SXM 2,19), el plan `final` cuesta unos 3,8–6,1 USD en H100 NVL, 5,3–8,3 en H100 SXM y 2,8–4,4 en A100 SXM.

Tarjetas más baratas (`--gpu`): el paso pide unos 12,4 GB sin checkpointing (medido), así que basta con 20–24 GB. Precios
de la nube segura en el catálogo del 2026-11-25; el tiempo incluye unos 40 min fijos.

| `--gpu` | Tarjeta | Memoria | USD/h | Tiempo estimado | Coste estimado | Tope por defecto |
|---|---|---|---|---|---|---|
| `l40s` | L40S | 48 GB | 1,09 | 1,5–2,1 h | 1,6–2,3 | 3 h |
| `a40` | A40 | 48 GB | 0,59 | 2,2–3,2 h | 1,3–1,9 | 4 h |
| `rtx4090` | RTX 4090 | 24 GB | 0,89 | 2,0–2,7 h | 1,8–2,4 | 4 h |
