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

1. **Saldo** en RunPod. El tope de esta corrida son 5 h de pod.
2. **API key de RunPod** con permiso de lectura y escritura (consola → Settings → API Keys), guardada con entrada oculta:

   ```bash
   python scripts/runpod/lanzar.py clave
   ```

3. **Secreto de RunPod** llamado `hf_token` (consola → Secrets → Create Secret) con tu token de Hugging Face de escritura.
   El pod lo recibe como `HF_TOKEN`; el lanzador solo envía la referencia `{{ RUNPOD_SECRET_hf_token }}`.

Después:

```bash
python scripts/runpod/lanzar.py simulacro            # precios y stock de H100, pod que se crearía, coste; no crea nada
python scripts/runpod/lanzar.py crear --confirmar    # crea el pod: una H100 bajo demanda, tope 3 USD/h y 5 h
python scripts/runpod/lanzar.py vigilar              # sigue los hitos y detiene el pod al terminar o al pasar el tope
python scripts/runpod/lanzar.py estado               # estado y coste por hora
python scripts/runpod/lanzar.py eliminar --confirmar # borra el pod y su volumen cuando el modelo ya esté publicado
```

El pod clona el commit exacto desde el que se lanzó (`FUGA_COMMIT`), así que hay que lanzar con todo subido a GitHub.

Seguros contra el gasto: la cadena detiene el pod al terminar o fallar; el contenedor lleva un temporizador que lo
detiene a las 5 h; `vigilar` lo detiene si no da señales en 35 min, si termina y no se apaga solo, o si pasa el tope; y
el precio se comprueba antes y después de crear (si sale por encima del tope, el pod se elimina).

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
| `FUGA_MAX_HORAS` | `5` | tope de vida del pod |
| `FUGA_PRECIO_HORA` | `3.0` | precio del pod, solo para la proyección de coste |
| `FUGA_PRUEBA` | `0` | `1` ensaya la cadena en miniatura sin publicar ni apagar |

## Si algo falla

- La cadena es reanudable: volver a arrancar el pod salta lo terminado (cada corrida con `log.json`).
- Todo queda en `/workspace/fuga/runs/final_h100/` (el volumen sobrevive a detener el pod) y los registros, incluido el
  del arranque si falla, en `esraderey/fuga-datos`, carpeta `corridas/final_h100/`.
- Datos que no coinciden con el manifiesto: `python scripts/runpod/datos.py estado`.

## Tiempo y coste previstos

Estimación con 0,10–0,20 s por paso en la H100 (7 a 9 veces la RTX 4060 medida); la sonda la sustituye por una medida.

| Plan | Pasos | Entrenamiento | Total con arranque, evaluación, muestras y subida |
|---|---|---|---|
| `final` | 36 880 | 60–125 min | 1 h 30 – 2 h 30 |
| `completo` | 18 380 | 30–60 min | 1 h – 1 h 30 |
