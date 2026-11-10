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

## Pasos

1. **Crear el pod** en https://console.runpod.io → Pods → Deploy:
   - GPU: H100 (PCIe o SXM, 80 GB), una sola, bajo demanda (no interrumpible).
   - Plantilla: «Runpod Pytorch 2.x» (cualquiera con PyTorch ≥ 2.4 y CUDA 12).
   - Disco: 20 GB de contenedor y 40 GB de volumen en `/workspace`.
   - Variable de entorno: `HF_TOKEN` = tu token de Hugging Face con escritura (se introduce ahí, no en ningún archivo).
2. **Abrir la terminal web** del pod y pegar:

   ```bash
   cd /workspace && git clone -b runpod-h100 https://github.com/esraderey/fuga && cd fuga && bash scripts/runpod/arranque.sh
   ```

   El arranque instala dependencias, descarga los datos del dataset privado `esraderey/fuga-datos` y los verifica contra
   `datos.sha256.json`, pasa los tests (unos 2 min) y lanza la cadena en segundo plano. Se puede cerrar la terminal.
3. **Seguir el progreso** (opcional): `tail -f runs/final_h100.salida.log`. La proyección aparece en el primer minuto.
4. **Al terminar** el modelo queda en https://huggingface.co/esraderey/fuga-s-eco-rel y el pod se detiene solo. Comprueba
   en la consola que figura detenido y, cuando hayas visto el modelo publicado, elimínalo (*Terminate*) para dejar de
   pagar el volumen. Si la imagen no trae `runpodctl`, el registro lo avisa y hay que detenerlo a mano.

## Opciones (variables antes de `bash scripts/runpod/arranque.sh`)

| Variable | Por defecto | Efecto |
|---|---|---|
| `FUGA_PLAN` | `final` | `completo` entrena solo la corrida larga y publica su mejor checkpoint |
| `FUGA_REPO_HF` | `esraderey/fuga-s-eco-rel` | repositorio donde publicar; vacío para no publicar |
| `FUGA_APAGAR` | `1` | `0` deja el pod encendido al terminar |
| `FUGA_PRECIO_HORA` | `3.0` | precio del pod, solo para la proyección de coste |
| `FUGA_PRUEBA` | `0` | `1` ensaya la cadena en miniatura sin publicar ni apagar |

## Si algo falla

- La cadena es reanudable: volver a ejecutar el mismo comando salta lo terminado (cada corrida con `log.json`).
- Todo queda en `/workspace/fuga/runs/final_h100/` (el volumen sobrevive a detener el pod) y una copia de los registros
  se sube a `esraderey/fuga-datos`, carpeta `corridas/final_h100/`, termine bien o mal.
- Datos que no coinciden con el manifiesto: `python scripts/runpod/datos.py estado`.

## Tiempo y coste previstos

Estimación con 0,10–0,20 s por paso en la H100 (7 a 9 veces la RTX 4060 medida); la sonda la sustituye por una medida.

| Plan | Pasos | Entrenamiento | Total con arranque, evaluación, muestras y subida |
|---|---|---|---|
| `final` | 36 880 | 60–125 min | 1 h 30 – 2 h 30 |
| `completo` | 18 380 | 30–60 min | 1 h – 1 h 30 |
