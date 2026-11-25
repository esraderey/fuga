# Informe: entrenamiento final de FUGA-S y publicación

- Fecha: 2026-11-25. Pre-registro: `ensayo/prerregistro_final.md` (fijado el 2026-11-09, notas del 2026-11-25 antes de lanzar)
- Modelo publicado: https://huggingface.co/esraderey/fuga-s-eco-rel (pesos safetensors, SHA-256 `7355ffe953e09c93…`, licencia MIT)
- Registros de la corrida: `runs/final_h100/registros/` (copia de `corrida/` del repositorio publicado y del dataset privado `esraderey/fuga-datos`)
- Código: commit `28c64b1` de la rama `runpod-h100`; cadena `scripts/cadena_final.py`, lanzada con `scripts/runpod/lanzar.py`
- Máquina: pod de RunPod con una **L40S** (48 GB, nube segura, 16 vCPU, driver CUDA 13.2), imagen PyTorch 2.4.1+cu124

## 1. Resultado en una frase

El modelo publicado es el checkpoint del paso 4 250 del presupuesto corto de 4 500 pasos: **3,362 bits por nota en
validación y 3,645 en test**, el mejor de todo el proyecto, y gana por pieza al mejor checkpoint de la corrida larga con
un intervalo enteramente por debajo de cero (KF1 pasa). Todas las corridas tocaron su mínimo en el mismo paso, el 4 250.

## 2. Candidatos y elección (validación completa: 617 trozos; test completo: 631 trozos de 475 piezas)

| Corrida | Checkpoint | Paso | Val bits/nota | Test bits/nota | |
|---|---|---|---|---|---|
| `completo` | `modelo.pt` | 18 380 de 18 380 | 8,186 | 8,591 | |
| `completo` | `modelo_mejor.pt` | 4 250 de 18 380 | 3,442 | 3,745 | |
| `wsd_4500` | `modelo.pt` | 4 500 de 4 500 | 3,374 | 3,655 | |
| `wsd_4500` | `modelo_mejor.pt` | 4 250 de 4 500 | **3,362** | **3,645** | **publicado** |
| `wsd_6000` | `modelo.pt` | 6 000 de 6 000 | 3,600 | 3,874 | |
| `wsd_6000` | `modelo_mejor.pt` | 4 250 de 6 000 | 3,480 | 3,778 | |
| `wsd_8000` | `modelo.pt` | 8 000 de 8 000 | 4,498 | 4,769 | |
| `wsd_8000` | `modelo_mejor.pt` | 4 250 de 8 000 | 3,448 | 3,746 | |

Regla aplicada, la fijada antes de entrenar: menor validación completa entre los ocho checkpoints. El test no intervino.

Referencias anteriores sobre los mismos conjuntos (val / test): corrida local de 6 h cortada, paso 4 250: 3,433 / 3,718;
barrido de dropout 0,3 a 3 000 pasos: 3,443 / 3,737; corrida de 3 h: 3,533 / 3,848; FUGA-S a 700 pasos: 4,49 / 4,87.

## 3. Veredictos sobre los criterios pre-registrados

| Id | Criterio | Resultado | Veredicto |
|---|---|---|---|
| KF1 | IC95 pareado por pieza en test, mejor corto − mejor largo, < 0 | −0,090 [−0,104; −0,076], 475 piezas | **pasa** |
| K6-1 | val del elegido < 3,40 | 3,362 (test 3,645) | **mejora declarada** |
| K6-2 | mínimo de `completo` y sobreajuste | mínimo en el paso 4 250; val completa 3,442 frente a 8,186 del final (+4,74) | informativo |
| KZ1 | almacén Z-Space < 60 % del equivalente bf16 | 8,02 GB en 37 versiones: 71,5 % del bf16, 35,7 % del fp32 | **falla** |
| KZ2 | la última versión reconstruye bit a bit `modelo.pt` en bf16 | exacta; `modelo_mejor.pt` sin versión en su paso (4 250) | **pasa** |
| KZ3 | segundos por versión < 3 % del reloj de referencia (21 600 s) | 328 s en 36 guardados (media 9,1 s, máx 18 s) = 1,5 %; 9,4 % del reloj real de `completo` en la L40S | **pasa** con el criterio original |
| KZ4 | no se repite | (medido antes: +0,00004 bits por nota al redondear a bf16) | — |
| K6-3 | muestras: copia, escala, repetición | ver §4 | sin copia |

## 4. Las 12 muestras del modelo publicado (`registros/analisis_muestras.json`)

| Medida | Generado | Real (test frente a train) |
|---|---|---|
| Copia de ventanas de 4 compases presentes en train | 0,000 (máximo por pieza) | 0,052 |
| Copia de ventanas de 2 compases | 0,000 | 0,092 |
| Notas en la escala de la tonalidad declarada | 0,968 | 0,957 |
| Repetición de segmentos (ventana de 32 compases) | 0,177 | 0,243 |
| Modo | 9 mayores, 3 menores | 73 % mayores en el corpus |
| Compases | 5 de 12 llegan a 16; 6 cierran en 9 o 10 y una en 4 | |
| Textura | 4 polifónicas (4 a 6 voces), 8 monódicas | |

Frente al modelo de 3 h (repetición 0,116; 9 de 12 piezas completas), la repetición se acerca a la de la música real,
pero más piezas cierran antes de los 16 compases: el modelo imita mejor la longitud típica del corpus, con muchas
canciones populares de unos 8 compases. Ilustran la ficha las muestras 1 (Re menor, 5 voces) y 8 (Re mayor, 4 voces).

## 5. Hipótesis declarada del director, contrastada

| Predicción | Resultado | |
|---|---|---|
| Elegido: final de `wsd_6000`, 3,28–3,38 en val | `modelo_mejor.pt` de `wsd_4500` (paso 4 250), 3,362 | rango acertado, corrida no |
| Mejor de `completo` entre los pasos 4 000 y 6 000, 3,40–3,46 | paso 4 250, 3,442 | acertada |
| Final de `completo` sobreajustado (> 5) | 8,186 | acertada |
| KF1 pasa con −0,03 a −0,10 | −0,090 | acertada |
| Z-Space entre 72 % y 80 % del bf16, deltas menores al final | 71,5 %; deltas de 220 MB bajan a 142 MB en el decaimiento | casi; KZ1 falla como se predijo |

El fallo de la predicción enseña algo: el mínimo cae en el paso 4 250 en las cuatro corridas, con cualquier calendario.
Lo que mejora el modelo no es un presupuesto «algo mayor» sino uno cuyo decaimiento termine justo en ese punto; con 6 000
y 8 000 pasos la meseta dura demasiado y el decaimiento llega cuando el modelo ya memoriza (finales de 3,60 y 4,50).

## 6. Coste y tiempo

| Medida | Valor |
|---|---|
| Velocidad (sonda de 100 pasos) | 0,164 s por paso; con evaluaciones y guardados, 0,19 s en `completo` |
| Cadena completa | 118 min (sonda, cuatro corridas, verificación, evaluación, muestras, export y subida) |
| Pod | 2,0 h encendido, de 08:35 a 10:35 hora local; se detuvo solo |
| Coste de GPU | ~2,18 USD a 1,09 USD/h; el volumen de 40 GB se factura mientras el pod exista |
| Proyección previa | 1,64–2,29 USD estimado; 2,76 USD proyectado por la sonda con el tope de 1,20 USD/h |

## 7. Desviaciones

- **Hardware:** L40S en vez de H100, por disponibilidad y coste (nota del 2026-11-25 en el pre-registro, antes de lanzar).
- **Lanzador:** RunPod retiró su API v1 el 2026-11-15; el lanzador pasó a la v2. La clave usada no podía leer el catálogo
  de precios: el pod se creó sin consultarlo y se comprobó su precio real (1,09 USD/h) nada más crearlo.
- **Muestras de la ficha:** las genera el pod; regeneradas en local desde el repositorio publicado, las piezas no son
  idénticas bit a bit (otra GPU y otra versión de PyTorch), como ya pasaba con los MIDI de ensayo.

## 8. Verificación de la publicación

- Descarga completa del repositorio publicado: el SHA-256 de `model.safetensors` coincide con el de `config.json`.
- `python -m fuga.sample --ckpt .` dentro del repositorio descargado, con su propio paquete `fuga/`, carga los pesos y
  genera 6 piezas de 16 compases (`muestras/final_publicado/`).

## 9. Límites

- Una semilla por corrida; los intervalos son entre piezas, no entre semillas.
- Validación y test salen del mismo corpus (partición por libro): generalización dentro de sus estilos, no fuera.
- La calidad musical se mide con métricas automáticas; no hay escucha ciega.
- Los checkpoints no publicados y el almacén Z-Space viven solo en el volumen del pod mientras no se elimine.
