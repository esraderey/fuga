# FUGA-S eco_rel: 3 horas en la RTX 4060

- Fecha: 2026-10-30, 13:58 a 16:55 (hora local)
- Comando: `python -m fuga.train --datos data/corpus_pd.jsonl --modo eco_rel --preset fuga-s --ctx 1024 --cache cache/eco_rel_1024.pkl --device cuda --ckpt --tokens_por_lote 8192 --minutos 180 --lr 6e-4 --wd 0.1 --dropout 0.1 --eval_cada 250 --guardar_mejor --semilla 0 --out runs/fuga_s_rel_3h`
- Registros: `runs/fuga_s_rel_3h/{log.json, cadena.log, eleccion.json, analisis_muestras.json}`; checkpoints `modelo_mejor.pt` (el útil) y `modelo.pt` (final, sobreajustado)
- Muestras: `muestras/fuga_s_rel_3h/romantico12/` (semillas 0 a 11, 16 compases, temperatura 1,0, top-p 0,95; render plano e interpretación romántica)
- Máquina: RTX 4060 (8,6 GB), Windows 11, torch 2.6+cu124, bf16 con TF32, checkpointing de activaciones
- Gates: `.forja/gates/fuga_s_3h.json` (G0, G1, G3 con 60 tests)

## 1. Resultado en una frase

En 3 horas FUGA-S (151,6M) baja la validación a 3,53 bits por nota, casi un bit por debajo del mismo modelo a 700 pasos
(4,49) y 1,7 por debajo del nano (5,25); pero el mínimo llega en el paso 3 250, a las 5,4 épocas, y desde ahí el modelo
memoriza: el checkpoint final da 6,84. La opción `--guardar_mejor`, añadida hoy a `fuga/train.py` con dos tests ancla,
es lo que convierte estas 3 horas en un modelo utilizable.

## 2. Presupuesto

| Medida | Valor |
|---|---|
| Datos | 15 834 secuencias, 4,95M tokens de entrenamiento, 617 trozos de validación |
| Pasos / tokens vistos | 9 436 pasos de 8 192 tokens; 75,3M tokens; 15,2 épocas |
| Reloj | 10 817 s de entrenamiento (1,15 s por paso; 6 960 tokens/s) + 53 s de elección + 77 s por tanda de 4 muestras |
| GPU | 5,2 GB usados, utilización 100 %, 69 °C (nvidia-smi en el paso 50) |
| Tasa de aprendizaje | WSD: 5 % de calentamiento, meseta en 6e-4, 20 % final de decaimiento lineal |

## 3. Curva de validación (subconjunto de 200 trozos cada 250 pasos)

| Paso | 250 | 1 000 | 2 000 | 3 000 | **3 250** | 4 000 | 5 000 | 6 000 | 7 000 | 8 000 | 9 250 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bits/nota | 5,96 | 4,58 | 4,10 | 3,93 | **3,87** | 3,93 | 4,29 | 4,53 | 5,31 | 5,61 | 7,81 |

La pérdida de entrenamiento (nats por token) baja de 0,92 en el paso 1 000 a 0,16 en el 9 000: el modelo se aprende el
corpus de memoria mientras la validación se dispara. El decaimiento final de la tasa no lo frena.

## 4. Elección del checkpoint

Validación completa: 617 trozos, 96 089 notas. Test: 631 trozos de 475 piezas, 70 358 notas (`cache/test_eco_rel_1024.pkl`).

| Checkpoint | Paso | Val bits/nota | Val bits/token | Test bits/nota | Test bits/token |
|---|---|---|---|---|---|
| `modelo_mejor.pt` | 3 250 | **3,533** | 1,374 | **3,848** | 1,374 |
| `modelo.pt` (final) | 9 436 | 6,844 | 2,662 | 7,216 | 2,576 |

Referencias sobre los mismos conjuntos: FUGA-S denso a 700 pasos (ensayo MZTrain) 4,489 en val y 4,86 a 4,89 en test;
FUGA-M MZTrain 4,80 a 4,82 en test; nano a 360 pasos 5,253 en val. El checkpoint elegido gana alrededor de un bit por
nota a todo lo entrenado antes en este proyecto.

## 5. Las muestras (12 semillas, `analisis_muestras.json`)

| Medida | Generado (media de 12) | Real (test contra train) |
|---|---|---|
| Copia de ventanas de 4 compases presentes en train | 0,000 (0 de 12 piezas) | 0,052 |
| Copia de ventanas de 2 compases | 0,000 | 0,092 |
| Notas en la escala de la tonalidad declarada | 0,973 | 0,957 |
| Repetición de segmentos (ventana de 32 compases) | 0,116 | 0,243 |
| Modo | 7 mayores, 5 menores | 73 % mayores en el corpus |
| Compases completos | 9 de 12 llegan a 16; 3 cierran antes (5, 9 y 11) | |
| Textura | 7 monódicas, 5 a 4-6 voces | |

Lectura: nada de copia literal (el detector de 4 compases, que da 5 % entre piezas reales distintas, da 0 aquí),
tonalidad bien sostenida, y menos repetición estructural que la música real, que es el defecto audible: las piezas varían
más de lo que repiten. El tempo es 94 en 10 de 12, como en el corpus (92 % de las piezas a 90).

Pieza entregada: muestra 6, La mayor, 127 ppm, 4 voces, 249 notas, 16 compases completos, repetición 0,283 (la más
cercana a la real), 95 % en escala, ámbito MIDI 33 a 88, 23,5 s. Alternativa: muestra 0, Re menor, 6 voces, 201 notas.

## 6. Decisión y seguimientos

1. `runs/fuga_s_rel_3h/modelo_mejor.pt` pasa a ser el modelo de referencia local para generar; `modelo.pt` se conserva
   solo como evidencia del sobreajuste.
2. Con este corpus, FUGA-S no necesita más de una hora: el presupuesto razonable es 3 000 a 3 500 pasos con
   `--guardar_mejor`. Las horas extra no compran calidad; la compran más datos o más regularización.
3. Seguimientos que el resultado justifica: (a) barrido corto de dropout 0,2 a 0,3 y decaimiento de pesos a 3 000 pasos
   para ver si el mínimo baja de 3,5; (b) ampliar el corpus; (c) evaluar la estructura con `scripts/ensayo.py` cuando
   `music21` esté instalado (juez tonal).

## 7. Límites

- Una sola semilla; la elección del checkpoint usa el subconjunto de 200 trozos de validación y se confirma con el completo.
- Validación y test salen del mismo corpus (partición por libro): miden generalización dentro de los estilos del corpus, no fuera.
- Sin barrido de hiperparámetros: lr 6e-4, dropout 0,1 y decaimiento 0,1 heredados del ensayo MZTrain.
- La calidad musical de las muestras se midió con métricas automáticas; no hay escucha ciega.
