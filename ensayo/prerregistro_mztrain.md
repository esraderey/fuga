# Pre-registro: FUGA-M factorizado con MZTrain frente a FUGA-S denso — fijado el 2026-10-30 antes de entrenar

## Pregunta

En la GPU disponible (RTX 4060, 8,6 GB, Windows 11, torch 2.6+cu124), ¿un FUGA-M (386,7M parámetros densos equivalentes)
entrenado con MZTrain 1.5.0 (factores SVD de rango fijo y optimizador con estados INT8) obtiene menos bits por nota que
un FUGA-S denso (151,6M) con el mismo bucle, los mismos datos y el mismo número de tokens?

## Hipótesis declarada

El director espera que **no**: el peaje de calidad iso-parámetro que publica el propio MZTrain (≈10 % de BPC; +0,051 BPC
en su medición T20) es del mismo orden que la ganancia esperable al pasar de 152M a 387M, y la regla de uso de MZTrain
(«si cabe denso, entrena denso») sugiere que el único caso útil es el modelo que no cabe. Esta hipótesis se declara
para que el resultado no se interprete a conveniencia en ninguna dirección.

## Lo ya medido antes de este pre-registro (memoria y velocidad, no calidad)

Sondas de 3 pasos con lotes sintéticos, ctx 1024, bf16, checkpointing de activaciones de FUGA (`scripts/train_mztrain.py --probar`):

| Brazo | Parámetros entrenables | Tokens por lote | s por paso | tokens/s | Pico asignado | Pico reservado |
|---|---|---|---|---|---|---|
| FUGA-S denso | 151,6M | 8 192 | 1,19 | 6 884 | 3,59 GB | 4,18 GB |
| FUGA-S denso | 151,6M | 16 384 | 2,36 | 6 941 | 4,37 GB | 4,97 GB |
| FUGA-M MZTrain r = 512 | 277,7M | 8 192 | 2,38 | 3 450 | 6,38 GB | 6,74 GB |
| FUGA-M MZTrain r = 512 | 277,7M | 16 384 | 8,15 | 2 010 | 7,74 GB | 8,68 GB (cruza el acantilado WDDM) |
| FUGA-M MZTrain r = 256 | 139,3M | 8 192 | 1,94 | 4 219 | 3,91 GB | 4,55 GB |
| FUGA-M denso | 386,7M | 8 192 | 17,2 | 476 | 8,32 GB | 9,66 GB (pagina: no cabe) |

Conclusiones operativas: FUGA-M denso no cabe; FUGA-M factorizado con rango 512 es la mayor configuración que queda bajo
los 7 GB reservados con lotes de 8 192 tokens, y es la que se compara. El rango 512 sustituye las 160 capas `nn.Linear`
de FUGA-M (proyecciones q, kv, o y las dos del SwiGLU) y deja intacto el embedding atado.

## Diseño

| Elemento | Valor común a los dos brazos |
|---|---|
| Datos | `cache/eco_rel_1024.pkl`: modo `eco_rel`, ctx 1024, 15 834 secuencias, 4 948 693 tokens, 617 trozos de validación |
| Presupuesto | 700 pasos × 8 192 tokens ≈ 5,7M tokens (≈ 1,16 épocas); igual número de tokens en ambos brazos |
| Optimización | lr pico 6e-4, WSD (calentamiento 5 %, decaimiento 20 %), β = (0,9; 0,95), eps 1e-8, decaimiento de pesos 0,1 solo en parámetros de dimensión ≥ 2 (regla de `fuga.train`), recorte de gradiente 1,0, dropout 0,1, bf16 con TF32, checkpointing de activaciones de FUGA |
| Semillas | 0 y 1 por brazo (las mismas para datos y modelo) |
| Evaluación | bits por nota en validación (617 trozos) y en test (478 piezas, mismos compases para ambos brazos), NLL acumulada por pieza |

| Brazo | Modelo | Optimizador |
|---|---|---|
| `s_denso` | FUGA-S denso (preset `fuga-s`) | `torch.optim.AdamW` |
| `m_mztrain` | FUGA-M con `ZTrainEngine` de MZTrain 1.5.0, `initial_rank = max_rank = 512`, sin crecimiento de rango ni ElasticRank, sin su checkpointing propio ni su GradScaler | `mztrain.optimizer.ZCompressedAdam` con `compress_states=True` y los mismos grupos e hiperparámetros |

Calibración informativa (no decide): `nano_denso` frente a `nano_mztrain` con rango 48, 360 pasos, semilla 0, mismos datos.

Lo que difiere entre brazos y no se controla: la forma del modelo (profundo y ancho factorizado frente a denso y estrecho),
el optimizador (estados INT8 frente a fp32), y la tasa de aprendizaje óptima de cada uno (no hay barrido: se usa la misma).

## Criterios fijados antes de ver resultados

| Id | Hipótesis a prueba | Criterio | Consecuencia si falla |
|---|---|---|---|
| KM1 | FUGA-M con MZTrain modela mejor que FUGA-S denso a igual número de tokens | IC95 bootstrap pareado por pieza de bits/nota en test (m_mztrain − s_denso), con las dos semillas agrupadas, enteramente < 0 | MZTrain no se adopta para FUGA en esta GPU; FUGA-S denso sigue siendo el modelo de referencia local |
| KM2 | El brazo MZTrain cabe en la GPU durante todo el entrenamiento | pico reservado < 7 GB y sin OOM ni paginación (tokens/s dentro del ±25 % de la sonda) | Se informa como no reproducible en 8,6 GB |
| KM3 | El modelo exportado a denso es utilizable por FUGA | `export_full_model` → `runs/mz/m_mztrain_*/modelo.pt` carga con `fuga.sample.cargar_modelo` y genera sin fallos de gramática | Defecto de integración; se documenta |
| KM4 (informativo) | El peaje en tokens musicales a escala nano es del orden del publicado | bits/nota nano_mztrain − nano_denso, 1 semilla | Solo contexto |

Se reportan además: tokens/s reales, segundos totales por brazo, curva de pérdida, bits/nota de validación por tramos y
el cociente de coste (segundos de m_mztrain / segundos de s_denso) para una lectura a igual reloj, que no decide.

## Límites declarados

- Una sola GPU de consumo y 5,7M tokens: régimen subentrenado para 150–280M parámetros; el resultado habla de este
  presupuesto, no de FUGA-M a escala completa.
- Sin barrido de tasa de aprendizaje por brazo: la misma lr puede favorecer a uno.
- El corpus es el de FUGA (folk monofónico y polifonía renacentista, 4,9M tokens); con menos de 1,2 épocas no hay
  repetición de datos, pero el dropout 0,1 queda activo en ambos brazos por paridad con la línea base sana de MZTrain.
- MZTrain se evalúa en su versión PyPI 1.5.0 (idéntica a `D:\mztrain` en la etiqueta v1.5.0), con factorización de rango
  fijo; no se prueban su crecimiento de rango, ElasticRank, GaLore ni su gobernador de VRAM.

## Desviaciones

Ninguna registrada antes de entrenar. Las que surjan durante la ejecución se anotan aquí con fecha, antes de ver los
resultados de calidad del brazo afectado.
