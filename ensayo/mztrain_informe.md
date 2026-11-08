# Informe del ensayo MZTrain: FUGA-M factorizado frente a FUGA-S denso

- Fecha: 2026-10-30
- Pre-registro: `ensayo/prerregistro_mztrain.md` (fijado antes de entrenar; sin desviaciones registradas)
- Resultados crudos: `ensayo/mztrain_resultados.json`; registros por corrida en `runs/mz/*/log.json`; registro de la cadena en `runs/mz/ensayo.log`
- Script: `scripts/train_mztrain.py` (mismo bucle para los dos brazos); tests: `tests/test_train_mztrain.py`
- MZTrain evaluado: PyPI 1.5.0 (MIT), copia aislada; idéntico a `D:\mztrain` en la etiqueta v1.5.0
- Máquina: RTX 4060 (8,6 GB), Windows 11 (WDDM), torch 2.6+cu124, bf16 con TF32

## 1. Resultado en una frase

A igual número de tokens, FUGA-M factorizado con MZTrain (rango 512, 277,7M parámetros entrenables) modela mejor que
FUGA-S denso (151,6M) en las dos semillas, con un intervalo pareado por pieza enteramente por debajo de cero; pero
reserva 8,8–8,9 GB, cae en el acantilado de paginación de WDDM y tarda entre 6,6 y 7,2 veces más en el reloj.
La hipótesis que el director declaró antes de entrenar (que no ganaría) cae.

## 2. Veredictos sobre los criterios pre-registrados

| Id | Criterio | Resultado | Veredicto |
|---|---|---|---|
| KM1 | IC95 pareado por pieza de bits/nota en test (m_mztrain − s_denso), dos semillas agrupadas, < 0 | −0,036 [−0,045; −0,028], 475 piezas × 2 semillas; por semilla −0,035 [−0,048; −0,024] y −0,037 [−0,049; −0,027] | **pasa** |
| KM2 | pico reservado < 7 GB y tokens/s dentro del ±25 % de la sonda | 8,77 y 8,89 GB reservados; mediana de 4,6 y 5,2 s por paso frente a 2,38 s de la sonda, con picos de 32 s | **falla** |
| KM3 | el export denso carga con `fuga.sample.cargar_modelo` y genera bajo la gramática | 386,7M parámetros, sin capas factorizadas, 2 de 2 generaciones válidas en cada semilla | **pasa** |
| KM4 (informativo) | peaje nano | nano factorizado rango 48 (1,0M entrenables) frente a nano denso (2,5M): +0,58 bits/nota [+0,55; +0,61] | peaje del 10 % con el 40 % de los parámetros |

## 3. Cifras por corrida

| Corrida | Parámetros entrenables | Val bits/nota | Test bits/nota | Segundos | s/paso (mín / mediana / máx) | Pico asignado | Pico reservado |
|---|---|---|---|---|---|---|---|
| s_denso_0 | 151,6M | 4,489 | 4,889 | 821 | 1,11 / 1,13 / 1,33 | 3,60 GB | 5,66 GB |
| s_denso_1 | 151,6M | 4,538 | 4,864 | 825 | 1,11 / 1,13 / 1,35 | 3,60 GB | 5,73 GB |
| m_mztrain_0 | 277,7M (386,7M densos equivalentes) | 4,428 | 4,824 | 5 917 | 2,29 / 4,62 / 31,97 | 6,47 GB | 8,77 GB |
| m_mztrain_1 | 277,7M | 4,432 | 4,802 | 5 413 | 3,22 / 5,19 / 24,05 | 6,46 GB | 8,89 GB |
| nano_denso_0 | 2,5M | 5,253 | 5,695 | 26 | 0,06 / 0,07 / 0,09 | 0,23 GB | 0,62 GB |
| nano_mztrain_0 | 1,0M | 5,918 | 6,391 | 45 | 0,12 / 0,12 / 0,14 | 0,21 GB | 0,60 GB |

Tokens vistos: 5,59M por brazo grande (700 pasos × 8 192), 2,87M por brazo nano. Tokens por segundo reales: FUGA-S
denso 6 800; FUGA-M MZTrain 944 y 1 032 (la sonda de tres pasos daba 3 450).

## 4. Lectura

- **Calidad a igual número de tokens.** El brazo factorizado gana 0,06 bits por nota agregados (1,3 %) y 0,036 por pieza
  en media, con las dos semillas de acuerdo. Las curvas de validación parciales lo muestran desde el paso 200. Conviene
  no sobreinterpretar el tamaño: el brazo MZTrain tiene 1,83 veces más parámetros entrenables y gasta cerca del doble de
  operaciones por token; la comparación pre-registrada es a igual número de tokens, no a igual cómputo.
- **Coste.** La sonda de tres pasos con lote sintético reservó 6,74 GB, bajo el acantilado; con lotes reales de longitud
  variable y la compactación periódica de los estados INT8, la memoria reservada subió a 8,8–8,9 GB y la GPU paginó. La
  mediana de paso se duplicó respecto a la sonda y los picos llegaron a 24–32 s. Es exactamente el comportamiento que
  documenta el README de MZTrain para Windows («no hay OOM, hay un acantilado silencioso ~7 GB»). A igual reloj, FUGA-S
  denso habría visto siete veces más tokens.
- **Integración.** La factorización respetó las 160 capas lineales de FUGA-M y dejó intacto el embedding atado; el
  checkpointing de activaciones de FUGA (`torch.utils.checkpoint`) funcionó con las capas factorizadas; `ZCompressedAdam`
  aceptó los grupos de parámetros de FUGA (decaimiento solo en matrices); el export denso es un `Transformer` normal que
  `fuga.sample` carga y con el que genera. No hizo falta tocar `fuga/`.
- **Calibración nano.** Con rango 48 el nano factorizado pierde el 10 % con el 40 % de los parámetros, en línea con el
  peaje iso-parámetro que publica MZTrain; no es iso-parámetro y solo sirve de contexto.

## 5. Decisión y seguimientos

1. MZTrain **no** sustituye a FUGA-S denso como modelo de referencia local mientras KM2 falle: siete veces más reloj por
   un 1,3 % de bits por nota no es un cambio razonable para el ciclo de experimentos del proyecto.
2. El resultado de KM1 sí justifica un seguimiento barato que el pre-registro no cubría: repetir el brazo MZTrain con
   lotes de 4 096 tokens (misma cantidad de tokens en el doble de pasos) para quedar bajo los 7 GB reservados y medir si
   el reloj vuelve a la razón de 2 a 1 de la sonda. Si lo hace, FUGA-M factorizado pasaría a ser la opción local para
   entrenar «el modelo que no cabe», que es el caso de uso que MZTrain reclama.
3. Queda sin medir lo que separaría arquitectura de optimizador: FUGA-M denso no cabe en esta GPU, así que no se puede
   saber cuánto de la ganancia viene de la forma profunda y ancha y cuánto pierde la factorización frente a ese denso.

## 6. Límites

- 5,6M tokens y una sola GPU de consumo: régimen subentrenado para 150–280M; una época larga y dropout 0,1.
- Sin barrido de tasa de aprendizaje: la misma lr (6e-4) para los dos brazos.
- Dos semillas por brazo, como exigía el pre-registro; con ese número, el acuerdo entre semillas es evidencia, no prueba.
- Solo rango fijo 512; no se probaron el crecimiento de rango, ElasticRank, GaLore ni el gobernador de VRAM de MZTrain.
