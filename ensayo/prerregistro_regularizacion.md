# Pre-registro: barrido corto de dropout en FUGA-S eco_rel (3 000 pasos)

- Fecha de fijación: 2026-10-30, antes de lanzar ninguna corrida
- Motivo: `ensayo/fuga_s_3h.md` mostró que FUGA-S toca su mínimo de validación hacia el paso 3 250 (5,4 épocas) y
  después memoriza el corpus. Pregunta: ¿más dropout baja ese mínimo, o solo lo retrasa?
- Máquina y pila: RTX 4060 (8,6 GB), Windows 11, torch 2.6+cu124, bf16 con TF32, checkpointing de activaciones

## Brazos (orden de ejecución)

| Brazo | Dropout | Resto | Salida |
|---|---|---|---|
| d02 | 0,2 | igual que el control | `runs/reg/d02` |
| d03 | 0,3 | igual que el control | `runs/reg/d03` |
| d01 (control) | 0,1 | — | `runs/reg/d01` |

Común a los tres: `python -m fuga.train --datos data/corpus_pd.jsonl --modo eco_rel --preset fuga-s --ctx 1024
--cache cache/eco_rel_1024.pkl --device cuda --ckpt --tokens_por_lote 8192 --pasos 3000 --lr 6e-4 --wd 0.1
--eval_cada 250 --guardar_mejor --semilla 0`. Una sola semilla por brazo (presupuesto: ~1 h por brazo).

El control a 3 000 pasos es necesario porque la corrida de 3 h tenía el mínimo durante la meseta de la tasa de
aprendizaje, sin decaimiento; un presupuesto de 3 000 pasos con WSD completo puede mejorar por sí solo. La referencia
de 3 h (`runs/fuga_s_rel_3h/modelo_mejor.pt`, 3,533 val / 3,848 test) se incluye como fila informativa.

## Métricas y criterios (se fijan ahora)

Para cada brazo se evalúa `modelo_mejor.pt` (mejor evaluación parcial) y `modelo.pt` (final) sobre la validación
completa (617 trozos) y el test completo (631 trozos de 475 piezas, `cache/test_eco_rel_1024.pkl`), en bits por nota.

- **KR1 (principal).** Para cada brazo con dropout alto, IC95 bootstrap (10 000 remuestreos, semilla 0) de la diferencia
  pareada por pieza de bits por nota en test entre su `modelo_mejor.pt` y el del control (brazo − control). Pasa si el
  IC95 queda enteramente por debajo de 0. Si queda enteramente por encima, el dropout alto **perjudica**. Si cruza 0, sin
  efecto detectable con una semilla.
- **KR2 (informativa).** Paso del mínimo de validación parcial en cada brazo: si crece con el dropout, el dropout retrasa
  la memorización aunque no mejore el mínimo.
- **KR3 (informativa).** Distancia entre `modelo.pt` y `modelo_mejor.pt` en val completo por brazo: mide cuánto
  sobreajuste queda a 3 000 pasos con cada dropout.

## Hipótesis declarada del director

Con 5M de tokens y 151M de parámetros, espero que dropout 0,2 mejore ligeramente al control (IC95 < 0, del orden de
0,05 a 0,15 bits por nota) y que 0,3 ya no mejore o perjudique por subajuste a 3 000 pasos. Se registra para poder
estar equivocado.

## Desviaciones

Cualquier desviación respecto a este documento se anota en `ensayo/regularizacion_informe.md` con su motivo.
