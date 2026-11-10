# Pre-registro: FUGA-S eco_rel, 6 horas con checkpoints periódicos en Z-Space

- Fecha de fijación: 2026-11-08, antes de lanzar la corrida
- Antecedentes: la corrida de 3 h (`ensayo/fuga_s_3h.md`) tocó mínimo a las 5,4 épocas con dropout 0,1 y luego memorizó;
  el barrido de dropout (`ensayo/prerregistro_regularizacion.md`) a 3 000 pasos no mostró sobreajuste con 0,2 ni 0,3.
- Novedad de ingeniería: `fuga/train.py --guardar_cada N` guarda cada N pasos una versión bf16 de los pesos en un almacén
  Z-Space (`fuga/zcheckpoints.py`, sobre z-space-core 0.1.0, MIT), con deltas XOR+zstd sin pérdida y un checkpoint
  completo cada 8 versiones; `--inicial` permite arrancar en caliente desde cualquier versión exportada.

## Corrida

`python -m fuga.train --datos data/corpus_pd.jsonl --modo eco_rel --preset fuga-s --ctx 1024 --cache cache/eco_rel_1024.pkl
--device cuda --ckpt --tokens_por_lote 8192 --minutos 360 --lr 6e-4 --wd 0.1 --dropout D --eval_cada 250 --guardar_mejor
--guardar_cada 500 --semilla 0 --out runs/fuga_s_6h`

**Regla para D, fijada ahora:** el dropout del brazo del barrido (d01 = 0,1, d02 = 0,2, d03 = 0,3) cuyo `modelo_mejor.pt`
tenga menor bits por nota en la validación completa, según `ensayo/regularizacion_resultados.json`. La cadena lo elige
sola y lo anota en `runs/fuga_s_6h/eleccion_dropout.json`. Una semilla.

## Métricas y criterios

- **K6-1 (calidad).** Bits por nota en validación completa (617 trozos) y test completo (631 trozos) de `modelo_mejor.pt`.
  Referencias: 3 h con dropout 0,1 → 3,533 val / 3,848 test; barrido a 3 000 pasos → 3,484 (0,2) y 3,443 (0,3) en val.
  Se declara mejora si val < 3,40.
- **K6-2 (sobreajuste).** Paso del mínimo de validación parcial y distancia val(`modelo.pt`) − val(`modelo_mejor.pt`).
- **KZ1 (tamaño).** Bytes del almacén Z-Space frente a los bytes que ocuparían las mismas versiones con `torch.save` en
  bf16 y en fp32. Se considera útil si el almacén baja del 60 % del equivalente bf16 (30 % del fp32).
- **KZ2 (exactitud).** La última versión del almacén reconstruye bit a bit `modelo.pt` redondeado a bf16; si existe
  versión en el paso del mínimo, también `modelo_mejor.pt`. Debe ser exacto.
- **KZ3 (coste).** Segundos por versión guardada, medidos en el registro; aceptable si la suma es < 3 % del reloj total.
- **KZ4 (utilidad bf16).** Bits por nota del export Z-Space (pesos bf16 → fp32) frente al `.pt` fp32 del mismo paso:
  diferencia esperada < 0,001.
- **K6-3 (muestras).** 12 piezas con `modelo_mejor.pt`: copia de ventanas de 2 y 4 compases frente a train (se espera 0),
  notas en escala, repetición, modo y textura, como en `ensayo/fuga_s_3h.md`.

## Hipótesis declarada del director

Con el dropout ganador del barrido, el mínimo llegará más tarde que a las 5,4 épocas pero bastante antes de las 6 h, y
quedará entre 3,30 y 3,45 en val; el modelo final volverá a estar sobreajustado. El almacén Z-Space quedará entre el 50 %
y el 80 % del equivalente bf16: los deltas de 500 pasos en bf16 no son tan dispersos como los del benchmark del paquete
(50 pasos). Se registra para poder estar equivocado.

## Desviaciones

- **2026-11-08, corrida cortada.** D resultó 0,3 (`runs/fuga_s_6h/eleccion_dropout.json`). El equipo se reinició sin
  apagado limpio (evento Kernel-Power 41) con el entrenamiento en el paso 5 575, el 31 % del presupuesto. Sin error de
  Python. Se conservan `modelo_mejor.pt` (paso 4 250: 3,433 val / 3,718 test) y el almacén Z-Space, que se reabre y
  verifica tras el corte (11 versiones, 78,9 % del equivalente bf16); detalle en `runs/fuga_s_6h/rescate.json`.
- **2026-11-09.** La corrida completa se traslada a una H100 con el presupuesto traducido a pasos; el plan, la regla de
  elección y los criterios vigentes están en `ensayo/prerregistro_final.md`.
