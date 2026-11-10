# Pre-registro: entrenamiento final de FUGA-S en una H100 (RunPod) y publicación

- Fecha de fijación: 2026-11-09, antes de lanzar ninguna corrida en la H100
- Sustituye, para la corrida completa, a `ensayo/prerregistro_6h.md`, cuya ejecución local quedó cortada (ver «Estado»)
- Cadena: `scripts/cadena_final.py`; arranque en el pod: `scripts/runpod/arranque.sh`; datos: dataset privado
  `esraderey/fuga-datos`, verificados contra `scripts/runpod/datos.sha256.json`

## Estado de partida (evidencia ya disponible)

- **Corrida local de 6 h, cortada.** El equipo se reinició sin apagado limpio el 2026-11-08 (evento Kernel-Power 41) con
  el entrenamiento en el paso 5 575 de unos 18 380 (31 %). No hay error de Python. Quedan `runs/fuga_s_6h/modelo_mejor.pt`
  (paso 4 250) y el almacén Z-Space con 11 versiones hasta el paso 5 500 (`runs/fuga_s_6h/rescate.json`):
  - mejor checkpoint parcial (paso 4 250, dropout 0,3, en plena meseta de la tasa): **3,433 val / 3,718 test**;
  - la validación parcial sube después (3,99 en el paso 5 000) con la tasa aún en el pico;
  - el almacén Z-Space se reabre y verifica tras el corte; 78,9 % del equivalente bf16; ritmo 1,1752 s/paso.
- **Barrido de dropout a 3 000 pasos** (`ensayo/regularizacion_resultados.json`): con decaimiento completo el mejor
  checkpoint es el final; 3,443 val con dropout 0,3 y 3,486 con 0,1; ningún brazo se sobreajusta.
- **Corrida de 3 h** (`ensayo/fuga_s_3h.md`): mínimo en el paso 3 250, en la meseta; 3,533 val; el final memoriza (6,84).

Lectura: con este corpus el mínimo llega hacia los 4 000–5 000 pasos. Una corrida larga pasa el 70 % de su presupuesto
memorizando y su mejor checkpoint es una instantánea de la meseta, sin el beneficio del decaimiento; un presupuesto corto
cuyo decaimiento termine cerca de ese punto debería dar un modelo mejor.

## Corridas (todas: FUGA-S, `eco_rel`, ctx 1024, lotes de 8 192 tokens, AdamW lr 6e-4, decaimiento 0,1, dropout 0,3, WSD
con 5 % de calentamiento y 20 % final de decaimiento, bf16, semilla 0, evaluación parcial cada 250 pasos, `--guardar_mejor`)

| Corrida | Pasos | Qué es |
|---|---|---|
| `completo` | 18 380 | La corrida de 6 h pre-registrada, traducida a pasos: 21 600 s / 1,1752 s por paso medidos en la RTX 4060. Con `--guardar_cada 500` (almacén Z-Space bf16). |
| `wsd_4500` | 4 500 | Presupuesto corto; la meseta acaba en el paso 3 600. |
| `wsd_6000` | 6 000 | Presupuesto corto; la meseta acaba en el paso 4 800. |
| `wsd_8000` | 8 000 | Presupuesto corto; la meseta acaba en el paso 6 400. |

Desviaciones respecto a `prerregistro_6h.md`, fijadas ahora: presupuesto en pasos en vez de minutos (el reloj de la H100
no es comparable); sin checkpointing de activaciones (80 GB de memoria; mismo cálculo, salvo redondeo); Linux y CUDA en
vez de Windows. El dropout 0,3 es el que eligió la regla del pre-registro anterior.

## Regla de elección del modelo publicado (fijada ahora)

Entre `modelo_mejor.pt` y `modelo.pt` de las cuatro corridas, el de menor bits por nota en la **validación completa**
(617 trozos). El test completo (631 trozos de 475 piezas) se calcula para todos y no interviene en la elección. El
elegido se exporta a safetensors con su ficha y se publica en `esraderey/fuga-s-eco-rel`; los registros van en `corrida/`.

## Criterios

- **KF1 (principal).** IC95 bootstrap (10 000 remuestreos, semilla 0) de la diferencia pareada por pieza de bits por nota
  en test entre el mejor checkpoint de los presupuestos cortos y el mejor de la corrida `completo` (corto − largo), ambos
  elegidos por validación. Pasa si el IC95 queda entero por debajo de 0.
- **K6-1.** Val y test del elegido; mejora declarada si val < 3,40 (referencias: 3,433 parcial local, 3,443 barrido).
- **K6-2.** En `completo`: paso del mínimo de validación parcial y val(`modelo.pt`) − val(`modelo_mejor.pt`).
- **KZ1.** Almacén Z-Space de `completo` frente al equivalente `torch.save` en bf16 y en fp32; útil si < 60 % del bf16.
- **KZ2.** La última versión reconstruye bit a bit `modelo.pt` redondeado a bf16 (y `modelo_mejor.pt` si hay versión en su paso).
- **KZ3.** Segundos por versión guardada. Criterio original (< 3 % del reloj) referido al reloj equivalente de la RTX 4060
  (21 600 s); se informa además la fracción del reloj real de la H100, que será mayor por construcción.
- **KZ4.** No se repite (medido: +0,00004 bits por nota al redondear a bf16).
- **K6-3.** 12 piezas con el elegido: copia de ventanas de 2 y 4 compases frente a train (se espera 0), notas en escala,
  repetición frente a la música real. Ilustran la ficha las dos piezas completas, sin copia, de repetición más parecida
  a la real (`elegir_muestras` en `scripts/cadena_final.py`).

## Hipótesis declarada del director

El elegido será el modelo final de `wsd_6000`, con 3,28–3,38 en val. El mejor checkpoint de `completo` caerá entre los
pasos 4 000 y 6 000 con 3,40–3,46, y su modelo final estará sobreajustado (> 5). KF1 pasará con un efecto pequeño, de
−0,03 a −0,10 bits por nota. El almacén Z-Space acabará entre el 72 % y el 80 % del equivalente bf16 (KZ1 falla), con
deltas más pequeños en el 20 % final. Se registra para poder estar equivocado.

## Tiempo previsto (estimación; la sonda del primer minuto la sustituye por una medida)

36 880 pasos en total. Con 0,10–0,20 s por paso en una H100 (7 a 9 veces la RTX 4060 medida, sin checkpointing de
activaciones): 60–125 min de entrenamiento, más unos 25 min de arranque, guardados Z-Space, evaluación, muestras, export
y subida. Solo `completo`: 30–60 min de entrenamiento.
