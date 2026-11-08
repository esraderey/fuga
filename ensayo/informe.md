# Informe de cierre del ensayo FUGA en CPU

- Fecha: 2026-10-25
- Pre-registro: `ensayo/prerregistro.md` (fijado el 2026-10-24, antes de ver resultados)
- Resultados crudos: `ensayo/resultados.json` (brazo A `base`+`eco`, 533 s; brazo B `fant`+`eco_rel`, 586 s; fusionados)
- Código evaluado: el posterior a la revisión adversarial G4 (`expedicion/revision_g4.md`), con M1–M6 corregidos

## 1. Diseño ejecutado

| Elemento | Valor |
|---|---|
| Corpus | music21 core, dominio público: 14 955 piezas (train 14 102 / val 375 / test 478), partición por archivo |
| Modelo | preset `nano`, 2,50 M parámetros, ctx 512, vocab 723 |
| Entrenamiento | 360 pasos × 8 192 tokens, AdamW, WSD, lr 2e-3, dropout 0,0, CPU |
| Brazos | `base` (literal), `eco`, `fant` (ECO + fantasmas), `eco_rel` (ECO en alturas relativas) |
| Evaluación | bits/nota en test (mismos compases en los cuatro brazos); 60 prompts de 4 compases → continuación hasta 16; adherencia tonal 24 tonalidades × 10 generaciones con juez Temperley–Kostka–Payne; perilla de forma 16 muestras × 3 sesgos |

## 2. Veredictos sobre las hipótesis pre-registradas

| Id | Hipótesis | Criterio | Resultado | Veredicto |
|---|---|---|---|---|
| K1 | ECO modela mejor que literal | IC95 de bits/nota (eco − base) < 0 | eco − base = −0,80 [−0,89, −0,72]; fant − base = −0,63 [−0,71, −0,55] | **pasa** |
| K2 | ECO acerca la estructura a la real | IC95 de D (eco − base) < 0 | eco − base = +0,11 [−0,04, +0,25]; fant − base = +0,12 [−0,05, +0,29] | **falla** |
| K3 | Materializar fantasmas ayuda | IC95 de bits/nota (fant − eco) < 0 | +0,18 [+0,15, +0,21] | **falla** (informativo) |
| K4 | Coordenadas relativas ahorran datos | bits/nota (eco_rel − eco) < 0 y adherencia ≥ | −0,05 [−0,07, −0,03]; adherencia exacta 0,56 frente a 0,31 | **pasa** (informativo) |
| K5 | La gramática garantiza decodificación | 0 fallos | 0 fallos en todas las generaciones de los cuatro brazos | **pasa** |
| K6 | La perilla controla la forma | máx P_trans crece de −3 a +3 con IC disjuntos | eco: 0,00 → 0,19 [0,11, 0,29] → 0,39 [0,25, 0,52]; fant: 0,01 → 0,10 → 0,22 | **pasa** en `eco` y `fant`; **falla** en `eco_rel` (0,23 → 0,23, satura) |

Consecuencias pre-registradas que se aplican:

- K2 falla: se retira la reclamación de que ECO resuelve la estructura de largo plazo (casilla F1 del mapa). Lo que queda demostrado es compresión (K1) y control de forma en inferencia (K6).
- K3 falla: el RFC no exige fantasmas. Quedan como opción de representación.
- K5 pasa: no hay defecto de implementación que bloquee el RFC.

## 3. Hallazgos no pre-registrados

| Métrica | base | eco | fant | eco_rel | real (test) |
|---|---|---|---|---|---|
| Adherencia tonal exacta (juez TKP) | 0,058 | 0,308 | 0,158 | 0,558 | techo del juez 0,76 |
| Fracción de notas en la escala declarada | 0,72 | 0,94 | 0,84 | 0,97 | 0,94–0,95 |
| JS de clases de altura (gen frente a real) | 0,088 | 0,016 | 0,035 | 0,003 | — |
| Copia literal de 4 compases presente en train | 0,005 | 0,005 | 0,004 | 0,005 | 0,052 |
| Compases completos generados (media, tope 16) | 13,1 | 13,4 | 13,4 | 14,6 | — |

- El efecto más grande del ensayo no estaba pre-registrado: la adherencia a la tonalidad pedida pasa del 6 % en `base` al 56 % en `eco_rel`, con un techo del juez del 76 % sobre música real. El marco relativo a la tónica convierte la tonalidad en una coordenada y no en algo que el modelo tenga que aprender.
- No hay memorización detectable: la copia literal de ventanas de 4 compases del train queda diez veces por debajo de la que exhibe la propia música real del test.
- El ruido de la métrica D entre dos piezas reales es 0,93 [0,65, 1,26], mayor que cualquier D entre generación y original (0,44–0,62). D mide sobre todo idiosincrasia por pieza y tiene poco poder para K2. El pre-registro ya anticipaba que faltaba el destructor D7.

## 4. Limitaciones que condicionan la lectura

1. **K3 está confundido.** El brazo `fant` recibió 1,90 M tokens supervisados frente a 2,87 M en los demás, porque los fantasmas consumen presupuesto de lote sin aportar pérdida. El contraste mide «igual cómputo», no «igual supervisión». Con igual supervisión el resultado podría invertirse; no se midió.
2. **Régimen subentrenado.** 2,5 M parámetros y menos de una época. Un «sobrevive» habilita la escalera de GPU; no confirma nada a 150–400 M.
3. **Corpus sesgado.** El 87 % de las piezas son monofónicas y el 85 % son canciones folk; por notas, Palestrina aporta un tercio. ECO gana en repertorio repetitivo y cuantizado. La transferencia a Lakh o PDMX no está medida.
4. **Métrica de estructura sin destructor D7.** Cualquier operador de copia puede ganar una métrica de repetición por lag sin función musical. Hasta tener D7, la estructura no se reclama.
5. **Truncado por contexto.** Con ctx 512, parte de las continuaciones no llegó a 16 compases (media 13–15); la comparación pareada usa solo compases completos comunes, pero la longitud difiere entre brazos.

## 5. Decisiones que pasan a la serie RFC

1. ECO se mantiene como representación (K1) y como perilla de forma (K6). No como solución de estructura (K2).
2. El marco relativo a la tónica (`eco_rel`) pasa a ser el modo recomendado por defecto, por bits/nota y por adherencia tonal.
3. Los fantasmas quedan opcionales (`--modo fant`), pendientes de un ensayo con igual supervisión.
4. La siguiente escalera es en GPU con FUGA-S, corpus con polifonía real y el destructor D7 antes de volver a medir F1.
5. El intérprete FUGA-I (RFC-FUGA-2) consume la salida simbólica tal cual. Su capa determinista garantiza por construcción que el audio contiene exactamente las notas de la partitura; la coherencia de doble salida no depende de un transcriptor.

## 6. Reproducción

```bash
python -m fuga.train --datos data/corpus_pd.jsonl --modo eco_rel --preset nano --ctx 512 --pasos 360 --dropout 0.0 --device cpu --out runs/eco_rel
python scripts/ensayo.py --datos data/corpus_pd.jsonl --runs runs/base runs/eco runs/fant runs/eco_rel --out ensayo/resultados.json --audio_dir ensayo/audio
```

El ensayo requiere `music21` (juez tonal). Los cuatro checkpoints evaluados están en `runs/*/modelo.pt`.

## 7. Ensayos posteriores de la serie RFC

- E1 de RFC-FUGA-2 (pares por transcripción sobre FMA CC BY/CC0, 2026-10-25): **MUERE (D3)**. La alineación y la compuerta pasan (mediana de |desfase| 15 ms; 55 % pasa τ = 0,6) pero solo quedan 231 h útiles proyectadas (< 300). Informe `ensayo/e1_informe.md`, datos `ensayo/e1_pares.json`, pre-registro `ensayo/e1_prerregistro.md`, script `scripts/e1_pares.py`.

- 2026-10-30, ensayo MZTrain (`ensayo/mztrain_informe.md`): FUGA-M factorizado con MZTrain 1.5.0 (rango 512) bate a FUGA-S denso a igual número de tokens (−0,036 bits/nota por pieza, IC95 [−0,045; −0,028], 2 semillas), pero reserva 8,8–8,9 GB, cae en el acantilado WDDM y tarda 6,6–7,2 veces más; no sustituye a FUGA-S denso como referencia local.
- 2026-10-30, FUGA-S eco_rel 3 h en la RTX 4060 (`ensayo/fuga_s_3h.md`): 9 436 pasos, 75M tokens (15 épocas); el mínimo de validación llega en el paso 3 250 (3,53 bits/nota en val completo y 3,85 en test, frente a 4,49 y 4,87 de FUGA-S a 700 pasos) y después el modelo memoriza (6,84 al final). `--guardar_mejor` (nuevo en `fuga/train.py`) conserva el checkpoint útil; 12 muestras sin copia de 2 ni 4 compases del train.
