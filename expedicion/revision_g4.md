# Revisión adversarial G4 — paquete FUGA (/home/claude/fuga)

Alcance: fuga/*.py, scripts/*.py, tests/test_fuga.py. Solo lectura; scripts de prueba en el scratchpad (a1.py, a2.py, b.py, c.py, d.py, e.py, f.py, g.py). Tool calls usadas: 29 de 40.
Línea base: `pytest -q -x tests` -> `16 passed in 24.96s`.
Resultado: 0 hallazgos ALTA, 6 MEDIA, 11 BAJA. Cada hallazgo indica CONFIRMADO (ejecutado) o SOSPECHA (solo lectura).

Contexto del ensayo real (de ensayo/prerregistro.md y runs/*.log): ctx 512, preset nano, 360 pasos, 4 brazos (base, eco, fant, eco_rel), vocab 723, corpus de 14955 piezas.

---------------------------------------------------------------------------------------------------

## MEDIA

### M1. La gramática rechaza lo que `codificar` produce: mismo (pos, pitch) con duraciones distintas en una parte
- Clase: 5 (y 4).
- Ubicación: fuga/tokenizer.py:351 (`out += self._ids("PITCH_", self.ult_pitch + 1)`), junto con fuga/tokenizer.py:158 (`s[p] = sorted(set(s[p]))`).
- Causa: `set` solo elimina duplicados exactos. Dos notas con el mismo (pos, pitch) y DUR distinta sobreviven. `literal()` emite `POS PITCH_60 DUR_a PITCH_60 DUR_b`, pero tras un DUR la gramática exige un pitch estrictamente mayor.
- Entrada: `notas=[(0,12,60,0),(0,24,60,0),(12,12,62,0)]`, `compases=[0,48]`.
- CONFIRMADO:
  ```
  dup_pitch_dur_distinta     DEC-EXC ValueError('token PITCH_60 no permitido en fase dur')
  dup_pitch_dur_distinta_eco DEC-EXC ValueError('token PITCH_60 no permitido en fase dur')
  ```
- Incidencia real: 20 de 14955 piezas (19 train, 0 test, 0 en los 60 prompts del ensayo), p. ej. beethoven/movement1.mxl#0, haydn/movement3.mxl#0.
  Con una pieza así como prefijo:
  `generar con prefijo de haydn/movement3.mxl#0 -> ValueError prefijo inválido en PITCH_68` (fuga/sample.py:56).
- Efecto: rompe el invariante central "permitidos() acepta todo lo que codificar produce". Esas piezas no se pueden usar como prompt. En ensayo.py cuentan como "fallos" de decodificación y violan K5. Impacto actual en el ensayo: nulo, porque el test no contiene piezas así.
- Severidad: MEDIA.

### M2. Off-by-one en el tope de compases: el último compás sale vacío
- Clase: 2 (y 6).
- Ubicación: fuga/sample.py:51 — `if dec.bar + 1 >= max_compases and voc["<eos>"] in perm and dec.fase in (...)`.
- Causa: `dec.bar + 1` cuenta los compases abiertos (BAR emitidos). La condición se cumple justo después del BAR número `max_compases`, en fase `post_bar`, y fuerza `<eos>` antes de que ese compás reciba contenido.
- Entrada: `generar(..., max_compases=N)`.
- CONFIRMADO (prefijo de 12 compases forzado):
  ```
  max_compases 3 | compases abiertos (BAR) 3 | compases con notas 2 | ultimo vacio: True | fin True
  max_compases 6 | compases abiertos (BAR) 6 | compases con notas 5 | ultimo vacio: True | fin True
  ```
- Efecto: en scripts/ensayo.py, (b) pide "compases 4..15" (`max_compases=16`) y produce 4..14, mientras `real_lag` y `nr` usan 4..15. (c) y (d) piden 8 y 16 compases y producen 7 y 15. Afecta por igual a todos los brazos, así que es un sesgo sistemático, no una diferencia entre brazos. El `D_gen_vs_real` compara generado con 11 compases contra real con 12. La condición correcta sería `dec.bar >= max_compases` (cerrar el compás N antes del eos).
- Severidad: MEDIA.

### M3. `tasas_lag`: "near" y "trans" son triviales con segmentos cortos o sin fuente
- Clase: 10 (tasas por lag).
- Ubicación: scripts/ensayo.py:95 — `elif 1 <= len(set(f) ^ set(s)) <= 2:`; y scripts/ensayo.py:97-99 (`if f and len(f) == len(s) and [(x[0], x[2]) ...`).
- Causas:
  1. Si la parte no existe en `b-lag`, `f=[]` y `|f Δ s| = |s|`. Todo segmento de 1-2 notas cuenta como "near" de un segmento inexistente.
  2. Cambiar una nota da `|Δ| = 2`. Por tanto, dos segmentos de 1 nota cualesquiera que sean distintos son siempre "near".
  3. `P_trans` no exige un mínimo de notas, mientras que el codificador ECO exige `lit > 4`. Dos segmentos de 1 nota con el mismo pos y dur cuentan como "copia transpuesta".
- CONFIRMADO:
  ```
  tasas_lag fuente ausente (ex,tr,near) lag1: (0.5, 0.5, 0.5)     # parte 1 sin fuente -> near=0.5
  tasas_lag 1 nota distinta (ex,tr,near): (0.0, 1.0, 1.0)         # 60 vs 62 -> trans=1 y near=1
  tasas_lag 2 notas, ambas distintas:     (0.0, 0.0, 0.0)
  ```
- Efecto: cada segmento de 1-2 notas (compases con notas largas) o sin fuente suma a P_near y P_trans sin que haya repetición real; el peso depende de la fracción de esos segmentos, que no medí. `D` (K2) y `max P_trans` de la perilla (K6) incorporan ese ruido estructural. Afecta a real y generado, pero amortigua la señal que K2 y K6 quieren medir.
- Severidad: MEDIA.

### M4. En modo relativo la gramática acepta alturas cuyo valor absoluto decodificado sale de 0..127
- Clase: 5 (y 7).
- Ubicación: fuga/tokenizer.py:346 y :364 (`return self._ids("PITCH_")`, todas las 128) y fuga/tokenizer.py:473 (`pitch + self.desp`).
- Entrada: `relativo=True`, `KEY_3_major` (desp=+3), `PITCH_126`.
- CONFIRMADO:
  ```
  rel: notas decodificadas [(0, 4, 129, 0)]
  a_midi: ValueError data byte must be in range 0..127
  ```
- Efecto: una generación del brazo `eco_rel` puede producir pitch > 127 (tónicas 1..5) o < 0 (tónicas 7..11). `a_midi` lanza ValueError, sin captura en ensayo.py (e) ni en sample.main, y aborta el resto del ensayo. `test_generacion_siempre_decodificable` solo prueba el modo absoluto (aserto `0 <= p <= 127`). Con un modelo entrenado y top_p=0.95 es poco probable; con top_p=1.0, temperatura alta o un modelo débil, es posible.
- Severidad: MEDIA.

### M5. Ensayo (b): la generación se corta por contexto de forma distinta en cada brazo
- Clase: 6 (distintos compases).
- Ubicación: scripts/ensayo.py:188 (`max_tokens=ctx, max_compases=16`) y :194 (`tasas_lag(dec.segs, 4, 16)`).
- Causa: la generación se detiene en `t == max_tokens = ctx = 512` (fuga/sample.py:47) sin importar cuántos compases lleve. Los brazos tienen longitudes en tokens muy distintas. `dec.segs` queda más corto y con un último compás parcial. `tasas_lag` usa `min(b_fin, len(segs))` y no lo reporta.
- CONFIRMADO (longitudes reales, 60 prompts del ensayo, 16 compases, ctx 512):
  ```
  base | frac(16 compases > 512 tok)=0.10 | compases que caben en 512 (media)=20.8
  eco  | frac(16 compases > 512 tok)=0.07 | compases que caben en 512 (media)=21.3
  fant | frac(16 compases > 512 tok)=0.13 | compases que caben en 512 (media)=20.3
  ```
- Efecto (SOSPECHA, no se corrió el ensayo con modelos): 6, 4 y 8 de 60 prompts quedan truncados en base, eco y fant respectivamente. `fant` sale perjudicado por construcción, porque sus tokens fantasma consumen contexto. La comparación pareada por prompt (K2) mezcla ventanas de distinta longitud.
- Severidad: MEDIA.

### M6. Ensayo (b): `except Exception` mal etiquetado y descarte de todo el brazo ante un solo fallo
- Clase: 9 (y 6).
- Ubicación: scripts/ensayo.py:190 (`except Exception as e:  # se cuenta: la gramática debería impedirlo`) y :204-205 (`... if not fallos else {}`).
- Causa: cualquier excepción (prefijo inválido por M1, RuntimeError de forma, IndexError de RoPE, etc.) se cuenta como "fallos_decodificacion" y se continúa. Si `fallos > 0`, `d_por_brazo[nombre] = {}`, y todas las comparaciones D pareadas de ese brazo se vuelven `nan` (`comunes` vacío), en lugar de usar los prompts comunes.
- Efecto: un único prompt problemático anula K2 para un brazo. La gran mayoría de los fallos que no sean de decodificación se confunden con K5. La excepción se imprime pero se pierde el tipo en el JSON.
- SOSPECHA (lectura del flujo; no se ejecutó con un modelo).
- Severidad: MEDIA.

---------------------------------------------------------------------------------------------------

## BAJA

### B1. `trozos_eval`: un compás único que no cabe en ctx se trunca, pero se cuentan todas sus notas
- Clase: 6 y 10 (bits por nota).
- Ubicación: fuga/data.py:82-84 — `notas = cods["base"].notas` / `ids, comp, per = c.ids[:ctx], c.compas[:ctx], c.perdida[:ctx]`; salida del bucle en :79 (`if b2 == b + 1 or ...`).
- Entrada: un compás con 2 partes x 40 notas, ctx=200.
- CONFIRMADO: `trozos_eval base [(200, 80, None)]  eco [(132, 80, None)]  fant [(200, 80, 132)]`. Es decir, base y fant quedan cortados a 200 tokens (sin `<eos>`), eco cabe entero (132), y las tres cuentan 80 notas.
- Efecto: el denominador de bits/nota incluye notas cuyos tokens no se puntúan, y la truncación difiere entre representaciones.
- Incidencia actual: 0 compases con >= 160 notas en los 11273 compases del test (ctx 512). Latente.

### B2. Ida y vuelta no exacta si `compases[0] != 0` o hay notas antes del primer compás
- Clase: 4 (anacrusa).
- Ubicación: fuga/tokenizer.py:147 (`if parte >= MAX_PARTES or on < ini[0]:`) y :303 (`self.inicio = 0`).
- Entrada: `compases=[12,60,108]`, notas en 0,6,12,60.
- CONFIRMADO: `compases[0]=12 MISMATCH exp=[(0,6,60,0),(6,6,62,0),(12,12,64,0),(60,12,65,0)] got=[(0,12,64,0),(48,12,65,0)]`. Se pierden las notas previas y los onsets quedan desplazados en `ini[0]`.
- Incidencia: 0 piezas en el corpus (min(compases)=0 siempre). Latente. ensayo.py:198 y :148 (`sum(largos[:4])`) heredan el supuesto `ini[0]=0`.

### B3. Modo relativo descarta notas extremas en silencio (ida y vuelta no exacta)
- Clase: 4 (notas fuera de rango).
- Ubicación: fuga/tokenizer.py:153-154 — `q = pitch + t - (desplazamiento_tonica(...) if relativo else 0)` / `if 0 <= q <= 127:`.
- Entrada: tónica 6, pitch 125 (q=131); tónica 5, pitch 3 (q=-2).
- CONFIRMADO: `rel_pitch125_T6 MISMATCH ... got=[(12,12,66,0),(24,12,60,0)]` y `rel_pitch3_T5 MISMATCH`.
- Incidencia: 0 piezas del corpus con notas <= 4 o >= 122. Latente. Efecto colateral: `notas` difiere ligeramente entre brazos relativo y absoluto, y ensayo.py:270 normaliza por las notas del primer brazo.

### B4. Estilo fuera del vocabulario se codifica como `<pad>` y la gramática lo rechaza
- Clase: 5.
- Ubicación: fuga/tokenizer.py:226 — `... in voc.stoi else voc["<pad>"],`.
- CONFIRMADO: `estilo_desconocido ids[1]= <pad>` y `DEC-EXC ValueError('token <pad> no permitido en fase sty')`.
- Incidencia: nula en los runs reales (vocab 723 incluye las 34 fuentes). Latente si se usa `--max_piezas` o un corpus nuevo con el vocab de otro checkpoint. En el ensayo caería en M6.

### B5. Segmentos ECOR decodificados pueden quedar con orden de altura no canónico
- Clase: 10 (tasas por lag) y 5.
- Ubicación: fuga/tokenizer.py:459-462 (asigna alturas en el orden del ritmo fuente) con la gramática de :363-364 (cualquier PITCH).
- CONFIRMADO: ECOR sobre un acorde, con pitches `[64, 60]` aceptados. Segmento decodificado `[(0,64,8),(0,60,8)]` frente al canónico `[(0,60,8),(0,64,8)]`.
- Efecto: `list(f) == list(s)` (ensayo.py:93) y el hash de `_clave_ventana` (metrics.py:115) dependen del orden, de modo que P_exact y la copia de n-gramas subestiman coincidencias con música real, que siempre viene ordenada. Solo afecta a acordes dentro de una misma parte.

### B6. Adherencia tonal: el denominador excluye las generaciones vacías
- Clase: 6.
- Ubicación: scripts/ensayo.py:222-224 — `if not notas: continue` / `cuenta["n"] += 1`.
- Efecto: un brazo que emite `<eos>` antes de la primera nota queda con `n` menor y no se penaliza. Se reporta `n` en el JSON, pero las tasas no son comparables entre brazos si `n` difiere. SOSPECHA (lectura).

### B7. Convención de signo y estimador de las diferencias pareadas
- Clase: 10.
- Ubicación: scripts/ensayo.py:269-270 — `pares_b[f"{nombres[i]} - {nombres[j]}"] = bootstrap(... (A[k][0] - B[k][0]) / A[k][1]`.
- Detalle 1: con `--runs base eco fant eco_rel` las claves salen "base - eco", etc. El pre-registro define K1 como (eco − base) < 0, K3 como (fant − eco) < 0 y K4 como (eco_rel − eco) < 0. Leída mecánicamente, la clave "base - eco" tiene el signo invertido respecto a K1.
- Detalle 2: la media de cocientes por pieza no pondera por notas, y no coincide con la diferencia de los `test_bits_por_nota` agregados que el mismo JSON reporta. Es un estimador legítimo, pero distinto del titular.
- SOSPECHA (lectura).

### B8. Corpus: excepciones tragadas y orden no determinista
- Clases: 9 y 8.
- Ubicación: scripts/extraer_corpus.py:24-27 (`except Exception: pass` tras `stripTies()`), :50-54 (`except Exception: tonica, modo = 0, "major"`), :63 (`modo if modo in ("major", "minor") else "major"`) y :100 (`pool.imap_unordered(...)`).
- Efecto: si falla la análisis de tonalidad, la pieza queda etiquetada C mayor sin avisar. Eso contamina el token KEY, las decisiones TRD de ECO y el "techo" del juez. Los modos dorian/mixolydian se fuerzan a major. `imap_unordered` hace que el orden de `corpus_pd.jsonl` varíe entre extracciones con procs > 1, y de él dependen `--max_piezas` (train.py:78) y el `rng.shuffle(prompts)` de ensayo.py:136. SOSPECHA (lectura).

### B9. Entrenamiento por reloj de pared no es reproducible
- Clase: 8.
- Ubicación: fuga/train.py:110 — `f = (time.time() - t_ini) / presupuesto if a.minutos else paso / a.pasos`.
- Efecto: con `--minutos` (el ejemplo del docstring) el número de pasos, el calendario WSD y los tokens vistos dependen de la carga de la máquina. Los brazos reciben presupuestos distintos de pasos. El ensayo pre-registrado usa pasos, así que no le afecta. SOSPECHA (lectura).

### B10. Los valores por defecto de generación superan el contexto de entrenamiento del ensayo
- Clase: 1.
- Ubicación: fuga/sample.py:91 — `ap.add_argument("--max_tokens", type=int, default=1024)`.
- Efecto: los checkpoints del ensayo se entrenan con ctx=512. `sample.main` genera hasta 1024 tokens, con posiciones RoPE no vistas en entrenamiento, sin recortar a `model.c.ctx`. La tabla RoPE no desborda (4096), así que no hay error visible, solo degradación. ensayo.py sí usa `max_tokens=ctx`. SOSPECHA (lectura).

### B11. `copia_ngramas` forma ventanas con compases no contiguos
- Clase: 10.
- Ubicación: fuga/metrics.py:94-95 — `bs = sorted({b for b, _ in segs})` / `for i in range(len(bs) - n + 1):`.
- Efecto: la ventana de "n compases" son n compases con contenido, aunque haya compases vacíos entre medias; los índices `j` de la clave tratan esos compases como adyacentes. Aplica por igual al índice de train y al generado, pero la métrica no es "4 compases consecutivos". SOSPECHA (lectura).

---------------------------------------------------------------------------------------------------

## Revisado sin hallazgos

- **Clase 1, posiciones y compás entre entrenamiento y generación.** Se interceptó `model.forward` durante `generar(prefijo_ids=...)` y se comparó con `colar`:
  ```
  base L 402  posiciones iguales a entrenamiento: True | max|KVcache - pase completo| = 1.4e-06
  eco  L 173  posiciones iguales a entrenamiento: True | max|KVcache - pase completo| = 1.7e-06
  fant L 518  posiciones iguales a entrenamiento: True | max|KVcache - pase completo| = 1.4e-06
  ```
  El compás de la cabecera es 0, BAR y su contenido llevan `bar+1`, y `<eos>` repite el del último token. Solo queda B10.
- **Clase 2, desplazamientos de objetivos y máscaras.** Los desplazamientos `[:, :-1]` y `[:, 1:]` de train.py:118, `bits_por_nota` y `nll_por_trozo` son correctos, y `perdida` queda alineada con el token objetivo (fantasma = 0, pitches de ECOR = 1). `secuencias`/`codificar` reservan el `<eos>` y la ventana siguiente arranca en `max(b2, b+1)`. El único off-by-one es M2.
- **Clase 3, fuga del futuro.** El modelo es causal y respeta la invarianza por desplazamiento:
  ```
  causalidad: max|dif| en posiciones <30 al cambiar futuro = 0.0
  invariancia por desplazamiento (RoPE relativo) = 9.5e-07
  lote vs individual = 0.0
  ```
  El eje de compás de las cabezas kv usa solo el número de BAR ya emitidos. Las referencias ECO apuntan solo a compases previos o a partes q<p del mismo compás. Las etiquetas de cabecera (STY, KEY, TEMPO) derivan de la pieza entera, pero son tokens predichos con pérdida y su coste cuenta en bits/nota; es idéntico en todos los brazos. `transposiciones_validas` solo afecta a la aumentación de train. La partición por archivo (SHA-1) es estable. No hay fuga.
- **Clase 4, ida y vuelta.** Sin fallos en lo siguiente:
  - 400 piezas reales x {base, eco, fant, eco_rel}, los primeros 10 compases: 0 excepciones y 0 discrepancias.
  - Ventanas de entrenamiento `secuencias` (b0>0, ctx 512, 2 transposiciones) en 300 piezas x {base, eco, fant} x {abs, rel}: 3.828 secuencias (842+421 base, 788+394 eco, 922+461 fant) aceptadas con gramática estricta.
  - Fuzz de 400 piezas sintéticas, con anacrusa, compases de hasta 384 ticks, 1-4 partes, acordes y 3.515 ECO + 1.079 ECOR por configuración, en {base, eco, fant, eco_rel, fant_rel}: 0 excepciones, 0 discrepancias y `d.fin` verdadero.
  - Casos a mano: `compas_400` OK, `anacrusa` OK, `una_parte_eco` OK, `acorde_3_notas` OK.
  - No hay piezas con partes >= 16 en el corpus. 29 piezas del corpus tienen compases > 192 ticks; no las probé una por una, pero el fuzz y `compas_400` cubren compases de hasta 400 y 384 ticks.
- **Clase 5, gramática vs codificar.** Aparte de M1, M4 y B4, la gramática acepta todo lo que `codificar` produce en el muestreo anterior. Las transiciones ECO/ECOR/fantasma/eos son consistentes. `eco_tr` nunca queda vacío porque TRC_0 siempre es válido, por lo que `generar` no puede cortar por `perm` vacío.
- **Clase 7, formas y dtypes.** Sin errores en `colar`, `bits_por_nota`, `nll_por_trozo`, RoPE por eje, GQA ni `generar`. La invarianza por desplazamiento simultáneo token y compás valida además que `eje_q` y `eje_kv` casen con `repeat_interleave`. `np.int16` basta para el vocab (723) y el índice de compás.
- **Clase 8, aleatoriedad.** `torch.Generator` con semilla en `generar`, `random.Random` con semilla en `lotes`/`transposiciones` y `bootstrap` con `default_rng(0)` son deterministas; `hash()` solo se aplica a tuplas de enteros. Lo no determinista es B8 y B9.
- **Clase 10, métricas.** `_js` es correcto (JS en bits, disjuntos = 1). `bootstrap` remuestrea con reemplazo y los percentiles son 2.5/97.5. `tonalidad_kk` (rotación del perfil, `desp ∈ [-6,5]`) y `en_escala` son correctos. `tonalidad_juez` (TKP de music21) devuelve (0, 'major') para una escala de Do. El pareo por pieza usa `comunes` y las mismas ventanas. Las ventanas de `trozos_eval` coinciden entre modo absoluto y relativo (0 de 450 diferencias en 150 piezas del test) y entre las tres representaciones. Salvo M3 y B7.

---------------------------------------------------------------------------------------------------

## Notas para memoria del usuario (proyecto FUGA)
- Paquete /home/claude/fuga: LM compositor con operadores ECO; ensayo CPU pre-registrado (K1-K6), ctx 512, 4 brazos.
- Revisión G4 entregada en este archivo; sin ALTA. Los puntos a corregir antes de interpretar resultados del ensayo son M2 (off-by-one de compases), M3 (métricas por lag) y M5 (truncado por contexto en (b)).
