# Innovador A — el compositor (Expedición FUGA)

Ángulo: representación, arquitectura, objetivo, decodificación y control del LM compositor de 150–400M en 1×24 GB. Única entrada leída: MAPA.md.

## Plataforma común (no reclamada; sale del mapa §3, solo ancla costes)

- Decoder de ~290M: 24 capas, d=1024, 16 cabezas, GQA con 4 kv-heads de 64, SwiGLU 2816, RoPE, QK-norm, embeddings atados, vocabulario ≤16K (REMI por compás + BPE restringida a tokens literales), ctx 4096, AdamW o Muon, WSD con 20% de decaimiento y dropout por repetición de datos.
- Memoria: estados ≈4.6 GB (pesos fp32 + gradientes + Adam) más activaciones con checkpointing a micro-lote 8×4096 ≈2–4 GB, total <12 GB [V0-conjetura].
- Cómputo: ≈2.3–2.9 GFLOP/token, que dan 21–30K tok/s en una 4090 a ~40% de MFU; 8B tokens tardan 3–4.5 días. Una 3090 va ≈2.3× más lenta [V0-conjetura].
- Datos: Lakh + PDMX ≈2–4B tokens sin BPE [V0-conjetura]. 8B tokens de entrenamiento son entonces 2–4 épocas, en el borde de las "~4 épocas casi gratis" del mapa §3.
- Representación base, común a A1–A3 y no reclamada: `BAR` [`TS_x`] [`TEMPO_x`] y, por cada pista activa en orden fijo, `TRK_p` seguido de sus notas ordenadas por (onset, altura): `POS_k` (48 por compás en 4/4, solo cuando cambia), `PITCH_m`, `DUR_d` (32 cubetas), `VEL_v` (8 cubetas). Un par pista-compás en silencio se omite. A cada par (compás, pista) lo llamo **segmento**.

---

## A1 — ECO: operador explícito de copia con transformación (LZ musical con puntero)

**Casilla atacada.** La "Cota del barrido" del mapa §2: no se localizó un mecanismo explícito de COPIA o REFERENCIA a compases previos con transformación (S2-A48). También F1 y, en la variante C, F2.

### (1) Mecanismo

**Tokens nuevos (~115).**
- `REF` y `PTR`.
- `XF_*` (≈40): `XF_ID`; `XF_TC±1..±12` (transposición cromática); `XF_TD±1..±7` (transposición diatónica en la escala del marco local, ver A2 o un estimador global); `XF_RHY` (ritmo prestado).
- `DEL_0..63`, `ADD`, `EOSEG`.
- Tres embeddings aditivos que no son vocabulario: `E_ghost` (bandera de fantasma), `E_idx[0..63]` (índice de nota en la plantilla) y `E_lag[16 cubetas logarítmicas de compases]`.

**Forma de un segmento Eco en la secuencia:**
```
TRK_p  REF  PTR  XF_τ  ⟨fantasma: plantilla τ(fuente) en tokens literales⟩  DEL_i…  ADD POS PITCH DUR…  EOSEG
```

- **Decisión explícita.** Tras `TRK_p`, la cabeza del LM predice `REF` o un `POS` (literal). Es una sola decisión categórica por segmento.
- **Cabeza puntero.** En la posición de `REF`, puntúa las anclas, que son el último token de cada segmento previo dentro de la ventana:
  `s_j = (W_q h_REF)·(W_k h_{a_j})/√d_p + b_lag[cubeta(lag_j)] + b_trk[misma_pista_j]`, con d_p=128.
  - Pérdida de conjunto: `L_ptr = −log Σ_{j∈J*} softmax(s)_j`. J* son todas las anclas cuya expansión bajo τ da exactamente la misma plantilla. Así desaparece la ambigüedad espuria del tipo "el estribillo aparece 3 veces, ¿a cuál apunto?".
  - El input de `PTR` es `E[PTR] + E_lag[lag elegido] + E_trk[pista fuente]`.
  - Los sesgos b_lag y b_trk aprenden en pocos pasos el apunte más común (misma pista, 4 u 8 compases atrás).
- **`XF_τ`.** Lo predice la cabeza LM normal. Las τ inválidas para la fuente elegida (alturas fuera de 0..127) se enmascaran; es una máscara token-local que el runtime calcula en CPU.
- **Fantasma.**
  - El runtime expande τ(fuente) de forma determinista, en CPU y en µs, y lo inserta como tokens literales con `E_ghost + E_idx[i]` en la nota i.
  - En entrenamiento los fantasmas tienen pérdida 0. En inferencia se precargan en un solo forward paralelo, como la salida de una herramienta.
  - El modelo ve la plantilla antes de editarla, así que `DEL_i` (borrar la nota i) y `ADD` (añadir notas literales) se aprenden con facilidad. Los `DEL_i` con i ≥ |plantilla| se enmascaran.
  - Con `XF_RHY` el fantasma muestra la fuente y el modelo emite exactamente n tokens `PITCH`: alturas nuevas sobre el ritmo prestado.
- **τ diatónica.** La altura se pasa a (grado, alteración, octava) en la escala del marco; se suma n al grado y se conserva la alteración.
- **Parser offline, determinista y por MDL:**
```
para cada segmento s con ≥3 notas, en orden temporal:
  cand = {misma pista a lags 1,2,4,8,16} ∪ hash(patrón de onsets) ∪ hash(intervalos)   # ventana W=64 compases
  para cada (j, τ) aplicable:
     T = τ(contenido_final(j))                 # contenido tras las ediciones de j
     dels = T∖s ; adds = s∖T                   # igualdad en (onset, altura, dur); velocidad copiada de T
     L_ref = C_REF + C_PTR(lag) + C_XF[τ] + C_DEL·|dels| + Σ C_lit(adds)
  tomar el mínimo; si L_ref < L_lit(s) y |dels|+|adds| ≤ 0.25·|s| → Eco; si no → literal
  costes C_* = −log2 frecuencia (2 pasadas: unigrama literal; luego re-estimación con la salida de la 1ª)
  desempate: menor lag, luego τ = ID; J* = todos los j con la misma plantilla bajo la τ elegida
```
- **Pérdida.** `L = CE(tokens no fantasma) + λ_ptr·L_ptr`, con λ_ptr=1.
- **Puntero en código:**
```
q = Wq(h[ref_pos]); k = Wk(h[anchor_pos])               # [B,R,dp], [B,A,dp]
s = q @ k.mT / dp**.5 + b_lag[lagb] + b_trk[same]       # [B,R,A]
s = s.masked_fill(~valid, -inf)                         # ancla anterior y dentro de ventana
L_ptr = (s.logsumexp(-1) - s.masked_fill(~Jstar, -inf).logsumexp(-1)).mean()
```
- **Variante Eco-C (fase 2, ataca F2).**
  - Los fantasmas tienen visibilidad local: solo los ven los tokens de su propio segmento (máscara de atención por bloques).
  - Tienen posición fraccionaria: comparten el índice RoPE de `XF_τ`, con p + i/(n+1).
  - En inferencia se expulsan del KV-cache tras `EOSEG`. La ventana de 4096 posiciones cubre entonces ≈1/(1−c) más compases, siendo c la cobertura Eco.
- **Extensión no evaluada (Eco-M).** Las claves del puntero de segmentos anteriores a la ventana se guardan en un banco; al elegir una, el runtime vuelve a materializar su contenido como fantasma. Permitiría copiar desde fuera de la ventana de atención.
- **Interfaz con el render.** El compositor exporta el grafo de REF: qué segmento repite a cuál y con qué τ. El render puede reutilizar el audio de las repeticiones exactas.

### (2) Novedad contra el mapa

- **Museformer** elige a qué compases atender (es blando) y sigue generando cada token, sin transformaciones. En Eco la copia es una decisión discreta con transformación y residuo: una repetición exacta cuesta ~3 decisiones en vez de ~40 tokens, y no deriva.
- **Theme Transformer** reutiliza por compuerta un tema que le dan. En Eco el modelo elige su propia fuente en cualquier punto y le aplica τ.
- **BPE** (−64–69% de tokens) es un diccionario global del corpus. Eco son retro-referencias dentro de la pieza con un grupo de transformaciones musicales. Las dos cosas componen: la BPE se aplica solo dentro de literales y fantasmas.
- **NotaGen y MuPT/SMT-ABC** aportan jerarquía por compás y alineación de compases, pero la copia sigue siendo implícita, vía atención.
- **Antecedentes fuera del mapa que recuerdo sin verificar [V0]:**
  - CopyNet y pointer-generator en NLP: copian tokens, sin grupo de transformaciones ni residuo de edición.
  - Desarrollo de motivos tipo MeloForm: sistema experto más refinamiento; no es un LM que decide el puntero.
- **Novedad reclamada.** La combinación de cinco piezas en un LM multipista: parse MDL automático del corpus, cabeza puntero con pérdida de conjunto, τ musicales (cromática, diatónica, rítmica), edición residual y expansión fantasma determinista.

### (3) Coste en 1×24 GB

- **Parámetros:** +0.47M (0.16% de 290M).
  - W_q y W_k: 2×1024×128 = 0.26M.
  - Vocabulario nuevo: 0.12M.
  - `E_idx`, `E_lag`, `E_ghost` y sesgos: 0.08M.
- **Memoria:** los logits del puntero son [B, R≈60, A≈250] por ventana de 4096; despreciable.
- **Entrenamiento:**
  - Eco-X: la secuencia crece un 3–8% por las cabeceras [V0-conjetura]; el throughput por token no cambia.
  - Eco-C: mismo coste de entrenamiento. En inferencia el KV por compás baja a ≈(1−c) del original.
- **Parser:** O(segmentos × candidatos × |G|) usando hash. Lakh completo ≈0.5–5 h de CPU, paralelizable [V0-conjetura].
- **Inferencia:** los pasos muestreados se multiplican por (1 − c·(1−e)), con e la fracción de edición. Con c=0.5 y e=0.15 el muestreo va ≈1.7× más rápido, acumulable con la BPE [V0-conjetura].

### (4) Riesgos

1. **Riesgo principal:** a 150–400M las cabezas de inducción quizá ya copian casi perfecto, y la ventaja de Eco sería un artefacto de modelos pequeños [V0-conjetura]. Hace falta la escalera 3M→25M→85M antes de comprometer el modelo grande.
2. **Poca palanca:** la cobertura puede ser baja en material melódico.
3. **Sobre-copia estructural** (ABABABAB), porque copiar sale barato. Mitigación: temperatura propia para el token `REF` y potenciales de forma (A3).
4. **Repeticiones mecánicas:** la velocidad se copia de la plantilla. Mitigación: tokens `VELΔ` o humanización en el render.
5. **Parse poco musical:** el MDL premia la edición mínima, no lo musicalmente significativo. Las secuencias por debajo del compás no se capturan (hay opción de segmentos de medio compás).
6. **Memorización (Eco-C):** con menos posiciones por pieza, a cómputo fijo hay más pasadas sobre cada pieza y sube el riesgo de memorización (fallo GEMA, mapa §4).
7. **Ensayo de CPU subentrenado.** Puede favorecer a Eco (la base aún no aprendió a copiar) o perjudicarlo (el puntero aún no se aprendió). Por eso se registra la tendencia de la brecha entre los checkpoints de 5 y 10 min.

### (5) Falsación: ensayo E-A1 en CPU, ~2.8M parámetros, ≤30 min

El ensayo mata A1 si no gana ni en un régimen favorable, y lo filtra en contra al quitarle la victoria trivial.

- **Datos (≤4 min).**
  - 450 MIDI de Lakh (LMD-full, CC-BY) con 4/4 constante y ≥24 compases.
  - Dos pistas con altura: melodía (mayor altura media, ≥2 notas por compás de media) y bajo (GM 32–39, o la de menor altura media).
  - **Sin batería**, que sería la victoria trivial de Eco; así el ensayo tiene más poder para matar.
  - Cuantización a 48 por compás, sin velocidad, 16 cubetas de duración. Split 400/50 por archivo. Lectura con `symusic` o `miditoolkit`.
- **K0 (≤1 min, antes de entrenar).** Cobertura = fracción de notas de melodía y bajo que caen en segmentos Eco con edición ≤25%. **Muere si es < 20%.**
- **Modelo, idéntico en ambos brazos.**
  - 6 capas, d=192, 6 cabezas, FFN 768, RoPE, ctx 1024 (≈25 compases de 2 pistas).
  - ≈2.7M parámetros más 25K de la cabeza puntero (d_p=64).
  - AdamW, lr 1e-3, warmup 100, coseno, lote 4×1024.
- **Brazos.** B (literal) y E (Eco-X), 10 min cada uno. Ambos con el mismo número de pasos, el que alcance el más lento. Checkpoints a 5 y 10 min.
- **Métricas (≤5 min).**
  - **Bits por nota** en las 50 piezas de validación, primeros 20 compases.
    - B suma −log2 p de todos sus tokens.
    - E suma todos sus tokens no fantasma, incluido −log2 de la distribución del puntero, sobre la derivación canónica.
    - Para E es una cota SUPERIOR de sus bits (las demás derivaciones solo añaden probabilidad), así que la comparación es conservadora a favor de B.
    - Diferencia pareada, con bootstrap de 1,000 réplicas por pieza.
  - **Estructura.**
    - 30 prompts de 4 compases, continuados 12 compases con T=1 y top-p 0.95.
    - Por pista y lag ℓ∈{1,2,4,8} se miden dos tasas: P_exact(ℓ), compás idéntico al de ℓ compases antes; y P_near(ℓ), compases que difieren en 1–2 notas (diferencia simétrica), la firma de la copia descuidada.
    - D = Σ|P_gen − P_real| sobre las 16 cifras (2 pistas × 4 lags × 2 tipos). P_real sale de las continuaciones verdaderas. Bootstrap por prompt.
- **Criterios de muerte pre-registrados.** A1 muere si falla K0, o si fallan a la vez K1 y K2:
  - K1: el IC95 de bits/nota(E) − bits/nota(B) no excluye 0, o favorece a B.
  - K2: D(E) ≥ D(B), con IC95.
- **Si sobrevive:** escalera en GPU 3M→25M→85M. Muere para el rango objetivo si la ventaja en bits/nota, extrapolada log-lineal en N, se anula antes de 150M.
- **Predicción declarada [V0-conjetura]:** cobertura de bajo 50–70% y de melodía 30–50%; E mejora ≥0.1 bits/nota; D baja ≥30%.

---

## A2 — MARCO: la tonalidad como sistema de coordenadas que el propio LM declara

**Casilla atacada.** F4 y la línea de text2midi del mapa §2 (acierta la tonalidad pedida 7.69% de las veces). También F6: con pocos datos legales conviene explotar la simetría. Se combina con F1 a través de A1.

### (1) Mecanismo

- **Tokens `FR_{t,m}` (36).**
  - t ∈ {−6..+5} es el desplazamiento de la tónica respecto a Do, el de |t| mínimo.
  - m ∈ {mayor, menor, indet}.
  - Se emiten al inicio y en cada cambio de marco, siempre en un `BAR`.
- **Alturas relativas.** `RP = pitch − t` en las pistas con altura; la batería queda intacta.
  - Es biyectivo dado el marco, así que no hay pérdida.
  - Como |t| ≤ 6, el registro absoluto se conserva a ±tritono, y el embedding de FR le da al modelo el desplazamiento exacto, de modo que los rangos de instrumento siguen siendo correctos.
- **Etiquetado de marcos (offline):**
  1. Histograma de clases de altura, ponderado por duración, en ventanas de 4 compases. Se correlaciona con los 24 perfiles de Krumhansl–Kessler, más un estado "indet" de puntaje constante θ.
  2. Viterbi sobre los 25 estados, con penalización de cambio π y permanencia mínima de 4 compases. Así una modulación queda como decisión de sección y no como ruido.
  3. En PDMX se bonifican los estados compatibles con la armadura de MusicXML (tónica ∈ {mayor de la armadura, su relativa menor}).
  4. **Desambiguación relativa por verosimilitud (hard-EM).** En las secciones donde |E_mayor − E_relativa menor| < ε, se elige el marco bajo el que un LM proxy de 25M da menor NLL. Luego se reentrena el proxy y se re-etiqueta, ≤3 rondas. Solo hay 2 candidatos por sección.
- **Control.**
  - Si el usuario pide una tonalidad, se fuerza `FR` con una máscara token-local. El LM genera en coordenadas relativas y el render convierte con pitch = RP + t.
  - La tónica pedida se cumple por construcción; el modo queda como control blando (ver A2-b).
  - Si no se pide tonalidad, el LM muestrea los `FR` y así planifica sus modulaciones.
- **Sinergia con A1.** `XF_TD` es una suma de grados dentro del marco. Un estribillo repetido un tono más arriba ("modulación de camionero") es un `FR` nuevo seguido de `REF XF_ID`: una copia idéntica en coordenadas.
- **A2-b (secundario): margen contrafactual de modo con negativo exacto.**
  - En una fracción f=0.25 de las secuencias se construye x̃ = x con el modo de todos los FR invertido (misma t; paralelo mayor↔menor).
  - `Δ = (log p(x) − log p(x̃)) / n_notas` y se añade `L += β·softplus(m − Δ)`, con β=0.1 y m=0.05 nats/nota [V0-conjetura].
  - Obliga al modelo a usar el token de modo: maximiza la PMI condicional del modo.
- **Texto (interfaz, no reclamado como novedad).**
  - Texto → ranuras (tonalidad, modo, tempo, instrumentos, género) con un analizador entrenado sobre captions tipo MidiCaps.
  - La tonalidad va a FR exacto; el resto va a tokens de control y a restricciones de A3.
  - El estilo residual entra como K=4 vectores de un codificador de texto congelado, precomputados offline (0 GB de GPU en entrenamiento).

### (2) Novedad contra el mapa

- **Todas las representaciones del mapa codifican altura absoluta:**
  - AMT: nota + instrumento.
  - REMI+ en text2midi.
  - ABC en MuPT y NotaGen: letra absoluta; la armadura solo implica las alteraciones.
  - Moonbeam: clase de altura y octava absolutas. Su atención relativa compara tokens entre sí, no respecto a una tónica declarada.
- **En el mapa el control tonal es condicionamiento:** FLAN-T5 en text2midi, croma en MusicGen. El modelo tiene que APRENDER el mapeo tonalidad→alturas, y a 272M falla (7.69%). En MARCO no hay mapeo que aprender: la tónica es el origen de coordenadas.
- **Antecedente fuera del mapa [V0]:** transponer todo a Do mayor / La menor como normalización de datos en generación melódica. Es global, pierde información (se descarta la tonalidad), no sirve de mecanismo de control y no admite modulaciones.
- **MARCO aporta:** marcos por sección, sin pérdida, generados por el LM, con hard-EM para la ambigüedad relativa y control tonal exacto.
- **Efecto sobre F6:** el aumento de datos por transposición deja de hacer falta. La simetría se paga una vez en la representación y no 12 veces en cómputo.

### (3) Coste en 1×24 GB

- **Parámetros:** +36 embeddings (37K).
- **Etiquetado:** milisegundos por pieza.
- **Hard-EM:** proxy de 25M, 2 forwards por sección ambigua, ≤3 rondas. Son ≈0.3·178K piezas × 2 × 3 × 15K tokens × 2·25M FLOP ≈ 2.4·10^17 FLOP, ≈1 h en una 4090 [V0-conjetura].
- **Entrenamiento del LM:** igual que la base, y sin el aumento por transposición.
- **A2-b:** forward/backward adicional en la fracción f de secuencias: +25% con f=0.25, o +12.5% con f=0.125.
- **Inferencia:** sin cambio.

### (4) Riesgos

1. **La causa del 7.69% puede no ser la representación.** Si está en la vía texto→LM, un LM absoluto con token de tonalidad explícito ya adhiere bien y MARCO aporta poco. El ensayo E-A2 discrimina justo esto.
2. **Errores del estimador** (pop modal, mixolidio, confusión con la relativa) dan marcos ruidosos y secuencias relativas menos regulares: sin pérdida, pero peor modelado.
3. **Música cromática o jazz:** el marco ayuda menos; ahí el estado indet equivale a absoluta.
4. **Tónica cumplida solo en código:** el modelo podría generar contenido centrado en el grado 6 bajo un FR mayor (es decir, en la relativa menor). Se mide con el estimador.
5. **Hard-EM** puede reforzar los sesgos del proxy, hasta colapsar a un modo.
6. **Fuentes con distinta calidad de marco:** PDMX trae armadura; Lakh es estimado.

### (5) Falsación: ensayo E-A2 en CPU, ~2.7M parámetros, ≤30 min

- **Datos:** los mismos 450 archivos y las mismas 2 pistas que E-A1. Marcos con KS + Viterbi en numpy (ms por pieza). Sin hard-EM: aquí se prueba la tesis central.
- **Brazo ABS, una base fuerte y no la de text2midi:**
  - altura absoluta;
  - tokens `KEY` con la MISMA segmentación Viterbi, para que haya paridad de información;
  - aumento por transposición uniforme en [−5, +6], desplazando `KEY` en consecuencia.
- **Brazo REL:** FR + RP, sin aumento.
- Mismo modelo que E-A1 y mismos pasos, 10 min por brazo.
- **Métricas:**
  - (a) Bits por nota en validación; cada brazo paga sus propios tokens KEY o FR.
  - (b) Adherencia: 24 tonalidades forzadas × 3 muestras × 8 compases sin prompt (72 muestras por brazo). Se aplica el estimador KS a los 8 compases y se mide acierto exacto (24 clases), acierto de tónica y acierto de modo, separando tonalidades frecuentes y raras en el corpus.
- **Muerte:**
  - **K-A:** si la adherencia exacta de ABS es ≥ la de REL − 5 pts **y** los bits/nota de REL son ≥ los de ABS − δ (IC95 bootstrap), la base convencional ya resuelve el control y MARCO no aporta. Muere.
  - **K-B:** si la adherencia exacta de REL es < 80%, la tesis "por construcción" es falsa (deriva a la relativa). Muere en su forma fuerte.
- **Sub-ensayos, ≤30 min cada uno:**
  - Hard-EM (2 rondas, con el propio modelo de 3M como proxy): se descarta si no baja los bits/nota más allá de δ.
  - A2-b: se descarta si la adherencia de modo no sube ≥5 pts.
- **Predicción [V0-conjetura]:** REL ≥90% de acierto exacto; ABS 60–85%, con caída en tonalidades raras.

---

## A3 — ENJAMBRE DE COMPASES: SMC sincronizado por compás con verificadores musicales

**Casilla atacada.** F4 (adherencia al control) y F1 (la forma como restricción ejecutable). En el mapa, el control solo aparece como condicionamiento (T5, CLAP, croma, controles intercalados de AMT) o como RL de entrenamiento (CLaMP-DPO). En decodificación el mapa solo trae técnicas de velocidad: especulativa exacta y MAGNeT.

### (1) Mecanismo

Objetivo: muestrear de `π(x) ∝ p_LM(x|c) · 1[x cumple las restricciones duras] · Φ(x)` sin reentrenar.

```
entrada: LM, N partículas, máscaras duras token-locales M, potenciales acumulados Φ_b
x_i = prompt; logw_i = 0
para b = 1..B:
  para cada partícula (en lote, con KV-cache), muestrear hasta BAR:
     p = softmax(logits/T); A_t = tokens permitidos por M
     Z_t = Σ_{v∈A_t} p(v)                        # masa que sobrevive a la máscara
     v ~ p restringida a A_t (renormalizada); logw_i += log Z_t
  logw_i += log Φ_b(x_i) − log Φ_{b−1}(x_i)      # incremento torcido (telescópico)
  ESS = (Σw)²/Σw²; si ESS < N/2: remuestreo sistemático (index_select del KV-cache por partícula)
salida: partícula de mayor peso, o muestreo ∝ w (top-k para el usuario)
```

- **La corrección `log Z_t` es la diferencia con el enmascarado ingenuo.** El enmascarado local muestrea de una distribución sesgada y miope aunque la restricción se cumpla. Con los pesos, el conjunto de partículas converge a la condicional correcta.
- **Restricciones duras** (token-locales, con corrección exacta):
  - conjunto de instrumentos (`TRK_p`);
  - rango por instrumento (`PITCH`, dados TRK y FR);
  - marco forzado (A2);
  - **plan de forma con A1.** Con un plan por compás, p. ej. `A A B A'`, en cada compás de etiqueta repetida la pista melódica solo puede emitir `REF`, con el puntero enmascarado a anclas de la misma etiqueta, y `XF_ID/TC/TD`. `A'` permite ≤k ediciones: tras k se enmascaran `DEL` y `ADD`.
- **Potenciales blandos Φ_b:**
  - curva de densidad dada por el usuario d(b): `Φ = exp(−Σ(n_b − d_b)²/2σ²)`;
  - cadencia final, solo en B: la raíz del último acorde es la tónica del marco;
  - presencia de los instrumentos pedidos, con penalización por plazo;
  - tonalidad cuando no hay A2: correlación KS sobre 4 compases, con κ templada.
- **Templado.** El κ de cada potencial crece linealmente con b, para evitar el colapso temprano de los pesos.

### (2) Novedad contra el mapa

- **El mapa no trae nada en decodificación.** El control es condicionamiento (text2midi falla con 7.69%) o RL de entrenamiento (CLaMP-DPO). No hay un mecanismo que cumpla restricciones conservando la distribución, ni una forma musical expresada como restricción ejecutable.
- **Antecedentes fuera del mapa [V0]:** SMC para LMs de texto (steering con programas probabilísticos, SMC torcido), y CFG / negative prompts.
- **Novedad reclamada:**
  - la transferencia a música, sincronizada por compás, que es la unidad natural de verificación;
  - potenciales musicales;
  - con A1, la forma convertida en máscara token-local exacta sobre el puntero. En un LM sin operador de copia solo podría ser un potencial blando y caro de similitud.

### (3) Coste en 1×24 GB

- Sin entrenamiento y sin parámetros nuevos.
- **KV-cache:** a ctx 2048 (24 capas, 4 kv-heads × 64, bf16) son 2×24×256×2 B ≈ 24.6 KB/token, 50 MB por partícula y 0.8 GB con N=16. Los pesos ocupan 0.6 GB.
- **Cómputo:** los FLOPs se multiplican por N. La decodificación en lote pequeño está limitada por memoria, así que el tiempo de reloj crece ~1.3–2.5× con N≤16 [V0-conjetura].
- **Remuestreo:** copiar ≤0.8 GB de KV lleva milisegundos.
- **Potenciales:** se evalúan en CPU, µs por compás.

### (4) Riesgos

1. **Degeneración de pesos** con potenciales afilados: el ESS colapsa, todas las partículas acaban compartiendo ancestro y baja la diversidad.
2. **Best-of-N puede bastar** en horizontes cortos. La falsación lo prueba.
3. **Garantía solo asintótica** para restricciones que no son token-locales (p. ej. un rango sobre una frase entera): solo pueden ser potenciales blandos.
4. **Autoevaluación sesgada** si la fidelidad se mide con el mismo LM. Por eso se usa un juez independiente.
5. **Sin A1**, la forma solo es un potencial blando y caro.

### (5) Falsación: ensayo E-A3 en CPU, ≤30 min

Reutiliza modelos si se corre después de E-A1.

- **Modelos:**
  - Generador: el brazo B de E-A1 (LM literal incondicional de 3M).
  - Juez: otro LM de 3M con otra semilla, entrenado 10 min sobre un split de entrenamiento disjunto. Mide fidelidad sin autoevaluación.
- **Tareas**, sobre 30 prompts de 4 compases:
  - T1: densidad de la melodía siguiendo una curva lineal creciente del percentil 25 al 90 a lo largo de B compases, con tolerancia de ±2 notas por compás.
  - T2: una tonalidad objetivo aleatoria (de las 24), para un modelo que no condiciona tonalidad; potencial KS.
  - T3: ambas a la vez.
- **Decodificadores, con el mismo presupuesto de tokens generados:**
  - (i) best-of-N: N muestras independientes, se elige la de mayor Φ;
  - (ii) local: sesgo de logits token a token hacia el objetivo (altura dentro de la escala; inicio de nota o `BAR` según el déficit de densidad);
  - (iii) SMC con N partículas.
- **Barrido:** N ∈ {4, 8, 16} y B ∈ {4, 8, 16}.
- **Métricas:**
  - adherencia: fracción de compases dentro de tolerancia en T1, acierto KS en T2, ambas en T3;
  - fidelidad: bits/nota bajo el juez;
  - diversidad: distancia de edición media entre 4 salidas del mismo prompt.
- **Muerte:**
  - **K1:** en N=8 y B=16, si la adherencia de SMC supera a la de best-of-8 en menos de 10 pts, lo simple basta. Muere.
  - **K2:** a adherencia emparejada, si los bits/nota de SMC bajo el juez son peores que los del decodificador local por más de δ. Muere.
  - **K3:** si con N=16 la diversidad de SMC cae por debajo del 50% de la incondicional, hay colapso de linaje. Muere, salvo que el templado lo corrija dentro del mismo ensayo.
- **Predicción [V0-conjetura]:** la ventaja de SMC sobre best-of-N crece con B, porque el éxito de best-of-N decae como ~q^B. En B=16, SMC ≥85% frente a best-of-8 ≤50%.
- **Tiempo:**
  - Decodificación: ≈30 prompts × 16 compases × 40 tok × 8 partículas × 3 decodificadores ≈ 0.46M pasos-token en lote, ≈3–8 min en CPU [V0-conjetura].
  - Total con el entrenamiento del juez: ≈25 min.

---

## Cómo componen y en qué orden ensayar

- **Composición:**
  - A2 hace triviales las τ diatónicas de A1 y convierte las repeticiones moduladas en copias idénticas.
  - A1 convierte el plan de forma de A3 en una restricción dura token-local.
  - A3 impone sobre A2 los controles del usuario que A2 no fija: instrumentos, densidad, forma, cadencia.
  - Interfaz con el render: MIDI + grafo de REF + marcos.
- **Orden de ensayos más barato:**
  1. K0 de A1 (≈2 min, sin entrenar).
  2. E-A1.
  3. E-A2, que reutiliza los datos.
  4. E-A3, que reutiliza los modelos.

---

## Supuestos declarados

- **S1 [V0-conjetura]. Tamaño de datos.** Lakh ≈2–3B tokens REMI multipista sin BPE (≈0.8–1.1B con BPE), PDMX ≈1B; total ≈2–4B.
- **S2 [V0-conjetura]. GPU.** Una 4090 da ≈165 TFLOP/s pico en bf16 con acumulación fp32; una 3090, ≈71. MFU realista: 35–45%.
- **S3 [V0-conjetura]. CPU.** Un modelo de 3M con ctx 1024 entrena a 2–5K tokens/s con PyTorch en un CPU de 8–16 núcleos. Si el CPU es más lento, cada brazo se recorta a 8 min sin cambiar los criterios.
- **S4 [V0-conjetura]. Repetición en Lakh.** Cobertura en melodía y bajo de 30–70%; en batería es mayor.
- **S5 [V0-conjetura]. Estimador de tonalidad.** KS con suavizado Viterbi acierta la tonalidad del pop en ~70–80%; los errores los dominan la relativa y los modos.
- **S6 [inferencia sobre el mapa].** El 7.69% de text2midi no distingue entre un fallo de la vía texto y un fallo de representación. E-A2 está diseñado para separarlos.
- **S7 [V0-conjetura].** A 150–400M la copia por cabezas de inducción mejora con la escala. Es el riesgo principal de A1.
- **S8 [V0-conjetura].** La decodificación en lote pequeño está limitada por memoria: N≤16 partículas cuestan 1.3–2.5× en tiempo de reloj.
- **S9 [V0, sin verificar, sin web]. Antecedentes recordados:**
  - CopyNet y pointer-generator;
  - MeloForm;
  - normalización "todo a Do mayor";
  - SMC para LMs de texto;
  - CFG y negative prompts;
  - entrenamiento MMI en reconocimiento de voz;
  - codificaciones relativas al acorde en melodía de jazz.
  Deben pasar por el chequeo de novedad de la expedición antes de dar por buena cualquier reclamación.
- **S10 [fuera de mi ángulo, lo señalo].** El "Lakh CC-BY 4.0" del mapa cubre la compilación, pero muchas piezas son transcripciones de canciones pop protegidas. Con el criterio del fallo GEMA (mapa §4), la memorización de las composiciones subyacentes es un riesgo que ninguna de mis propuestas elimina. Eco-C lo agrava levemente: a cómputo fijo hay más pasadas por pieza.
- **S11 [V0].** La licencia de las captions tipo MidiCaps no está en el mapa. La vía de texto de A2 depende de verificarla.
- **S12.** Los ensayos de CPU son filtros de muerte, no confirmaciones. Un modelo de 3M con 1–3M tokens está subentrenado; un "sobrevive" solo habilita la escalera en GPU (3M→25M→85M).
