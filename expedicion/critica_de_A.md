# Crítica cruzada ciega del revisor A al lote X1–X4 (Expedición FUGA, F5)

Fuentes leídas: lote_para_A.md y MAPA.md, nada más. No hice búsquedas web. Todo antecedente que cito de memoria va marcado [V0].

Conflicto de interés declarado: X4 coincide en lo esencial con mi propuesta A1, incluso en el nombre (ECO). Donde comparo X4 con A1 argumento con hechos técnicos verificables, pero el director debería pedir una segunda opinión independiente sobre ese punto.

## Veredictos

| Id | Veredicto | Motivo en una línea |
|---|---|---|
| X1 PxT | SOBREVIVE, con 3 enmiendas obligatorias | Tal como está escrito, el puente determinista devuelve la media condicional y el audio sale difuminado. La pieza útil, pares fabricados por transcripción, se mantiene. |
| X2 GIM | SOBREVIVE, recortada | Su argumento cuantitativo central (el 97%) se cae con una representación métrica y relativa a la tónica, y su ensayo tiene una contradicción entre deduplicación, DF y canarios. Lo valioso es la deduplicación invariante y el cruce PDMX×Lakh. |
| X3 ECH | SOBREVIVE, degradada | La métrica casi seguro es un refrito del perfil de lags de la matriz de auto-similitud (SSM), y la línea base de su criterio de muerte está amañada. Lo valioso es la escalera metamórfica. |
| X4 ECO | FUSIONAR-CON-A1 | Mismo núcleo que A1, pero sin materializar el contenido copiado, sin ediciones residuales y con una falsación débil. Además atribuye al mapa cosas que el mapa no contiene. |

---

## X1 — PxT: pares por transcripción + puente render→real

### (a) Debilidades graves

1. **El puente determinista desde un punto aprende la media condicional, y el audio sale difuminado.** Es un error técnico, no un riesgo.
   - El muestreo son 4–8 pasos de Euler de una EDO que parte de `z_r`, que es fijo dadas las condiciones. Una EDO determinista no puede transportar una masa puntual a una distribución. La salida es una función determinista f(z_r, c) entrenada con MSE, y el óptimo en MSE de una función determinista es la media condicional E[z_y | z_r, c].
   - Ese condicional es muy multimodal: mezcla, sala, producción, detalle tímbrico y las notas que el AMT perdió. Promediar latentes DAC produce timbre apagado y transitorios borrados.
   - Hay un segundo defecto. El término de ruido σ·√(t(1−t))·ε tiene derivada temporal, y el objetivo `v = z_y − z_r` la omite. Ese campo no transporta las marginales del interpolante estocástico; es una regresión de desplazamiento medio con ruido de regularización.
   - Consecuencia directa: diversidad cero por construcción. Para un mismo `m` solo varía la salida si cambia la referencia.
   - Enmienda: puente estocástico con su término de score, muestreado como EDS (familia I²SB/DDBM [V0]), o un SDEdit desde `z_r` + ruido fuerte. Eso sube los pasos a unos 20–50 NFE [V0-conjetura]. La latencia sigue siendo aceptable, pero el coste de inferencia declarado ("segundos") se multiplica por 3–6.
2. **La palanca contra la alucinación de notas apunta mal, y la proyección de ida y vuelta rompe la coherencia entre las dos salidas.**
   - La CFG sobre el piano-roll (20% de descarte) casi no tiene palanca: `z_r` ya codifica las notas. La rama "sin piano-roll" sigue viendo el render, y la alucinación (rellenar lo que el AMT suele perder) se aprende como función de `z_r`, no del piano-roll.
   - La palanca correcta es `c`, pero `c = 1` es el extremo de su distribución. En mezclas densas de FMA, casi ningún clip de entrenamiento tendrá `c ≈ 1`, así que el embedding escalar extrapola fuera de distribución.
   - Además `c` solo cuenta onsets cubiertos. No mide alturas erróneas, notas espurias ni confusión de instrumento.
   - La ida y vuelta `m̃ = AMT(render(m))` hace que el audio corresponda a `m̃` y no a la partitura oficial `m`, en contra de la restricción F0 de doble salida coherente. Y no cierra la brecha: los errores del AMT sobre un render limpio de FluidSynth no son los errores sobre mezclas reales. La entrada de inferencia sigue fuera de distribución. Debe salir del camino por defecto, no quedar como ablación.
   - Con la vía barata (familias batería, bajo y otros), la elección de instrumento del LM (violín frente a piano) se colapsa en "otros". El timbre lo decide la referencia, y la mitad de audio deja de obedecer a la partitura. R3 lo reconoce pero no lo mide como criterio de muerte.
3. **La evaluación es circular y el umbral de muerte es demasiado fácil.**
   - La note-F1 se mide con AMT(salida) usando la misma familia de transcriptor que fabricó los pares. El refinador aprende a producir lo que ese AMT transcribe como `m̂`, puntos ciegos incluidos. Hace falta un transcriptor independiente, de otra familia y con otros datos.
   - "Mejora de KAD ≥20% sobre el render crudo" lo cumple cualquier refinador entrenado sobre audio real, incluso el control barajado. El control barajado es el bueno, y no figura como criterio de muerte.
   - El humanizador H aprende como "expresividad" el error temporal del AMT: decenas de ms de jitter de transcripción sobre música de FMA que en su mayor parte es secuenciada. Y luego lo aplica a PDMX, que es otro dominio.

Costes: el disco de los códigos DAC (≈7 GB) y el tiempo del refinador (≈4 h en 4090) son coherentes. Pero 100K pasos × 16 × 10 s equivalen a ≈4.400 h vistas sobre 300–600 h útiles, o sea 7–15 épocas, más allá del régimen de ≤4 épocas del mapa §3. Eso eleva el riesgo de memorizar FMA, que solo se auditaría a posteriori con MiRA. El volumen útil tras filtrar licencia, voz y compuerta es una incógnita, y puede quedar en el borde inferior [V0-conjetura].

### (b) ¿Refrito?
- No del mapa: CoSaRef evita los pares; PxT los fabrica.
- Fuera del mapa hay antecedentes cercanos [V0]: síntesis multi-instrumento por difusión de espectrograma entrenada con pares alineados automáticamente (alineación por DTW o transcripción en conjuntos tipo MusicNet), y la traducción sintético→real por puente o SDEdit.
- Condicionar por un escalar de calidad de datos (`c`) es un patrón conocido en otros dominios (puntuación estética como condición [V0]).
- Lo que queda como novedad es la combinación: pares por AMT sobre audio CC, puente local desde el render y refinador pequeño.

### (c) Solapamiento con A1–A3
- No compite; hay una interfaz con A1. PxT procesa ventanas de 10 s de forma independiente, así que dos estribillos idénticos en la partitura suenan distintos: deriva de producción entre ventanas.
- El grafo de REF que exporta A1 lo resuelve sin coste:
  - una repetición exacta (`XF_ID`, misma pista) reutiliza el latente refinado de la ventana fuente;
  - una repetición transformada se refina con inpainting usando la ventana fuente como referencia.
- En ese punto es mejor la combinación que cualquiera de las dos sola.

### (d) Veredicto: SOBREVIVE, con enmiendas obligatorias
1. Puente estocástico o SDEdit en lugar de la EDO desde un punto.
2. Note-F1 con un transcriptor independiente, y "emparejado ≤ barajado" como criterio de muerte.
3. Quitar la ida y vuelta del camino por defecto y medir la adherencia por familia de instrumento como criterio de muerte.

Es la única propuesta del lote que ataca F3 dentro del presupuesto.

### (e) Residuo útil
Aunque caiga el puente, quedan tres piezas aprovechables: la fábrica de pares (AMT → render → compuerta de croma y onsets), la compuerta de consistencia como filtro de datos y la idea del índice de completitud, medido mejor como F1 por nota contra tallos.

---

## X2 — GIM: guardia invariante de memorización

### (a) Debilidades graves

1. **El ensayo E2-1 se contradice: deduplicación, DF y canarios chocan.**
   - El paso c (MinHash por clúster, que conserva 1–2 versiones) borra justamente la duplicación d ∈ {4, 16, 64} que miden los canarios. Si los canarios sobreviven a la deduplicación como clústeres distintos, DF(κ) = d > f_max = 3, y la exención por frecuencia los deja sin defensa precisamente a alta duplicación. Si se agrupan, d deja de existir.
   - Hace falta declarar que los canarios esquivan la deduplicación pero cuentan como un solo clúster para DF.
   - Falta el brazo competidor real: "solo deduplicación invariante, sin goldfish". Tras deduplicar, la amenaza residual es d ≈ 1–2 con ≤4 épocas, más la repetición dentro de la pieza. Si el brazo A no memoriza en ese régimen, GIM-d sobra.
2. **El argumento (i), el del 97%, solo vale para una tokenización concreta.**
   - Exige altura absoluta con aumento por transposición. En tokens métricos (REMI, POS por compás) el tempo ya es invariante, porque solo cambia el token TEMPO. La instrumentación tampoco entra en un n-grama dentro de la pista.
   - Con una representación relativa a la tónica (mi A2) desaparece el aumento por transposición, y el goldfish crudo pasa a ser casi tan invariante como el de GIM.
   - Lo que de verdad aporta GIM queda en tres cosas: la deduplicación por clúster melódico entre versiones distintas, la exención por DF y el cruce PDMX×Lakh. E2-1 necesita un brazo con tokens relativos; si no, la ventaja medida es un artefacto de la tokenización.
3. **La defensa es débil en música por diseño, y la cobertura es parcial.**
   - Con k = 4 se borra 1 de cada 4 notas idiosincráticas. En música el contorno es muy predecible: notas repetidas, secuencias, la repetición interna de la pieza. Las notas sin gradiente se recuperan por generalización (R1). El canario realista lo mediría, pero en GPU y tarde.
   - La variante de "primera aparición" deja supervisadas la segunda y siguientes apariciones del estribillo. En el pop de Lakh el gancho, que es lo más protegible, aparece 3 o más veces por pieza, así que esa variante anula casi toda la protección justo donde importa.
   - φ solo protege líneas monofónicas extraídas (skyline, bajo). Los riffs, las voces interiores y el arreglo, que en Lakh es en sí una obra derivada, quedan fuera.
   - El sesgo de la máscara por DF no es neutral: borra gradiente sistemáticamente de las continuaciones raras y nunca de las comunes. Eso empuja la distribución hacia el cliché [V0-conjetura], y la val-NLL agregada no lo detecta.
   - Coste: con 176K archivos multipista, 4–7·10⁸ notas parece corto, quizá por 2× [V0-conjetura]. Como todo corre en CPU, no es grave.

### (b) ¿Refrito?
El mapa solo trae auditorías en audio o CLAP. Pero casi todas las piezas existen fuera del mapa [V0]:
- pérdida goldfish con hash de contexto (Hans et al., 2024);
- veto de n-gramas con filtro de Bloom en decodificación (MemFree, Ippolito et al., 2022): GIM-f es esto mismo sobre rasgos invariantes;
- deduplicación contra la memorización (Lee et al., 2022);
- canarios (Carlini et al., 2019);
- n-gramas de intervalos y contorno para similitud melódica y detección de versiones (MIR clásico, código Parsons);
- auditoría de originalidad en música generada calibrada contra un nulo de música humana ("originality report", Yin et al., ~2021–2022), muy cercana a GIM-e.

La novedad que queda es la combinación: grupo de invariancia, exención por DF y cruce de procedencia PDMX×Lakh.

### (c) Solapamiento con A1–A3
- **A2:** neutraliza el argumento (i), como se explicó arriba.
- **A1:** resuelve de forma estructural el dilema de la "primera aparición". Con A1 las repeticiones son tokens REF y no notas objetivo, así que la máscara cae sobre la única aparición literal, y copiar del contexto sigue supervisado a nivel de operador. En ese punto A1 es mejor que la variante de GIM.
- **A3:** el veto GIM-f es una máscara dura token-local, y como enmascarado ingenuo produce casi-copias: se bloquea la nota y el modelo toma la vecina a un semitono, que es el caso R2.
  - Dentro de A3, la corrección `log Z_t` penaliza a las partículas que "querían" copiar (Z_t pequeño) y el remuestreo las elimina, en vez de dejarlas convertirse en copias con una nota cambiada.
  - En el veto, A3 es mejor; GIM-f debería implementarse como restricción de A3.

### (d) Veredicto: SOBREVIVE, recortada
- GIM-c (deduplicación invariante y cruce PDMX×Lakh) y GIM-e (auditoría con nulo humano) se mantienen como requisitos.
- GIM-d queda a prueba frente al brazo "solo deduplicación" y frente a tokens relativos.
- GIM-f pasa a ser una restricción dentro de A3.

### (e) Ensayo de muerte más barato que E2-1 (CPU, menos de 30 min)
- Un LM de 3M sobre 400 piezas más 20 melodías canario de dominio público, retenidas y duplicadas 16 veces con transposición.
- Cuatro brazos: sin defensa, goldfish crudo, goldfish invariante y goldfish crudo sobre tokens relativos.
- Validez: el brazo sin defensa tiene que extraer.
- Muere si la extracción exacta e invariante de los canarios con el goldfish invariante no baja al menos un 50% frente al goldfish crudo sobre tokens relativos.

---

## X3 — ECH: espectro de compresión por horizonte + escalera metamórfica

### (a) Debilidades graves

1. **La línea base del criterio de muerte (d) está amañada.**
   - gzip sobre tokens crudos falla T1 (transposición) por construcción, así que (d) no puede matar a ECH.
   - Las líneas base justas son dos:
     - la razón total C_64/C_0 sobre las MISMAS vistas invariantes A y B, que dice si la resolución por horizonte añade algo a la compresibilidad total;
     - el perfil de lags de la matriz de auto-similitud por compás, que es un descriptor estándar de estructura en MIR (representación tiempo-lag [V0]), con el mismo MMD.
   - Si cualquiera de las dos pasa la escalera con igual sensibilidad, la métrica ECH no aporta.
2. **La métrica se puede ganar por arquitectura, no solo por recompensa.**
   - La regla anti-Goodhart solo prohíbe usar ECH como recompensa. Un modelo con operador de copia explícito (A1 o X4) aprende la distribución de lags del corpus en su prior de puntero (`b_lag`, `DIST_d`) y calca el espectro g_v(h) sin que la repetición tenga función musical.
   - ECH sería entonces juez y parte justo en la comparación que la expedición necesita para F1: modelos con copia contra la base literal.
   - Falta un destructor D7, "repetición con lags preservados y fuente aleatoria": se conserva el histograma de lags de las repeticiones, pero el material que se repite se elige al azar dentro de la pieza.
   - Una métrica que no separe D7 del real no puede arbitrar entre ECO y la base. Mi propia métrica D de E-A1 (P_exact y P_near por lag) tiene el mismo defecto: lo admito y la ato a D7.
3. **Las invariancias que exhibe son vacuas, y las que importan son frágiles.**
   - T1 y T2 se cumplen por construcción en las vistas A y B (Δaltura y razón de IOI); pasarlas no es evidencia.
   - El punto débil real es el microtiempo. ρ = log2(IOI_j/IOI_{j−1}) se cuantiza en pasos de 0.5. Con semicorcheas a 120 bpm (125 ms), un jitter de ±10 ms mueve ρ hasta ≈±0.23 y cruza cubetas cerca de los bordes.
   - Eso rompe las coincidencias exactas del LZ en MIDI interpretado (parte de Lakh), y ECH confunde expresividad con falta de forma. T5 (±10 ms) probablemente falla.
   - T3 y T6 (reinstrumentación, doblaje a la octava) desestabilizan la extracción skyline. Ahí está la información, no en T1 y T2.
   - Problemas menores:
     - el coste del puntero, log2 del tamaño de ventana, crece con h y se mezcla con la señal de repetición;
     - el LZ voraz no es monótono en h;
     - el ancla humana (20 oyentes × 24 pares) mezcla pares real–destructor con pares sistema–sistema y tiene poco poder para ordenar más de dos sistemas.

### (b) ¿Refrito?
- **Métrica.** La versión por horizonte equivale, en gran parte, al perfil de lags de la SSM (Goto, 2003; "structure features" de Serrà et al., 2012 [V0]) expresado como compresión. Añádanse Information Rate y Variable Markov Oracle de Dubnov, que miden la ganancia de compresión por repetición [V0], y el análisis por compresión de conjuntos de puntos de Meredith (COSIATEC [V0], que el autor cita). El MMD sobre rasgos es una prueba de dos muestras estándar.
- **Escalera.** Las listas de perturbaciones para validar métricas existen en evaluación de NLG (Sai et al., 2021 [V0]). Aplicada a métricas simbólicas de forma, con destructores musicales y un criterio de admisión, no la conozco y la considero la parte nueva.

### (c) Solapamiento con A1–A3
- **A1:** el parse LZ con horizonte de ECH es casi el parser MDL de A1, solo que usado como medida y no como representación. Se puede compartir el código. Como métrica general, ECH es mejor que mi D: tiene tres vistas, dos colas y MMD. Como juez de A1, ninguna de las dos sirve sin D7.
- **A2 y A3:** ECH-g (adherencia verificable más muestreo por rechazo, acierto = 1 − (1 − p)^N) es exactamente el brazo best-of-N de mi E-A3, y supone independencia entre muestras.
  - Para un atributo global verificable al final, el rechazo basta y es más simple: ahí X3-g es mejor.
  - Para restricciones por compás sobre B compases, el éxito decae como q^B, y A3 es mejor.
  - Para la tónica, A2 da p ≈ 1 por construcción y el cálculo de 0.49 pierde sentido.

### (d) Veredicto: SOBREVIVE, degradada
La reclamación de novedad de la métrica cae. Quedan:
- la escalera (e), como protocolo de admisión obligatorio, con D7 añadido;
- ECH como métrica candidata, que se admite solo si vence al perfil de lags de la SSM y a C_64/C_0 en la escalera, incluido D7.

### (e) Ensayo de muerte barato (CPU, menos de 30 min)
- 500 piezas retenidas pasadas por D2–D5 y D7, más T5.
- Se comparan tres métricas: ECH, perfil de lags de la SSM y C_64/C_0 invariante.
- Muere si ECH no supera a ambas en AUC pareado en D4 y D7, o si falla T5.
- Residuo seguro: la coherencia de doble salida (f): F1 por nota normalizada por el techo del transcriptor, y ECH(MIDI) frente a ECH(AMT(audio)). Usar un transcriptor independiente, por la misma objeción que en X1.

---

## X4 — ECO: operadores de referencia con transformación dentro del vocabulario

### (a) Debilidades graves

1. **El contenido copiado no se materializa: el modelo tiene que resolver referencias encadenadas de memoria.**
   - Tras `ECO DIST_d FROM_q TRC_k`, las notas copiadas no están en la secuencia. El autómata las conoce, pero solo para la máscara de gramática; el modelo no las ve.
   - Para continuar, el modelo tiene que saber qué suena: para la conducción de voces, para armonizar una melodía literal sobre un bajo que es ECO del mismo compás, o para apuntar a un compás que a su vez es ECO de otro (cadenas 9→5→1).
   - Eso exige resolver por atención una indirección de varios saltos más una transposición. A 2.5M parámetros es casi seguro que falla [V0-conjetura]. Con eso, el ensayo mezclaría "copiar ayuda" con "copiar sin ver el contenido estorba" y no sería interpretable en ningún sentido.
   - A1 lo resuelve con la expansión fantasma: las notas se insertan sin pérdida, de forma determinista. El coste es una secuencia más larga, y se recupera en inferencia con Eco-C. Es el compromiso explícito que X4 ni menciona.
2. **Solo copia exacta, sin ediciones residuales, y una evidencia que no se transfiere.**
   - Sin DEL/ADD, cualquier compás que difiera en una nota (rellenos, variaciones, ruido de cuantización en el MIDI secuenciado de Lakh) se escribe literal, y la cobertura cae en el corpus objetivo.
   - El −28.5% de tokens por nota se midió en 1.186 piezas folk de dominio público. Es el régimen más favorable posible: monofónico, cuantizado y muy repetitivo, y no dice nada de Lakh multipista.
   - Además, reducir tokens no es evidencia de modelar mejor, y −28.5% queda dominado por el −64–69% de la BPE (mapa §2) como argumento de contexto. X4 no dice si compone con la BPE.
3. **Falsación débil y atribuciones al mapa sin respaldo.**
   - Los criterios de muerte no tienen IC ni umbral pre-registrado; "no menores" lo decide el ruido.
   - No hay filtro previo de cobertura y no excluye la batería, que es la victoria trivial de cualquier operador de copia.
   - Mide una sola "tasa de repetición", ciega a las casi-copias, que son justo la firma de un modelo sin copia.
   - Ninguna ablación separa "ahorro de tokens" de "decisión explícita".
   - Dice que "el mapa localiza LZMidi y R-Transformer". MAPA.md no menciona ninguno de los dos, así que la atribución no tiene respaldo en la fuente común.
   - "Medido en 1.186 piezas, ida y vuelta exacta en las 1.186" no trae script, salida ni ruta. El director debe exigir el artefacto antes de darle nivel V; hasta entonces es V0.
   - Menor: "cero parámetros extra" con unos 50 tokens nuevos es casi cierto, pero no exacto. El sesgo aditivo sobre los logits de ECO como "perilla de forma" distorsiona la distribución sin corrección; A3 da la versión con pesos.

### (b) ¿Refrito?
- No es refrito del mapa, porque ataca la misma cota S2-A48 que A1. Es una convergencia con A1: mismo nombre, tokens de distancia, fuente y transposición cromática/diatónica, y su ECOR equivale a mi `XF_RHY`.
- Fuera del mapa [V0]: compresión gramatical y LZ de música (Sequitur, el propio LZMidi que cita), redes puntero y CopyNet.

### (c) Solapamiento con A1, punto por punto (con el conflicto de interés declarado)

| Aspecto | Mejor | Por qué |
|---|---|---|
| Contenido copiado visible para el modelo | A1 | La expansión fantasma; X4 no la tiene (ver a.1). |
| Copias aproximadas | A1 | DEL/ADD con máscaras de índice; X4 solo copia exacto. |
| Ambigüedad entre fuentes equivalentes | A1 | Su pérdida de conjunto no obliga a aprender el desempate. El "más cercano" de X4 se puede aprender, pero obliga al modelo a comparar igualdades entre todas las candidatas. |
| Simplicidad | X4 | DIST/FROM como vocabulario, sin cabeza puntero. Bien para d ≤ 32. |
| Posición por compás | X4 | Una cabeza kv con RoPE por índice de compás. A1 no la tiene, y debe adoptarla. |
| Evidencia ejecutada | X4 (si existe el artefacto) | Ida y vuelta exacta y parse canónico ya escrito. |
| Doblajes entre partes | Empate | `DIST_0` en X4; anclas de pistas anteriores del mismo compás con `XF_TC±12` en A1. |
| Transposición diatónica con modulaciones | A1+A2 | TRD de X4 usa una tonalidad declarada global; con el marco por sección de A2 se resuelve. |
| Falsación | A1 | K0, sin batería, cota superior declarada, IC bootstrap, K1∧K2 pre-registrados, métrica por lag con casi-copias. |

### (d) Veredicto: FUSIONAR-CON-A1

### (e) Residuo útil para la fusión
- La cabeza RoPE por índice de compás.
- DIST/FROM como vocabulario, en un brazo de ablación "A1 sin puntero" que diga si la cabeza puntero merece su complejidad.
- El script de parse e ida y vuelta, si existe, como base de K0.
- Un tercer brazo en E-A1: literal / X4 (sin fantasma, solo exacto) / A1-X (fantasma + ediciones), 8 min cada uno, para que siga cabiendo en 30 min. Así el ensayo mata, de paso, la hipótesis "materializar no hace falta".

---

## Hallazgos transversales para el director

1. **Dos atribuciones al mapa no aparecen en MAPA.md:** LZMidi y R-Transformer, en X4. Además, una medición "ejecutada" (X4, 1.186 piezas) llega sin artefacto. Hay que tratarlas como V0 hasta ver la fuente.
2. **Dos defectos que comparte el lote:**
   - X1 y X3 evalúan la coherencia de doble salida con la misma familia de AMT con la que se fabricaron los pares. El lote necesita un transcriptor independiente.
   - Toda métrica de estructura basada en repetición (ECH, mi D) es ganable por modelos con operador de copia. D7 es obligatorio antes de comparar ECO o A1 con la base.
3. **Dependencias que cambian conclusiones:**
   - A2 (tokens relativos) anula el argumento central de X2.
   - A3 debe absorber el veto de X2 y el muestreo por rechazo de X3-g cuando las restricciones son por compás.
   - El grafo REF de A1 da consistencia de repeticiones al audio de X1.
