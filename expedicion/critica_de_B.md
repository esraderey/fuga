# Crítica cruzada ciega de B sobre Y1–Y4 (Expedición FUGA, F5)

Entradas: lote_para_B.md y MAPA.md. Sin web. Los antecedentes de fuera del mapa los cito de memoria, van marcados V0 y deben pasar por el chequeo de novedad.

## Veredictos

| Id | Veredicto | Motivo en una línea |
|---|---|---|
| Y1 | **SOBREVIVE** (solo Eco-X; base de la fusión con Y4) | Ataca una casilla libre del mapa y trae un ensayo de muerte barato y bien planteado. Eco-C cae. |
| Y2 | **CAE** (como solución de F4) | La tónica se consigue con transposición post hoc a coste cero; el modo, que es lo difícil, queda como control blando. |
| Y3 | **FUSIONAR-CON-Y1** | El núcleo es SMC de LMs ya conocido. Lo único nuevo (forma como máscara sobre el puntero) no se ensaya. |
| Y4 | **FUSIONAR-CON-Y1** | Duplicado inferior de Y1: sin edición, parseo por tokens y no por bits, evidencia y citas no verificables. |

---

## Y1 — ECO (operador de copia con puntero, transformación y edición)

### (a) Debilidades graves

1. **XF_TD no es causal.** La τ diatónica usa "la escala del marco local, Y2 **o un estimador global**". Un estimador global se calcula sobre la pieza entera, que en inferencia todavía no existe. En entrenamiento eso es fuga de información del futuro; en inferencia el fantasma de un `XF_TD` no se puede expandir. Y1 depende entonces de un token de marco declarado y causal que el LM muestree antes de usarlo (el FR de Y2 o un KEY inicial). Hoy esa dependencia está escrita como opcional y no lo es.
2. **Eco-C rompe la dependencia vertical entre pistas.** Con fantasmas visibles solo dentro de su segmento, las pistas siguientes del mismo compás y todos los compases posteriores ven de un segmento Eco solo `REF PTR XF DEL ADD`, nunca sus notas.
   - Para armonizar la melodía sobre un bajo Eco, el modelo tiene que resolver el puntero por atención indirecta.
   - Si la fuente también es Eco (cadenas Eco→Eco), el contenido no está en ningún token visible.
   - El propio autor predice una cobertura de bajo del 50–70%: en Eco-C la mayor parte del bajo sería invisible para la melodía. Se gana contexto (F2) a costa del modelado contrapuntístico. Eco-C tiene que salir.
3. **Criterio de muerte laxo y K2 inclinado a favor de E.**
   - Y1 muere solo si fallan K1 **y** K2. La reclamación es F1 (estructura). Si E gana en bits/nota pero no en D, Eco es compresión y no una solución de estructura, así que esa reclamación tendría que caer por sí sola.
   - D suma |P_gen − P_real| sobre 16 cifras con 30 prompts × 12 compases. El valor absoluto da sesgo positivo con muestras pequeñas: D(real, real) > 0. Falta la línea de ruido (D entre dos mitades del real).
   - El brazo B, de 3M parámetros entrenado 10 min, casi no produce copias exactas; E las produce por construcción. K2 está sesgado hacia E.
   - Hay que exigir K2 para reclamar F1 y medir D contra su línea de ruido.

**Menores:**
- J* solo marginaliza fuentes bajo la τ elegida. Las derivaciones equivalentes con otra τ (la fuente A con TC+2 frente a la fuente B con ID) o con otro conjunto de ediciones siguen penalizadas por la CE de `XF`/`DEL`. La cota de bits sigue siendo conservadora, así que esto no invalida nada.
- Las 0.5–5 h del parser solo son realistas con claves de hash invariantes a τ (la transposición cromática sale de la primera nota; no hace falta probar 40 τ) y código vectorizado. Con el bucle literal "para cada (j, τ)" en Python puro serían días.

### (b) ¿Refrito?
Lo es parcialmente. La combinación es nueva; las piezas no lo son.
- **La notación ya tiene un operador de copia exacta:** barras de repetición, casillas de 1ª y 2ª, D.C./D.S. y el signo de compás símil. El MusicXML de PDMX los codifica, y el ABC (el formato de MuPT y NotaGen en el mapa) tiene `|: :|` y `[1 [2`. Si MuPT o NotaGen entrenan con ABC sin expandir las repeticiones (V0, no sé si lo hacen), la "cota del barrido" del mapa es falsa para la copia exacta adyacente. Lo que queda como novedad es el lag arbitrario, la copia entre pistas, la transformación y la edición.
- **MELONS** (V0): grafo de estructura con relaciones por compás de repetición, secuencia rítmica y secuencia transportada, y generación condicionada a ese grafo. El conjunto de τ de Y1 (ID, TC/TD, RHY) es ese mismo conjunto de relaciones: "τ musicales" no es novedad. Sí lo son el puntero dentro del LM, el fantasma determinista y la edición residual.
- LZ77/MDL es compresión clásica; LZMidi (lo cita Y4, V0) sería un antecedente directo de "LZ aplicado a música". El autor ya cita pointer-generator y CopyNet.

### (c) Solapamiento con B1–B3
- **B3 (ECH).** Es el mismo motor: parseo LZ con horizonte frente a parseo MDL con lag.
  - Para evaluar, ECH es mejor que D: es invariante a transposición y tempo, cubre escalas hasta 64 compases, es distribucional con IC y está validada por la escalera metamórfica. D son 16 cifras sin validar.
  - Y1 tiene algo que ECH no tiene: P_near, la firma de la "copia descuidada" (una o dos notas distintas). Lo adopto como rasgo en ECH.
  - Eco y ECH comparten punto ciego: ambos tratan la estructura como repetición literal. Que Eco gane en ECH no demuestra forma percibida, así que para evaluar Y1 el ancla humana de B3-h es obligatoria.
- **B2 (GIM).** Hay sinergia: los fantasmas con pérdida 0 hacen innecesaria mi variante "primera aparición". La repetición interna se supervisa como decisión (REF/PTR), no como notas, y la máscara goldfish solo actúa sobre literales y `ADD`. Pero Eco reduce los bits necesarios para almacenar una canción (el estribillo se guarda una vez), así que a capacidad fija caben más canciones memorizadas (V0). Con Y1 sobre Lakh, la auditoría DER de B2 deja de ser recomendable y pasa a ser requisito.
- **B1.** El grafo de REF permite reutilizar latentes ya refinados en las ventanas donde **todas** las pistas repiten exactamente: da consistencia y ahorra cómputo. Con repeticiones parciales no sirve (colas de reverberación, notas que cruzan la barra de compás). Si Eco copia velocidades (R4 del autor), el humanizador H de B1 deja de ser opcional.

### (d) Veredicto
**SOBREVIVE** (núcleo Eco-X), como base de la fusión con Y4. Eco-C cae.

Condiciones:
- token de marco causal obligatorio para TD;
- K2 obligatorio para reclamar F1, con línea de ruido;
- chequeo de novedad contra MELONS y contra las repeticiones notacionales sin expandir.

### (e) Residuo
De Eco-C no se salva nada sin resolver antes el problema vertical. Eco-M queda para la escalera solo si Eco-X sobrevive a 85M.

---

## Y2 — MARCO (tonalidad como sistema de coordenadas)

### (a) Debilidades graves

1. **La tónica "por construcción" se consigue sin MARCO.** Basta generar con cualquier modelo, estimar la tonalidad de la salida y transponer a la tonalidad pedida: diez líneas y coste cero.
   - E-Y2 no incluye ese brazo (ABS + transposición post hoc), y con el mismo juez acertaría la tónica casi por definición.
   - El 7.69% de text2midi no demuestra que haga falta una representación nueva; demuestra que no se aplicó la transposición post hoc.
   - Lo que la transposición **no** arregla es el modo (mayor/menor) y las modulaciones planificadas. MARCO deja el modo como control blando (Y2-b): resuelve por construcción la parte trivial y deja blanda la difícil.
2. **Evaluación circular y sin potencia estadística.**
   - La adherencia se mide con el mismo estimador KS + Viterbi que fabricó las etiquetas: REL aprende "lo que KS llama tónica" y KS lo confirma. Los errores sistemáticos del estimador (mixolidio, relativa menor) quedan invisibles.
   - Con 72 muestras por brazo y p ≈ 0.8, el error estándar es ≈4.7 pts y el IC95 ≈ ±9 pts. El umbral de K-A ("ABS ≥ REL − 5 pts") no se puede resolver con ese n, y separar tonalidades frecuentes y raras con 3 muestras por tonalidad es ruido.
   - Falta un juez independiente del etiquetador (otro estimador, o las armaduras de PDMX retenidas como verdad) y hace falta n ≥ 400 por brazo, que en CPU sigue siendo barato.
3. **Supuestos y costes erróneos.**
   - "La simetría se paga una vez y no 12 veces en cómputo" es falso. El aumento por transposición no multiplica los FLOPs: los pasos son los mismos y las muestras aumentadas sustituyen a las originales. Lo que cuesta es capacidad y eficiencia muestral.
   - 3M parámetros y 10 min es el régimen más favorable posible a REL: ABS no tiene capacidad ni pasos para aprender las 12 copias. El resultado no extrapola a 290M con 8B tokens.
   - El hard-EM optimiza las etiquetas para la NLL de un proxy, no para la verdad musical. "Menor" acaba significando "lo que al proxy le salió más barato etiquetar como menor", y el control de modo pierde semántica. Además, el coste omite el reentrenamiento del proxy en cada ronda: ≈3× lo declarado, aunque siguen siendo horas.
   - **Idiomas absolutos.** Las cuerdas al aire de guitarra y bajo, los pedales en Mi y La o los metales en Si♭ y Mi♭ codifican el idioma en la altura absoluta. REL obliga al modelo a reconstruir t + RP para recuperarlo. El autor lo reduce a "rango a ±tritono", pero el problema es de idioma, no de rango.

### (b) ¿Refrito?
En gran parte.
- **Normalizar a una tónica y guardar la tonalidad como token** es la normalización "todo a Do", ya conocida. El autor la descarta con argumentos falsos: si la tonalidad se guarda como token, no se pierde información y sí sirve como control.
- **Codificar por grado relativo a una tonalidad declarada, con cambios por sección,** es el formato de datos de lead sheets tipo Hooktheory/TheoryTab (V0) y de las codificaciones funcionales.
- **Perfiles KK con HMM/Viterbi** para detectar la tonalidad: método clásico (Temperley, V0).
- **Y2-b** es un objetivo contrastivo del tipo MMI/unlikelihood; el propio autor cita MMI.
- **Dentro del mapa:** Moonbeam ya tiene atención relativa en altura, es decir, invarianza a transposición en las interacciones. "Todas las representaciones del mapa son absolutas" es cierto para los tokens, no para el sesgo inductivo.
- **Lo que no es refrito:** marcos seccionales que emite el propio LM como plan de modulación.

### (c) Solapamiento con B1–B3
- **B3(g).** Yo propuse muestreo con rechazo con comprobador exacto (≈0.49 con p = 0.08 y N = 8). Para la tónica, MARCO es mejor que mi rechazo y la transposición post hoc es mejor que ambos. Retiro el rechazo para la tónica y lo dejo para el modo.
- **B2.** MARCO vuelve casi invariante a transposición la tokenización, y eso debilita mi argumento (d-i) de que el aumento por transposición multiplica las copias. Pero solo si las etiquetas son coherentes entre versiones: dos versiones de la misma canción etiquetadas en Do mayor y en La menor dan RP desplazadas 3 semitonos, y por tanto hashes distintos. La φ por intervalos de B2 es robusta a errores de etiqueta y MARCO no. En ese punto B2 es mejor; MARCO es más barato.

### (d) Veredicto
**CAE** como propuesta para F4.

### (e) Residuo útil
1. El token de marco seccional, declarado y causal, es exactamente lo que necesita el XF_TD de Y1. Se integra en Y1, en coordenadas absolutas o relativas según decida la ablación.
2. REL frente a ABS + aumento se queda como ablación de eficiencia de datos en la escalera de GPU (25M→85M), no en CPU.
3. Regla para cualquier evaluación de control: incluir el brazo de transposición post hoc y un juez de tonalidad independiente del etiquetador.

---

## Y3 — ENJAMBRE DE COMPASES (SMC sincronizado por compás)

### (a) Debilidades graves

1. **La pieza nueva no se ensaya.** Lo único que no es SMC genérico es la forma como máscara exacta sobre el puntero de Y1. E-Y3 prueba densidad (T1) y tonalidad (T2/T3) sobre el brazo B de E-Y1, el literal, que no tiene Y1.
   - T2 es además un hombre de paja: la tonalidad se resuelve con transposición post hoc, sin decodificador.
   - Si E-Y3 sobrevive, prueba algo ya conocido; si cae, no dice nada de la forma.
2. **La salida "partícula de mayor peso" no muestrea de π: busca la moda.**
   - En música empuja hacia salidas de alta probabilidad (repetitivas, planas) y anula la reclamación de que "conserva la distribución". La salida tiene que ser muestreo ∝ w.
   - K2 ("a adherencia emparejada") no dice cómo se empareja (¿con κ?, ¿con N?).
   - El juez de 3M premia la baja entropía: un decodificador que colapsa gana en fidelidad. K3 solo lo frena con N = 16.
3. **Degeneración y potenciales de golpe.**
   - La cadencia "solo en B" no tiene torsión intermedia. En el último compás el SMC queda en un simple best-of-N, y ninguna partícula puede rehacer el compás anterior para preparar la cadencia (un V–I hay que prepararlo).
   - Las restricciones que no son token-locales (rango por frase, presencia de instrumentos con plazo) solo tienen garantía asintótica; con N = 8–16 la "corrección" es nominal.
   - Con ±2 notas de densidad y KS por compás, el ESS se hunde en pocos compases. El templado lineal de κ es ad hoc.
   - Generador y juez son LMs de 3M entrenados 10 min: la tasa de éxito por compás (q) de un modelo así no representa la del de 290M.

### (b) ¿Refrito?
Sí, en el núcleo.
- SMC para dirigir LMs con máscaras locales, corrección por la masa superviviente (log Z_t) y potenciales torcidos: SMC steering con programas probabilísticos (Lew et al. 2023, V0), twisted SMC (Zhao et al. 2024, V0), control sintáctico y semántico por SMC (Loula et al. 2025, V0). El propio autor cita esa familia.
- Muestreo exacto bajo restricciones en música: Markov constraints de Pachet y Roy (V0) y las restricciones de DeepBach (V0).
- Sincronizar por compás es la versión musical de sincronizar por palabra: una instanciación, no un mecanismo.
- Lo único no refrito es la forma como máscara sobre un operador de copia, y eso depende de Y1.

### (c) Solapamiento con B1–B3
- **B3(g).** Mi muestreo con rechazo es best-of-N con comprobador. Para restricciones que se acumulan compás a compás (éxito ~q^B), SMC es mejor. Lo concedo.
- **B2(f).** Mi veto en decodificación es justo el enmascarado ingenuo que Y3 critica: es sesgado y miope, y el modelo lo esquiva cambiando una nota y siguiendo con la copia. Con la corrección log Z_t, las partículas que "quieren" copiar (mucha masa en tokens vetados) pierden peso y desaparecen al remuestrear. Además, log Z_t en los pasos vetados es por sí mismo una señal de memorización en vivo: mide la masa que el modelo pone en continuaciones del corpus. En este punto Y3 mejora a B2; lo adopto.
- **B3 anti-Goodhart.** Si KAD-forma o ECH se usan como potencial Φ, la evaluación tiene que pasar a otra familia de métricas admitida.

### (d) Veredicto
**FUSIONAR-CON-Y1.**

### (e) Residuo útil
1. Un decodificador SMC por compás con corrección log Z_t para **todas** las máscaras duras del sistema: instrumentos, rango, marco y el veto de B2.
2. El ensayo que falta: plan de forma AABA como máscara exacta sobre REF/PTR. Se mide el cumplimiento del plan y la calidad (juez, ECH y 2AFC) frente a Y1 con solo la perilla de temperatura de REF.
3. Salida ∝ w, nunca argmax.

---

## Y4 — ECO en el vocabulario (DIST/FROM/TRC/TRD/ECOR)

### (a) Debilidades graves

1. **Es un duplicado inferior de Y1:** mismo nombre y mismo operador.
   - Sin edición residual (DEL/ADD) solo captura copias exactas. Las repeticiones casi exactas de Lakh (relleno al final de frase, una anacrusa, una nota cambiada) pasan a literal, y son justo las que Y1 recupera con ≤25% de edición.
   - Sin pérdida de conjunto, la ambigüedad de fuente se resuelve con una regla de desempate que el LM tiene que aprender a imitar.
2. **El parseo cuenta tokens, no bits.** La regla es "ECO si ahorra tokens (literal > 4)".
   - Un puntero sobre d ∈ [0, 32] × partes × ~40 τ elige entre ~1.3K × P opciones, ≈10–12 bits sin un buen prior. Un literal de 2 notas cuesta del mismo orden bajo un LM razonable (V0).
   - Con ventanas de 33 compases, todas las partes y 40 transformaciones, las coincidencias casuales de segmentos cortos son frecuentes. El parseo mete punteros espurios de alta entropía que el LM no puede predecir, así que la NLL de ECO puede empeorar por construcción.
   - Y1 lo mitiga con MDL y un mínimo de 3 notas; Y4 no tiene nada equivalente.
3. **Evidencia no verificable y atribuciones al mapa que no existen.**
   - "El mapa localiza … LZMidi … R-Transformer": ninguno de los dos aparece en MAPA.md. O el autor leyó fuera de contrato o la cita es inventada; en ambos casos no cuenta como novedad "contra el mapa".
   - "Medido en 1,186 piezas folk: 3.48→2.49 tokens/nota, ida y vuelta exacta" llega sin script, sin fuente y sin dataset nombrado, bajo un contrato sin ejecución: es V0 hasta que se reproduzca.
   - Aunque sea real: tokens/nota no son bits/nota; el folk AABB es el mejor caso posible; −28.5% es menos de la mitad de lo que ya da BPE (−64 a −69%, mapa §2) y nadie sabe si las dos ganancias se suman. La ida y vuelta exacta es una comprobación de sanidad de un parseo sin pérdida, no evidencia.
   - La falsación no tiene umbrales, n, IC ni definición de "tasa de repetición". Con "el mismo presupuesto de tiempo", ECO ve un 28% más de música por paso y la ganancia en bits/nota se confunde con haber visto más notas.

### (b) ¿Refrito?
- Es refrito de Y1, dentro del propio lote. Fuera, comparte los antecedentes de Y1: repeticiones notacionales, MELONS (V0), LZ y LZMidi si existe.
- La cabeza kv con RoPE por índice de compás es una adopción declarada por el autor. Con GQA y 4 kv-heads, quita el orden dentro del compás al 25% de las cabezas de consulta.

### (c) Solapamiento con B1–B3
Si LZMidi existe (V0), también es antecedente de "LZ sobre música" para mi ECH. Mis diferencias (horizontes, invariancia, uso exclusivo para evaluar, escalera de admisión) tienen que pasar el chequeo de novedad contra él. Con B1 y B2, lo mismo que en Y1.

### (d) Veredicto
**FUSIONAR-CON-Y1.**

### (e) Residuo útil
1. Un brazo de ablación "solo vocabulario" (DIST/FROM como tokens, sin cabeza puntero ni pérdida de conjunto). Responde si la cabeza puntero de Y1 justifica sus 0.26M parámetros y su complejidad.
2. La copia explícita entre partes del mismo compás (d = 0), para doblajes a la octava.
3. Un autómata de gramática que garantice que el 100% de las salidas se pueden decodificar. Y1 lo necesita para sus máscaras.
4. La perilla de sesgo sobre ECO, equivalente a la temperatura de REF de Y1.
5. Si el autor entrega el script, la medición sobre 1,186 piezas sirve como test de regresión del parser.

---

## Hallazgos transversales
- **T1. Los filtros del ensayo inflan la cobertura.** Todos los ensayos de CPU usan Lakh con "4/4 constante y ≥24 compases". Ese filtro selecciona pop secuenciado con copiar-pegar de secuenciador, que maximiza la cobertura Eco y la regularidad tonal. Un positivo ahí no se extrapola a PDMX: su clásica trae repeticiones notadas que, expandidas, son copias triviales. Hace falta un segundo conjunto de prueba: PDMX con repeticiones expandidas y sin expandir.
- **T2. La expresividad queda sin dueño.** La interfaz "MIDI + grafo REF + marcos" copia velocidades de la plantilla y cuantiza a 48 posiciones por compás, y ninguna Y dice qué pasa con la expresividad. Con B1, el humanizador H pasa a ser obligatorio.
- **T3. Discrepancia en el tamaño de los datos.** El lote estima Lakh + PDMX en 2–4B tokens; mi S9 suponía ~1.5B. Si el lote tiene razón, mi supuesto era bajo: hay menos épocas y menos riesgo de memorización por repetición. Se resuelve con un conteo real en CPU, en minutos.
- **T4. Legal.** El lote (S10) y yo (B2/S5) coincidimos en que la CC-BY de Lakh no cubre las obras subyacentes. Eco abarata en bits la memorización (V0), así que B2 es requisito si se usa Y1 sobre Lakh.

## Lo que esta crítica me obliga a cambiar en B1–B3
- **B3(g).** Para la tónica, transposición post hoc en lugar de rechazo; para el modo y las demás restricciones, SMC (residuo de Y3).
- **B2(f).** El veto pasa a ser una máscara con corrección log Z_t dentro del SMC, y log Z_t se usa como señal de memorización en vivo.
- **B3.** Añadir P_near como rasgo, declarar el punto ciego que ECH comparte con Eco y hacer obligatoria el ancla humana cuando se evalúe Y1.
- **B2.** Si se adopta MARCO, la φ por intervalos sigue haciendo falta por los errores de etiqueta.
