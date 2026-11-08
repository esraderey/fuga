# Innovador B — El sistema completo (Expedición FUGA)

Ángulo: de la salida simbólica al audio, reparto del presupuesto, evaluación y salvaguardas. Fuente única: MAPA.md. Lo que no viene del mapa va como V0-conjetura, incluidos los antecedentes que recuerdo de memoria fuera del mapa. Los cito para no vender como nuevo algo que quizá ya existe.

Tesis del sistema: el LM de 150–400M compone en simbólico y carga toda la estructura de largo plazo. El audio lo produce un refinador local de ≤120M que parte del render y no necesita estructura (ventanas de 10 s). Las salvaguardas y la evaluación trabajan en un mismo espacio invariante de melodía y ritmo.

---

## B1 — PxT: pares por transcripción + puente render→real

**Casilla atacada:** F3 (simbólico→audio sin pares alineados; CoSaRef refina con un modelo de ~1B) y la mitad de audio de F2 (un LM de audio de ~300M exige ≥20K h y decenas de GPUs).

### (1) Mecanismo

**Datos (se preparan una sola vez):**
- a. Audio: pistas de FMA con licencia CC-BY o CC0 (filtrado por metadatos), mono, 44.1 kHz, clips de 10 s. Filtro de voz: un separador de fuentes abierto descarta los clips cuya pista vocal supere −30 dB respecto a la mezcla. El objetivo `y` es la mezcla original, no la separada, para no aprender los artefactos de la separación.
- b. Pseudo-MIDI `m̂ = AMT(y)` con un transcriptor multi-instrumento abierto (clase MT3). Vía barata: tallos + transcriptor ligero, etiquetando por familia (batería, bajo, otros). Si el AMT no entrega velocidad, se estima del RMS del tallo en el frame de ataque.
- c. Render `r = FluidSynth(m̂, SoundFont permisiva)`. `r` e `y` quedan alineados por construcción, salvo el error temporal del AMT.
- d. Compuerta de consistencia: similitud coseno de croma (CQT) por ventana de 1 s entre `r` e `y`, más correlación de las envolventes de onset. El clip se conserva si la mediana es ≥ τ, con τ fijado para que pase ~60%.
- e. Índice de completitud `c ∈ [0,1]` por clip: fracción de los onsets detectados en `y` que tienen una nota de `m̂` a ≤50 ms. Se guarda como etiqueta.
- f. `r` e `y` se codifican con un codec abierto congelado (DAC-44k, 86 Hz). Se guardan los códigos (9×10 bits/frame) y se decuantizan al vuelo al latente continuo `z_q`. Así se evitan ~635 GB de latentes en coma flotante por cada 1.000 h y por lado.
- g. Opcional: Slakh2100 (no está en el mapa; V0) da pares perfectos pero sintéticos. Sirve para pre-entrenar, no como objetivo.

**Refinador F_θ (puente de flujo emparejado; no genera desde ruido):**
- `x_t = (1−t)·z_r + t·z_y + σ·√(t(1−t))·ε`, con σ pequeño. F_θ predice `v = z_y − z_r`. El muestreo son 4–8 pasos de Euler **desde z_r**, no desde ruido.
- Condiciones:
  - (i) piano-roll de `m̂` remuestreado a 86 Hz (128 alturas × 16 familias → conv 1-D), con 20% de descarte;
  - (ii) referencia tímbrica: 5 s de **otra** pista del mismo artista (si no hay, la misma pista a ≥30 s), codificados con el mismo codec y resumidos en 8 tokens por un pooler de atención, con 30% de descarte (CFG);
  - (iii) `c` como embedding escalar.
- Red: transformer 1-D sobre frames, atención local de 4 s y una de cada 4 capas global, ancho 768, 16 capas ≈ 100M (rango 60–120M).

**Humanizador H (opcional, 10–30M):** se entrena con pares `(q(m̂), m̂)`. `q` cuantiza los onsets a 1/16 sobre el pulso rastreado en `y` y aplana las velocidades por compás. H predice, por nota, Δonset (ms), razón de duración y velocidad. Convierte una salida tipo partitura (estilo PDMX) en interpretación, con expresividad aprendida de grabaciones CC reales.

**Inferencia:**
1. LM → `m`, la salida simbólica oficial.
2. H → render.
3. Proyección de ida y vuelta `m̃ = AMT(render(m))`: iguala formato y etiquetas a los del entrenamiento. Una ablación decide si hace falta.
4. `r = render(m̃)`.
5. F_θ con `c = 1` ("la partitura está completa: no inventes notas") y una referencia CC que elige el usuario.
6. Decodificar.

Las piezas largas se procesan en ventanas de 10 s con 1 s de solape. En cada paso, el solape se fija con el resultado de la ventana anterior (inpainting) y se cruza en 50 ms. La forma del audio es la de la partitura, así que el refinador nunca necesita memoria de largo plazo. Por eso cabe pequeño: el mapa §1 dice que los LM de audio no resuelven la estructura.

### (2) Novedad contra el mapa
- CoSaRef (mapa §2 [V2]) esquiva los pares refinando sin supervisión con Stable Audio Open (~1B; F3). Un refinado sin pares cambia fidelidad a las notas por realismo con una sola perilla de ruido. PxT **fabrica** pares alineados a partir de audio CC real, sin anotación humana. Así el refinador aprende supervisado, parte del render y cabe en ≤120M.
- MusicGen-melody (mapa §4) ya extrae un control (croma) del propio audio. La diferencia: aquí el control es la transcripción completa nota a nota, renderizada, y el generador es un puente local, no un LM entrenado con 20K h.
- El mapa no cubre ninguna de estas tres piezas: el índice de completitud `c` como perilla contra la alucinación de notas, la proyección de ida y vuelta para cerrar la brecha entre entrenamiento e inferencia, y la interpretación expresiva aprendida de transcripciones de audio CC.
- Antecedente posible fuera del mapa (V0-conjetura): síntesis MIDI→audio entrenada con alineaciones automáticas (p. ej., DTW partitura–grabación). El chequeo de novedad debe buscar "pseudo-pares por AMT para síntesis".

### (3) Coste en 1×24 GB
Supuesto: ~25 TFLOPs bf16 efectivos en una 3090 y ~60 en una 4090 (V0). Las horas salen de 6·N·tokens.

| Pieza | Parámetros | Tiempo | Memoria / disco |
|---|---|---|---|
| AMT sobre ~1.000 h | ~50–60M, congelado (V0) | 20–100 GPU-h (V0: 10–50× tiempo real) | <8 GB |
| Separación (filtro de voz) | congelado | 20–50 GPU-h; opcional (se puede filtrar por la etiqueta "instrumental") | <8 GB |
| Render FluidSynth | — | ~1 h en 16 núcleos de CPU (V0: ~100× tiempo real por núcleo) | — |
| Codificación DAC de r + y (2.000 h) | ~75M, congelado (V0) | 5–10 GPU-h | códigos ≈ 7 GB |
| Refinador | 60–120M | ≈8·10¹² FLOP/paso (batch 16×860 frames); 100K pasos ≈ 4 h (4090) / 9 h (3090); ≤1 día con carga y atención | AdamW fp32 ≈ 1.6 GB + activaciones bf16 ≈ 6 GB → <16 GB |
| Humanizador | 10–30M | 2–4 GPU-h | <6 GB |
| LM compositor (lo diseña el otro innovador; aquí solo para el encaje) | 150–400M | 300M × 6B tokens ≈ 1.1·10¹⁹ FLOP → ~2 días (4090) / ~5 días (3090) | ≤24 GB |

- **Inferencia completa:** LM de 300M + AMT + DAC + refinador en bf16 caben en <8 GB. Tres minutos de música son ~18 ventanas × 8 NFE: segundos (V0).
- **Reparto del presupuesto:** el 100% del presupuesto "LM" va al compositor simbólico; el audio no lleva LM. Todo lo que se entrena suma ≈0.25–0.55B parámetros y se entrena en ~1 semana de GPU, en secuencia.

### (4) Riesgos
- **R1 (principal): incoherencia entre las dos salidas.** El AMT pierde notas en mezclas densas. El refinador aprende a rellenar lo que falta en `m̂` y, en inferencia, puede añadir notas que el LM no escribió: el audio deja de corresponder a la partitura. `c` y la CFG sobre el piano-roll lo atacan. Si no bastan, la propuesta cae (lo mide E1-1).
- **R2: brecha de dominio.** FMA (electrónica, rock, folk) no es Lakh (pop) ni PDMX (clásica). Un cuarteto de cuerdas puede quedarse sin cobertura tímbrica. Mitigación: ponderar por familias y elegir la referencia.
- **R3: el timbre lo decide la referencia más que la etiqueta.** Si el AMT confunde instrumentos, el refinador aprende a ignorar la etiqueta y se pierde la elección de instrumento del LM. Hay que medir la adherencia por familia.
- **R4: fuga desde la referencia.** El refinador puede copiar la referencia en vez de seguir `z_r`. Por eso la referencia sale de otra pista.
- **R5: legal y procedencia.** Las licencias de FMA son heterogéneas: hay que excluir NC y ND, y SA es dudosa. CC-BY exige un manifiesto de atribución. Los pesos del AMT y del separador se entrenaron con datos que tienen licencias propias, algunas NC (V0): usarlos como herramientas de preprocesado es zona gris. La memorización del refinador frente a FMA se audita con MiRA (mapa §4).
- **R6: techo del codec.** La calidad la acota DAC (mapa §1: buena en música).

### (5) Falsación
- **E1-0 (CPU, ≤1 día).** 300 clips FMA CC-BY; transcriptor ligero en CPU → render. Se mide:
  - (a) el desfase de onsets entre `r` e `y` (correlación cruzada de envolventes);
  - (b) la similitud de croma;
  - (c) la fracción de clips que pasa τ = 0.6.

  **Muere** si la mediana de |desfase| supera 2 frames de latente (~23 ms) o si pasa la compuerta menos del 30% de los clips. En ese caso los pares por transcripción no están alineados para un puente síncrono.
- **E1-1 (GPU, 4–6 h).** 30–50 h de pares, refinador de 20M, 20K pasos, 200 clips retenidos. Se mide KAD contra audio FMA real retenido (2 embeddings) y la note-F1 entre el MIDI de entrada y AMT(salida), normalizada por la F1 de AMT(render crudo).
  - Controles: (i) render crudo; (ii) refinador entrenado con pares barajados (`y` de otro clip). Si el emparejado no gana al barajado, la alineación no aporta nada y PxT se reduce a un filtro de timbre.
  - Prueba decisiva de dominio: las mismas medidas sobre MIDI de Lakh (fuera de dominio), con y sin ida y vuelta, con `c = 1`.

  **Muere** si en Lakh la mejora de KAD sobre el render crudo es menor del 20% relativo o si la note-F1 cae más de 0.10 absoluto.

---

## B2 — GIM: guardia invariante de memorización

**Casilla atacada:** F5 (riesgo legal de memorización; según el mapa, en GEMA v. Suno almacenar canciones protegidas en el modelo infringe el derecho de reproducción) y F6 (datos "legales"). El mapa §4 solo trae auditorías en espacio de audio/CLAP (MiRA, SAO) y un ataque fonético. No hay nada simbólico, nada preventivo durante el entrenamiento y nada invariante a la transposición.

**Premisa que el mapa no recoge** (V0-conjetura; no es asesoría legal):
- La CC-BY 4.0 de Lakh cubre la compilación, no las obras. Sus MIDI son en su mayoría arreglos de canciones comerciales; el subconjunto emparejado con el Million Song Dataset lo muestra. Memorizar una melodía de Lakh es memorizar una obra protegida.
- Dos rasgos del entrenamiento musical rompen las defensas importadas del texto: (a) el aumento por transposición (12 tonos) y por tempo; (b) las muchas versiones del mismo tema dentro de Lakh. Una deduplicación o un hash sobre tokens crudos trata las copias transpuestas como textos distintos.

### (1) Mecanismo
- **a. Extractor invariante φ.** Por cada pista no percusiva se extraen líneas monofónicas: skyline, bajo y separación de voces por proximidad de altura.
  - Vista A: cada nota j es un símbolo (Δaltura recortada a ±12 → 25 valores) × (ρ = log2(IOI_j / IOI_{j−1}) cuantizado a pasos de 0.5 en [−2, 2] → 9 valores) = 225 símbolos.
  - Vista B (gruesa): (signo de Δaltura, ρ) → 27 símbolos.
  - Ambas son invariantes a transposición, tempo, instrumentación y velocidad.
- **b. Índice.**
  - Todos los h-gramas: h = 8 en la vista A y h = 16 en la vista B, con hash de 64 bits.
  - Para cada uno, DF(g) = número de clústeres-canción distintos que contienen g. Se guarda en arrays numpy ordenados con búsqueda binaria.
  - Para las auditorías, un arreglo de sufijos sobre la concatenación de líneas (alfabeto ≤256, 1 byte por símbolo).
- **c. Deduplicación por canción.**
  - MinHash/LSH sobre los 8-gramas A de la línea principal de cada archivo agrupa en clústeres las versiones del mismo tema. Se conservan 1–2 por clúster (las más completas).
  - Cruce PDMX × Lakh: una pieza de PDMX que cae en un clúster de Lakh emparejado con una grabación comercial se marca como sospechosa. Es una partitura "de dominio público" que en realidad es el arreglo de una canción comercial (ataca F6).
- **d. Pérdida goldfish invariante con exención por frecuencia.**
  - Para la nota j de la línea ℓ, κ = hash(s_{j−7..j}) en la vista A.
  - Se anula la pérdida de **todos** los tokens que codifican esa nota si `hash(κ‖sal) mod k = 0` y `DF(κ) ≤ f_max`. Por defecto, k = 4 y f_max = 3.
  - Efecto: todas las copias transpuestas, a otro tempo o rearregladas de una melodía pierden **las mismas** notas, y esas notas no reciben gradiente nunca. Las fórmulas comunes (escalas, arpegios, cadencias: DF alto) se aprenden enteras.
  - Como la máscara es invariante, se calcula una vez por archivo y sirve para todas las transposiciones: coste cero en el bucle.
  - Variante "primera aparición": dentro de una pieza solo se anula la primera aparición de cada κ. Las repeticiones internas (el segundo estribillo) quedan supervisadas porque enseñan a copiar del contexto, que es estructura (F1) y no memorización. La ablación es obligatoria: la segunda aparición puede filtrar memorización.
  - Interfaz con el tokenizador del otro innovador: solo necesita saber qué posiciones codifican cada nota y a qué pista pertenece.
- **e. Auditoría calibrada contra un nulo humano.**
  - λ(x) = longitud de la coincidencia invariante más larga entre una pieza y el corpus de entrenamiento (estadísticas de coincidencia sobre el arreglo de sufijos).
  - Nulo: H = piezas humanas de clústeres retenidos (Lakh y PDMX); n* = percentil 99 de λ en H. **Tasa de réplica en exceso:** ERR = P_G(λ > n*) − 0.01. El nulo es necesario porque en música los fragmentos cortos que comparten obras independientes son mucho más frecuentes que en texto (V0-conjetura; E2-0 lo mide).
  - **Extracción descubrible (DER):** 1.000 clústeres estratificados por número de versiones (1, 2–5, 6–20, >20). El prompt son los primeros 8 compases en tono y tempo aleatorios; se generan 8 compases (voraz y T = 1). Cuenta como éxito si la coincidencia invariante con la continuación real de cualquier miembro supera n*. Se reporta la curva DER frente a duplicación.
  - **Canarios:** 200 aleatorios (alta entropía) + 200 realistas (melodías de dominio público de PDMX retenidas, con predictibilidad real), duplicados d ∈ {1, 4, 16, 64} con transposición, tempo e instrumento aleatorios. Los realistas son imprescindibles: los aleatorios sobreestiman la defensa.
- **f. Veto en decodificación** (defensa en profundidad; no sustituye a d).
  - Un filtro de Bloom guarda los n*-gramas idiosincráticos (DF ≤ f_max) de las vistas A y B. Al muestrear el token de altura de una nota se enmascaran los candidatos que completarían uno: ≤128 consultas por nota, en CPU.
  - La vista B, con ventana más larga, atrapa copias con una nota alterada.
  - El veto no quita lo que ya está almacenado en los pesos, que es justo lo que señala el fallo de Múnich (mapa §4). Por eso la defensa principal es d.

### (2) Novedad contra el mapa
- En el mapa, la memorización solo se audita en audio o en espacio CLAP; F5 queda abierto, sin nada simbólico ni preventivo.
- Antecedentes fuera del mapa, de memoria y sin verificar (V0): deduplicación contra memorización, pérdida goldfish con hash de contexto y decodificación con filtro de n-gramas, todos en LMs de texto. Lo nuevo frente a ellos:
  - (i) El hash vive en el grupo de invariancia musical (transposición × tempo × instrumentación), que es justo donde el aumento de datos musical multiplica las copias. Predicción falsable: con aumento de 12 tonos y k = 4, el goldfish sobre tokens crudos deja supervisada al menos una vez ~97% de cada nota idiosincrática (1 − (3/4)¹² ≈ 0.968). El invariante deja el 25% sin supervisar nunca.
  - (ii) La exención por DF dirige los huecos a lo idiosincrático, que es lo protegible, y deja intacto el vocabulario común (V0 legal).
  - (iii) Los umbrales se calibran con un nulo humano en lugar de un n fijo.
  - (iv) La variante de primera aparición separa copiar del contexto de recordar del corpus.

### (3) Coste
- **Extracción, índice y DF:** en CPU. Con ~4–7·10⁸ notas (V0), ordenar 4–7·10⁸ uint64 ocupa ≈3–6 GB de RAM y lleva 1–3 h. El arreglo de sufijos ocupa ≈5 bytes por símbolo (2–3.5 GB) y tarda <1 h. MinHash: 1–2 h de CPU.
- **Entrenamiento:** la máscara está precalculada, así que el sobrecoste de cómputo es ≈0. La señal que se pierde es ≈ (1/k) × fracción idiosincrática ≈ 15–20% de las notas (V0). Igualar la pérdida exige +15–20% de pasos, dentro del régimen de ≤4 épocas (mapa §3).
- **Auditoría:** ~6·10⁵ tokens generados con un LM de 300M son minutos de GPU; las búsquedas en CPU, segundos.
- **Veto:** ~128 consultas Bloom por nota, despreciable. Un Bloom de ~10⁸ elementos al 1% ocupa ≈120 MB.

### (4) Riesgos
- **R1 (principal): la música es predecible.** Una nota sin gradiente puede recuperarse por generalización (secuencias, notas repetidas), así que el hueco no rompe la extracción tanto como en texto. Lo mide el canario realista.
- **R2: literal no es lo mismo que similitud sustancial.** Bajar la extracción literal no prueba que no haya copia reconocible: con un hueco cada ~4 notas, una melodía famosa puede seguir reconociéndose.
- **R3: calidad.** Un f_max demasiado bajo deja huecos en patrones semi-comunes; uno demasiado alto deja sin defensa. Se fija con val-loss y DER.
- **R4: cobertura.** Los riffs de acompañamiento y las voces interiores que el skyline extrae mal quedan sin protección.
- **R5: legal.** El fallo de Múnich no es firme y su alcance para melodías simbólicas es incierto (V0). Más segura que cualquier mecanismo: excluir el subconjunto de Lakh emparejado con grabaciones comerciales y apoyarse en PDMX. GIM sigue sirviendo para medir lo que quede.

### (5) Falsación
- **E2-0 (CPU, ~2 h).** Sobre Lakh se mide:
  - (a) la tasa de duplicación invariante (clústeres con ≥2 versiones; fracción de los 8-gramas idiosincráticos de cada archivo que aparecen en otro archivo del mismo clúster) frente a la cruda (hash de archivo; n-gramas de tokens);
  - (b) λ del nulo humano y n*.

  **Muere**, o baja a accesorio, si la duplicación invariante no duplica al menos (≥2×) a la cruda, o si n* sale tan grande (>40 notas) que ninguna auditoría práctica discrimina.
- **E2-1 (GPU, 4–8 h en total).** Tres brazos con un LM de ~50M sobre ~200M tokens de Lakh deduplicado más los canarios, todos con aumento de 12 tonos:
  - A: sin defensa;
  - B: goldfish crudo, k = 4;
  - C: goldfish invariante, k = 4, con exención por DF.

  Para que el ensayo valga, A tiene que memorizar (DER_A ≥ 0.3 en algún d); si no, se sube d o el número de pasos. **Muere** si C no reduce la DER de los canarios realistas en ≥50% relativo frente a B, en d = 16 o en el menor d en que A memorice. También muere si la penalización de val-NLL de C supera a la de B en más de 2 puntos porcentuales. Predicción declarada: B ≈ A para d ≥ 4.

---

## B3 — ECH: espectro de compresión por horizonte + escalera metamórfica

**Casilla atacada:** F5 (métricas simbólicas sin validar; su correlación con el juicio humano queda en la cota del barrido) y la medición de F1 (estructura). En audio: el FAD está sesgado por el tamaño de muestra y depende del embedding (mapa §4).

### (1) Mecanismo
- **a. Secuencias.** Se usan las vistas A y B de φ (de B2) por línea, más una vista armónica Hm: por pulso, el conjunto de clases de altura relativo a la tónica estimada (perfil de Krumhansl), como máscara de 12 bits hasheada. La rejilla de compases sale de los tokens de compás del LM o del MIDI; para audio re-transcrito, de un rastreador de pulso.
- **b. Código con horizonte.** C_h(S) = bits de un parseo LZ77 voraz que solo permite copiar de fuentes situadas a ≤h compases, con h ∈ {0, 1, 2, 4, 8, 16, 32, 64}.
  - Literal: modelo adaptativo de orden 0, idéntico para todo h.
  - Puntero: log2(número de posiciones en la ventana) + Elias-γ(longitud); longitud mínima 3.
  - C_0 = solo literales.
- **c. Espectro de forma por vista.** g_v(h) = max(0, C_{h/2} − C_h) / C_0 para h = 1..64 (7 valores), más la razón total C_64 / C_0: 3 vistas × 8 = 24 dimensiones. Tiene dos colas: tanto la falta de forma (masa solo en h ≤ 2) como la sobre-repetición degenerada (masa excesiva en una sola escala) aparecen como distancia a lo real.
- **d. Comparación distribucional ("KAD-forma").** MMD insesgado con kernel RBF (ancho por mediana) entre los vectores estandarizados de G y de un conjunto real retenido R, estratificado por fuente (Lakh/PDMX) y por longitud. Se acompaña de prueba de permutación, IC por bootstrap y un diagnóstico por horizonte (p. ej., "falta forma a 8–16 compases").
- **e. Escalera metamórfica.** Admite o rechaza **cualquier** métrica: la propia ECH, la entropía de clases de altura, la tasa de repetición o puntuaciones tipo CLaMP.
  - Destructores de estructura que conservan lo local:
    - D1: barajar pulsos dentro del compás;
    - D2: barajar compases dentro de la frase de 4;
    - D3: barajar frases de 4 compases;
    - D4: des-repetir (cada sección repetida, detectada por la matriz de auto-similitud, se sustituye por otro segmento no repetido de la misma pieza);
    - D5: sobre-repetir (bucle de los 4 primeros compases);
    - D6: grados (x% de frases barajadas, x ∈ {0, 25, 50, 75, 100}).
  - Transformaciones que conservan la estructura:
    - T1: transposición de ±1 a 6 semitonos;
    - T2: tempo ×0.8–1.25;
    - T3: reinstrumentación;
    - T4: ruido de velocidad;
    - T5: ruido temporal de ±10 ms;
    - T6: doblaje a la octava.
  - Criterio de admisión: AUC pareado ≥0.8 en D2–D5; Spearman ≥0.9 a lo largo de D6; y, bajo T1–T6, una diferencia que no se distinga de la que hay entre dos mitades del mismo real (permutación, p > 0.05).
  - Salida: una ficha de validez por métrica. Solo las métricas admitidas entran en la comparación de sistemas.
- **f. Coherencia de doble salida** (enlaza con B1).
  - note-F1 entre el MIDI del LM y AMT(audio), normalizada por la F1 de AMT(render crudo), que es el techo del transcriptor.
  - Distancia entre ECH(MIDI) y ECH(AMT(audio)): la forma tiene que sobrevivir al render.
  - Calidad de audio: KAD en ≥2 embeddings (uno general y uno musical; V0), con N fijo e IC por bootstrap.
- **g. Adherencia verificable** (integración; no se reclama novedad). Tonalidad, tempo, métrica e instrumentación se comprueban de forma exacta en lo simbólico. La tasa de acierto es la métrica de F4, y el mismo comprobador permite muestreo con rechazo: acierto = 1 − (1 − p)^N. Con p = 0.08 (cerca del 7.69% de text2midi) y N = 8, sale ≈0.49.
- **h. Ancla humana mínima.** Prueba 2AFC con ~20 oyentes × 24 pares (real frente a D3/D4; sistema frente a sistema). Se exige que el orden de KAD-forma coincida con la preferencia humana por encima del azar.

**Regla anti-Goodhart:** ECH solo sirve para evaluar. Si se usa como recompensa (al estilo de CLaMP-DPO en NotaGen, mapa §2), la evaluación pasa a otra familia de métricas admitida.

### (2) Novedad contra el mapa
- En el mapa, las métricas simbólicas no están estandarizadas ni validadas, y KAD (MMD) solo está validado en audio. Museformer usa la serie de compases 1, 2, 4, …, 32 como diseño de atención, no como medida.
- ECH aporta a la vez varias propiedades:
  - no usa modelo, así que no hay juez que optimizar;
  - es multiescala y de dos colas;
  - es invariante a transposición y tempo, y capta la variación mediante la vista gruesa;
  - es distribucional (MMD);
  - trae un protocolo de admisión por evidencia (la escalera) aplicable a cualquier métrica.
- Antecedente posible fuera del mapa (V0): el análisis musical por compresión (familia de descubrimiento de patrones por compresión de conjuntos de puntos) y las métricas de paquetes tipo MusPy. Hasta donde sé, ninguno resuelve por horizonte ni se valida con una escalera metamórfica; el chequeo de novedad debe buscarlo.

### (3) Coste
- Todo en CPU. El LZ voraz con cadenas de hash es O(N·W) acotado. Con ~10³–10⁴ símbolos por pieza son milisegundos por pieza y horizonte; 10K piezas × 8 horizontes × 3 vistas son minutos en un portátil.
- MMD con 1–5K muestras: segundos. Escalera: minutos.
- En audio, el AMT y los embeddings de 1–2K clips llevan menos de 1 GPU-h.
- Ancla humana: voluntarios, sin GPU.

### (4) Riesgos
- **R1: repetición no es estructura percibida.** El desarrollo temático, la aumentación y disminución y la estructura armónica de largo alcance pueden escaparse. La música clásica de desarrollo puede parecer "sin forma".
- **R2: depende de la rejilla de compases.** Un LM que rompe la métrica produce horizontes falsos; el respaldo es medir por pulsos.
- **R3: Goodhart** si se convierte en recompensa (de ahí la regla).
- **R4: mezcla de géneros.** Con un R mal estratificado, KAD-forma se vuelve un detector de género.
- **R5: la escalera valida la sensibilidad a destrucciones sintéticas,** no a los fallos reales de un LM, que pueden ser otros. Por eso hace falta el ancla humana.

### (5) Falsación
- **E3-0 (CPU, ≤4 h).** 2.000 piezas retenidas (1K de Lakh, 1K de PDMX) pasadas por la escalera. **Muere** si se cumple cualquiera de estas condiciones:
  - (a) AUC pareado < 0.8 en D3 o en D4;
  - (b) T1 o T2 producen una diferencia detectable (p < 0.05) frente a la partición real–real;
  - (c) D5 no se distingue del real, es decir, la métrica no es de dos colas;
  - (d) una línea base trivial (gzip sobre tokens crudos, o tasa de repetición tipo MusPy) pasa la misma escalera con igual sensibilidad e invariancia, y entonces ECH no aporta.
- **E3-1 (GPU barata, si existen dos checkpoints).** Un LM con contexto recortado (p. ej., 512 frente a 2.048 tokens) debe perder masa en h ≥ 8. Si no se distingue, ECH no ve lo que dice ver.

---

## Supuestos declarados
- **S1.** FMA trae la licencia de cada pista en los metadatos; la fracción CC-BY/CC0 es desconocida (V0). B1 necesita ≥300–1.000 h útiles después de los filtros de licencia, voz y consistencia.
- **S2.** Existe un transcriptor multi-instrumento abierto (clase MT3) y otro ligero que corre en CPU, con licencias que permiten usarlos como herramienta (V0). Su calidad en mezclas es la incógnita que mide E1-0.
- **S3.** Codec DAC-44k con pesos de licencia permisiva (V0), 86 Hz y 9 libros de códigos. Se guardan los códigos y se decuantizan al vuelo.
- **S4.** Hay disponible una SoundFont General MIDI con licencia permisiva (V0).
- **S5.** La CC-BY de Lakh no licencia las composiciones subyacentes (V0; no es asesoría legal). Si se entrena con Lakh, B2 pasa de buena práctica a requisito.
- **S6.** El tokenizador del LM (otro innovador) expone la correspondencia token→nota→pista y las marcas de compás. B2 y B3 solo dependen de esa interfaz.
- **S7.** Hardware: 1 GPU de 24 GB + estación con ≥32 GB de RAM (mejor 64), ≥16 núcleos y ~1–2 TB de disco (FMA completo ≈ 0.9 TB; V0).
- **S8.** Rendimiento bf16 efectivo de ~25 TFLOPs (3090) a ~60 (4090), con 35–40% de MFU (V0). Todas las horas del reporte se derivan de ahí con 6·N·tokens.
- **S9.** El LM compositor ve ~1.5B tokens únicos (Lakh + PDMX) × ≤4 épocas (V0). La cifra solo se usa para encajar el presupuesto.
- **S10.** Ningún número de calidad de este reporte está medido. Todos son predicciones que los ensayos E1–E3 confirman o matan.
