# RFC-FUGA-2 — Intérprete FUGA-I: de la partitura al audio en dos capas (ejecución determinista y refinador PxT)

- Estado: ACEPTADO (aprobado para construir el 2026-10-25)
- Fecha: 2026-10-25
- Firmado: OMRI
- Expedición: «cómo construir un modelo generativo de música tipo LM de 150–400M parámetros, innovador e implementable
  en código, que produzca a la vez representación simbólica y audio escuchable, entrenable en una GPU de 24 GB con datos
  legalmente utilizables» (F0, 2026-10-24), ampliada el 2026-10-25 con los sectores S5 (render neuronal y puentes) y S6
  (interpretación expresiva y render determinista).
- Depende de: RFC-FUGA-1 (compositor FUGA-C). Se relaciona con: RFC-FUGA-3 (medición y salvaguardas).
- Regla de tinta: cada párrafo va marcado **HECHO.** (afirmación con fuente o con ejecución registrada) o **PROPUESTA.**
  (decisión nuestra). Las cotas del barrido se redactan como «no localizado», nunca como inexistencia.

## 1. Resumen

FUGA-C emite partituras cuantizadas sin dinámica ni microtiempo. FUGA-I las convierte en audio en dos capas separables.
La capa I-1, **implementada y verificada** en `fuga/interprete.py`, es un intérprete determinista por reglas de la familia
KTH que produce una ejecución expresiva con invariantes comprobados por tests: no añade ni quita notas, cada ataque
queda a menos de 50 ms de su posición en el mapa de tempo, el orden de ataques por voz se conserva y la misma semilla da
la misma salida. Su render es síntesis aditiva propia; el MIDI con dinámica que emite admite cualquier sintetizador externo. La capa I-2, **propuesta**, es
PxT enmendado: un refinador de 20–120M parámetros que parte del latente de codec del render y lo acerca al audio real
mediante un puente estocástico, entrenado con pares fabricados por transcripción de audio con licencia Creative Commons.
La coherencia entre partitura y audio queda garantizada por construcción en I-1 y se mide en I-2 con un transcriptor
independiente. Se dan presupuestos para 24 GB y para la GPU de 8,6 GB disponible, y cinco experimentos asesinos
ordenados por coste, el primero ya ejecutado.

## 2. Estado del arte (del Mapa del Territorio, § 1–§ 2 y § 7)

Solo afirmaciones V2 o V3. Las anclas de § 7 del mapa (S5 y S6) pasaron por verificación ciega el 2026-10-25: 45
muestreadas, 42 confirmadas, 3 parciales (dos se señalan en § 6 con (p); la tercera es el detalle N = 250, n = 150
de SDEdit aplicado a audio en [R3], que no se cotejó), 0 falsas (`expedicion/verificacion_S5_S6.md`). Las citas vienen
de la lectura de las páginas por el extractor de WebFetch, no de un cotejo carácter a carácter.

### 2.1 Render neuronal de partitura a audio

- **HECHO.** MIDI-DDSP se entrena con pares nota-audio alineados de URMP (13 instrumentos, unas 3 h) y está limitado a
  instrumentos monofónicos; sus autores proponen, solo como trabajo futuro, obtener etiquetas de nota con transcriptores
  [R1].
- **HECHO.** Spectrogram Diffusion (ISMIR 2022) usa pares audio-MIDI de los conjuntos de MT3 y 34 clases de instrumento
  más batería; su modelo base tiene 412M parámetros; la F1 de transcripción de su salida es 0,36 frente a 0,63 del audio
  real, con artefactos y sonoridad inconsistente declarados [R2].
- **HECHO.** CoSaRef (2024) evita los pares: afina el DiT de Stable Audio Open con audio sin anotar y usa MT3 solo para
  evaluar. En URMP mejora FAD y F1 frente a MIDI-DDSP (0,235 y 0,412 frente a 0,526 y 0,323), pero el MOS de realismo
  favorece a MIDI-DDSP (3,27 frente a 2,57–2,84). Evaluación solo monofónica [R3].
- **HECHO.** Performance Conditioning (ICASSP 2024) entrena con unas 58 h de grabaciones reales de música clásica y MIDI
  alineado preexistente, no transcrito [R4].
- **HECHO.** Multi-Aspect Conditioning (TASLP 2025, página de proyecto de los autores; el texto del artículo no pudo
  abrirse) declara una variante condicionada por predicciones de un transcriptor en lugar de alineaciones, con resultados
  comparables, entrenada solo con datos reales (~58 h): «For score conditions we use predictions of a transcriber, rather
  than alignments, and obtain comparable results» [R5]. Es el antecedente más cercano a la fábrica de pares de PxT.
- **HECHO.** RenderBox (2025) va de partitura MIDI más texto directamente a audio multiinstrumento; FAD 340,8 frente a
  1017,5 de MusicGen-Melody; prueba de escucha con 23 participantes; limitaciones de calidad acústica declaradas; tamaño
  del modelo no publicado [R6].
- **HECHO.** Tang et al. (2025) encadenan un transformer de interpretación expresiva (partitura MIDI a MIDI expresivo) con
  un sintetizador neuronal: correlación de velocidad 0,831 e IOI 0,990 en ATEPP; en escucha (13 participantes, MOS sobre
  10) la salida expresiva con Pianoteq (6,69) gana a la partitura con Pianoteq (5,07) y ambas ganan al sintetizador
  neuronal ajustado (3,87) [R7]. Lectura nuestra, no afirmación de los autores: la capa de interpretación aporta más que la capa de
  timbre neuronal, y un sintetizador neuronal pequeño puede empeorar el resultado.
- **HECHO.** CPR (septiembre de 2026) va de MIDI a audio en tres etapas (Composer, Performer y Refiner), con timbre de una
  grabación de referencia; solo se leyó el resumen [R8].

### 2.2 Puentes sintético→real

- **HECHO.** I²SB parte de la señal degradada, no de ruido, es estocástico por defecto, usa 1000 pasos por defecto y
  informa de 2–10 NFE en inpainting; sus autores declaran que requiere datos emparejados [R9].
- **HECHO.** DDBM usa un muestreador híbrido ODE y Euler-Maruyama porque seguir solo la ODE resulta problemático; N = 40 en
  traducción 64×64 [R10].
- **HECHO.** SDEdit añade ruido a la señal guía hasta t0 ∈ [0,3; 0,6] y elimina el ruido; CoSaRef aplica el mismo mecanismo a
  audio (N = 250, n = 150). La fuente dice «we should choose t0 appropriately», no «must» (ancla parcial) [R11].

### 2.3 Interpretación expresiva

- **HECHO.** El sistema de reglas KTH (Friberg, Bresin, Sundberg) incluye phrase arch, duration contrast, melodic charge y
  final ritard; en la evaluación de la regla «phrase final note», 15 oyentes prefirieron la versión con regla en el 70 % de los juicios.
  El total de 25 reglas es un recuento del extractor, no una cifra declarada por la tesis (ancla parcial) [R12].
- **HECHO.** DExter (2024), la comparación abierta más reciente leída: con 82 oyentes y escala de 100, la ejecución humana
  puntúa 51,81; DExter 48,54; VirtuosoNet 48,31 y Basis Mixer 46,33; las correlaciones de la curva de tempo son bajas
  (0,19; 0,09 y 0,02); inferencia de 40 s para 95 s de música en una RTX 6000; asincronía y estructura vertical sin
  resolver [R13].
- **HECHO.** Datos de ejecución alineada: ASAP (>92 h, piano) y MAESTRO (198,7 h, piano) son CC BY-NC-SA 4.0; ATEPP
  (~1000 h) es CC BY 4.0 pero su MIDI está transcrito de grabaciones, no capturado [R14] [R15] [R16].

### 2.4 Herramientas, codecs, transcriptores y licencias

- **HECHO.** FluidSynth es LGPL-2.1 [R17]. MuseScore_General es MIT [R18]. Salamander Grand Piano es CC BY 3.0 [R19].
  GeneralUser GS y FluidR3_GM: licencia no confirmada en las páginas leídas (cota del barrido).
- **HECHO.** DAC publica sus pesos bajo MIT, modelo de 44,1 kHz a 8 kbps; su README no indica tasa de frames ni número de
  codebooks [R20]. Mimi: pesos CC BY 4.0, 12,5 Hz, entrenado con voz [R21].
- **HECHO.** MT3: código Apache-2.0, licencia de los checkpoints no declarada [R22]. YourMT3+: código GPL-3.0, 44,7–45,8M
  parámetros, F1 de onset en Slakh 84,14 frente a 75,20 de MT3 y por debajo del 10 % en instrumentos no principales
  [R23]. Basic Pitch: Apache-2.0, agnóstico al instrumento, sin F1 publicada en el README [R24].
- **HECHO.** FMA: 106 574 pistas tras excluir 2616 que prohibían la redistribución; el artículo no desglosa cuántas son
  CC BY o CC0 frente a NC o ND [R25].
- **HECHO.** Modelos de audio condicionados por rasgos extraídos del propio audio (no por notas): JASCO alcanza 49,1 % de
  exactitud de melodía con unos 330M; MuseControlLite 61,1 % con 85M entrenables en una RTX 3090 [R26] [R27]. Estas dos
  anclas no entraron en la muestra del verificador.

### 2.5 Resultado del ensayo de FUGA-C que condiciona a FUGA-I

- **HECHO.** (`ensayo/informe.md`). La gramática del decodificador no produjo ningún fallo en todas las generaciones; el
  brazo `eco_rel` acierta la tonalidad pedida en el 56 % de los casos con un techo del juez del 76 %. Consecuencia para el
  diseño: la entrada de FUGA-I es siempre una partitura válida con tonalidad y tempo declarados.

### 2.6 Frontera (cotas del barrido)

- **HECHO.** No localizado en el barrido (S6, tres consultas dirigidas) un sistema publicado que encadene un LM simbólico,
  un intérprete expresivo y un refinador neuronal de timbre. Los vecinos cubren tramos: Tang et al. (intérprete y
  sintetizador, entrada partitura), RenderBox (partitura a audio directo), CPR (MIDI a audio con refinador).
- **HECHO.** La idea de fabricar pares por transcripción automática para entrenar síntesis **sí tiene antecedente**:
  Multi-Aspect Conditioning (2025) condiciona por predicciones de un transcriptor sobre ~58 h de grabaciones clásicas
  reales [R5]. No localizado (S5, cuatro consultas y una consulta de cierre) su uso a escala sobre audio «in the wild»
  con filtro de licencia, ni como puente desde un render determinista, ni con un refinador de ≤120M.
- **HECHO.** El mapa (§ 1) recoge que los LM de audio de ~300M exigen ≥20 000 h y decenas de GPU, y que no resuelven la
  estructura de largo plazo. Decisión de diseño que se sigue de ello: FUGA-I no entrena ningún LM de audio.

## 3. Propuesta

### 3.1 Arquitectura en dos capas y contrato de interfaz

**PROPUESTA.** FUGA-I recibe exactamente lo que emite el decodificador de FUGA-C: la lista de notas `(ataque, duración,
altura, parte)` en ticks, los inicios y largos de compás, el tempo y la tonalidad declarados, más un estilo y una
semilla. Devuelve dos artefactos. La capa I-1 devuelve eventos `(t0_s, dur_s, altura, velocidad, parte)` alineados uno
a uno con la nota de origen (`Interpretacion.origen`), un mapa de tempo por pulso y, opcionalmente, MIDI con dinámica y
WAV. La capa I-2 devuelve audio a 44,1 kHz a partir de esos eventos.

El invariante de coherencia de doble salida (restricción F0) se reparte así. **HECHO.** En I-1 se cumple por
construcción: los tests de `tests/test_interprete.py` comprueban que el multiconjunto de notas de salida es el de
entrada y que ningún ataque se desvía más de `max_desvio_s`. **PROPUESTA.** En I-2 no puede garantizarse y por eso se mide (§ 3.3 y criterio de muerte de E3 en § 5).

### 3.2 Capa I-1: intérprete determinista (implementada)

**HECHO.** (`fuga/interprete.py`; 55 tests en verde el 2026-10-25, 27 de ellos sobre el intérprete, tras una
revisión adversarial independiente en dos pasadas: `expedicion/revision_g4_interprete.md` y `_v2.md`). Las reglas son adopción de la familia KTH [R12], no
novedad: acento métrico (compás y pulso), arco de frase en dinámica y en tempo sobre frases de 4 compases, high-loud,
ritardando final cuadrático sobre los últimos pulsos, articulación por voz, adelanto de la voz principal (la más aguda en
promedio, criterio skyline) y microtiempo gaussiano con semilla, un desvío por ataque para no deshacer acordes. Invariantes comprobados por test: ninguna nota distinta se añade ni se
pierde (dos notas de la misma voz con igual ataque y altura se fusionan, como hace el tokenizador); cada ataque queda a ≤ `max_desvio_s` de su posición en el mapa de tempo; el orden de ataques por voz se
conserva y las notas de un acorde comparten ataque; una altura deja de sonar cuando la misma voz la vuelve a
atacar; el estilo `plano` reproduce la partitura tal cual, también las notas de menos de 40 ms. Todo está
parametrizado por `Estilo`; se incluyen `plano`, `neutro`, `romantico` y `mecanico`. El render por defecto es síntesis
aditiva propia con dinámica y reverberación corta (`render.a_audio_eventos`), sin dependencias externas; el MIDI con
dinámica (`render.a_midi_eventos`) permite cualquier sintetizador aguas abajo.

**PROPUESTA.** Lo que esta capa aporta frente a los antecedentes no es la regla sino el contrato: invariantes
ejecutables por test, alineación nota→evento como interfaz para las métricas de coherencia de RFC-FUGA-3 y la decisión
de que el estilo sea una perilla declarada (como la perilla de forma de FUGA-C) y no un parámetro aprendido opaco. Para
mayor realismo tímbrico sin modelo neuronal, FUGA-I admite FluidSynth (LGPL-2.1) con MuseScore_General (MIT) como
render alternativo [R17] [R18]; la elección de SoundFont queda restringida a licencias confirmadas (Salamander CC BY 3.0
para piano [R19]); GeneralUser GS y FluidR3_GM quedan fuera hasta confirmar su licencia.

**PROPUESTA (I-1b, aplazada).** Un humanizador aprendido sobre ATEPP (CC BY 4.0, ~1000 h) sustituiría las reglas por
predicciones por nota. Se aplaza por dos hechos y una inferencia. Hechos: los modelos aprendidos siguen por debajo del
humano y con correlaciones de tempo bajas [R13]; ATEPP está transcrito de grabaciones, así que un humanizador
aprendería el jitter del transcriptor como «expresividad» (objeción de la crítica cruzada X1). Inferencia nuestra a
partir de Tang et al. [R7]: la capa de interpretación aporta más ganancia perceptiva que un timbre neuronal pequeño.
Entra solo como brazo de ablación en E2.

### 3.3 Capa I-2: refinador PxT enmendado (propuesta)

**PROPUESTA.** Se parte de la propuesta B1 de la expedición con las tres enmiendas que la crítica cruzada declaró
obligatorias (`expedicion/critica_de_A.md`, X1) y con la frontera corregida por [R5].

**PROPUESTA (datos).** Pistas de FMA cuya licencia por pista sea CC BY o CC0 según los metadatos (la fracción es desconocida: E1 la mide
[R25]); filtro de voz por separación de fuentes; mono 44,1 kHz; clips de 10 s. Pseudopartitura `m̂ = AMT(y)` con un
transcriptor cuya licencia permita el uso como herramienta: Basic Pitch (Apache-2.0) o MT3 (código Apache-2.0,
checkpoints sin licencia declarada: hay que confirmarla antes) [R22] [R24]; YourMT3+ (GPL-3.0) solo si se acepta su
licencia para el preprocesado y nunca para la evaluación. Render `r` con FluidSynth y MuseScore_General. Compuerta de
consistencia por similitud de croma y correlación de envolventes de onset entre `r` e `y`; índice de completitud `c` por
clip (fracción de onsets de `y` con nota de `m̂` a ≤50 ms). Codificación de `r` e `y` con DAC congelado (pesos MIT [R20]);
se guardan códigos, no latentes continuos.

**PROPUESTA (modelo).** Puente estocástico en el latente de DAC desde `z_r` hacia `z_y`, en la familia de I²SB con muestreo SDE
(enmienda 1: el puente determinista aprende la media condicional y difumina). Se parte del render, no de ruido [R9].
Pasos de muestreo 20–50 NFE como punto de partida; I²SB informa de 2–10 NFE en inpainting, pero esa cifra no se transfiere
sin medirla. Condiciones: piano-roll de `m̂` a la tasa de frames del codec, referencia tímbrica de 5 s de otra pista del
mismo artista resumida por un pooler de atención, e índice `c` como escalar. Tamaño 60–120M para 24 GB y 20–40M para
8 GB. La ida y vuelta `AMT(render(m))` sale del camino por defecto (enmienda 3): el audio debe corresponder a la
partitura oficial `m`, no a su transcripción.

**PROPUESTA (inferencia).** FUGA-C → I-1 (eventos) → render → ventanas de 10 s con 1 s de solape → puente con `c = 1` → DAC decodifica.
**HECHO.** El decodificador de FUGA-C conoce la fuente exacta de cada segmento ECO (`DIST`, `FROM`, transformación).
**PROPUESTA.** Una repetición exacta reutiliza el latente refinado de su fuente y una repetición transformada se refina
con la fuente como referencia, de modo que dos estribillos iguales suenen iguales (residuo de la crítica X1 sobre la
deriva entre ventanas).

**PROPUESTA (coherencia).** Note-F1 entre la partitura `m` y la transcripción del audio final con un transcriptor **de otra familia**
que el que fabricó los pares (enmienda 2), normalizada por la F1 del transcriptor sobre el render crudo (techo), y
desglosada por familia de instrumento: si el refinador ignora la etiqueta de instrumento del LM, la propuesta cae.

### 3.4 Presupuesto

Las filas marcadas **HECHO** están medidas en esta máquina; el resto de celdas son **PROPUESTA** (estimaciones
derivadas de `scripts/escala.py` o de la propuesta B1).

| Pieza | 24 GB (F0) | 8,6 GB (RTX 4060 disponible) | Base |
|---|---|---|---|
| FUGA-S paso fwd+bwd, 512 tokens, bf16 | — | 0,45 s, pico 1,57 GB (**HECHO**, `ensayo/escala_cuda.json`) | medido 2026-10-25 |
| FUGA-S entrenamiento ctx 4096, lote 8, con checkpointing | 3,6 GB (estimación) | 3,6 GB (estimación; lote menor si hace falta) | `scripts/escala.py` |
| FUGA-M paso fwd+bwd, 512 tokens | — | 0,25 s, pico 3,55 GB (**HECHO**) | ídem |
| Refinador PxT | 60–120M, <16 GB (B1) | 20–40M (**PROPUESTA**, escala lineal de B1) | `expedicion/innovador_B.md` |
| AMT + separación + codec | congelados, <8 GB | congelados, <8 GB | B1 |
| Humanizador aprendido (I-1b) | 10–30M | 10–30M | B1, aplazado |

**HECHO.** La capa I-1 corre en CPU en tiempo despreciable (16 s de música en menos de un segundo, en la demo de
`fuga.interprete` del 2026-10-25).

## 4. Alternativas consideradas

- **Render directo partitura→audio con difusión** (Spectrogram Diffusion, RenderBox): **PROPUESTA.** Derribado por tamaño (412M en el
  primero, no publicado en el segundo) y porque no garantiza las notas (F1 0,36 frente a 0,63) [R2] [R6]. Residuo: sus
  métricas entran en E2 (escucha) y E4 (distancias por embedding).
- **CoSaRef sin pares** (afinar el DiT de Stable Audio Open con audio sin anotar): **PROPUESTA.** Derribado por presupuesto y por evaluación solo
  monofónica [R3]. Residuo: SDEdit desde el render es la línea base barata de E3 si existe un modelo de audio con pesos
  permisivos.
- **Puente determinista desde un punto** (PxT original, B1): **PROPUESTA.** Derribado en la crítica cruzada por aprender la media
  condicional. Residuo: la fábrica de pares, la compuerta de consistencia y el índice `c`.
- **Humanizador aprendido como capa por defecto**: **PROPUESTA.** Aplazado por [R13] y por el jitter de ATEPP; pasa a ablación.
- **LM de audio de ~300M**: **PROPUESTA.** Derribado por datos y cómputo (≥20 000 h, decenas de GPU; mapa § 1).
- **Ida y vuelta `AMT(render(m))`**: **PROPUESTA.** Eliminada del camino por defecto; rompe la correspondencia con la partitura oficial.
- **Reclamar la fábrica de pares como novedad**: **PROPUESTA.** Retirado tras [R5]; la novedad que queda es la combinación descrita en
  § 2.6.

## 5. Riesgos, costes y preguntas abiertas

Experimentos asesinos, por coste ascendente (todos **PROPUESTA** salvo E0, ya ejecutado):

- **E0 (CPU, ejecutado).** **HECHO.** Invariantes de I-1: 27 tests en `tests/test_interprete.py` (conservación de
  notas, tope de desvío con jitter grande y con ataques a 1 tick, orden y cohesión de acordes por voz, sin solapes de
  la misma altura, determinismo, arco de tempo y ritardando, acento métrico, high-loud, legato, adelanto, anacrusa,
  validación de entradas, render WAV y MIDI con ticks, canales y orden off/on comprobados). Pasan el 2026-10-25 y
  matan los mutantes que la primera revisión G4 encontró vivos.
  **HECHO.** E1 se ejecutó el 2026-10-25 (`ensayo/e1_informe.md`, `ensayo/e1_pares.json`): la alineación pasa (mediana
  de |desfase| 15,1 ms) y la compuerta también (54,7 % de 300 clips con τ = 0,6), pero FMA CC BY/CC0 solo proyecta 231 h
  útiles, por debajo de las 300 h exigidas. Por el criterio pre-registrado, la vía de datos de I-2 muere tal como está
  escrita: hace falta otra fuente de audio con licencia permisiva o un refinador de menor escala.
- **E1 (CPU, ≤1 día).** Fábrica de pares sobre 300 clips FMA: (a) fracción real de pistas CC BY/CC0 según metadatos;
  (b) desfase de onsets entre `r` e `y`; (c) similitud de croma; (d) fracción que pasa la compuerta. **Muere** I-2 si la
  mediana de |desfase| supera 23 ms (dos frames de latente) o si pasa la compuerta menos del 30 %, y queda sin datos si
  la fracción CC BY/CC0 útil no permite ≥300 h.
- **E2 (oyentes, sin GPU).** 2AFC con ~20 oyentes × 24 pares sobre salidas reales de FUGA-C `eco_rel`: `plano` frente a
  `neutro` y `romantico`, más un brazo con el humanizador aprendido de I-1b si llega a entrenarse. **Muere** I-1 (vuelve a `plano`) si
  las reglas no son preferidas por encima del azar con IC95; referencia: 70 % de los juicios de 15 oyentes en KTH [R12].
- **E3 (GPU 8 GB, 4–6 h).** Refinador de 20M sobre 30–50 h de pares, 20 000 pasos, 200 clips reservados. Controles:
  render crudo; refinador con pares barajados; SDEdit desde render con modelo preentrenado permisivo si existe. **Muere**
  si el emparejado no gana al barajado, si la note-F1 con transcriptor independiente cae más de 0,10 absoluto respecto al
  render crudo, o si la adherencia por familia de instrumento cae por debajo de la del render crudo.
- **E4 (GPU 24 GB, ~1 día).** Escala a 60–120M con 300–600 h; auditoría de memorización frente a FMA (MiRA, mapa § 4);
  KAD con dos embeddings; prueba fuera de dominio con partituras de PDMX y Lakh.

Riesgos (valoración nuestra, **PROPUESTA**):

- **RI-1 Alucinación de notas en I-2.** El refinador aprende a rellenar lo que el transcriptor pierde. Palancas: `c = 1`,
  piano-roll con descarte y el criterio de muerte de E3. Si no basta, FUGA-I se queda en I-1 con SoundFont.
- **RI-2 Licencias.** Checkpoints de MT3 sin licencia declarada; YourMT3+ GPL-3.0; fracción CC BY/CC0 de FMA desconocida;
  ASAP y MAESTRO son no comerciales y quedan fuera del entrenamiento de cualquier pieza de producción [R14] [R15].
- **RI-3 Dominio.** FMA es electrónica, rock y folk; el corpus de FUGA-C es folk monofónico y polifonía renacentista. La
  cobertura tímbrica de cuerdas y voces puede faltar. Se mide por familia en E3.
- **RI-4 Techo del codec.** La calidad la acota DAC; Mimi no sirve (voz, 12,5 Hz) [R21].
- **RI-5 Novedad estrecha.** Tras [R5], PxT es una combinación, no un mecanismo nuevo. Si E3 muere, el residuo útil es I-1.
- **RI-6 Hardware local.** FUGA-M no cabe en 8,6 GB para entrenar (el paso de 512 tokens de § 3.4 no incluye el
  estado del optimizador ni el contexto de 4096); FUGA-S cabe con checkpointing y lote reducido (estimación, no
  medida).

Preguntas abiertas (**PROPUESTA**): ¿qué fracción de FMA es CC BY o CC0 (E1)? ¿Cuántos NFE necesita un puente en latente de audio para
no difuminar (E3)? ¿Existe un modelo de audio con pesos permisivos para la línea base SDEdit? ¿Se puede confirmar la
licencia de los checkpoints de MT3 y leer el texto completo de [R5]?

## 6. Referencias

Fecha de consulta de todas: 2026-10-25. Las anclas marcadas (v) fueron confirmadas por el verificador ciego; (p),
parcialmente; (no muestreada), la fuente no entró en la muestra del verificador y se apoya solo en el explorador.

- [R1] Wu et al., MIDI-DDSP. https://ar5iv.labs.arxiv.org/html/2112.09312 (arXiv 2112.09312, dic. 2021, inferido). (v)
- [R2] Hawthorne et al., Multi-instrument Music Synthesis with Spectrogram Diffusion, ISMIR 2022.
  https://ar5iv.labs.arxiv.org/html/2206.05408. (v)
- [R3] CoSaRef. https://arxiv.org/html/2410.16785v2 (v2, 11 jun. 2025). (v)
- [R4] Maman y Bermano, Performance Conditioning, ICASSP 2024. https://ar5iv.labs.arxiv.org/html/2309.12283. (v)
- [R5] Maman, Zeitler, Müller, Bermano, Multi-Aspect Conditioning for Diffusion-Based Music Synthesis, IEEE/ACM TASLP
  33:68–81, 2025, DOI 10.1109/TASLP.2024.3507553. Página de proyecto leída:
  https://benadar293.github.io/multi-aspect-conditioning. Texto del artículo no abierto (IEEE Xplore vacío, PDF
  ilegible). (v, consulta de cierre)
- [R6] Zhang, Maezawa, Dixon, RenderBox. https://arxiv.org/html/2502.07711v1 (11 feb. 2025). (no muestreada)
- [R7] Tang et al., EPR Transformer / M2M. https://arxiv.org/html/2501.10222v1 (17 ene. 2025). (no muestreada)
- [R8] Jing, Zhang, Wu, CPR. https://arxiv.org/abs/2609.18216 (v2, 18 sep. 2026; solo resumen). (no muestreada)
- [R9] Liu et al., I²SB. https://ar5iv.labs.arxiv.org/html/2302.05872. (v)
- [R10] Zhou et al., DDBM. https://ar5iv.labs.arxiv.org/html/2309.16948. (no muestreada)
- [R11] Meng et al., SDEdit. https://ar5iv.labs.arxiv.org/html/2108.01073. (p: «should», no «must»)
- [R12] Friberg, tesis KTH (1995; web 1997). https://www.speech.kth.se/music/publications/thesisaf/sammfa2nd.htm.
  (p: el total de 25 reglas lo derivó el extractor)
- [R13] Zhang et al., DExter. https://arxiv.org/html/2406.14850v1 (21 jun. 2024). (v)
- [R14] ASAP dataset, README. https://github.com/fosfrancesco/asap-dataset. (v)
- [R15] MAESTRO dataset. https://magenta.tensorflow.org/datasets/maestro. (v)
- [R16] ATEPP, README. https://github.com/BetsyTang/ATEPP. (v)
- [R17] FluidSynth, README. https://github.com/FluidSynth/fluidsynth. (v)
- [R18] MuseScore, SoundFonts (manual de MuseScore 2). https://musescore.org/en/handbook-sound-and-playback/soundfonts.
  (v)
- [R19] Salamander Grand Piano V3. https://sfzinstruments.github.io/pianos/salamander/. (v)
- [R20] Descript Audio Codec, README. https://github.com/descriptinc/descript-audio-codec. (v)
- [R21] Kyutai Mimi, ficha de modelo. https://huggingface.co/kyutai/mimi. (v)
- [R22] MT3, README. https://github.com/magenta/mt3. (v)
- [R23] YourMT3+. https://arxiv.org/html/2407.04822 (v3, 1 ago. 2024) y https://github.com/mimbres/YourMT3. (v)
- [R24] Basic Pitch, README. https://github.com/spotify/basic-pitch. (v)
- [R25] Defferrard et al., FMA. https://ar5iv.labs.arxiv.org/html/1612.01840. (v)
- [R26] JASCO. https://arxiv.org/html/2406.10970v1 (16 jun. 2024). (no muestreada)
- [R27] MuseControlLite. https://arxiv.org/html/2506.18729v2 (24 jun. 2025). (no muestreada)

Fuentes internas del proyecto: `expedicion/MAPA.md` (§ 1–§ 7), `expedicion/S5.md`, `expedicion/S6.md`,
`expedicion/verificacion_S5_S6.md`, `expedicion/innovador_B.md` (B1), `expedicion/critica_de_A.md` (X1),
`ensayo/informe.md`, `ensayo/escala_cuda.json`.
