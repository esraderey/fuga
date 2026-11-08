# Mapa del Territorio — Expedición FUGA (2026-10-24)

Pregunta: cómo construir un modelo generativo de música tipo LM de 150–400M parámetros, innovador e implementable en código.

Restricciones fijadas por el usuario (F0):
- El LM tiene 150–400M parámetros.
- Se entrena en **una GPU de 24 GB** (RTX 3090/4090 o similar).
- Debe producir **ambas salidas**: representación simbólica (MIDI/partitura) Y audio escuchable.
- Implementable en PyTorch; datos legalmente utilizables.

Etiquetas: [V2] fuente primaria abierta y leída; [V3] contrastada en ≥2 fuentes independientes. "Cota del barrido" = no localizado tras consultas dirigidas, no inexistencia.

## 1. Audio por tokens de codec / latentes
- MusicGen 300M/1.5B/3.3B: 20K h de música licenciada; 1M pasos, batch 192, 32 GPUs para el de 300M; EnCodec 32 kHz, 50 Hz, 4 codebooks×2048 con delay pattern; flattening mejora un poco la FAD (0.86 vs 0.96) a 4× coste [V2, arXiv:2306.05284].
- MAGNeT-small 300M (no autorregresivo, enmascarado por spans): FAD 3.3 vs 3.4 de MusicGen-large en MusicCaps; latencia ≈3.3–4.4× menor según su tabla [V2, arXiv:2401.04577].
- Magenta RealTime: 220M (Base) / ~770M, ~190K h de música de stock, TPU v6e 128–256 chips; streaming por chunks de 2 s con 10 s de contexto; declara que NO crea estructura de canción de largo plazo [V2, arXiv:2508.04651]. MRT2 (jun-2026): versión pequeña de 230M en tiempo real en MacBook [V2].
- Stable Audio Open Small: DiT 0.34B, 8 pasos, ~12 s de audio en ~75 ms en H100, entrenado en efectos/loops [V2, arXiv:2505.08175].
- Modelos de canción con voz (YuE, LeVo, SongBloom, ACE-Step): ≥1.3B parámetros y 100K–650K h de audio [V2].
- La estructura de largo plazo no se resuelve nativamente en el LM de audio (tres equipos lo declaran) [V3].
- Codecs: WavTokenizer, 1 codebook, 75 tokens/s (0.9 kbps), MUSHRA en música 92.9 ≈ DAC a 9 kbps; adoptado como tokenizador por InspireMusic [V3]. DAC: 44.1 kHz, 86 Hz, 9 codebooks [V3]. Codecs semánticos ponen la información semántica en los primeros codebooks [V3].

## 2. Música simbólica con LMs
- Anticipatory Music Transformer: 128M/360M/780M sobre Lakh (~178K MIDI); tripletas (llegada, duración, nota+instrumento), vocab 27,512, ctx 1024; intercala controles cerca de los eventos que afectan; en acompañamiento empata con el humano (18-31-11) [V2, arXiv:2306.08620].
- MuPT: 190M–4.23B sobre 33.6B tokens ABC; propone ley de escalado simbólica; sobreajuste tras ~3 épocas a 1.07B; alinea compases de todas las pistas (SMT-ABC) [V2, arXiv:2404.06393].
- NotaGen 516M: dos decodificadores jerárquicos (parche por compás + carácter), ctx 1024; RL sin anotación humana (CLaMP-DPO) sube CLaMP 2 Score 0.570→0.730 [V2, arXiv:2502.18008].
- Moonbeam ~309M/839M, ~18B tokens: token compuesto (onset, duración, octava, clase de altura, instrumento, velocidad) con atención relativa multidimensional sin parámetros extra; quitar sus embeddings musicales sube la perplejidad 2.42→4.22 [V2, arXiv:2505.15559].
- Museformer (16M): atención fina a compases previos 1,2,4,8,12,16,24,32 + resúmenes gruesos del resto; modela secuencias >3× más largas [V2, arXiv:2210.10349].
- Theme Transformer: atención paralela con compuerta para que un tema dado reaparezca [V2, arXiv:2111.04093].
- text2midi: FLAN-T5 congelado + decodificador 272M, REMI+, 2,048 tokens, MidiCaps (168K MIDI con captions); acierta la tonalidad pedida solo 7.69% de las veces [V2, arXiv:2412.16526].
- MIDI-LLM: Llama 3.2 1B con tokens MIDI añadidos [V2].
- BPE sobre tokens MIDI: −64 a −69% de tokens por tiempo y ≥2× de velocidad de inferencia; los tokens compuestos (CP Word, Octuple) rinden peor en escucha humana porque muestrean atributos en paralelo [V2, arXiv:2301.11975].
- **Cota del barrido:** no se localizó un mecanismo explícito de COPIA o REFERENCIA a compases previos (tokens/operadores que repitan un compás anterior con transformación) en LMs simbólicos; lo hallado es atención selectiva a compases (Museformer) y condicionamiento por tema [3 consultas dirigidas; S2-A48].
- Datos simbólicos: Lakh ~176–178K archivos, CC-BY 4.0 [V2]; GigaMIDI 1.4M, solo investigación [V2]; Aria-MIDI >1M (~100K h), CC-BY-NC-SA [V2]; PDMX >250K partituras MusicXML de dominio público (conflicto interno CC-0/CC-BY) y su subconjunto calificado y deduplicado rinde mejor con 15% de los datos [V2, arXiv:2409.10831].
- Render simbólico→audio: CoSaRef (síntesis concatenativa con muestras + refinamiento con Stable Audio Open) no necesita pares MIDI–audio y supera a MIDI-DDSP (FAD 0.235 vs 0.526); falló en una configuración [V2, arXiv:2410.16785].

## 3. Técnicas de LM pequeño
- Profundo y estrecho gana a ancho a 125M/350M (+0.9/+1.1 pts); atar embeddings cuesta −0.2 pts; GQA con 4 kv-heads casi gratis; compartir capas adyacentes +0.7/+0.8 [V2, MobileLLM arXiv:2402.14905].
- Muon vs AdamW bien ajustado: ganancia moderada, hasta 1.4× a 130M, ~1.1× a 1.2B [V3]. µP transfiere LR, no regularización [V2].
- Predicción multi-token PERJUDICA a 0.3–0.6B; ayuda desde ~7B [V2, arXiv:2404.19737].
- Looped desde cero a pequeña escala: evidencia solo a 12M, con limitaciones [V2].
- MoE pequeño: ≥2% sobre denso a igual FLOPs, pero más memoria y menos tokens/s [V2].
- Varios streams por paso: depth transformer (RQ-Transformer, Moshi), enmascarado paralelo (MAGNeT), decodificación especulativa exacta 2–3× [V2].
- Repetir datos: hasta ~4 épocas casi gratis; ~40 inútil; dropout eficaz con repetición [V3/V2].
- Gemma 3: 5 capas locales (ventana 1024) por 1 global; QK-norm [V2].
- SmolLM2: 135M con 2T tokens, 360M con 4T; WSD con 20% de decaimiento [V2].

## 4. Evaluación, memorización, datos, legal
- FAD: sesgo por tamaño de muestra y dependencia del embedding [V3]; KAD (MMD) se alinea mejor con juicio humano [V2]; CLAP no es específico de música [V2]; MusicEval: 2,748 clips con 13,740 notas de 14 expertos [V2].
- Métricas simbólicas (entropía de clases de altura, repetición) no estandarizadas; correlación con juicio humano: cota del barrido.
- Memorización: MiRA (réplica sobre audio) [V2]; auditoría de SAO en espacio CLAP [V2]; ataque fonético extrae canciones de Suno/YuE con 91% de similitud [V2].
- Legal: UMG–Udio acuerdo (oct-2025), WMG–Suno acuerdo (nov-2025); tribunal de Múnich en GEMA v. Suno (31-jul-2026, no firme): almacenar canciones protegidas en el modelo infringe el derecho de reproducción [V2]. EU AI Act: obligaciones de modelos de propósito general desde ago-2025, sanciones desde ago-2026 [V2, fuente secundaria].
- Condicionamiento: a 300M, T5≈FLAN-T5 > CLAP como encoder de texto; croma como condicionamiento de melodía sube la similitud 0.10→0.66 [V2, MusicGen].
- Audio abierto: FMA 106K pistas CC; MTG-Jamendo 55K pistas solo no comercial; Song Describer para evaluación [V2].

## 5. Frontera abierta (casillas que el barrido deja libres o débiles)
- F1 Estructura/repetición: los LMs de audio no la resuelven [V3]; los simbólicos la abordan con atención o jerarquía, sin operador explícito de copia/variación (cota del barrido).
- F2 Presupuesto: los simbólicos del rango 150–400M usan contextos de 1024–2048 tokens; los de audio de ~300M necesitan ≥20K h y decenas de GPUs (inferencia del director a partir de las cifras: incompatible con 1×24 GB).
- F3 Simbólico→audio sin pares alineados: CoSaRef existe pero refina con un modelo de ~1B.
- F4 Adherencia al control débil (text2midi: tonalidad correcta 7.69%).
- F5 Evaluación simbólica sin validar; riesgo legal de memorización (fallo GEMA).
- F6 Datos legales para un desarrollador pequeño: Lakh (CC-BY), PDMX (dominio público); Aria/GigaMIDI restringidos.

## 6. Resultado de la verificación ciega (F3) y consultas asesinas adicionales
- Muestra de 39 anclas (20% estratificado + todas las de carga): 33 confirmadas, 5 parciales (atribución/alcance), 0 falsas, 0 no verificables. Correcciones: la estructura de largo plazo no nativa la declaran directamente 2 equipos (Magenta RT, ACE-Step), sigue en V3; el WER de YuE baja por escalado, no por CoT; 168,407 es Lakh, no MidiCaps.
- Copia explícita (8 consultas en total, dos agentes): no localizada. Adyacentes leídos por el director: LZMidi (arXiv:2503.17654, Stanford) — generación NO neuronal por árbol LZ78 sobre notas monofónicas, sin transposición ni referencias a compases; R-Transformer (arXiv:2209.11426) — generador de variaciones de motivo (encoder + aprendiz de repetición) con tipos estricta/transpositiva aplicados al motivo de entrada, no tokens de copia dentro del vocabulario de un LM.
- Ejes musicales en la posición: existe trabajo — Moonbeam (atención relativa multidimensional), Rotary Informational Embeddings (RoPE sobre altura, tiempo absoluto y posición intra-compás; PMLR v318, 2026), "Of All Stripes" (codificación posicional informada por estructura, 2025). Un RoPE por índice de compás NO es novedad por sí solo.


## 7. Ampliación del 2026-10-25 — intérprete simbólico→audio (sectores S5 y S6)

Barrido de dos sectores nuevos, ortogonales a §1–§4: S5 (render neuronal de partitura a audio y puentes sintético→real) y S6 (interpretación expresiva y render determinista). Reportes: `S5.md` (43 anclas) y `S6.md` (30 anclas). Verificación ciega sobre 45 anclas: 42 confirmadas, 3 parciales, 0 falsas (`verificacion_S5_S6.md`). Las citas proceden del extractor de WebFetch, no de un cotejo carácter a carácter.

### 7.1 Render neuronal de partitura a audio
- MIDI-DDSP: pares nota-audio alineados de URMP (13 instrumentos, ~3 h), solo monofónico; transcriptores para etiquetar quedan como trabajo futuro [V2, arXiv:2112.09312].
- Spectrogram Diffusion (ISMIR 2022): pares audio-MIDI de los conjuntos de MT3, 34 clases más batería, modelo base 412M; F1 de transcripción de su salida 0,36 frente a 0,63 del audio real; artefactos declarados [V2, arXiv:2206.05408].
- CoSaRef (2024): sin pares; afina el DiT de Stable Audio Open con audio sin anotar; en URMP mejora FAD y F1 frente a MIDI-DDSP pero el MOS de realismo favorece a MIDI-DDSP (3,27 frente a 2,57–2,84); solo monofónico [V2, arXiv:2410.16785v2].
- Performance Conditioning (ICASSP 2024): ~58 h de grabaciones clásicas reales con MIDI alineado preexistente [V2, arXiv:2309.12283].
- Multi-Aspect Conditioning (IEEE/ACM TASLP 2025): variante condicionada por predicciones de un transcriptor en vez de alineaciones, resultados comparables, solo datos reales (~58 h); leído en la página de proyecto de los autores, texto del artículo no abierto [V2, https://benadar293.github.io/multi-aspect-conditioning].
- RenderBox (2025): partitura MIDI + texto → audio directo; FAD 340,8 frente a 1017,5 de MusicGen-Melody; 23 oyentes; tamaño no publicado [V2, arXiv:2502.07711].
- Tang et al. (2025): transformer de interpretación (partitura → MIDI expresivo) + sintetizador neuronal; corr. velocidad 0,831; en escucha, M2M+Pianoteq 6,69 > partitura+Pianoteq 5,07 > M2M+sintetizador neuronal 3,87 (13 oyentes) [V2, arXiv:2501.10222].
- CPR (sep-2026): MIDI → audio en etapas Composer/Performer/Refiner; solo resumen leído [V2, arXiv:2609.18216].

### 7.2 Puentes sintético→real
- I²SB: parte de la señal degradada, estocástico por defecto, 1000 pasos por defecto, 2–10 NFE en inpainting, requiere pares [V2, arXiv:2302.05872].
- DDBM: muestreador híbrido ODE + Euler-Maruyama; N = 40 en 64×64 [V2, arXiv:2309.16948].
- SDEdit: ruido hasta t0 ∈ [0,3, 0,6] desde la guía; CoSaRef lo aplica a audio (N = 250, n = 150) [V2, arXiv:2108.01073; cita «should», no «must»].

### 7.3 Interpretación expresiva y datos
- Sistema KTH (Friberg, Bresin, Sundberg): phrase arch, duration contrast, melodic charge, final ritard; 70 % de 15 oyentes prefiere la versión con regla «phrase final note» [V2, tesis KTH 1995/1997; el total de 25 reglas es recuento del extractor].
- DExter (2024), 82 oyentes: humano 51,81, DExter 48,54, VirtuosoNet 48,31, Basis Mixer 46,33; corr. tempo 0,19 / 0,09 / 0,02; inferencia 40 s por 95 s de música en RTX 6000 [V2, arXiv:2406.14850].
- Datos alineados: ASAP (>92 h) y MAESTRO (198,7 h) CC BY-NC-SA 4.0; ATEPP (~1000 h) CC BY 4.0 con MIDI transcrito de grabaciones [V2, repos y página oficial].

### 7.4 Herramientas y licencias
- FluidSynth LGPL-2.1; MuseScore_General MIT; Salamander Grand Piano CC-BY-3.0; GeneralUser GS y FluidR3_GM sin licencia confirmada (cota del barrido) [V2].
- DAC: pesos MIT, 44,1 kHz a 8 kbps. Mimi: pesos CC-BY-4.0, 12,5 Hz, entrenado con voz [V2].
- AMT: MT3 código Apache-2.0 y checkpoints sin licencia declarada; YourMT3+ código GPL-3.0, 44,7M, Slakh onset F1 84,14 frente a 75,20 de MT3; Basic Pitch Apache-2.0 sin F1 en el README [V2].
- FMA: 106 574 pistas; el artículo no desglosa licencias por pista [V2, arXiv:1612.01840].

### 7.5 Frontera revisada
- **F3 (simbólico→audio sin pares alineados) se corrige:** la fábrica de pares por transcripción automática ya existe a pequeña escala (Multi-Aspect Conditioning, ~58 h clásica). Cota del barrido: no localizado su uso sobre audio «in the wild» con filtro de licencia, como puente desde un render determinista, ni con refinador ≤120M (4 consultas + consulta de cierre).
- **F7 (nueva) Cadena LM simbólico → intérprete expresivo → refinador de timbre:** no localizado en el barrido tras 3 consultas; los vecinos (Tang et al., RenderBox, CPR) cubren tramos.
- **Evidencia a favor de interpretar por reglas antes que aprender:** los modelos aprendidos siguen por debajo del humano y con correlaciones de tempo bajas (DExter); la capa de interpretación aporta más que el timbre neuronal pequeño (Tang et al.).

### 7.6 No cubierto
EnCodec, WavTokenizer, SNAC y X-Codec (pesos y tasas); Onsets and Frames; ScorePerformer y VirtuosoNet de primera mano; licencias de (n)ASAP, Vienna 4x22, GeneralUser GS y FluidR3_GM; desglose de licencias de FMA y MTG-Jamendo; texto completo de Multi-Aspect Conditioning (TASLP 2025). Detalle en «No cubierto» de `S5.md` y `S6.md`.

Salida de esta ampliación: `rfc/RFC-FUGA-2.md`.
