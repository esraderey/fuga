## Reporte: verificador-anclas

- Estado: cumplido (39/39 anclas con veredicto; B1 y B2 ejecutadas). Presupuesto: 55 llamadas incluyendo Write y este handback.
- Método: cada fuente abierta con WebFetch (PDF o abs de arXiv; HTML/URL cuando la ancla lo indicaba). WebFetch resume con un extractor, no es grep byte a byte: las cifras de tabla y las citas se reportan tal como las devolvió el extractor.

### Tabla

| ID | veredicto | nota |
|---|---|---|
| S1-01 | confirmada | Cita literal presente; 300M/1.5B/3.3B, 1M pasos, batch 192, 32/64/96 GPUs presentes. Sello arXiv:2306.05284v3, 30 Jan 2024, NeurIPS 2023. |
| S1-25 | confirmada | Tabla 2: SAO 96.51/0.55/0.41; SA 2.0 (190 s) 71.25/0.37/0.42; MusicGen-large-stereo 190.47/0.52/0.31. Cita en pie de Tabla 2 y Sec. 5.1. v2, 31 Jul 2024. |
| S1-26 | confirmada | "up to 47s" presente (Sec. 2). "cannot produce intelligible speech or singing" (Sec. 5.3; también 6.1). |
| S1-32 | parcial | Cita y "extend no integrado nativamente en el entrenamiento" (Sec. 7) y "gacha-like" (lotería) confirmados. Sobre DCAE: el texto dice "does not surpass the reconstruction quality upper bound of 1D VAE" y está en Sec. 5.2.1 (resultados), no en limitaciones. Significado distinto de "no alcanza la cota". Sec. 7 solo señala que depender de un DCAE sobre mel puede limitar fidelidad. |
| S1-34 | confirmada | Abstract: "<2 s por canción en A100, <10 s en RTX 3090", "<4GB VRAM". PDF Sec. 4.1: "200x speedup ... 240-second tracks in ~1 second on an NVIDIA A100", mismo párrafo baja de 50 a 8 pasos. Enviado 2026-01-31; v3 2026-02-06 (la fecha de la ancla es la de v3). |
| S1-41 | confirmada | Fase 3: "increase the maximum positional embedding and sequence length to 16384". Abstract: hasta cinco minutos. Fecha por ID, el PDF no trae sello. |
| S1-42 | parcial | Cita 3K/6K tokens presente (discusión del decaimiento de RoPE). WER ~70% (0.5B) a ~20% (7B): el texto lo atribuye a escalar el modelo, no al CoT; además está confundido con presupuesto de entrenamiento (0.5B: 500B+200B tokens; 7B: 1.75T, Sec. 8.4). Existe ablación del CoT (Sec. 8.3, Fig. 12, solo en 0.5B vs Vanilla/Curriculum/ABF) sin cifras de WER en texto. Atribuir la caída al CoT no está soportado. |
| S1-52 | confirmada | WavTokenizer 1 codebook 0.9 kbps, 75 Hz; Qwen2.5-0.5B/1.5B (totales 0.8B/1.8B en Tabla 1); flow matching a HiFi-Codec 150 Hz; vocoder. |
| S1-53 | parcial | 29B tokens (~100,000 h), clips de 30 s y "up to 8 minutes" confirmados. "Long afinada con audio completo": Sec. 5.1 menciona una etapa general "fine-tuning with full-length audio waveforms" pero no la ata a la variante Long; Sec. 3.4 solo dice que el modelo previo se entrenó con segmentos de 30 s. Es inferencia. |
| S1-59 | confirmada | MUSHRA MusicDB: 92.9±2.2; DAC 9 kbps 92.6±2.4; DAC 1 kbps 57.6±2.1; EnCodec 6 kbps 76.9±1.6. Cita presente (Sec. 4.2). Lo de "0.3 dentro del error" es derivado correcto. |
| S1-60 | confirmada | "current acoustic codec models lack the understanding capabilities (ASR)" en Apéndice F; validar en modelos grandes es trabajo futuro (Sec. 4.2). |
| S1-62 | confirmada | Sec. 4.3 (Limitation): "additional computational overhead". Tabla 6: Mel 0.54 (baseline n_q=8) vs 0.62 (X-Codec n_q=8). Matiz: son 1000 muestras de habla LibriTTS, no música; el paper lo presenta como compromiso compensado por UTMOS (3.72 a 4.01). |
| S1-68 | confirmada | YuE: "codebook-0 alone captures rich semantic information such as melody and vocal content" (Sec. 3.4). OmniCodec: la frase NO está en el abstract, sí en Sec. 4 Conclusions; cabecera lista ASLP@NPU / Northwestern Polytechnical University; abs enviado 2026-03-21. No verificada en esta pasada la afiliación HKUST/M-A-P de YuE. |
| S1-71 | parcial | Magenta RT: frase en Apéndice B (Limitations), sello v3 5 Nov 2025: confirmada. ACE-Step: Sec. 7, v1 28 May 2025: confirmada. InspireMusic: la cita existe (Sec. 5.5, "generated tokens may include more errors"), pero habla de errores en la variante Long, y el propio paper presenta Long como mejora de coherencia estructural. Solo 2 de 3 equipos dicen directamente que la estructura no se resuelve nativamente. |
| S1-72 | confirmada | DAC: cita en abstract; Tabla 1 "44.1 / 8 kbps / stride 512 / 86 Hz / 9 codebooks"; v2, 26 Oct 2023. SSM-LM: "restrict our modeling to the first-layer codebook of DAC", usa DAC 44.1 kHz (~86 Hz); v1, 9 Jul 2025. |
| S2-01 | confirmada | Frase literal presente. TMLR (04/2024); sello v2, 25 Jul 2024. |
| S2-07 | confirmada | Frase en la Conclusión. Venue ICLR 2025 no visible en el PDF (no verificado). |
| S2-08 | confirmada | Sec. 4.1: "33.6 billion tokens derived from a diverse collection of monophonic and polyphonic...". |
| S2-11 | confirmada | Abstract con "Symbolic Music Scaling Law (SMS Law)"; Tabla 3 fila SMS Law R² 0.9780 (train) y 0.9612 (test). |
| S2-13 | confirmada | "totaling 516M parameters" (Sec. 4.1); abstract con 1.6M piezas ABC y ~9K composiciones clásicas. PDF sin sello de fecha. |
| S2-21 | confirmada | MRA en espacio 5-D (onset, duration, octave, pitch class, velocity); "without additional trainable parameters" en abstract y Sec. 3.2.1. v1, 21 May 2025. |
| S2-22 | confirmada | Tabla 1 (Moonbeam S): 2.423 / 4.216 sin FME / 3.245 sin GRU. Tabla 4 "Overall": 3.600±1.047 vs 2.885±1.131 (baseline con tokenizador REMI-like); frase de Sec. 5.3 presente. Significancia por Wilcoxon. |
| S2-24 | confirmada | Sec. 3.3: "we carefully select 8 bars – the previous 1st, 2nd, 4th, 8th, 12th, 16th, 24th, and 32nd bars" como por defecto; grano grueso a resúmenes de los demás compases (abstract). NeurIPS 2022 no visible en el PDF; solo corroborado por un resultado de búsqueda (papers.neurips.cc) no abierto. |
| S2-32 | parcial | "MidiCaps is a dataset of 168,401 unique MIDI files" presente (sec. de datasets); 168,385 aparece en Related Work como MidiCaps. 168,407 aparece en la Introducción referido a Lakh MIDI, NO a MidiCaps. Pseudo-captions: atributos time signature, key signature, BPM, instrumentos con Music21; 10 frases plantilla. El extractor no confirmó la mención explícita de SymphonyNet. Sin fecha en el HTML. |
| S2-48 | confirmada (negativa; no refutada) | Mis 5 consultas B1 no localizaron mecanismo de copia explícito. Abiertos de los citados: Museformer (atención selectiva a compases previos, ver A24), text2midi, 2407.19900 (papers.cool: "two structural embeddings that do not rely on domain-specific annotations"; la página no describe los embeddings, como dice la ancla). No abiertos: Music Transformer 1809.04281 y XMusic 2501.08809. Un negativo no se puede confirmar del todo: ver B1 y "No cubierto". |
| S3-01 | confirmada | Tablas 10 y 11: 125M 44.8 (30L/d512) vs 43.9 (12L/d768); 350M 49.8 (32L/d896) vs 48.7 (15L/d1280). Frase en Sec. 2.2.2. v2, 27 Jun 2024. |
| S3-04 | confirmada | Tabla 3: 46.3 a 47.0 y 51.3 a 52.1; Tabla 7: 15.6 a 16.0 ms (iPhone 13); frases en Sec. 3.2 y 3.6. Ablación 125M: +0.4 (Tablas 2 y 14) vs +1.1 (Tabla 10, 45.0 a 46.1); el paper atribuye 0.25T tokens solo al +1.1. |
| S3-07 | confirmada | 2509.02046 v2, 4 Sep 2025: frase literal en pie de Fig. 4. Matiz: el 1.4x es "highest speedup ratio" y, en el abstract, de optimizadores matriciales en general (Muon incluido), no exclusivo de Muon; ≈1.3x bajo 520M y ≈1.1x a 1.2B confirmados. Tamaños reales 130M-1.2B. 2505.02222 v4, 20 May 2025: frase literal en Conclusión; cinco tamaños 100M, 500M, 1B, 2B, 4B. |
| S3-11 | confirmada | Tabla S7 (n=1 vs n=4): 0.3B MBPP 1.8/1.0, HumanEval 1.9/1.2; 0.6B MBPP 4.7/3.0 (HumanEval 2.9/2.7); 1.3B MBPP 6.8/7.4 (HumanEval 4.6/4.8); 13B MBPP 26.0/30.5, HumanEval 14.1/15.8. Pie de Fig. 3 literal. Aviso: en la ancla el primer número es el baseline, aunque la oración nombra primero a MTP. |
| S3-29 | confirmada | "2X-3X acceleration ... with identical outputs"; T5-XXL 11B con 77M/250M/800M; Corolario 3.6 y Teorema 3.8 coinciden. v2, 18 May 2023. |
| S3-32 | confirmada | "At 40 epochs, repeating is worthless" solo en el callout de la Fig. 1 de 2305.16264; el cuerpo dice ganancias significativas hasta ~16 épocas; rango 10M-9B. v5, 28 Jun 2025. 2305.13230 v2, 5 Oct 2023: T5 78M-2.8B, 2^27 tokens x 2^8 épocas, dataset mayor alivia, "larger models are more susceptible to overfitting" literales. |
| S3-33 | confirmada | Cita literal en la Introducción; requiere ajuste a escala XL. "El más eficaz de los probados" no se dice así: el abstract dice que la mayoría de regularizadores no mejoran significativamente salvo dropout. |
| S3-39 | confirmada | Tabla 3 con 0-6T, 6-8T, 8-10T, 10-11T; Sec. 4.5: Stack-Edu 24%, matemáticas 14%, web 58%; ~2 épocas; frase de 4-5 épocas literal (principio 4). v1, 4 Feb 2025. Leído completo. |
| S4-01 | confirmada | Gui et al. (Gui, Gamper, Braun, Emmanouilidou): frase en abstract; enviado 2023-11-02, v2 2024-03-05. KAD (Chung, Eu, Choi, Chon; Lee, Nam; Gaudio Lab/KAIST): "provides unbiased results for all sample sizes" en Conclusión y "FAD is an inherently biased metric" en Sec. 2.1. Autores distintos, sin solapamiento visible. La cita de KAD no está en un abstract; fecha y venue (ICML 2025) no aparecen en el PDF (Apéndice C trae fecha de acceso 2025-02-03). |
| S4-20 | confirmada | Munich Regional Court I, 31 July 2026; "storing protected songs in the model infringes the reproduction right"; "not yet final"; demanda 21 January 2025. La página trata la memorización solo como argumento de la demandante. |
| S4-22 | confirmada | FAD T5 3.12, Flan-T5 3.36, CLAP 4.16; ambas citas literales (Tabla A.1). Fecha por ID. |
| S4-23 | confirmada | Tabla 2: 0.66 y 0.10 presentes; citas literales. El extractor ubica el 0.66 en Text+Chroma y el 0.10 en evaluación con solo texto. Cifras vía extractor, no cotejadas celda a celda. |
| S4-26 | confirmada | Abstract con la cita, 161 géneros, 16,341 artistas, 14,854 álbumes, "full-length"; enviado 2016-12-06, v3 2017-09-05. |
| S4-29 | confirmada | Ambas citas del abstract, metadatos de tags e interacción; el abstract no reporta licencia por partitura; 2024-09-17, v2 2025-03-17. |

### Resumen por reporte

| Reporte | Anclas | Confirmadas | Parciales | Falsas | No verificables |
|---|---|---|---|---|---|
| S1 | 15 | 11 | 4 (S1-32, S1-42, S1-53, S1-71) | 0 | 0 |
| S2 | 10 | 8 + 1 negativa no refutada (S2-48) | 1 (S2-32) | 0 | 0 |
| S3 | 8 | 8 | 0 | 0 | 0 |
| S4 | 6 | 6 | 0 | 0 | 0 |
| Total | 39 | 33 + 1 negativa | 5 | 0 | 0 |

Ninguna fuente inexistente ni cita ausente. Los fallos son de atribución o alcance: S1-42 (WER atribuido a CoT cuando el texto lo atribuye a escalado), S1-32 (DCAE reubicado y reinterpretado), S1-53, S1-71, S2-32 (168,407 es Lakh, no MidiCaps).

### Tarea B1: copia/referencia a compases previos en música simbólica

Consultas exactas (5):
1. symbolic music generation copy mechanism repeat previous bars transposed copy operator tokens transformer
2. pointer network symbolic music generation copy earlier measures repetition structure
3. Lempel-Ziv OR grammar compression of musical score as training representation language model symbolic music repetition macro tokens
4. music generation "copy" bar "repeat" token structure-aware symbolic music language model reference to previous section transposition
5. retrieval-augmented OR repetition-aware symbolic music generation explicit repeat of earlier bars motif reuse copy module MIDI language model

Resultado: operador explícito de copia/puntero a compases con transposición, o pointer network sobre compases: no localizado. Las búsquedas devuelven solo títulos y URL, sin fragmentos; solo leí lo que sigue.

Lo más cercano que sí leí (ninguno es copia explícita):
- LZMidi, compresión LZ78 para generación simbólica: modelo no neuronal, árbol de prefijos con conteos; no es representación para entrenar un modelo neuronal ni referencia a compases. Fuente secundaria (reseña, sin ID de arXiv visible): https://www.themoonlight.io/en/review/lzmidi-compression-based-symbolic-music-generation. Cita: "constructing the LZ78 tree based on the input training corpus".
- Byte Pair Encoding for Symbolic Music (Fradet et al., arXiv:2301.11975, enviado 2023-01-27): compresión de la secuencia de tokens por fusión de pares frecuentes, para LMs. Cita: "Byte Pair Encoding, a compression technique widely used for natural language". No referencia compases ni transpone.
- Museformer (arXiv:2210.10349): atención selectiva a compases previos. Cita: "the previous 1st, 2nd, 4th, 8th, 12th, 16th, 24th, and 32nd bars".
- Retrieval Augmented Generation of Symbolic Music with LLMs (arXiv:2311.10384, Jonason et al.): recuperación de ejemplos externos; la página no menciona copia de compases propios. Cita: "using a retrieval system to select relevant examples".
- Segment-Factorized Full-Song Generation (arXiv:2510.05881, Chen, Tan, Yang): atención a segmentos relacionados, mecanismo no descrito en el abstract. Cita: "selective attention to related segments".
- 2407.19900 (Rhyu et al.): embeddings estructurales sin anotaciones, sin descripción de copia.

Apareció en las búsquedas y no abrí: Enhanced Memory Network 2110.03392; The Beauty of Repetition in Machine Composition Scenarios 2209.11426; Motifs, Phrases, and Beyond (survey, QMUL); N-Gram Unsupervised Compoundation 2312.08931; MuseTok 2510.16273; BEAT 2604.19532; SymPAC 2409.03055; The Effect of Explicit Structure Encoding 1811.08380; entrada de blog "Why GPT-Style Models Do Not Directly Transfer to Symbolic Music: Compression in the Wrong Coordinate System".

### Tarea B2: RoPE / atención relativa sobre ejes musicales

Consultas exactas (2):
1. rotary position embedding RoPE bar position beat position symbolic music transformer
2. relative attention musical time metrical position bar-level relative positional encoding music generation transformer

Localizado:
- Rotary Informational Embeddings for Symbolic Music Generation (Schön, Tompits; CAIAC 2026, PMLR 318, pp. 1181-1185): https://proceedings.mlr.press/v318/schon26c.html. Extiende RoPE a información entera: "pitch, absolute time, or intra-bar positions". Resultados preliminares en Lakh MIDI y POP909, menor perplejidad que Transformer, Music Transformer y RoPE base, sobre todo en secuencias largas no vistas. La página no describe ejes de compás o tiempo musical como tales, solo posición intra-compás y tiempo absoluto.
- Of All Stripes: Investigating Structure-informed Positional Encoding For Efficient Music Generation (2025; sin autores ni ID visibles): https://awesomepapers.io/speech-audio/papers/agarwal2025of. Cita: "inserting musically-relevant structural information into the positional encoding (PE) module". Compara con RoPE y propone RoPEPool; el texto visible no dice qué estructura (compás, tiempo).
- De la muestra, ya verificado: Moonbeam (https://arxiv.org/pdf/2505.15559), atención relativa multidimensional sobre onset, duración, octava, pitch class y velocidad, "without additional trainable parameters". El extractor no dijo que use RoPE.
- Apareció y no abrí: Music Transformer (1809.04281, atención relativa), 2402.13301, 2410.18067, 2504.05364.

### No cubierto

- Fechas "(ID)" y venues: las anclas con fecha inferida del ID (S1-41, 42, 52, 53, 59, 60, 68-YuE, S2-07, 08, 11, 13, 32, S4-22, 23) no pude cotejarlas, porque los PDF vía extractor no traen sello. Venues no vistos en la fuente: ICLR 2025 (MuPT), NeurIPS 2022 (Museformer, solo vía lista de búsqueda), ICML 2025 (KAD).
- Cifras de tabla y citas llegan vía el extractor de WebFetch, no por coincidencia de cadenas exacta (S1-25, S1-59, S2-11, S2-22, S3-11, S4-22, S4-23).
- YuE: el extractor leyó los primeros 100,000 de 106,765 caracteres en ambas consultas. S1-42 se dictamina sobre ese tramo; los últimos 6,765 no se leyeron.
- Independencia de grupos en S1-68 y S4-01: afiliaciones confirmadas solo para OmniCodec (ASLP@NPU) y KAD (Gaudio Lab/KAIST). No verifiqué afiliaciones de YuE (HKUST/M-A-P) ni de Gui et al.
- S1-34: el "8 pasos" lo confirma el extractor en el mismo párrafo del "~1 s en A100" (de 50 a 8 pasos); no tengo la frase literal con el 8.
- S2-48 (negativa): no se puede confirmar una ausencia. No abrí XMusic 2501.08809 ni Music Transformer. B1 solo con 5 consultas en WebSearch (solo EE. UU., solo títulos), sin recorrer ISMIR, ACL Anthology ni listados arXiv. Quedan sin abrir los candidatos listados en B1.
- LZMidi: no abrí el paper primario, solo una reseña de terceros; el ID de arXiv no se vio.
- Primera consulta a arxiv.org/abs/2502.15602 devolvió contenido sin título ni autores; los datos del KAD salen del PDF.
- No evalué si las cifras de las anclas son las más relevantes para el diseño, ni la calidad de las fuentes (por contrato).
