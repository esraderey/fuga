## Reporte: verificador-anclas-S5-S6
- Estado: parcial (45 anclas verificadas; JASCO, MuseControlLite y Tang et al. no se abrieron por presupuesto; el texto del artículo TASLP no fue legible, solo su página de proyecto)
- Método: cada URL abierta con WebFetch, pidiendo cita literal y cifras. WebFetch devuelve la lectura de un modelo extractor, no el HTML crudo: las «citas literales» son las que ese extractor declaró literales. En las anclas con varias afirmaciones, el veredicto cubre lo comprobado; lo no comprobado queda en la nota.
- Presupuesto: 31 llamadas de herramienta en total (1 por encima del máximo de 30: un intento fallido de escribir el archivo con Bash consumió una llamada, y se reintentó con Write). No se leyó ningún archivo local salvo la lista de anclas.

### Tabla
| ID | veredicto | nota |
|---|---|---|
| S5-39 | confirmada | Frase «Weights are released as part of this repo under MIT license.» presente; barra lateral MIT. Contenido de LICENSE no mostrado. |
| S5-41 | confirmada | Metadatos «License: cc-by-4.0» presentes; la ficha no declara licencia del código. |
| S5-42 | confirmada | «12.5Hz» y «1.1kbps» presentes; 96.2M params; la ficha no indica número de codebooks. |
| S5-43 | confirmada | «It was trained on speech data, which makes it particularly adapted…» presente; ausencia de mención a música no comprobada explícitamente. |
| S5-31 | confirmada | Apache-2.0 en la página; «not an officially supported Google product» presente; sin licencia aparte para checkpoints. |
| S5-32 | confirmada | Modelo de piano (ISMIR 2021) y multi-instrumento (ICLR 2022) presentes; «we do not (easily) support training» literal. |
| S5-33 | confirmada | GPL-3.0 en barra lateral; frase del README presente; sin licencia de pesos; última noticia «2024.11». |
| S5-36 | confirmada | Solo licencia arXiv del texto; código y datasets «disponibles» en GitHub sin licencia declarada. Cita exacta no cotejada. |
| S5-37 | confirmada | Apache-2.0 y frase «instrument-agnostic and supports polyphonic instruments» literales; sin licencia de modelos ni F1. Extractor señala mención suelta a «GPL License». |
| S5-38 | confirmada | «note accuracy competes with much larger and more resource-hungry AMT systems» presente; sin cifra. |
| S6-19 | confirmada | LGPL-2.1 en barra lateral; frase de licencia y LicensingFAQ presentes; nada sobre licencia de SoundFonts. |
| S6-20 | confirmada | «released under the MIT license», 35,9 MB y 208 MB presentes; «Updated 6 years ago»; el enlace de licencia apunta a FluidR3Mono_License.md. |
| S6-21 | confirmada | sfzinstruments: CC-BY-3.0, Alexander Holm, V3, 20161209, 394 MB/1,18 GB. MuseScore: «Creative Commons Attribution 3.0». |
| S6-23 | confirmada | FluidR3Mono_GM.sf3 (13,8 MB) sin licencia en la página; FluidR3Mono_License.md solo aparece como enlace bajo MuseScore_General. |
| S6-14 | confirmada | CC BY-NC-SA 4.0 literal; 1067/1068, 222/236, >92 h y audio vía MAESTRO v2.0.0 coinciden. Ausencia de nASAP no comprobada. |
| S6-15 | confirmada | CC BY-NC-SA 4.0 (Google LLC) ahora literal; v3 1.276/198,7 h/7,04 M; v2 1.282/201,2 h; «~3 ms» y cita de versión presentes. |
| S6-16 | confirmada | CC BY 4.0, «piano transcription of existing audio recordings» y disclaimer presentes; v1.2 11.674/1.595 (v1.0 11.742/1.580 también figura). |
| S6-24 | confirmada | Frase literal presente; el paper no da desglose de pistas por licencia. |
| S6-25 | confirmada | «licensed under the CC BY 4.0 and MIT licenses» presente (datos y código producidos por los autores). |
| S6-26 | confirmada | Tabla 5 coincide (8.000/25.000/106.574/106.574; full 917 GiB, 343 días); frase sobre Jamendo presente. |
| S5-12 | confirmada | «1000 sampling time steps for all tasks with quadratic discretization» literal. «Estocástico por defecto» no está en esas palabras; se infiere (DDPM; variante OT-ODE). |
| S5-13 | confirmada | Literal en Sec. 3.3. |
| S5-14 | confirmada | «2∼10 NFEs» literal en inpainting Freeform 20-30 % (Sec. 5.2); «matches Pix2pix with only one NFE» en Sec. 5.3. |
| S5-15 | confirmada | Literal en la Introducción. |
| S5-16 | confirmada | Literal en Sec. 5.3, Limitation. |
| S5-20 | confirmada | K=1, N=500, t0=0,5 (VP) en Apéndice D.1 (trazos); t0∈[0,3; 0,6] recomendado; el paper no reporta NFE total. |
| S5-21 | parcial | Mecanismo SDEdit confirmado (añade ruido gaussiano a la guía); lado CoSaRef (N=250, n=150) no comprobado, por lo que no se sostiene V3. |
| S5-22 | parcial | Texto dice «we should choose t0 appropriately», no «must»; el contenido (elección de t0) sí está en Sec. 3. |
| S5-07 | confirmada | Cita presente, truncada antes de «, as shown in Fig. 2(b)»; lista de datasets y papel de MT3 como evaluador no comprobados. |
| S5-08 | confirmada | Cita literal presente; 62.000 pasos y lote 64 son del modelo 2 (multi-dataset); URMP: 8.000 pasos, lote 8. |
| S5-09 | confirmada | Tabla 2 FAD_r 0,526/0,235 y F1 0,323/0,412; MOS 3,267/2,570/2,837/2,534; 22 evaluadores p=0,009. Extractor sugiere etiqueta FAD_r/FAD_t inconsistente en la fuente. |
| S5-10 | confirmada | Cita literal presente; trabajo futuro (polifonía multipista, tiempo real) no comprobado. |
| S5-03 | confirmada | 412M literal; 45M AR pequeño, 85M difusión pequeña sin contexto, 104M con contexto; parámetros del inversor no comprobados. |
| S5-04 | confirmada | Cita literal; seis datasets coinciden; inversor con «internal dataset of 16k hours of uncurated music». |
| S5-05 | confirmada | Cita literal presente. |
| S5-06 | confirmada | F1 0,31/0,36/0,63 y RT 1,05 en Tabla 1; 0,63 figura como «Ground Truth Raw» (límite superior); tabla aplanada en la extracción. Cita literal presente. |
| S5-34 | confirmada | Seis cifras coinciden; la columna MT3 es la ejecución Colab de los autores; Tabla 3 da MT3 Slakh onset 76. MAPS no comprobado. |
| S6-01 | parcial | Las cuatro etiquetas de regla están; la tesis no declara el total de 25 (recuento derivado por el extractor). |
| S6-02 | confirmada | Literal; prueba PHRASE FINAL NOTE (Friberg 1987), cuatro versiones de dos melodías en pares. |
| S6-03 | confirmada | Cita literal; nº de sujetos no declarado en el pasaje (coincide con el ancla). Atribución «atonal/Paper II» no comprobada. |
| S6-04 | confirmada | Cita literal (Sundberg 1989); resultado de entonación (temperamento igual vs mixta) no comprobado. |
| S6-05 | confirmada | 82 participantes, escala de 100 puntos, verdad-terreno 51,81, frase literal con 48,54/48,31/46,33. |
| S6-06 | confirmada | Valores coinciden (tempo 0,19/0,09/0,02; velocidad 0,27/0,21/0,08); columnas de la tabla en orden Basis Mixer, VirtuosoNet, DExter. ± y cita no comprobados. |
| S6-07 | confirmada | Tabla coincide: Vienna4x22 4/88/2 h 18; (n)ASAP 235/1.067/94 h 30; ATEPP-subset 1.580/11.677/1.000 h transcrito. «170k segments» no comprobado. |
| S6-09 | confirmada | Nota literal «We applied the Basis Mixer model with LSTM architecture, trained on the ASAP corpus.» |

No verificadas por presupuesto (sin veredicto): JASCO (S5-26, S5-27), MuseControlLite (S5-28, S5-29), Tang et al. (S6-12, S6-13, S6-27, S6-28), RenderBox (S6-10, S6-11, S6-30), Zenodo Con Espressione (S6-17), GeneralUser GS repo (S6-22), CPR (S6-29), DDBM (S5-17 a S5-19), MIDI-DDSP (S5-01, S5-02, S5-23), ICASSP 2024 (S5-11, S5-24), S5-25, S5-30 (Crossref), S5-35, S5-40, S6-08, S6-18.

### Consulta de cierre (TASLP 2025, Maman et al.)
Pregunta: ¿pseudo-etiquetas de AMT sobre audio real o MIDI alineado preexistente?

Respuesta (de la página de proyecto de los autores; el texto del artículo NO se pudo leer): el artículo contempla ambas variantes. Hay un modelo con MIDI alineado y otro condicionado por predicciones de un transcriptor automático sobre audio real, con resultados comparables según los autores. Citas literales según el extractor de WebFetch, no cotejadas con el HTML crudo:
- «MIDI-aligned scores and automatic multi-instrument transcription based on neural networks» (resumen).
- «For score conditions we use predictions of a transcriber, rather than alignments, and obtain comparable results.»
- «Using predicted transcriptions is significantly more favourable from a practical point of view, since unaligned MIDIs do not always exist.»
- «train on real data alone (~58 hours)» (frente a Hawthorne et al., «mainly synthetic data (~1500 hours)»).
- Transcriptor para pop/rock: «trained mainly on rock albums, with unalinged MIDIs, following Maman and Bermano, 2022». La página no nombra los datasets de entrenamiento de la síntesis.

URL probadas y resultado:
1. https://doi.org/10.1109/TASLP.2024.3507553 -> 302 a https://ieeexplore.ieee.org/document/10771710/ (la redirección no se siguió automáticamente).
2. https://ieeexplore.ieee.org/document/10771990 (ID adivinado por mí, incorrecto) -> contenido vacío.
3. https://ieeexplore.ieee.org/document/10771710 (el ID correcto de la redirección) -> contenido vacío (página JS); sin resumen ni cuerpo.
4. https://www.audiolabs-erlangen.de/fau/professor/mueller/publications -> entrada #21 con autores, título, IEEE/ACM TASLP 33: 68–81, 2025; enlaces a DOI, IEEE 10771710, PDF ePrint y demo https://benadar293.github.io/multi-aspect-conditioning; sin enlace arXiv ni resumen. (La página tiene 409.268 caracteres; solo se leyeron los primeros 100.000.)
5. https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2025_MamanZMB_DiffusionMusic_TASLP_ePrint.pdf -> PDF binario con flujos Flate; solo metadatos legibles (título, DOI, TASLP vol. 33, pp. 68–81, 2025). Ninguna sección de datos. WebFetch guardó el binario en C:\Users\Raul\.claude\projects\D--fuga\7a4c3412-31dd-4934-bbb9-a0da03230d7d\tool-results\webfetch-1792950619362-3wcyay.pdf; no se extrajo por la restricción de no leer otros archivos locales.
6. WebSearch «Maman Zeitler Müller Bermano "Multi-Aspect Conditioning…" arXiv» -> no hay preprint arXiv de este artículo; salen la entrada de cris.iucc.ac.il, el ePrint y el artículo precursor arXiv 2309.12283.
7. https://benadar293.github.io/multi-aspect-conditioning -> legible; origen de las citas de arriba.

Corroboración incidental de S5-30: la entrada de audiolabs (título, autores, TASLP 33, pp. 68–81, 2025, DOI) y los metadatos del PDF coinciden con el ancla; la URL de Crossref no se abrió.

### Totales
Verificadas 45: confirmada 42, parcial 3, cita ausente 0, no soporta la afirmación 0, fuente inexistente 0, no verificable 0. Parciales: S5-21, S5-22, S6-01.
