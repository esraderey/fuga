# Pre-registro de E1 (RFC-FUGA-2 § 5) — fijado el 2026-10-25 antes de descargar los clips y de ver resultados

Experimento: fábrica de pares `r = FluidSynth(AMT(y))` ↔ `y` (audio real) sobre 300 clips de 30 s de FMA con licencia
CC BY o CC0 y sin voz declarada. Script: `scripts/e1_pares.py`. Resultado: `ensayo/e1_pares.json`, informe `ensayo/e1_informe.md`.

## Criterios de muerte (copiados de `rfc/RFC-FUGA-2.md` § 5, E1; no se tocan)

| Id | Criterio | Mata a |
|---|---|---|
| D1 | mediana de \|desfase\| entre `r` e `y` > 23 ms (dos frames de latente DAC a 86 Hz) | I-2 (puente síncrono) |
| D2 | menos del 30 % de los clips pasan la compuerta τ = 0,6 | I-2 |
| D3 | horas útiles proyectadas < 300 h | I-2 queda sin datos |

Veredicto: **SOBREVIVE** si no se dispara ninguno; **MUERE (Dk)** en otro caso. Se informan los tres aunque el primero ya mate.

## Definiciones operativas (fijadas aquí porque el RFC no las concreta)

**Licencia por pista** (`track.license` de `tracks.csv`, texto libre; clasificación por expresión regular sobre el texto
normalizado): CC0/dominio público si contiene «CC0» o «Public Domain»; en la familia CC, `by` = «attribution»,
`nc` = «noncommercial», `nd` = «noderiv», `sa` = «sharealike». Cubos del encargo: **CC BY** = by sin nc, nd ni sa;
**CC0** = CC0 o dominio público; **NC** = cualquier variante con nc; **ND** = by-nd sin nc; **otras** = BY-SA, Free
Art License, FMA-Limited, sin licencia y resto. CC BY-SA se informa aparte y **no** cuenta como CC BY (copyleft).

**Sin voz declarada** (solo metadatos; no garantiza instrumental, es «la ficha no declara voz»). Se excluye la pista si
se cumple cualquiera: (1) `track.language_code` no nulo; (2) `track.lyricist` no nulo; (3) algún género de
`track.genres_all` cuelga de Spoken (20) o Hip-Hop (21), o es Singer-Songwriter (103), Choral Music (443), Opera
(441), Rock Opera (440), Jazz: Vocal (37), Gospel (567), Easy Listening: Vocal (170) o Sound Poetry (186); (4) alguna
etiqueta de `track.tags` contiene, como palabra, vocal(s|ist), voice(s), singing, singer, sung, rap(per|ping), spoken,
lyric(s|al), acapella, «a cappella», choir, chant(ing), «voice-over»; (5) `instrumentalness` de Echonest < 0,5 cuando
existe (13 129 pistas).

**Conjunto elegible E** = (CC BY ∪ CC0) ∩ sin voz declarada. **Marco de muestreo F** = E ∩ `track.duration` ≥ 30 s.
**Muestra**: permutación de F con semilla 20261025 (`numpy.random.default_rng`); se recorre en orden y se toman los
300 primeros clips cuyo MP3 se lee de `fma_large.zip` (lectura remota por rangos HTTP), cuyo SHA-1 coincide con el
fichero `checksums` del propio zip y que decodifican con ≥ 25 s. Los saltados se registran con motivo.

**Transcriptor**: Basic Pitch 0.4.0 (Apache-2.0), modelo ICASSP 2022 en ONNX, parámetros por defecto
(onset 0,5, frame 0,3, nota mínima 127,7 ms, melodia_trick). **Render**: FluidSynth 2.6.1 (LGPL-2.1) con
MuseScore_General.sf3 v0.2.0 (MIT), programa GM 0 (piano), 22 050 Hz, reverberación y coro desactivados, ganancia 0,8;
MIDI a 960 ticks por negra y 120 bpm, velocidad = round(127 · amplitud de Basic Pitch).

**Señales**: `y` e `r` en mono a 22 050 Hz; `r` recortado o rellenado con ceros a la longitud de `y`.

**Desfase** (a): envolventes de onset de librosa (`onset_strength`, salto 128 muestras = 5,8 ms), suavizadas con
gaussiana σ = 20 ms y tipificadas; correlación normalizada ρ(τ) para |τ| ≤ 250 ms; τ* = argmax con interpolación
parabólica; desfase_ms = τ* (positivo = `r` va tarde respecto a `y`). D1 usa la mediana de |desfase| sobre los clips
con alguna nota transcrita. También se informa la mediana con signo y la fracción con |desfase| ≤ 23 ms.

**Croma** (b): `chroma_cqt` (salto 512) de `y` y `r`; vector medio por ventana de 1 s (43 frames completos); similitud
coseno por ventana; **C** = mediana de las ventanas del clip. Sin compensar el desfase.

**Correlación de onsets**: **R** = ρ(τ*) del apartado (a) (0 si `r` es silencio).

**Compuerta** (d): puntuación del clip **S = (C + R) / 2**; pasa si S ≥ τ = 0,6. Un clip sin notas transcritas no pasa.
Secundarias (informativas, no deciden): fracción con C ≥ 0,6; con R ≥ 0,6; con mín(C, R) ≥ 0,6.

**Completitud** c: onsets de `y` con `onset_detect` sobre la envolvente sin suavizar; c = fracción de esos onsets con
una nota de `m̂` a ≤ 50 ms. Se informa su distribución; no decide.

**Horas útiles proyectadas** (D3): primaria = Σ `track.duration` de E (pistas completas de `fma_full`) × fracción que
pasa la compuerta. Secundarias: la misma cuenta con clips de 30 s (|E| × 30 s) y la primaria incluyendo CC BY-SA.

## Lo que este experimento no mide

Ni la calidad del refinador ni la escucha: solo si los pares fabricados están alineados y cuántos datos quedan. El
filtro de voz es por ficha, no por separación de fuentes (la pipeline del RFC § 3.3 propone separación).
