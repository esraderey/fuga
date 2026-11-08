# Informe E1 — fábrica de pares por transcripción (RFC-FUGA-2 § 5)

- Fecha: 2026-10-25. Pre-registro: `ensayo/e1_prerregistro.md` (fijado antes de descargar y de ver resultados).
- Script: `scripts/e1_pares.py` (venv aparte `.venv_e1`, Python 3.10). Resultado crudo: `ensayo/e1_pares.json`.
- Datos: `fma_metadata.zip` (SHA-1 `f0df49ff…` cotejado con el README oficial de mdeff/fma) y 300 clips de 30 s leídos
  de `fma_large.zip` por rangos HTTP, SHA-1 cotejado con el `checksums` del propio zip. Todo en `data/fma_e1/` (no
  versionado).
- Herramientas: Basic Pitch 0.4.0 (Apache-2.0, modelo ICASSP 2022 en ONNX, onnxruntime 1.23.2), FluidSynth 2.6.1
  (LGPL-2.1, release oficial win10-x64), MuseScore_General.sf3 v0.2.0 (MIT), librosa 0.11.0, soundfile 0.14.0
  (libsndfile 1.2.2), numpy 2.2.6, scipy 1.15.3, mido 1.3.3.

## 1. Veredicto pre-registrado

| Id | Criterio (RFC-FUGA-2 § 5, E1) | Valor medido | Umbral | Dispara |
|---|---|---|---|---|
| D1 | mediana de \|desfase\| entre `r` e `y` | 15,1 ms (IC95 bootstrap 12,6–18,3) | > 23 ms | no |
| D2 | fracción de clips que pasa la compuerta τ = 0,6 | 54,7 % (164/300; IC95 Wilson 49,0–60,2) | < 30 % | no |
| D3 | horas útiles proyectadas (pistas completas CC BY/CC0 sin voz declarada × fracción que pasa) | 231 h (IC95 207–254) | < 300 h | **sí** |

**Veredicto: MUERE (D3).** La fábrica de pares funciona (los pares quedan alineados y más de la mitad pasa la
compuerta), pero con la fuente que propone el RFC, FMA restringido a CC BY o CC0, no llega a las 300 h útiles. Con las
palabras del RFC: I-2 «queda sin datos». D1 y D2 no se disparan.

## 2. (a) Licencias por pista en FMA

106 574 pistas (`tracks.csv`), 8 225 h de audio completo. Clasificación por texto de `track.license` (reglas en el
pre-registro).

| Cubo | Pistas | Fracción | Horas (fma_full) |
|---|---|---|---|
| CC BY (sin NC, ND ni SA) | 6 960 | 6,5 % | 504 |
| CC0 / dominio público | 1 820 | 1,7 % | 97 |
| NC (BY-NC-SA 43 512, BY-NC-ND 41 869, BY-NC 8 332) | 93 713 | 87,9 % | 7 353 |
| ND (BY-ND) | 962 | 0,9 % | 64 |
| otras (BY-SA 2 802, Free Art License 16, sin licencia 87, resto 214) | 3 119 | 2,9 % | 207 |

Embudo del conjunto elegible: 8 780 pistas CC BY o CC0 → 7 573 sin `language_code` → 7 554 sin letrista → 6 210 sin
géneros vocales (Spoken, Hip-Hop y ocho géneros con voz en el nombre) → 6 145 sin etiquetas de voz → **6 075 pistas
elegibles (E)** tras el filtro de `instrumentalness` de Echonest. E suma **421,8 h** de pistas completas (4 855 CC BY,
360,7 h; 1 220 CC0/PD, 61,2 h) y solo 50,6 h en clips de 30 s. Con CC BY-SA, 7 959 pistas y 553,4 h. Marco de muestreo
(duración ≥ 30 s): 5 654 pistas.

Consecuencia aritmética, visible antes de medir ningún par: para llegar a 300 h tendría que pasar la compuerta el
71,1 % de los clips (54,2 % contando BY-SA). D3 era el criterio más apretado desde el conteo de licencias.

Salvedad: 182 pistas de E (9,7 h; todas «Old-Time / Historic») llevan la etiqueta «Public Domain / Sound Recording
Common Law Protection». El pre-registro las cuenta como dominio público; el fonograma puede conservar protección
estatal pre-1972 en EE. UU. Quitarlas deja 412 h.

## 3. (b), (c), (d) Pares sobre 300 clips

Muestra: permutación con semilla 20261025 del marco F; 300 clips válidos, 7 saltados (5 que libsndfile no decodifica,
2 de menos de 25 s). 245 CC BY y 55 CC0/PD; 207 sin `genre_top`; 317 MB; 602 peticiones HTTP; 154 s. Basic Pitch
transcribió 52 498 notas (mediana 5,4 notas/s; ningún clip sin notas).

| Medida | Mediana | p25 | p75 | Notas |
|---|---|---|---|---|
| \|desfase\| `r`–`y` (ms) | 15,1 | 6,9 | 75,4 | p90 208,8; 63,0 % ≤ 23 ms; 41,7 % ≤ 11,6 ms; 4,3 % (13 clips) en el tope de ±250 ms |
| desfase con signo (ms) | +10,9 | | | media +13,2; positivo = el render va tarde |
| C, similitud de croma por ventana de 1 s | 0,921 | 0,887 | 0,943 | p10 0,847; mínimo 0,0 en un clip con una sola nota transcrita |
| R, correlación de envolventes de onset | 0,318 | 0,178 | 0,489 | p10 0,075; p90 0,697 |
| c, completitud (onsets de `y` con nota a ≤ 50 ms) | 0,61 | 0,35 | 0,83 | informativa |
| S = (C + R) / 2 | 0,613 | 0,535 | 0,712 | compuerta S ≥ 0,6 |

Compuerta: pasan 164/300 = **54,7 %** (IC95 49,0–60,2 %). Secundarias pre-registradas: C ≥ 0,6 en el 99,7 %; R ≥ 0,6
en el 17,3 %; mín(C, R) ≥ 0,6 en el 17,3 %.

Horas proyectadas: primaria **231 h** (421,8 h × 0,547; IC95 207–254). Secundarias: clips de 30 s, 28 h; incluyendo
CC BY-SA, 303 h (IC95 271–333).

## 4. Lectura de los resultados (análisis posterior, no pre-registrado)

- **La alineación es bimodal.** Tres de cada cinco clips quedan a ≤ 23 ms y uno de cada cinco supera 100 ms, donde la
  correlación cruzada ha elegido un pico espurio (música sin ataques claros, o transcripción pobre). Entre los clips
  que pasan la compuerta, la mediana de \|desfase\| baja a 10,5 ms y el 82 % queda a ≤ 23 ms (2 en el tope); entre los
  que no pasan, 53,6 ms y el 40 % (11 en el tope). Spearman entre \|desfase\| y R: −0,51. La compuerta actúa también
  como filtro de alineación, que es lo que el diseño de B1 pretendía.
- **Hay un sesgo sistemático de signo.** El render llega unos 11 ms tarde (mediana +10,9 ms; entre los clips con
  R ≥ 0,3, +9,7 ms). Es compatible con la cuantización de Basic Pitch a frames de 11,6 ms más el ataque de 5–7 ms de
  las muestras de piano medido en FluidSynth; no se ha comprobado cuál de los dos pesa más. Una corrección constante
  es trivial, pero no estaba pre-registrada y no se aplica al veredicto.
- **El croma no discrimina; la compuerta la decide la correlación de onsets.** C ≥ 0,6 en el 99,7 % de los clips
  (un render de piano de la transcripción comparte casi siempre las clases de altura con el original), mientras que R
  es baja (mediana 0,32) porque el render de piano no reproduce la envolvente de una mezcla con percusión. S promedia
  ambas; una compuerta solo de R dejaría el 17 %.
- **Por género (n pequeño).** Pasan Classical 8/10, Instrumental 9/11 y Folk 6/8; Electronic 6/21, Rock 5/15 y
  Experimental 2/14; el bloque sin `genre_top` (207 clips) 58 %. La fábrica funciona donde el material es de notas y
  falla donde es de texturas, lo que coincide con el riesgo RI-3 del RFC.
- **Sensibilidad de D3.** Con τ = 0,55 pasaría el 72 % (304 h); con τ = 0,5, el 83 % (352 h); exigiendo además
  \|desfase\| ≤ 23 ms sobre los que pasan, 190 h. Ninguna de estas variantes es la primaria pre-registrada, y 303 h
  con BY-SA es un valor en el filo (su IC95 cruza las 300 h). El veredicto no cambia con los datos de esta ejecución.
- **Correlaciones.** S crece con la densidad de notas (Spearman 0,62) y con la completitud c (0,78): los clips que
  Basic Pitch transcribe densamente son los que quedan consistentes.

## 5. Limitaciones

1. El filtro de voz es por ficha («la ficha no declara voz»), no por separación de fuentes como propone el RFC § 3.3;
   207 de los 300 clips no tienen `genre_top`, así que el dominio real de la muestra se conoce mal.
2. Un solo transcriptor (Basic Pitch, parámetros por defecto) y un solo timbre de render (piano GM 0). R castiga a la
   música percusiva; otro render (familias de instrumento) o un transcriptor multi-instrumento podrían mover la
   compuerta en cualquier sentido. No se ha medido.
3. La proyección de horas supone que la fracción que pasa en clips de 30 s se traslada a pistas completas y que la
   licencia por pista de los metadatos de 2017 sigue vigente. Las 9,7 h de «Public Domain / Common Law» son dudosas.
4. El desfase se mide por correlación de envolventes, no nota a nota; con pares poco correlacionados el pico es
   espurio y satura en ±250 ms (13 clips).
5. La compuerta y la definición de S se fijaron antes de ver datos, pero sin calibración previa: τ = 0,6 viene del RFC
   y B1 la quería «fijada para que pase ~60 %». Aquí pasó el 55 %, cerca de esa intención.

## 6. Consecuencias para RFC-FUGA-2

- Lo que sobrevive: los pares `FluidSynth(AMT(y))` ↔ `y` quedan alineados por debajo de dos frames de latente en la
  mediana (D1) y más de la mitad pasa la compuerta (D2). La fábrica de pares no es el cuello de botella.
- Lo que muere, tal como estaba pre-registrado: el volumen. FMA CC BY/CC0 sin voz declarada da 231 h útiles, no 300.
  El refinador I-2 descrito en § 3.3 no puede entrenarse solo con esa fuente y esas definiciones; E3 (30–50 h) sí tiene
  datos, E4 (300–600 h) no.
- Decisiones que quedan para el RFC, no tomadas aquí: aceptar CC BY-SA (303 h en el filo, con obligación de compartir
  igual sobre lo derivado), añadir otras fuentes con licencia por pista verificable, bajar τ con un nuevo pre-registro
  y una medida de calidad del refinador que lo justifique, o quedarse en I-1 con SoundFont como prevé RI-1.

## 7. Reproducción

```bash
py -3.10 -m venv .venv_e1
.venv_e1/Scripts/python.exe -m pip install basic-pitch librosa soundfile numpy scipy pandas requests mido
# fma_metadata.zip en data/fma_e1/meta/ (descomprimido en fma_metadata/), FluidSynth win10-x64 en data/fma_e1/tools/fluidsynth/,
# MuseScore_General.sf3 (+ License, Readme, VERSION) en data/fma_e1/tools/soundfont/
.venv_e1/Scripts/python.exe -I scripts/e1_pares.py todo
```

Coste medido en esta máquina (CPU): licencias 6 s; descarga 154 s; medición 1 746 s (render 1 446 s, casi todo carga
del SoundFont comprimido por clip; Basic Pitch 115 s; métricas 149 s). Versiones y hashes de modelo y SoundFont en
`versiones` dentro de `ensayo/e1_pares.json`.

## 8. Archivos

| Archivo | Contenido |
|---|---|
| `ensayo/e1_prerregistro.md` | criterios de muerte y definiciones operativas, fijados antes de ver datos |
| `ensayo/e1_pares.json` | licencias, embudo, descarga, 300 filas por clip, agregados, criterios y veredicto |
| `scripts/e1_pares.py` | pasos `licencias`, `descargar`, `medir`, `veredicto` |
| `data/fma_e1/` (no versionado) | metadatos, clips, MIDI transcritos, renders WAV, registros intermedios, herramientas |
