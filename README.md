# FUGA — compositor musical de 150–400M con operadores ECO e intérprete FUGA-I

Implementación de referencia de la serie RFC-FUGA (firmada OMRI):

- **RFC-FUGA-1** — compositor FUGA-C: representación por compás y parte, operadores ECO/ECOR con fantasmas opcionales,
  gramática, marco tonal relativo, núcleo FUGA-S (151,6M) y FUGA-M (386,7M). El texto del RFC no está en este árbol;
  la propuesta está en `expedicion/propuesta_D1.md` y su falsación en `ensayo/`.
- **RFC-FUGA-2** — intérprete FUGA-I: capa determinista de ejecución expresiva (implementada en `fuga/interprete.py`)
  y refinador PxT propuesto. Texto: [`rfc/RFC-FUGA-2.md`](rfc/RFC-FUGA-2.md).
- **RFC-FUGA-3** — medición (escalera metamórfica) y salvaguardas (guardia de memorización, datos, marco legal). Texto no
  incluido; los residuos de investigación están en `expedicion/innovador_B.md` (B2, B3) y en las críticas cruzadas.

## Instalación y pruebas

```bash
pip install -e ".[corpus,dev]"   # corpus = music21 (extracción y juez tonal); dev = pytest y ruff
pytest -q                        # 55 tests; dos de ellos solo corren con CUDA
```

Requisitos: Python ≥ 3.10, torch ≥ 2.4 (usa `F.rms_norm`), numpy, mido. El paquete, los tests y los scripts corren en
Windows, Linux y macOS; CUDA es opcional y se detecta sola. La extracción del corpus se corrigió para rutas de Windows,
pero en esta máquina no se ha ejecutado (falta `music21`).

## Uso rápido

```bash
# corpus de dominio público (music21) → JSONL de notas cuantizadas (12 ticks por negra), orden determinista
python scripts/extraer_corpus.py --out data/corpus_pd.jsonl --procs 2

# entrenar un brazo: base | eco | fant | base_rel | eco_rel | fant_rel  (eco_rel es el recomendado, ver ensayo/informe.md)
#   --device auto usa CUDA con bf16 si existe; --precision fp32 lo desactiva; --ckpt ahorra memoria de activaciones
#   los brazos de runs/ se entrenaron en CPU: para reproducirlos bit a bit fija --device cpu (la GPU mueve el 3.er decimal)
python -m fuga.train --datos data/corpus_pd.jsonl --modo eco_rel --preset nano --ctx 512 --pasos 360 \
  --dropout 0.0 --cache cache/eco_rel.pkl --out runs/eco_rel

# generar MIDI + WAV; --sesgo_eco sube o baja la repetición; --interpretar añade la ejecución expresiva
python -m fuga.sample --ckpt runs/eco_rel/modelo.pt --n 4 --compases 16 --sesgo_eco 1.5 --wav --interpretar neutro --out muestras/

# interpretar una pieza del corpus (plano y con estilo) para comparar
python -m fuga.interprete --datos data/corpus_pd.jsonl --indice 100 --estilo romantico --wav --out muestras/

# prueba de escala de FUGA-S y FUGA-M (CPU o GPU)
python scripts/escala.py --device cuda --precision bf16 --out ensayo/escala_cuda.json

# evaluación pre-registrada (ensayo/prerregistro.md) → ensayo/resultados.json
python scripts/ensayo.py --datos data/corpus_pd.jsonl --runs runs/base runs/eco runs/fant runs/eco_rel \
  --out ensayo/resultados.json --audio_dir ensayo/audio
```

## Representación

Cabecera `<bos> STY_* KEY_t_m TEMPO_*`; luego por compás `BAR [BL_n]` y por parte `PART_p` seguido de:

| Forma | Tokens | Significado |
| --- | --- | --- |
| literal | `(POS_x) PITCH_m DUR_d …` | notas |
| ECO | `ECO DIST_d FROM_q TRC_k\|TRD_k` | copia de la parte q, d compases atrás, transpuesta k semitonos o k grados |
| ECOR | `ECOR DIST_d FROM_q PITCH×n` | ritmo de la fuente, alturas nuevas |

Con `fantasma=True` el contenido referido se inserta como tokens literales con pérdida 0. En los modos `*_rel` las
alturas van relativas a la tónica declarada en `KEY`. `Decodificador.permitidos()` da la gramática: toda salida
generada se decodifica. Una pieza cuyo primer compás no empieza en 0 recibe un compás parcial inicial (anacrusa).

## Intérprete FUGA-I (capa determinista)

`fuga.interprete.interpretar(notas, inicios, largos, tempo_bpm, estilo, semilla)` convierte la partitura cuantizada en
eventos `(t0_s, dur_s, pitch, vel, parte)` aplicando reglas de la familia KTH parametrizadas por `Estilo`: acento
métrico, arco de frase en dinámica y tempo, high-loud, ritardando final, articulación por voz, adelanto de la voz
principal y microtiempo con semilla. Estilos incluidos: `plano` (reproduce la partitura tal cual), `neutro`,
`romantico`, `mecanico`.

Invariantes verificados por tests: no añade ni quita notas distintas (los duplicados de misma voz, ataque y altura se
fusionan); una altura no se solapa consigo misma en el MIDI; cada ataque queda a ≤ 50 ms de su posición en el mapa de
tempo; el orden de ataques dentro de cada voz se conserva; misma semilla, misma salida. La salida va a MIDI con
dinámica (`render.a_midi_eventos`, 960 ticks por negra) y a WAV por síntesis aditiva con dinámica y reverberación
corta (`render.a_audio_eventos`). El refinador neuronal de timbre (PxT) se propone en RFC-FUGA-2 y no está implementado.

## Estado

- **Ensayo en CPU cerrado** (`ensayo/informe.md`): K1 compresión, K4 marco relativo, K5 gramática y K6 perilla pasan;
  K2 estructura y K3 fantasmas no. La adherencia tonal exacta sube del 6 % (literal) al 56 % (`eco_rel`).
- **Verificado en GPU** (RTX 4060, 8,6 GB, torch 2.6 + CUDA 12.4): entrenamiento nano en bf16 con y sin
  checkpointing; paso completo de FUGA-S y FUGA-M a 512 tokens con picos de 1,6 y 3,6 GB (`ensayo/escala_cuda.json`).
  El entrenamiento real de FUGA-S a ctx 4096 queda pendiente (estimación con checkpointing: 3,6 GB para lote 8).
- **Ensayo MZTrain** (`ensayo/mztrain_informe.md`): FUGA-M factorizado (MZTrain 1.5.0, rango 512) gana 0,036 bits por
  nota por pieza a FUGA-S denso a igual número de tokens, pero reserva 8,8 GB y tarda siete veces más en esta GPU; FUGA-S
  denso sigue siendo la referencia local. Script: `scripts/train_mztrain.py`.
- **FUGA-S 3 h en GPU** (`ensayo/fuga_s_3h.md`): FUGA-S eco_rel entrenado 180 min en la RTX 4060 llega a 3,53 bits por
  nota en validación con el checkpoint del paso 3 250 (`--guardar_mejor`); más allá de ~5 épocas memoriza el corpus.
  Modelo de referencia local para generar: `runs/fuga_s_rel_3h/modelo_mejor.pt`.
- **Pendiente para producción**: cargadores de Lakh/PDMX, ediciones DEL/ADD, decodificador SMC, destructor D7 de la
  escalera metamórfica, salvaguardas de RFC-FUGA-3 y el refinador PxT de RFC-FUGA-2.

## Documentos

| Carpeta | Contenido |
| --- | --- |
| `rfc/` | RFC-FUGA-2 (intérprete) |
| `expedicion/` | `MAPA.md` (mapa del territorio, con la ampliación del intérprete), reportes S1–S6, propuestas, críticas cruzadas, verificaciones y la revisión adversarial G4 |
| `ensayo/` | pre-registro, resultados crudos, informe de cierre, pruebas de escala y audio de muestra |
| `runs/` | los cuatro checkpoints nano del ensayo (10 MB cada uno) |

## Notas de portabilidad

- `scripts/escala.py` mide el pico de memoria con psapi en Windows y `getrusage` en POSIX.
- `mido` es dependencia obligatoria (MIDI); `music21` solo hace falta para extraer el corpus y para el juez tonal del
  ensayo.
- El entrenamiento por `--pasos` es reproducible; `--minutos` depende de la carga de la máquina.
- `.gitignore` excluye datos, cachés, pesos y WAV regenerables. El árbol aún no es un repositorio git.

## Checkpoints periódicos (Z-Space) y publicación en Hugging Face

```bash
# cada 500 pasos guarda una versión bf16 sin pérdida en un almacén Z-Space (deltas XOR+zstd; requiere z-space-core)
python -m fuga.train ... --guardar_mejor --guardar_cada 500 --out runs/x
python -m fuga.zcheckpoints --raiz runs/x/zspace                 # lista versiones y tamaño frente a torch.save
python -m fuga.zcheckpoints --raiz runs/x/zspace --exportar 3000 # .pt normal de FUGA con los pesos de ese paso
python -m fuga.train ... --inicial runs/x/zspace/paso_3000.pt    # arranque en caliente (el optimizador empieza de cero)

# carpeta lista para Hugging Face: model.safetensors, config.json, ficha generada desde los registros, fuga/ y muestras
python scripts/exportar_hf.py --ckpt runs/x/modelo_mejor.pt --run runs/x --out hf/fuga-s-eco-rel --muestras muestras/x/*.wav
python scripts/subir_hf.py --dir hf/fuga-s-eco-rel --repo usuario/fuga-s-eco-rel            # ensayo en seco
python scripts/subir_hf.py --dir hf/fuga-s-eco-rel --repo usuario/fuga-s-eco-rel --confirmar
```

`fuga.sample --ckpt` acepta tanto el `.pt` como la carpeta exportada. Extras de `pip install -e .`: `zspace`, `hf`.
Licencia del código y de los pesos: MIT (`LICENSE`). Fallos encontrados en z-space-core 0.1.0 y sus rodeos:
`ensayo/reporte_zspace/` (issue esraderey/z-space-core#1).
