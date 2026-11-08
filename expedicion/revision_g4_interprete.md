# Revisión adversarial G4 — intérprete FUGA-I y brechas (2026-10-25)

**Alcance.** El diff `diff_final.patch` (antes → ahora) entero: `fuga/{tokenizer,model,train,sample,render,interprete,data,__init__}.py`, `scripts/{escala,extraer_corpus,ensayo}.py`, `tests/{test_fuga,test_interprete}.py`, `README.md` y `pyproject.toml`. Revisé las clases 1 a 8 del contrato en ese orden. No leí el razonamiento de quien implementó el cambio.

**Método.** Ejecuté todo con scripts propios en `C:\Users\Raul\AppData\Local\Temp\claude\D--fuga\7a4c3412-31dd-4934-bbb9-a0da03230d7d\scratchpad\g4\`, sin tocar `D:\fuga` salvo este reporte:
- `reg.py`: el mismo script contra `antes/` y contra `D:\fuga`. Codifica piezas del corpus y genera con los 4 checkpoints en CPU. Después comparé los hashes de los dos árboles.
- `interp_adv.py`: entradas adversas para el intérprete y para el render de eventos, más un barrido sobre el corpus.
- `mutar.py` y `mutar2.py`: mutantes sobre copias del árbol. En cada copia confirmé que `fuga` se importaba de la copia local (`import_local=True`).
- `cuda_chk.py`, más entrenamientos cortos con `python -m fuga.train` en CPU y en CUDA, antes y ahora.
- Humo de las CLI `fuga.sample --interpretar` (cpu y cuda), `fuga.interprete` y `scripts/escala.py` (cpu y cuda).

**Línea base.** En `D:\fuga`, `python -m pytest -q tests` → `36 passed in 9.25s`. `python -m ruff check --select E,F,W,B --ignore E501 fuga scripts tests` → `All checks passed!`. Entorno: torch `2.6.0+cu124`, RTX 4060, sin music21.

**Resultado: 0 ALTA / 4 MEDIA / 8 BAJA.** No hay regresiones en CPU para entradas que antes funcionaban. Los cuatro checkpoints cargan y generan bit a bit igual que antes. Los defectos están en el intérprete nuevo, en su oráculo y en una afirmación de portabilidad del README.

## ALTA

Ninguno.

## MEDIA

### G4I-M1. El MIDI interpretado corta notas repetidas: dos ataques de la misma tecla y canal se solapan
- **Clase:** 4 (y 2). **Ubicación:** `fuga/render.py:108-112` (`a_midi_eventos` emite on y off sin resolver solapes de la misma altura). El solape nace en `fuga/interprete.py:139-144`.
- **Causa:** el note_off de una nota A se calcula como `t0_A + dur_s`, y `dur_s` se mide desde la posición nominal `t_nom_A` y se multiplica por el legato (≥ 1). El ataque de la siguiente nota igual, B, lleva su propio desvío (jitter) y puede caer antes de ese off. Entonces `on_B < off_A`. En la semántica MIDI habitual (un note_off por canal y tecla libera la nota que suena), el off de A corta a B.
- **Entrada:** el corpus real con el estilo por defecto `neutro` y el tempo de cada pieza.
- **Evidencia (CONFIRMADO):**
  - `python interp_adv.py` → `'solapes_misma_altura_60_piezas': {'a_midi': 0, 'plano': 0, 'neutro': 338, 'romantico': 584}`.
  - `python cuda_chk.py …` → `"solape_ejemplo": {"pieza": "airdsAirs/book5.abc#0", "parte": 0, "pitch": 64, "on_A": 24259, "off_A": 24889, "on_B": 24866, "off_B": 25495, "B_suena_ticks_tras_off_A": 23, "B_debia_sonar_ticks": 629}`. A 1920 ticks/s, B suena ≈ 12 ms en vez de ≈ 328 ms.
  - El render plano anterior no tiene el problema: `render.py:41` cierra la nota un tick antes (`- 1`) y da 0 solapes.
  - El WAV (`a_audio_eventos`) no se ve afectado.

### G4I-M2. El orden de ataques por voz y la cohesión de los acordes se rompen cuando actúa la pinza de orden
- **Clase:** 2. **Ubicación:** `fuga/interprete.py:140-142`. Contradice el docstring (`interprete.py:8-9`) y `README.md:70`.
- **Causa:** `t0 = max(t0, ultimo[parte][1] + 1e-3)` solo se aplica a la primera nota de cada ataque (condición `on > ultimo[parte][0]`). Las demás notas del acorde se quedan con su `t0` sin corregir, y `ultimo[parte]` se sobrescribe con ese valor menor. Resultado: el acorde se parte, y el ataque siguiente se compara contra un tiempo equivocado.
- **Entrada:** `[(k,1,60,0),(k,1,64,0) for k in range(60)]`, `inicios=[0]`, `largos=[48]`, 200 bpm, tpq 12. Falla con `neutro` semilla 15 y con `romantico` semilla 0.
- **Evidencia (CONFIRMADO):** `python interp_adv.py` → `[A] {'neutro': {'semilla': 15, 'viol': (0, 9, 0.20031249385182767, 0.20954079885146623), 'test_ts_ordenado': False, 'acordes_partidos': 2}, 'romantico': {'semilla': 0, 'viol': (0, 17, 0.5027702351487144, 0.5056170567783488), 'test_ts_ordenado': False, 'acordes_partidos': 1}}`.
  - En `on=9` una nota ataca a 0,2003 s cuando un ataque anterior de la misma voz ya estaba en 0,2095 s.
  - La propia comprobación del test (`ts == sorted(ts)`, `tests/test_interprete.py:41`) falla con esta entrada.
- **Alcance:** hace falta tener ataques separados por 1 tick (≈ 25 ms a 200 bpm) y acordes en la misma parte. En 120 piezas del corpus × 3 tempos × 4 estilos no apareció ninguna violación.

### G4I-M3. Los tests del intérprete y del render de eventos no detectan la ruptura de las propiedades que dicen verificar
- **Clase:** 5. **Ubicación:** `tests/test_interprete.py:30-41`, `:54-61` y `:75-88`. Afirmación afectada: `README.md:70` («Invariantes verificados por tests»).
- **Causa:**
  - La pieza sintética de `_preparar` (100 bpm, ataques separados ≥ 6 ticks, sin acordes dentro de una parte) nunca activa el tope de desvío: σ 6–9 ms frente a un tope de 50 ms. Tampoco activa la pinza de orden.
  - El test de MIDI solo cuenta los note_on y compara el conjunto de velocidades.
- **Evidencia (CONFIRMADO):** `python mutar.py` y `python mutar2.py` → sobreviven (`VIVO | 12 passed`) los mutantes siguientes:
  - del intérprete: M1 sin pinza de orden; M2 sin tope de desvío; M3 jitter por nota (los acordes se deshacen); M4 legato ignorado; M5 high-loud invertido; M6 arco de tempo invertido (aunque el nombre `test_ritardando_final_y_arco` dice cubrir el arco); M7 adelanto invertido;
  - del render: M8 note_on antes que note_off en el mismo tick; M9 todas las partes en el canal 0; M10 el audio ignora la velocidad; M16 ticks escalados un 1 %; M17 todo el audio en t=0.
  - Controles: M12 (sin anacrusa), M14 (logits en la GPU) y M15 (`<pad>` silencioso) mueren (`1 failed`). M0 (sin cambio) sobrevive (`12 passed`).
  - M8 es justo el caso de «note_off antes de note_on en el mismo tick». El código actual lo resuelve bien, pero ningún test lo protege.

### G4I-M4. `extraer_corpus.py` aborta en Windows, y el README nuevo afirma que funciona en Windows
- **Clases:** 7 y 8. **Ubicación:** `scripts/extraer_corpus.py:80` (también `:86`, `ruta.rsplit('/', 1)`). La lista de rutas se arma en `:102` y se consume en `:107`. Afirmación afectada: `README.md:20` («Funciona en Windows, Linux y macOS»).
- **Causa:** `ruta.split("/corpus/")[1]` asume el separador `/`, pero `str(p)` de `corpus.getCorePaths()` da `\` en Windows. La línea 80 está fuera del `try` de la línea 81, así que la excepción escapa del worker, `pool.imap` la relanza en el proceso padre y la extracción entera se detiene.
- **Entrada:** cualquier ruta de music21 en Windows.
- **Evidencia:** CONFIRMADO para la expresión: `python -c "…str(pathlib.Path(r'C:\…\music21\corpus\bach\bwv1.6.mxl')).split('/corpus/')[1]…"` → `extraer_corpus.py:80 -> IndexError list index out of range`. SOSPECHA para el script completo, porque music21 no está instalado. El código es anterior al diff, pero el diff añade la afirmación de portabilidad y reescribe este archivo sin corregirlo.

## BAJA

### G4I-B1. El tope |t0 − mapa de tempo| ≤ `max_desvio_s` se puede violar
- **Clase:** 2. **Ubicación:** `fuga/interprete.py:141` (la pinza suma +1 ms y se acumula) y `:139` (`max(0.0, …)` cuando `on < 0`).
- **Entradas:**
  - (a) tpq=960, 200 bpm, 300 ataques separados por 1 tick, estilo `neutro`.
  - (b) una nota con `on=-6` a 90 bpm.
- **Evidencia (CONFIRMADO):** `[B] {'tpq960_max_desvio_s': 0.2007, 'tope': 0.05, 'on_negativo': [(0.0, -0.3508), (0.001, 0.0)]}`.
- **Alcance:** no se alcanza con el corpus actual (`min_on 0` y tpq 12 en las 14955 piezas).

### G4I-B2. El estilo «plano» no reproduce las notas de menos de 40 ms
- **Clase:** 2. **Ubicación:** `fuga/interprete.py:144` (`max(0.04, dur_s)`). Contradice `interprete.py:9` y `README.md:67`.
- **Entrada:** `(0,1,60,0)` a 200 bpm → `[C] {'dur_s': [0.04, 1.175], 'esperado': [0.025, 1.175]}`.
- **Evidencia (CONFIRMADO):** en el corpus, `'plano_dur_distinta': 10, 'plano_notas': 32961`, por ejemplo `('airdsAirs/book5.abc#10', 200.0, (42, 1, 69, 0), 0.04, 0.025)`. El test del estilo plano corre a 100 bpm y no alcanza el caso.

### G4I-B3. `largos` vacío o más corto que `inicios` da `IndexError` sin explicación; `inicios` desordenado pasa en silencio
- **Clase:** 2. **Ubicación:** `fuga/interprete.py:115` (`list(largos) or [4 * tpq]` solo sirve si `inicios` tiene un elemento) y `:80` (`largos[b]`).
- **Evidencia (CONFIRMADO):** `[D] {'largos_vacio': 'EXC IndexError: list index out of range', 'largos_corto': 'EXC IndexError: list index out of range', 'inicios_desordenados': 'ok n=3 desvio=0.0204', 'una_nota': 'ok n=1 desvio=0.0000'}`.

### G4I-B4. La anacrusa recibe acento de primer tiempo y cuenta como compás 0 de la frase
- **Clase:** 2. **Ubicación:** `fuga/tokenizer.py:124-125` (`compases_de` antepone el compás parcial `[0, ini0)`) combinado con `fuga/interprete.py:129` (`pos == 0` recibe `acento_compas`) y `:80` (`b % frase`).
- **Entrada:** `compases=[6,54,102]`, notas en los ticks 0, 6, 18 y 54, estilo `mecanico` → `'vel_por_on': [(0, 88), (6, 88), (18, 84), (54, 88)]`. La nota de levare (88) recibe el mismo acento que los primeros tiempos.
- **Evidencia (CONFIRMADO).** Hoy no tiene efecto: `n_anacrusa 0` en 14955 piezas.

### G4I-B5. El valor por defecto `--device auto --precision bf16` cambia el resultado del comando del ensayo pre-registrado
- **Clase:** 1. **Ubicación:** `fuga/train.py:78-79`. El pre-registro (`ensayo/prerregistro.md:1`) dice «(CPU)».
- **Entrada:** el mismo comando (`--modo eco --preset nano --ctx 128 --pasos 6 --max_piezas 40 --dropout 0.0`) en esta máquina.
- **Evidencia (CONFIRMADO):** `cpu1 vs cuda1: maxdif=5.882e-03`, `antes_cpu vs ahora_cpu1: maxdif=0.000e+00`, `cpu1 vs cpu2: 0.000e+00`, `cuda1 vs cuda2: 0.000e+00`. El cambio de valor por defecto está documentado (`README.md:30`), pero los comandos de entrenamiento del README no fijan `--device cpu` y en esta máquina no reproducen los brazos de `runs/`.

### G4I-B6. La bandera de entrenamiento `ckpt` queda guardada en la configuración del modelo, y los checkpoints 0.2.0 no cargan con 0.1.0
- **Clases:** 1 y 3. **Ubicación:** `fuga/model.py:26` y `fuga/train.py:124` y `:174`.
- **Evidencia (CONFIRMADO):** `"ck_config_ckpt": true`; cargar ese checkpoint con el código de `antes/` da `TypeError ConfigModelo.__init__() got an unexpected keyword argument 'ckpt'`. La compatibilidad en sentido contrario está bien: los checkpoints viejos cargan con el código nuevo.

### G4I-B7. El nuevo `ValueError` de estilo desconocido rompe `ensayo.py` para brazos entrenados con `--max_piezas`
- **Clase:** 1. **Ubicación:** `scripts/ensayo.py:210` (`trozos_eval` sin `try`) y `:226` (`codificar` fuera del `try` de `:227`). Los estilos se derivan del subconjunto en `fuga/train.py:109`.
- **Entrada:** un modelo entrenado con `--max_piezas 40` y una pieza de test de otra fuente.
- **Evidencia:** `"estilo_ajeno": "ValueError: estilo 'bach' fuera del vocabulario (estilos: ['airdsAirs']) (fuente=bach, particion=test)"`. Antes se codificaba `<pad>`. CONFIRMADO para `codificar`; SOSPECHA para el ensayo completo, que no ejecuté. El arreglo de B4 de la revisión anterior es correcto, pero su consumidor no se adaptó.

### G4I-B8. El oráculo de checkpointing y de dispositivo no cubre dropout ni `bits_por_nota` en GPU
- **Clase:** 5. **Ubicación:** `tests/test_fuga.py:233-236` (nano con dropout 0 en CPU) y `fuga/train.py:49`.
- **Evidencia (CONFIRMADO):** sobreviven `M11_ckpt_sin_rng: VIVO | 24 passed` y `M13_bpn_sin_device: VIVO | 24 passed`.
- **Comportamiento real:** es correcto (ver la clase 3 en «Revisado sin hallazgos»). La brecha es solo del test.

## Revisado sin hallazgos

- **Clase 1 (regresiones):**
  - `reg.py` contra los dos árboles: los cuatro checkpoints (`base`, `eco`, `fant`, `eco_rel`) cargan en CPU. Las 5 generaciones por brazo (3 sin prefijo y 2 con prefijo de una pieza de test) y la NLL de los prefijos son idénticas entre árboles (`base igual … eco_rel igual`).
  - `codificar` da hashes idénticos en 800 casos (200 piezas × 4 modos), con 0 excepciones y 0 avisos.
  - El corpus tiene `n_anacrusa 0` y `min_on 0`, así que la anacrusa no cambia ninguna pieza actual.
  - El entrenamiento con `--device cpu` es bit a bit igual a `antes` (`maxdif=0.000e+00`).
  - `.float().cpu()` y `device=dev` en `sample.py:83-84` no alteran la ruta en CPU.
- **Clase 2 (intérprete):**
  - 120 piezas × tempos 40, el de la pieza y 200 × 4 estilos: `'viol_orden': 0, 'viol_desvio': 0, 'viol_n': 0, 'acordes_partidos': 0`.
  - Una sola nota: bien. Sin notas ni compases: lo cubre el test.
  - `mapa_tempo` con `rit_pulsos > n` no indexa fuera de rango (lectura).
  - El estilo «plano» a 100 bpm reproduce exacto (test).
- **Clase 3 (dispositivo y precisión):**
  - El checkpoint entrenado en CUDA con bf16, `--ckpt` y dropout 0.1 se guarda en CPU y en float32 (`"ck_devices": ["cpu"]`, `"ck_dtypes": ["torch.float32"]`), y carga y genera tanto en cpu como en cuda (`gen_cpu`/`gen_cuda` `[1, 2, true, …, false]`).
  - Checkpointing con dropout 0.1 y autocast bf16 en CUDA: `ckpt_dropout_maxdif_misma_semilla: 0.0` (referencia con otra semilla: 3.68).
  - `bits_por_nota`: CPU 5.915177399, CUDA fp32 5.915177556, CUDA TF32 5.915162699. El docstring dice «fp32» aunque `train.py` activa TF32; la diferencia, 1,5e-5 bits, es irrelevante.
  - `non_blocking=True` sin memoria fijada es inocuo, porque la copia desde memoria paginable es síncrona.
  - `fuga.sample --device cuda` con `runs/eco_rel` da el mismo resumen que en cpu (`'notas': 53, 'compases': 4, 'tonica': 2`).
- **Clase 4 (render de eventos):**
  - El orden `(tick, es_on)` pone el off antes que el on en el mismo tick (`render.py:112`, lectura).
  - Cada nota dura al menos un tick (`:111`).
  - `_canal` es la misma expresión que había antes en `a_midi`.
  - En `a_audio_eventos` las longitudes de `y[i0:i0+n]` y de `[: len(y)-i0]` coinciden, y `fin` incluye +1,5 s.
  - Todo silencio da ceros sin NaN: `[F] {'silencio_nan': False, 'silencio_max': 0.0, 'len': 8001, 'neg_ok': True, 'len2': 6801}`.
- **Clase 5 (oráculo):**
  - El diff solo añade tests; ninguna aserción previa se debilitó.
  - El único `skipif` (CUDA) está justificado y aquí corre.
  - `test_anacrusa…`, `test_estilo_fuera…` y `test_cuda…` matan sus mutantes (M12, M15, M14).
- **Clase 6 (`escala.py`):**
  - Corre en Windows en CPU (nano: `"rss_max_gb": 0.51`) y en CUDA (fuga-s: `"cuda_pico_gb": 1.57`, `"total_gb": 3.63`, igual que `ensayo/escala_cuda.json`).
  - El orden de campos de `PROCESS_MEMORY_COUNTERS` y `restype=HANDLE` son correctos.
  - Rama POSIX: Linux en KB y macOS en bytes (lectura).
  - Ningún archivo del árbol usa las claves renombradas `seg_fwd_bwd_cpu` ni `estimacion_24gb…` (grep vacío).
- **Clase 7 (`extraer_corpus.py`):** por lectura, `imap` ordenado da un orden de salida independiente de `--procs`; `avisos` registra `stripTies`, la tonalidad no analizada y el modo forzado, y el forzado ocurre de verdad en el `return`. Fuera de G4I-M4, nada.
- **Clase 8 (documentación):**
  - Existen todos los archivos citados: `rfc/RFC-FUGA-2.md`, `.gitignore`, `expedicion/{MAPA,propuesta_D1,innovador_B}.md`, `ensayo/{informe.md,escala_cuda.json,prerregistro.md}`.
  - torch 2.6 + CUDA 12.4 coincide con el entorno.
  - «36 tests; uno solo con CUDA» coincide con la ejecución.
  - «6 % → 56 %» coincide con `ensayo/informe.md:45`.
  - «`--pasos` es reproducible» se cumple en CPU y en CUDA para 6 pasos.
  - Las CLI de `README.md` (`fuga.sample --interpretar romantico --wav` y `fuga.interprete --indice 100 --estilo romantico --wav`) escriben sus `.mid` y `.wav`.
- **Hallazgos B1–B11 de la revisión anterior que toca este diff:**
  - B2 (anacrusa): cerrado y cubierto por un test que mata M12.
  - B3 (aviso del modo relativo): emite el aviso.
  - B4 (estilo): falla claro; ver G4I-B7.
  - B8 (corpus): cerrado por lectura, salvo G4I-M4.
  - B9: solo documentación.
  - No revisé M1–M6, B1, B5–B7, B10 ni B11 de esa revisión porque este diff no los toca.
