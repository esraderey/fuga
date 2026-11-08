# Revisión G4, segunda pasada — verificación de arreglos (2026-10-25)

Método: scripts propios en `C:\Users\Raul\AppData\Local\Temp\claude\D--fuga\7a4c3412-31dd-4934-bbb9-a0da03230d7d\scratchpad\g4b\` (`chk.py`, `chk2.py`, `mut.py`; salidas en `chk.json`, `chk2.json`, `mut.json`, `chk.log`, `mut.log`). `D:\fuga` solo se leyó, salvo este reporte. Los mutantes corren sobre copias de `fuga/`, `tests/`, `scripts/` y `pyproject.toml`, y `fuga.__file__` se comprueba en el mismo proceso que lanza pytest.
Línea base: las copias de control del árbol actual dan `M0_control … "24 passed in 4.90s"` (test_interprete) y `M0_control_fuga … "28 passed in 9.92s"` (test_fuga), 52 en total, como dice el contrato. Entorno: CUDA disponible, mido instalado, music21 no instalado.
No reutilicé el razonamiento de quien implementó los arreglos. Las entradas son las de la revisión previa, más barridos del corpus completo (14955 piezas), de 300 piezas × 4 estilos con MIDI, de 400 casos aleatorios (tpq 12 y 960) y de las CLI (198 MIDI).

## Veredicto por hallazgo previo

| Id | veredicto | evidencia |
|---|---|---|
| G4I-M1 (corte de notas repetidas en MIDI) | **parcial** | Con ataques distintos queda corregido. `python chk.py` → `M1_solapes_misma_altura`: en 300 piezas, para `plano`, `neutro`, `romantico` y `mecanico`, `"solapes_eventos": 0, "midi_on_sonando_pista": 0, "midi_on_sonando_canal": 0`. El ejemplo previo da `"book5_neutro": {"solapes_eventos": 0, "midi": [0, 0, 0]}`. En los 100 MIDI interpretados de las CLI: `"interpretados": 0` con violación. **No se corrige** cuando dos notas de la misma parte y altura atacan en el mismo tick, como pasa en el corpus: `corpus_estructura → "pares_mismo_tick_misma_tecla": 341, "piezas_con_dup": 30`, y `python chk2.py` → `"corpus_dup_midi_viol_neutro": [318, 318, 0]`. Ver N1. |
| G4I-M2 (pinza de orden y acordes) | **corregido** | `M2_pinza_acordes`: con la entrada previa, `"neutro_s15": {… "orden_viol": 0, "orden_estricto_viol": 0, "acordes_partidos": 0 …}` y `"romantico_s0": {… 0, 0, 0 …}`. Un barrido de 300 semillas × 3 estilos da `"orden": 0, "orden_estricto": 0, "acordes": 0` en los tres. En el corpus completo con `neutro`: `"orden": 0, "acordes": 0`. |
| G4I-M3 (fuerza del oráculo) | **corregido** | `python mut.py`: los 12 mutantes que sobrevivían (M1–M10, M16, M17) mueren todos, con `import_local: true`. Ver la tabla de mutantes. El control M0 sobrevive (`24 passed`). |
| G4I-M4 (`extraer_corpus.py` en Windows) | **corregido** (script completo sin ejecutar) | `M4_fuente_y_nombre`: para `…\music21\corpus\bach\bwv1.6.mxl` la función nueva da `["bach", "bwv1.6.mxl"]` y la expresión vieja `"EXC IndexError"`. Las rutas POSIX dan lo mismo con la nueva y con la vieja (`["essenFolksong", "altdeu10.abc"]` en ambas). Una ruta sin `corpus` da `ValueError`, que ahora queda dentro del `try` (`extraer_corpus.py:88-90`, por lectura). El README (`:21-22`) ya dice que no se ha ejecutado. `"procesar_sin_corpus": "sin music21"` (SOSPECHA para el script entero). |
| G4I-B1 (tope de desvío) | **corregido** | `B1_tope`: `"tpq960_300ataques": {"desvio_max": 0.009429607776502674, "tope": 0.05, "orden_estricto_viol": 0}`, `"on_negativo": "ValueError: hay ataques en ticks negativos"`. En 400 casos aleatorios: `"exceso_tope": 7.077671781985373e-16` (un ulp), `"orden": 0, "acordes": 0, "n_mal": 0, "dur_no_pos": 0`. En el corpus: `"exceso_tope_max": -0.01876`. |
| G4I-B2 (`plano` < 40 ms) | **corregido** | `B2_plano`: `"dur_s": [0.025], "esperado": [0.025]`. En 300 piezas a 200 bpm: `"plano_notas": 43562, "plano_distintas": 0`. |
| G4I-B3 (validación de `largos`/`inicios`) | **corregido** | `B3_validacion`: `"largos_vacio": "EXC ValueError: inicios y largos deben tener la misma longitud (1 y 0)"`, `"largos_corto": "… (2 y 1)"`, `"inicios_desordenados": "EXC ValueError: inicios debe empezar en 0 e ir ordenado, y largos ser positivo …"`, `"largo_cero"` igual, `"una_nota": "ok n=1"`. |
| G4I-B4 (acento de la anacrusa) | **corregido** | `B4_anacrusa`: `"vel_por_on": [[0, 84], [6, 88], [18, 84], [54, 88]]`. La nota de levare (84) ya no recibe el acento de primer tiempo (88). El mutante N23 muere. |
| G4I-B5 (`--device auto` cambia el ensayo) | **aceptado sin cambio** | El valor por defecto sigue en `auto` (`train.py:78`). `B5_B6_entrenos`: `"maxdif_auto_vs_cpu": 0.005882399156689644` y `"maxdif_cpu_vs_antes_g4": 0.0`. El README añade (`:32`) «para reproducirlos bit a bit fija --device cpu», pero el comando de ejemplo (`:33-34`) sigue sin `--device cpu`. |
| G4I-B6 (`ckpt` en la configuración guardada) | **corregido** | `"ckpt_en_config": {"auto": false, "cpu": false, "cpu_ckpt": false …}`. El código anterior (`scratchpad/antes`) carga los tres checkpoints nuevos: `"antes_carga_cpu_ckpt": [0, "…\\scratchpad\\antes\\fuga\\__init__.py", ""]`. El código nuevo carga `runs/{base,eco,fant,eco_rel}` (`"OK ckpt=False"`) y el checkpoint intermedio que trae `ckpt: true` (`{"config_ckpt": true, "c.ckpt": true, "training": false}`, inocuo en inferencia). El mutante N22 muere. |
| G4I-B7 (estilos ajenos en `ensayo.py`) | **corregido** (por lectura y emulación; el ensayo no se ejecutó) | `ensayo.py:206` filtra `te_b` antes de `trozos_eval` y `:227-228` salta los prompts de estilo ajeno. `contrastes` (`:127`, `:133`) usa la intersección de claves. Emulación con un modelo `--max_piezas 40` (`"estilos_cpu": ["airdsAirs"]`): `"ajena": "ValueError: estilo 'bach' fuera del vocabulario …"` y `"exc_en_conocidas_200": 0`. El filtro excluye justo lo que fallaría. Queda un resto en N5. |
| G4I-B8 (oráculo de checkpointing y de dispositivo) | **parcial** | `M13_bpn_sin_device … "MUERTO", "1 failed, 27 passed"` (lo mata `test_bits_por_nota_en_cuda_coincide_con_cpu`). Pero `M11_ckpt_sin_rng … "VIVO", "28 passed in 10.59s"`: `test_checkpoint_con_dropout_misma_semilla_mismo_resultado` (`tests/test_fuga.py:273-284`) solo compara la pasada hacia delante. Para detectar que el recómputo del backward usa otras máscaras de dropout hay que comparar gradientes con dropout > 0. |

Recuento: 9 corregidos, 2 parciales, 0 sin corregir y 1 aceptado sin cambio.

## Defectos nuevos

### N1. MEDIA — Dos ataques de la misma tecla en el mismo tick siguen solapándose; el segundo dura un tick en el MIDI — CONFIRMADO
- **Ubicación:**
  - `fuga/interprete.py:186-187`: `_cortar_repeticiones` deja la primera nota en `max(1e-3, limite - t0)` = 1 ms, que sigue pasando del ataque siguiente (`limite == t0`).
  - `fuga/render.py:115`: `nota[1] = max(a + 1, min(b, siguiente[pitch]))` obliga a `off ≥ a + 1 > on` siguiente `= a`.
  - Contradice el docstring (`interprete.py:10-11`) y la propiedad «sin solapes» que comprueba `tests/test_interprete.py:38-44`. Ningún test usa esta entrada.
- **Entrada:** `[(0,6,60,0),(0,24,60,0),(24,12,62,0)]`, `[0]`, `[48]`, 100 bpm, `neutro`. Es el patrón del corpus: 341 pares en 30 piezas, por ejemplo `airdsAirs/book3.abc#155` y `bach/bwv248.23-2.mxl#0`.
- **Evidencia (`python chk.py`, `N_cortar_repeticiones`):**
  - `"dup_mismo_tick_neutro": {"eventos": [[0.0, 0.001, 60, 82, 0], [0.0, 1.3021036918489084, 60, 82, 0], …], "solapes": 1, "midi": [1, 1, 0], "msgs": [[0, "note_on", 60], [0, "note_on", 60], [1, "note_off", 60], [2461, "note_on", 62], [2500, "note_off", 60], …]}`.
  - Con la semántica que usó G4I-M1 (un note_off libera la tecla), la nota de 1,30 s suena un tick (≈ 0,5 ms).
  - La entrada directa a MIDI `[(0.0,0.5,60,80,0),(0.0002,0.5,60,80,0)]` da `[[0,"note_on"],[0,"note_on"],[1,"note_off"],[960,"note_off"]]`.
  - De punta a punta (`chk.py`, `corpus_dup_end_to_end`): `["airdsAirs/book3.abc#155", "neutro", 2, [2, 2, 0]]`. En el corpus completo (`corpus_completo_neutro`): `"solapes": 341, "piezas_con_solape": 30`. En MIDI (`chk2.py`): `"corpus_dup_midi_viol_neutro": [318, 318, 0]`.
- **SOSPECHA, por lectura del reporte previo:** antes del arreglo, la nota larga sonaba hasta el final de la corta (≈ 0,6 s). Ahora el arreglo la deja en un tick, así que este caso empeora. No pude ejecutar la versión revisada en la primera pasada, porque `scratchpad/ahora` ya incluye el arreglo y da los mismos mensajes.

### N2. BAJA — Con tpq alto, el piso de 1 ms de `_cortar_repeticiones` deja solapes cuando la misma altura se repite a menos de 1 ms — CONFIRMADO
- **Ubicación:** `fuga/interprete.py:187` (`max(1e-3, …)`), frente al `eps = min(1e-3, 0.5·Δ)` de `:162`.
- **Entrada:** `[(k,1,60,0) for k in range(300)]`, tpq 960, 200 bpm, `neutro`.
- **Evidencia:** `"tpq960_misma_altura": {… "solapes": 290, "dur_min": 0.001, "midi": [196, 196, 0], "primeros": [[0.0, 0.001, …], [0.0001644580078125, 0.001, …], …]}`. El tope y el orden se mantienen (`"exceso_tope": -0.0406`, `"orden_estricto_viol": 0`).
- **Alcance:** el corpus no lo alcanza (`"tpqs": [12]`), y el test `test_tope_con_tpq_alto…` usa alturas distintas.

### N3. MEDIA (preexistente, fuera de las zonas tocadas) — El render plano `a_midi` de `fuga.sample` corta notas repetidas en las muestras generadas — CONFIRMADO
- **Ubicación:** `fuga/render.py:38-42`. El off va a `(on+dur)·esc − 1` sin mirar el siguiente ataque de la misma tecla.
- **Entrada:** `fuga.sample` con los 4 checkpoints, `--n 12 --compases 6|16 --interpretar romantico`.
- **Evidencia (`python chk2.py`):**
  - Tienen violación 30 de los 98 `muestra_XX.mid` planos, por ejemplo `["cli_eco_16", "muestra_03.mid", [3, 3, 0], 0]` y `["cli_sample", "muestra_00.mid", [1, 1, 0], 0]` (el comando del contrato). Ninguna sale de ataques simultáneos (`"violaciones_sin_ataque_simultaneo": 30`): el modelo genera notas que pasan del siguiente ataque de la misma altura.
  - Los MIDI interpretados de esas mismas muestras salen limpios (`"interpretados": 0`).
  - Esto matiza la revisión previa («el render plano anterior no tiene el problema»), que solo miró el corpus. El corpus también tiene 1060 solapes de misma altura en 53 piezas (`"solapes_partitura": 1060`).

### N4. BAJA — `dur_min_s` ya no es la duración mínima de un evento — CONFIRMADO
- **Ubicación:** `fuga/interprete.py:45` («duración mínima de un evento»), frente a `:187`.
- **Evidencia:** con la entrada de G4I-M2 y `neutro` (`dur_min_s` 0,04), `"dur_min": 0.0010000000000000009`. El recorte es intencionado, pero el comentario no lo dice.

### N5. BAJA — `ensayo.py` divide por cero si un brazo no conoce ningún estilo de test — SOSPECHA (lectura)
- **Ubicación:** `scripts/ensayo.py:218-219`.
- **Causa:** si `te_b` queda vacío, `tot_n = 0` y `R["test_bits_por_nota"] = … / tot_n` lanza `ZeroDivisionError`. Además, `nll_por_trozo([])` y `bootstrap([])` reciben listas vacías.
- **Alcance:** pasa con un `--max_piezas` cuyos estilos no tengan ninguna pieza de test. Con `--max_piezas 40` no ocurre (1180 piezas del estilo conocido en el corpus). No ejecutado: falta music21.

## Mutantes

`python mut.py` (copias en `g4b\mut\<nombre>`). En todas, `import_local: true`.

| nombre | test | import_local | estado | lo mata |
|---|---|---|---|---|
| M0_control | test_interprete | true | VIVO (control) | `24 passed` |
| M0_control_fuga | test_fuga | true | VIVO (control) | `28 passed` |
| M1_sin_pinza_orden | test_interprete | true | MUERTO | `test_acordes_con_ataques_a_un_tick[neutro-15]`, `…[romantico-0]`, `test_tope_con_tpq_alto…` |
| M2_sin_tope_desvio | test_interprete | true | MUERTO | `test_tope_de_desvio_actua_con_jitter_grande` |
| M3_jitter_por_nota | test_interprete | true | MUERTO | `test_acordes_con_ataques_a_un_tick` ×3 |
| M4_legato_ignorado | test_interprete | true | MUERTO | `test_repeticion…`, `test_high_loud_legato_y_adelanto` |
| M5_high_loud_invertido | test_interprete | true | MUERTO | `test_high_loud_legato_y_adelanto` |
| M6_arco_tempo_invertido | test_interprete | true | MUERTO | `test_arco_de_tempo_y_ritardando_final` |
| M7_adelanto_invertido | test_interprete | true | MUERTO | `test_high_loud_legato_y_adelanto` |
| M8_on_antes_que_off | test_interprete | true | MUERTO | `test_midi_off_antes_que_on…` |
| M9_todo_canal_0 | test_interprete | true | MUERTO | `test_render_de_eventos_wav_y_midi` |
| M10_audio_sin_velocidad | test_interprete | true | MUERTO | `test_audio_respeta_dinamica_y_tiempo` |
| M16_ticks_1pct | test_interprete | true | MUERTO | `test_midi_off_antes_que_on…` |
| M17_audio_en_t0 | test_interprete | true | MUERTO | `test_audio_respeta_dinamica_y_tiempo` |
| N18_sin_cortar_repeticiones | test_interprete | true | MUERTO | 9 tests (`test_invariantes_en_pieza_sintetica…`) |
| N19_sin_recorte_off_midi | test_interprete | true | MUERTO | `test_midi_off_antes_que_on…` |
| N20_eps_fijo_1ms | test_interprete | true | MUERTO | `test_tope_con_tpq_alto_y_ataques_densos` |
| N21_dur_min_fija (0,04 también en plano) | test_interprete | true | MUERTO | `test_plano_conserva_notas_mas_cortas_de_40_ms` |
| N23_anacrusa_con_acento | test_interprete | true | MUERTO | `test_anacrusa_no_recibe_acento…` |
| N24_acepta_negativos | test_interprete | true | MUERTO | `test_validacion_de_entradas` |
| N25_cortar_tolerancia_10ms | test_interprete | true | MUERTO | 6 tests |
| N26_validar_sin_orden | test_interprete | true | MUERTO | `test_validacion_de_entradas` |
| M11_ckpt_sin_rng (`preserve_rng_state=False`) | test_fuga | true | **VIVO** | `28 passed in 10.59s` |
| M13_bpn_sin_device | test_fuga | true | MUERTO | `test_bits_por_nota_en_cuda_coincide_con_cpu` |
| N22_config_con_ckpt | test_fuga | true | MUERTO | `test_config_guardable_no_lleva_ckpt…` |

Recuento sin los controles: 22 muertos y 1 vivo (M11). De los 12 que pedía el contrato, mueren los 12.

## Revisado sin hallazgos

- **(a) `_cortar_repeticiones`:**
  - Nunca deja una duración ≤ 0: `"dur_no_pos": 0` en 400 casos aleatorios y en el corpus completo.
  - Solo toca la nota recortada, de su misma parte y altura: `"recorte_local": {"piezas_estilo": 450, "eventos_recortados": 2427, "anomalias": []}`. Comparé con el recorte anulado: en cada evento cambia solo la duración, que baja y termina en el siguiente ataque de esa tecla o en 1 ms.
  - El estilo `plano` con notas adyacentes de la misma altura sigue exacto (`"plano_distintas": 0` en 43562 notas).
  - No añade ni quita notas en el corpus completo, comparando con tuplas: `"n_mal_tuplas": {"neutro": 0, "romantico": 0}`. El `"n_mal": 14955` de `chk.json` es un artefacto de mi script, que comparaba listas JSON con tuplas.
- **(b) Pinza `eps = min(1e-3, 0.5·Δ)`:**
  - Por inducción, si t_prev ≤ t_nom_prev + D, entonces t_prev + eps ≤ t_nom − 0,5·Δ + D < t_nom + D. La cota inferior la da el jitter recortado.
  - En ejecución: tpq 960 a 1 tick da `desvio_max 0.0094`. Los 400 casos aleatorios dan un exceso máximo de 7,1e-16 (un ulp, dentro del 1e-9 del test) y `orden_estricto 0`.
- **(c) `_validar` frente a las CLI:**
  - `python -m fuga.sample --ckpt runs/eco_rel/modelo.pt --n 2 --compases 6 --interpretar neutro --out …` → `rc 0`, `{'muestra': 0, 'notas': 74, 'compases': 6, …}`.
  - `python -m fuga.interprete --datos data/corpus_pd.jsonl --indice 100 --estilo romantico --out …` → `rc 0`, `"duracion_s": 16.99`.
  - 8 barridos (4 checkpoints × compases 6 y 16 × 12 muestras): todos `rc 0`.
  - Corpus completo con `compases_de`: `"exc": 0` en 14955 piezas.
  - `sample.py:116-119` solo interpreta si hay notas, así que `dec.inicios` no está vacío y empieza en 0 (`tokenizer.py:443-447`).
- **(d) Recorte del off en `a_midi_eventos`:** con ataques en ticks distintos, el off nunca pasa del siguiente on de la misma tecla. Lo muestran los 300 × 4 piezas y los 100 MIDI interpretados de las CLI, todos con 0 violaciones, y el orden off antes que on a igual tick está protegido (muere M8). La única excepción es el mismo tick (N1).
- **(e) `config_guardable`:** carga cruzada correcta en los dos sentidos (ver G4I-B6). `train.py:174` es el único `torch.save`.
- **(f) Filtro de estilos de `ensayo.py`:** correcto por lectura en (a) y (b). (c), (d) y (e) generan sin prefijo de estilo y no codifican piezas de test.
- **Canales MIDI:** `_canal` se repite a partir de la parte 16, pero el máximo del corpus es 15 partes (`"max_partes": 15`), así que no se alcanza.
