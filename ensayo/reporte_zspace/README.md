# Reporte de fallos: z-space-core 0.1.0

- Repositorio: https://github.com/esraderey/z-space-core, commit `0f51a9e` (2026-09-19, «Revise README for improved documentation and examples»)
- Fecha del reporte: 2026-11-08
- Origen: integración de z-space-core como almacén de checkpoints periódicos en FUGA (`fuga/zcheckpoints.py`); los fallos aparecieron en los tests de esa integración y se redujeron a los dos scripts de esta carpeta.
- Entorno: Windows 11 Pro, Python 3.13.2, torch 2.6.0+cu124, numpy 2.3.5, mscs 2.4.0, zstandard 0.23.0, tensorly 0.9.0, lz4 4.4.4. Instalación: `pip install --no-deps --user <clon>`; el módulo instalado es byte a byte el del commit.

## Resumen

| # | Tipo | Título | Gravedad |
|---|---|---|---|
| 1 | Fallo | Cargar varias versiones de un checkpoint bloquea el proceso para siempre (deadlock en el pool de hilos) | Alta |
| 2 | Fallo | Los descriptores de checkpoint no se persisten: tras reabrir el packfile no hay forma de cargar ninguna versión | Alta |
| 3 | Observación | El ahorro de los deltas XOR+zstd depende del dtype: en fp32 es casi nulo; el README solo documenta bf16 | Documentación |
| 4 | Observación | `ZGen` nunca cierra su `ThreadPoolExecutor`; un pool atascado impide que el intérprete termine | Baja |

Los dos fallos se reproducen de forma determinista con `repro_bloqueo.py` y `repro_persistencia.py` (sin GPU, sin archivos grandes, en menos de un minuto).

## Fallo 1: bloqueo al cargar versiones

### Reproducción

```
python repro_bloqueo.py
carga v0: primera pasada, siembra el historial: ok en 0.05s; hilos del pool = 1; sintesis pendientes = 0
carga v1: vuelve a recorrer v0 cacheado y dispara prefetches: ok en 0.03s; hilos del pool = 4; sintesis pendientes = 4
BLOQUEO: la carga no termina en 30 s; hilos del pool = 4; sintesis pendientes = 5
```

Patrón mínimo: `register_checkpoint` (v0, RAW), `update_checkpoint` (v1, delta) y `update_checkpoint` con `full_every=2` (v2, completo); después `load_checkpoint_desc` de v0, v1 y v2 en ese orden. Con 40 tensores de 64×64 ya bloquea; con 200 también. Almacén en memoria, sin disco.

### Causa

1. `ZGen.__init__` (línea 1410) crea un único `ThreadPoolExecutor(max_workers=4)`.
2. `ZGen.load` (1536-1565) encola `self.synthesize` en ese pool (1555) y se queda esperando en `future.result()` (1559). Antes de volver llama a `_record_and_prefetch`.
3. `_record_and_prefetch` (1567-1578) pide al `MarkovPrefetcher` (1380-1401) los siguientes tensores previstos y, para cada uno no cacheado, encola **otra llamada a `self.load`** en el mismo pool (1578), en modo `APPROX`.
4. `_cache_key` (1533) incluye el modo en la clave, así que un tensor cargado en modo `EXACT` nunca satisface la comprobación de caché del prefetch `APPROX`: cada predicción encola siempre un `load`.
5. El predictor es de segundo orden: en cuanto una pareja consecutiva de accesos se repite (recorrer de nuevo los tensores del padre al cargar la versión siguiente), predice el siguiente en cada acceso. Una pasada sobre tensores cacheados tarda microsegundos y encola decenas de prefetches de golpe. Los cuatro primeros ocupan los cuatro hilos, cada uno encola su `synthesize` **detrás** de los prefetches restantes en la cola FIFO y se bloquea en `result()`. Ninguna síntesis puede ejecutarse jamás.
6. La siguiente carga no cacheada (los tensores completos de v2) encola su síntesis en la misma cola muerta y espera para siempre. Si no hay carga posterior, el proceso no termina al salir: los hilos del pool no son daemon y siguen bloqueados.

El bloqueo es estructural, no una carrera: depende solo del orden FIFO de la cola y de que haya más de cuatro prefetches pendientes, no de la velocidad de síntesis.

### Por qué no lo detecta la suite

`scripts/validation_suite.py` carga 25 versiones en cadena sin `full_every`: solo los tensores de v0 pasan por `ZGen.load`; todas las demás versiones son deltas que `load_checkpoint_desc` aplica con `store.get` directamente, fuera del pool. El bloqueo necesita una carga no cacheada **después** de que el prefetcher se haya activado, y eso ocurre justo con la política operativa recomendada en el README (`full_every=8`).

### Impacto

Cualquier programa que cargue dos o más versiones con tensores compartidos y después una versión completa se congela sin error ni traza. En FUGA se manifestó como un `pytest` que no terminaba nunca; el volcado de pila con `faulthandler` apuntaba a `future.result()` en la línea 1559.

### Corrección propuesta

- El prefetch no debe encolar una tarea bloqueante en el pool que ejecuta las síntesis. Opciones, de menor a mayor cambio: (a) que `_record_and_prefetch` encole `self.synthesize` directamente y guarde el resultado en caché desde un `add_done_callback`, sin esperar dentro de un hilo del pool; (b) un ejecutor separado y acotado para prefetch (`max_workers=1` o 2) con un semáforo que limite las predicciones pendientes; (c) desactivar el prefetch por defecto hasta que exista una de las dos anteriores.
- Añadir un test de regresión con el patrón del repro (tres versiones con `full_every=2`, cargas 0, 1 y 2) bajo un tiempo límite.
- Rodeo usado en FUGA mientras tanto: sustituir `space.gen.prefetcher` por un objeto cuyo `predict_next` devuelve siempre `[]`. La carga sigue siendo exacta.

## Fallo 2: los descriptores de checkpoint no se persisten

### Reproducción

```
python repro_persistencia.py
antes de cerrar: versiones = [0, 1] nodos = 16
tras reabrir: nodos en el packfile = 16 | checkpoints conocidos = [] | descriptores = 0
FALLO: load_checkpoint tras reabrir -> KeyError('Unknown checkpoint: m')
```

### Causa

`PackfileContentStore` persiste los nodos (tensores, bloques y deltas) y su índice, pero los `CheckpointDescriptor` solo existen en `ZSpace.checkpoint_table` y `ZSpace.checkpoint_index` (1640-1641), que se rellenan en memoria en `register_checkpoint` (1721-1722) y `update_checkpoint` (1778-1779, 1831-1832) y no se escriben nunca al almacén. `NodeKind` (84-91) no tiene ninguna clase de nodo para descriptores. Al reabrir, el packfile contiene todos los bytes necesarios, pero no hay ningún camino en la API para encontrar el grafo de versiones.

### Por qué no lo detectan la suite ni los tests

`scripts/validation_suite.py` (líneas 168-169) reinyecta a mano en `reopened_space.checkpoint_index` los descriptores que conservó en memoria antes de cerrar; en un reinicio real del proceso ese estado no existe. `tests/test_z_space_core.py::test_packfile_store_reopens_from_persisted_index` (177) comprueba solo que el almacén de nodos recupera su índice, que sí funciona.

### Impacto

El README promete «Exact reconstruction after restart» (línea 15) y «Durable append-only storage with recovery after interrupted writes» (16). En la práctica, tras un reinicio del proceso (o un corte de luz, el caso que motivó la integración en FUGA) los checkpoints son irrecuperables salvo que la aplicación haya serializado los descriptores por su cuenta. Es pérdida de datos desde el punto de vista del usuario, aunque los bytes estén en disco.

### Corrección propuesta

- Persistir cada `CheckpointDescriptor` (y el árbol de `ZDescriptor` que referencia) como nodo del almacén con una nueva clase (`NodeKind.CHECKPOINT`), serializado con `mscs` igual que los componentes; `canonical_bytes()` ya existe y da una dirección estable.
- Guardar en el diario del packfile una entrada de «cabeza» por nombre (`name → address`), y que `ZSpace.__init__` (o un `ZSpace.open(store)`) reconstruya `checkpoint_table`, `checkpoint_index` y `checkpoint_version_graph` al abrir.
- Mientras tanto, documentar la limitación y ofrecer `export_descriptors()` / `import_descriptors()` para que el llamador pueda hacerlo sin tocar atributos internos.
- Test: cerrar, crear un `ZSpace` nuevo sobre el mismo directorio y cargar por nombre, sin reinyectar nada.
- Rodeo usado en FUGA: `pickle` de `checkpoint_history(nombre)` escrito de forma atómica junto al packfile y reinyectado al abrir, más un `versiones.json` con el mapa paso → versión.

## Observación 3: el ahorro de los deltas depende del dtype

Medido con dos checkpoints reales de FUGA-S (151,6M parámetros, 218 tensores) separados 1 250 pasos de AdamW, tasa 6e-4:

| Almacenado | torch.save de una versión | Delta Z-Space (XOR+zstd, nivel 3) | Delta / torch.save |
|---|---:|---:|---:|
| fp32 | 606,4 MB | 563,4 MB | 92,9 % |
| bf16, preacondicionador `xor` | 303,2 MB | 238,4 MB | 78,6 % |
| bf16, preacondicionador `u16_sub` | 303,2 MB | 238,2 MB | 78,6 % |

Reconstrucción exacta en los tres casos. Los bits bajos de la mantisa fp32 cambian en casi todos los pesos tras cualquier paso de Adam, así que el XOR es ruido para zstd. El benchmark del README (Pythia-410M, bf16, checkpoints cada 50 pasos) está en el régimen favorable y no lo dice. Sugerencias: indicar en el README que las cifras son para bf16 y para intervalos cortos; recomendar guardar en bf16 (en FUGA el redondeo costó 0,00004 bits por nota); y considerar un preacondicionador para fp32 que separe los 16 bits altos (delta `u16_sub`) de los 16 bajos (crudos), que es donde está la estructura aprovechable.

## Observación 4: el pool nunca se cierra

`ZGen` crea el `ThreadPoolExecutor` en 1410 y ningún método lo apaga; `ZSpace` no tiene `close()`. Con el fallo 1 activo, `sys.exit` se queda colgado al final del programa aunque toda la carga útil haya terminado (el repro sale con `os._exit` por eso). Sugerencia: `ZSpace.close()` que llame a `store.close()` y `executor.shutdown(wait=False, cancel_futures=True)`, y marcar los hilos del pool como daemon.

## Archivos adjuntos

- `repro_bloqueo.py`: código de salida 2 si la tercera carga no termina en 30 s; imprime hilos del pool y síntesis pendientes.
- `repro_persistencia.py`: muestra el `KeyError` tras cerrar y reabrir un `PackfileContentStore` con dos versiones guardadas.
- Rodeos de referencia en FUGA: `fuga/zcheckpoints.py` (clase `_SinPrefetch` y persistencia de descriptores), tests en `tests/test_zcheckpoints.py`.
