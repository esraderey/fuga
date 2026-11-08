"""Repro: cargar versiones de un checkpoint bloquea el proceso (z-space-core 0.1.0, commit 0f51a9e).

Causa: MarkovPrefetcher predice el siguiente tensor cuando una secuencia de accesos se repite, y
ZGen._record_and_prefetch encola `self.load(desc, mode=APPROX)` en el MISMO ThreadPoolExecutor(max_workers=4) en el
que `load` encola `synthesize` y espera su resultado (`future.result()`). Una pasada sobre tensores ya cacheados
encola decenas de prefetches en microsegundos; los 4 primeros ocupan los 4 hilos y encolan sus sintesis DETRAS de los
prefetches restantes. Nadie puede avanzar. La siguiente carga no cacheada (un checkpoint completo nuevo) espera para
siempre; si no hay carga posterior, el proceso no termina porque los hilos del pool no son daemon.
Uso: python repro_bloqueo.py [n_tensores=40]   (codigo 2 si la carga no termina en 30 s)
"""
import os
import sys
import threading
import time

import torch

from z_space_core import DecompType, ZSpace

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
torch.manual_seed(0)
sd = {f"capa{i}.peso": torch.randn(64, 64) for i in range(n)}
space = ZSpace(cache_size=1 << 26)                                   # almacen en memoria; basta con esto
space.register_checkpoint("m", sd, decomp_type=DecompType.RAW, exact=True)            # v0 completo
sd1 = {k: v + 1e-3 for k, v in sd.items()}
space.update_checkpoint("m", sd1, full_every=2, parent_state=sd)                     # v1 delta sobre v0
sd2 = {k: v + 2e-3 for k, v in sd.items()}
space.update_checkpoint("m", sd2, full_every=2, parent_state=sd1)                    # v2 completo periodico
v0, v1, v2 = space.checkpoint_history("m")
hecho = threading.Event()


def cargas():
    for etiqueta, desc in (("v0: primera pasada, siembra el historial", v0),
                           ("v1: vuelve a recorrer v0 cacheado y dispara prefetches", v1),
                           ("v2: tensores completos no cacheados", v2)):
        t0 = time.perf_counter()
        space.load_checkpoint_desc(desc)
        hilos = sum(1 for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor"))
        print(f"carga {etiqueta}: ok en {time.perf_counter()-t0:.2f}s; hilos del pool = {hilos}; "
              f"sintesis pendientes = {len(space.gen.pending_synthesis)}", flush=True)
    hecho.set()


threading.Thread(target=cargas, daemon=True).start()
if not hecho.wait(30):
    hilos = sum(1 for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor"))
    print(f"BLOQUEO: la carga no termina en 30 s; hilos del pool = {hilos}; "
          f"sintesis pendientes = {len(space.gen.pending_synthesis)}", flush=True)
    os._exit(2)                                                      # sys.exit se quedaria colgado en el pool
print("sin bloqueo", flush=True)
os._exit(0)
