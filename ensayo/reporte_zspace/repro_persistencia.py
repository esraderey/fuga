"""Repro: tras cerrar y reabrir un PackfileContentStore, ZSpace no conoce ningun checkpoint (z-space-core 0.1.0).

Los nodos (tensores y deltas) se persisten en el packfile, pero los CheckpointDescriptor viven solo en
ZSpace.checkpoint_table / checkpoint_index (memoria). El README promete "exact reconstruction after restart";
scripts/validation_suite.py lo consigue reinyectando a mano los descriptores que conservo en memoria (lineas 168-169).
Uso: python repro_persistencia.py
"""
import tempfile

import torch

from z_space_core import DecompType, PackfileContentStore, ZSpace

torch.manual_seed(0)
sd = {f"capa{i}.peso": torch.randn(32, 32) for i in range(8)}
with tempfile.TemporaryDirectory() as raiz:
    store = PackfileContentStore(raiz)
    space = ZSpace(cache_size=1 << 24, store=store)
    space.register_checkpoint("m", sd, decomp_type=DecompType.RAW, exact=True)
    space.update_checkpoint("m", {k: v + 1e-3 for k, v in sd.items()}, parent_state=sd)
    print("antes de cerrar: versiones =", [d.version for d in space.checkpoint_history("m")],
          "nodos =", store.stats()["nodes"])
    store.close()

    reabierto = PackfileContentStore(raiz)
    space2 = ZSpace(cache_size=1 << 24, store=reabierto)
    print("tras reabrir: nodos en el packfile =", reabierto.stats()["nodes"],
          "| checkpoints conocidos =", list(space2.checkpoint_table), "| descriptores =", len(space2.checkpoint_index))
    try:
        space2.load_checkpoint("m")
        print("sin fallo")
    except KeyError as e:
        print("FALLO: load_checkpoint tras reabrir ->", repr(e))
    reabierto.close()
