"""Checkpoints periódicos en un almacén Z-Space (z-space-core, MIT): cada versión se guarda sin pérdida como delta
XOR+zstd respecto a la anterior, con un checkpoint completo cada `full_every` versiones. Los descriptores de versión,
que z-space-core mantiene solo en memoria, se persisten aquí (`descriptores.pkl`) junto con `versiones.json`."""
from __future__ import annotations

import argparse
import json
import os
import pickle

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _escribir_atomico(ruta: str, datos: bytes) -> None:
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        f.write(datos)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ruta)


class _SinPrefetch:
    """Sustituto del MarkovPrefetcher que nunca predice (evita el bloqueo descrito en AlmacenZ)."""

    def record_access(self, addr) -> None:
        pass

    def predict_next(self, curr, prev) -> list:
        return []


class AlmacenZ:
    """Almacén persistente de versiones de un state_dict. Reabrir con la misma raíz recupera el historial."""

    def __init__(self, raiz: str, nombre: str = "fuga", full_every: int = 8, dtype: str = "bf16", cache: int = 1 << 28):
        from z_space_core import PackfileContentStore, ZSpace  # dependencia opcional: pip install z-space-core
        self.raiz, self.nombre, self.full_every, self.dtype = raiz, nombre, int(full_every), dtype
        os.makedirs(raiz, exist_ok=True)
        self.store = PackfileContentStore(raiz)
        self.space = ZSpace(cache_size=cache, store=self.store)
        # z-space-core 0.1.0: el prefetcher Markov encola cargas bloqueantes en el mismo pool de 4 hilos que las
        # sintesis; con patrones de acceso repetidos (cargar varias versiones) se queda sin hilos y se bloquea.
        # Sin prefetch no hay bloqueo y la carga sigue siendo exacta.
        self.space.gen.prefetcher = _SinPrefetch()
        self.ruta_desc = os.path.join(raiz, "descriptores.pkl")
        self.ruta_meta = os.path.join(raiz, "versiones.json")
        self.meta: dict = {"nombre": nombre, "dtype": dtype, "full_every": self.full_every, "extra": {}, "versiones": []}
        self._ultimo: dict | None = None
        if os.path.exists(self.ruta_desc) and os.path.exists(self.ruta_meta):
            with open(self.ruta_desc, "rb") as f:
                descs = pickle.load(f)
            for d in descs:
                self.space.checkpoint_index[d.address] = d
            self.space.checkpoint_table[nombre] = descs[-1]
            with open(self.ruta_meta, encoding="utf-8") as f:
                self.meta = json.load(f)
            self.dtype = self.meta["dtype"]

    @property
    def versiones(self) -> list[dict]:
        return self.meta["versiones"]

    def _a_disco(self, state_dict: dict) -> dict:
        return {k: v.detach().to("cpu", DTYPES[self.dtype]).contiguous() for k, v in state_dict.items()}

    def guardar(self, state_dict: dict, paso: int, extra: dict | None = None) -> dict:
        """Añade una versión; devuelve su entrada de `versiones.json`."""
        from z_space_core import DecompType
        sd = self._a_disco(state_dict)
        if self.nombre in self.space.checkpoint_table:
            desc = self.space.update_checkpoint(self.nombre, sd, full_every=self.full_every, parent_state=self._ultimo)
        else:
            desc = self.space.register_checkpoint(self.nombre, sd, decomp_type=DecompType.RAW, exact=True)
        self._ultimo = sd
        if extra:
            self.meta["extra"] = extra
        equiv = sum(t.numel() * t.element_size() for t in sd.values())
        prev = self.versiones[-1]["bytes_tienda"] if self.versiones else 0
        tienda = int(self.store.stats()["compressed_bytes"])
        entrada = {"version": desc.version, "paso": int(paso), "direccion": desc.address_hex, "bytes_version": tienda - prev,
                   "bytes_tienda": tienda, "bytes_equivalente_torch": equiv * (desc.version + 1),
                   "completo": not desc.meta_view().get("requires_parent", True)}
        self.versiones.append(entrada)
        _escribir_atomico(self.ruta_desc, pickle.dumps(self.space.checkpoint_history(self.nombre)))
        _escribir_atomico(self.ruta_meta, json.dumps(self.meta, indent=1, ensure_ascii=False).encode("utf-8"))
        return entrada

    def cargar(self, version: int | None = None, paso: int | None = None) -> dict:
        """state_dict exacto (en el dtype guardado) de una versión, por número de versión o por paso; sin args, la última."""
        if paso is not None:
            version = next(v["version"] for v in self.versiones if v["paso"] == paso)
        hist = self.space.checkpoint_history(self.nombre)
        desc = hist[-1] if version is None else hist[version]
        assert version is None or desc.version == version
        return self.space.load_checkpoint_desc(desc, verify=True)

    def exportar(self, ruta: str, version: int | None = None, paso: int | None = None) -> str:
        """Escribe un checkpoint `.pt` normal de FUGA (pesos en fp32) que `fuga.sample.cargar_modelo` abre."""
        sd = {k: v.to(torch.float32) for k, v in self.cargar(version, paso).items()}
        torch.save({"modelo": sd, **self.meta["extra"]}, ruta)
        return ruta

    def cerrar(self) -> None:
        self.store.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Listar o exportar versiones de un almacén Z-Space de FUGA")
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--exportar", type=int, default=None, help="paso a exportar como .pt")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    alm = AlmacenZ(a.raiz)
    for v in alm.versiones:
        print(f"v{v['version']:3d} paso {v['paso']:6d} {'completo' if v['completo'] else 'delta   '} "
              f"{v['bytes_version']/1e6:8.1f} MB  tienda {v['bytes_tienda']/1e6:8.1f} MB "
              f"({100*v['bytes_tienda']/max(1, v['bytes_equivalente_torch']):.1f} % de torch.save)")
    if a.exportar is not None:
        print("exportado", alm.exportar(a.out or os.path.join(a.raiz, f"paso_{a.exportar}.pt"), paso=a.exportar))
    alm.cerrar()


if __name__ == "__main__":
    main()
