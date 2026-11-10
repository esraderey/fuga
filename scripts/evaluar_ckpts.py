"""Evalúa checkpoints de FUGA sobre la validación y el test completos (bits por nota) y, con --por_pieza, guarda la NLL
de test por pieza para contrastes pareados.

Uso:
  python scripts/evaluar_ckpts.py --ckpts runs/a/modelo_mejor.pt runs/a/modelo.pt --out runs/a/evaluacion.json --por_pieza
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys

import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from fuga.data import cargar, particion, trozos_eval  # noqa: E402
from fuga.sample import cargar_modelo  # noqa: E402
from fuga.train import bits_por_nota, resolver_device  # noqa: E402


def _mz():
    """scripts/train_mztrain.py como módulo (nll_por_pieza y bootstrap); no necesita mztrain instalado."""
    spec = importlib.util.spec_from_file_location("train_mztrain", os.path.join(RAIZ, "scripts", "train_mztrain.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def trozos_test(cache_test: str, datos: str, voc, op: dict, ctx: int) -> list:
    """[(id, (ids, compás, pérdida, notas))] del test; se lee de la caché o se construye desde el corpus y se guarda."""
    if cache_test and os.path.exists(cache_test):
        with open(cache_test, "rb") as f:
            return pickle.load(f)
    rep = "fant" if op["fantasma"] else "eco" if op["eco"] else "base"
    piezas = [p for p in cargar(datos) if particion(p) == "test" and f"STY_{p['fuente']}" in voc.stoi]
    out = []
    for p in piezas:
        out += [(p["id"], t) for t in trozos_eval([p], voc, ctx, op["relativo"])[rep]]
    if cache_test:
        os.makedirs(os.path.dirname(cache_test) or ".", exist_ok=True)
        with open(cache_test, "wb") as f:
            pickle.dump(out, f)
    return out


def evaluar(ckpts: list[str], cache: str, cache_test: str, datos: str = "", device: str = "auto",
            por_pieza: bool = False) -> dict:
    with open(cache, "rb") as f:
        _, ev, _ = pickle.load(f)
    dev = str(resolver_device(device))
    nll = _mz().nll_por_pieza if por_pieza else None
    res: dict = {"val_trozos": len(ev), "evaluaciones": {}}
    pares = None
    for ruta in ckpts:
        if not os.path.exists(ruta):
            print("no existe:", ruta, flush=True)
            continue
        m, voc, op = cargar_modelo(ruta, dev)
        if pares is None:
            pares = trozos_test(cache_test, datos, voc, op, m.c.ctx)
            res["test_trozos"] = len(pares)
            res["test_piezas"] = len({i for i, _ in pares})
        with torch.no_grad():
            fila = {"val": bits_por_nota(m, ev), "test": bits_por_nota(m, [t for _, t in pares])}
            if nll is not None:
                fila["por_pieza"] = nll(m, pares)
        clave = ruta.replace(os.sep, "/")
        res["evaluaciones"][clave] = fila
        del m
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"{clave}: val {fila['val']['bits_por_nota']:.4f} test {fila['test']['bits_por_nota']:.4f}", flush=True)
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Bits por nota en validación y test completos de uno o más checkpoints")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--cache", default="cache/eco_rel_1024.pkl")
    ap.add_argument("--cache_test", default="cache/test_eco_rel_1024.pkl")
    ap.add_argument("--datos", default="data/corpus_pd.jsonl", help="solo si hay que construir la caché de test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--por_pieza", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = evaluar(a.ckpts, a.cache, a.cache_test, a.datos, a.device, a.por_pieza)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
