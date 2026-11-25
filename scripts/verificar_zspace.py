"""Verifica el almacén Z-Space de una corrida: tamaño frente a `torch.save`, coste por versión y reconstrucción bit a
bit de `modelo.pt` (y de `modelo_mejor.pt` si hay versión en ese paso), ambos redondeados al dtype del almacén.

Uso: python scripts/verificar_zspace.py --run runs/x      # escribe runs/x/zspace_verificacion.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from fuga.zcheckpoints import DTYPES, AlmacenZ  # noqa: E402


def verificar(run: str) -> dict:
    with open(os.path.join(run, "log.json"), encoding="utf-8") as f:
        log = json.load(f)
    alm = AlmacenZ(os.path.join(run, "zspace"))
    vs = alm.versiones
    ult = vs[-1]
    eq = ult["bytes_equivalente_torch"]
    factor_fp32 = 4 // torch.empty(0, dtype=DTYPES[alm.dtype]).element_size()
    res = {"dtype": alm.dtype, "versiones": len(vs), "completas": sum(v["completo"] for v in vs),
           "bytes_tienda": ult["bytes_tienda"], "bytes_equivalente_torch": eq,
           "bytes_equivalente_torch_fp32": eq * factor_fp32,
           "ratio_vs_mismo_dtype": ult["bytes_tienda"] / eq, "ratio_vs_fp32": ult["bytes_tienda"] / (eq * factor_fp32),
           "bytes_por_version": [v["bytes_version"] for v in vs], "pasos": [v["paso"] for v in vs]}
    for nombre, paso in (("modelo.pt", log["final"]["paso"]), ("modelo_mejor.pt", log.get("mejor", {}).get("paso"))):
        ruta = os.path.join(run, nombre)
        if paso is None or not os.path.exists(ruta) or not any(v["paso"] == paso for v in vs):
            res[f"exacto_{nombre}"] = None
            continue
        ck = torch.load(ruta, map_location="cpu", weights_only=False)["modelo"]
        ref = {k: v.to(DTYPES[alm.dtype]) for k, v in ck.items()}
        z = alm.cargar(paso=paso)
        res[f"exacto_{nombre}"] = bool(ref.keys() == z.keys() and all(torch.equal(ref[k], z[k]) for k in ref))
    alm.cerrar()
    cadena = os.path.join(os.path.dirname(os.path.abspath(run)), "cadena.log")
    if os.path.exists(cadena):   # segundos por versión, tal como los imprime fuga.train
        with open(cadena, encoding="utf-8", errors="replace") as f:
            seg = [float(x) for x in re.findall(r"zspace v\d+ paso \d+: .*?, ([0-9.]+)s", f.read())]
        if seg:
            res["segundos_por_version"] = {"n": len(seg), "media": sum(seg) / len(seg), "max": max(seg), "suma": sum(seg)}
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Tamaño y exactitud del almacén Z-Space de una corrida")
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    res = verificar(a.run)
    with open(os.path.join(a.run, "zspace_verificacion.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("bytes_por_version", "pasos")}, indent=1), flush=True)
    return res


if __name__ == "__main__":
    main()
