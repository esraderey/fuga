"""Regenera las muestras de un checkpoint (mismas semillas que `fuga.sample`) y mide, pieza a pieza, copia de ventanas de
2 y 4 compases frente al train, notas en la escala declarada y repetición de segmentos; añade la base de la música real.

Uso: python scripts/analizar_muestras.py --ckpt runs/x/modelo_mejor.pt --n 12 --out runs/x/analisis_muestras.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from fuga.data import cargar, particion  # noqa: E402
from fuga.metrics import copia_ngramas, indice_ngramas, repeticion  # noqa: E402
from fuga.sample import cargar_modelo, generar  # noqa: E402
from fuga.tokenizer import ESCALAS, segmentar  # noqa: E402
from fuga.train import resolver_device  # noqa: E402


def segs_dict(segs) -> dict:
    return {(b, p): tuple(s) for b, sb in enumerate(segs) for p, s in sb.items() if s}


def en_escala(notas, tonica: int, modo: str) -> float:
    esc = ESCALAS[modo]
    return sum(((p - tonica) % 12) in esc for _, _, p, _ in notas) / len(notas) if notas else float("nan")


def _media(xs) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def analizar(ckpt: str, n: int, datos: str, device: str = "auto", compases: int = 16) -> dict:
    t0 = time.time()
    piezas = cargar(datos)
    tr = [segs_dict(segmentar(p)[1]) for p in piezas if particion(p) == "train"]
    te_p = [p for p in piezas if particion(p) == "test"]
    te = [segs_dict(segmentar(p)[1]) for p in te_p]
    idx4, idx2 = indice_ngramas(tr, 4), indice_ngramas(tr, 2)
    base = {"test_real_copia4": copia_ngramas(te, idx4), "test_real_copia2": copia_ngramas(te, idx2, 2),
            "test_real_repeticion": _media(repeticion(s) for s in te),
            "test_real_en_escala": _media(en_escala(p["notas"], p["tonica"], p["modo"]) for p in te_p)}
    print(f"índice: {len(idx4)} ventanas de 4 compases; {time.time() - t0:.0f}s", flush=True)
    model, voc, op = cargar_modelo(ckpt, str(resolver_device(device)))
    filas = []
    for k in range(n):
        dec = generar(model, voc, op["eco"], None, model.c.ctx, compases, 1.0, 0.95, 0.0, semilla=k,
                      relativo=op["relativo"], fantasma=op["fantasma"])
        notas = dec.notas()
        segs = segs_dict(dec.segs)
        filas.append({"muestra": k, "notas": len(notas), "compases": dec.bar + 1, "fin": dec.fin,
                      "partes": len({q for *_, q in notas}), "tonica": dec.tonica_abs, "modo": dec.modo,
                      "tempo": round(dec.tempo), "copia4": copia_ngramas([segs], idx4), "copia2": copia_ngramas([segs], idx2, 2),
                      "en_escala": round(en_escala(notas, dec.tonica_abs, dec.modo), 3) if notas else None,
                      "repeticion": round(repeticion(segs), 3),
                      "ambito": [min(p for *_, p, _ in notas), max(p for *_, p, _ in notas)] if notas else None})
        print(filas[-1], flush=True)
    return {"ckpt": ckpt.replace(os.sep, "/"), "compases": compases, "base_real": base, "muestras": filas}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Copia, escala y repetición de las muestras de un checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--compases", type=int, default=16)
    ap.add_argument("--datos", default="data/corpus_pd.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analizar(a.ckpt, a.n, a.datos, a.device, a.compases)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("BASE REAL", json.dumps(res["base_real"]), flush=True)
    return res


if __name__ == "__main__":
    main()
