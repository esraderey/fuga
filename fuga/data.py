"""Construcción de secuencias de entrenamiento/evaluación y lotes por presupuesto de tokens."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from .tokenizer import Vocab, codificar, segmentar

REPR = ("base", "eco", "fant")  # literal · ECO · ECO con contenido materializado


def cargar(ruta: str) -> list[dict]:
    with open(ruta) as f:
        return [json.loads(linea) for linea in f]


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def particion(pieza: dict) -> str:
    """train/val/test estable por archivo de origen (evita fuga entre piezas de un mismo libro)."""
    h = _h(pieza["id"].split("#")[0]) % 100
    return "val" if h < 5 else "test" if h < 10 else "train"


RANGO_TRANSP = range(-5, 7)


def transposiciones_validas(pieza: dict, rango=RANGO_TRANSP, lo: int = 21, hi: int = 108) -> list[int]:
    pmin = min(n[2] for n in pieza["notas"])
    pmax = max(n[2] for n in pieza["notas"])
    return [t for t in rango if pmin + t >= lo and pmax + t <= hi] or [0]


def transposiciones(pieza: dict, n: int) -> list[int]:
    """n transposiciones deterministas por pieza (mismas en todos los brazos)."""
    v = transposiciones_validas(pieza)
    return sorted(random.Random(_h(pieza["id"])).sample(v, min(n, len(v))))


def _flags(repr_: str) -> dict:
    return {"eco": repr_ != "base", "fantasma": repr_ == "fant"}


def secuencias(piezas, voc: Vocab, repr_: str, ctx: int, n_transp: int = 2, relativo: bool = False):
    """Trozos consecutivos (por compases) de cada pieza y transposición. Returns [(ids, compas, perdida|None)]."""
    fl = _flags(repr_)
    out = []
    for pz in piezas:
        for t in ([0] if relativo else transposiciones(pz, n_transp)):
            seg = segmentar(pz, t, relativo)
            b = 0
            while b < len(seg[1]):
                c, b2 = codificar(pz, voc, t=t, b0=b, max_tokens=ctx, segmentado=seg, relativo=relativo, **fl)
                if len(c.ids) > 8:
                    per = None if all(c.perdida) else np.array(c.perdida, np.bool_)
                    out.append((np.array(c.ids, np.int16), np.array(c.compas, np.int16), per))
                b = max(b2, b + 1)
    return out


def trozos_eval(piezas, voc: Vocab, ctx: int, relativo: bool = False):
    """Mismos rangos de compases en las tres representaciones: el rango debe caber en ctx en todas.

    Returns {repr: [(ids, compas, perdida|None, notas)]}."""
    res = {r: [] for r in REPR}
    for pz in piezas:
        seg_abs = segmentar(pz, 0)
        seg = segmentar(pz, 0, relativo) if relativo else seg_abs
        b = 0
        while b < len(seg_abs[1]):
            _, b2 = codificar(pz, voc, eco=False, b0=b, max_tokens=ctx, segmentado=seg_abs)
            b2 = max(b2, b + 1)
            while True:
                cods = {r: codificar(pz, voc, b0=b, b1=b2, segmentado=seg, relativo=relativo, **_flags(r))[0]
                        for r in REPR}
                if b2 == b + 1 or all(len(c.ids) <= ctx for c in cods.values()):
                    break
                b2 -= 1
            notas = cods["base"].notas
            for r, c in cods.items():
                ids, comp, per = c.ids[:ctx], c.compas[:ctx], c.perdida[:ctx]
                res[r].append((np.array(ids, np.int16), np.array(comp, np.int16),
                               None if all(per) else np.array(per, np.bool_), notas))
            b = b2
    return res


def lotes(seqs, tokens_por_lote: int, rng: random.Random):
    """Agrupa por longitud similar; genera índices de lote hasta agotar una época."""
    orden = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + rng.random())
    grupos, actual, maxlen = [], [], 0
    for i in orden:
        L = len(seqs[i][0])
        if actual and max(maxlen, L) * (len(actual) + 1) > tokens_por_lote:
            grupos.append(actual)
            actual, maxlen = [], 0
        actual.append(i)
        maxlen = max(maxlen, L)
    if actual:
        grupos.append(actual)
    rng.shuffle(grupos)
    return grupos


def colar(seqs, idx, pad: int = 0):
    """Returns ids (B,T), pos (B,2,T), mascara de pérdida (B,T): True donde el token se predice."""
    L = max(len(seqs[i][0]) for i in idx)
    B = len(idx)
    ids = torch.full((B, L), pad, dtype=torch.long)
    comp = torch.zeros((B, L), dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    for j, i in enumerate(idx):
        a, c = seqs[i][0], seqs[i][1]
        per = seqs[i][2] if len(seqs[i]) > 2 else None
        ids[j, : len(a)] = torch.from_numpy(a.astype(np.int64))
        comp[j, : len(c)] = torch.from_numpy(c.astype(np.int64))
        comp[j, len(c):] = int(c[-1])
        m[j, : len(a)] = True if per is None else torch.from_numpy(per)
    tok = torch.arange(L).expand(B, L)
    return ids, torch.stack([tok, comp], 1), m
