"""Métricas simbólicas calculadas sobre NOTAS decodificadas (comparables entre representaciones)."""
from __future__ import annotations

import math
from collections import Counter

from .tokenizer import ESCALAS


def _js(p: Counter, q: Counter) -> float:
    claves = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    js = 0.0
    for k in claves:
        a, b = p[k] / sp, q[k] / sq
        m = (a + b) / 2
        if a:
            js += 0.5 * a * math.log2(a / m)
        if b:
            js += 0.5 * b * math.log2(b / m)
    return js


def segmentos(notas, inicios_compas):
    """notas [(on,dur,pitch,parte)] → {(b,parte): tupla((pos,pitch,dur))}."""
    import bisect
    out = {}
    for on, dur, pitch, p in notas:
        b = bisect.bisect_right(inicios_compas, on) - 1
        out.setdefault((b, p), []).append((on - inicios_compas[b], pitch, dur))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def repeticion(segs: dict, ventana: int = 32) -> float:
    """Fracción de segmentos (≥2 notas) que repiten, salvo transposición, otro segmento previo en la ventana."""
    total = rep = 0
    vistos = {}
    for (b, p) in sorted(segs):
        s = segs[(b, p)]
        if len(s) < 2:
            continue
        total += 1
        forma = tuple((x[0], x[1] - s[0][1], x[2]) for x in s)
        if forma in vistos and b - vistos[forma] <= ventana:
            rep += 1
        vistos[forma] = b
    return rep / total if total else 0.0


def perfil(notas, inicios_compas, tonica: int, modo: str) -> dict:
    if not notas:
        return {}
    pcs = Counter((p - tonica) % 12 for _, _, p, _ in notas)
    esc = ESCALAS[modo]
    dur = Counter(d for _, d, _, _ in notas)
    ons = Counter(on for on, _, _, _ in notas)
    segs = segmentos(notas, inicios_compas)
    return {
        "pcs": pcs, "dur": dur,
        "en_escala": sum(v for k, v in pcs.items() if k in esc) / len(notas),
        "polifonia": len(notas) / len(ons),
        "repeticion": repeticion(segs),
        "notas_por_compas": len(notas) / max(1, len({b for b, _ in segs})),
        "ambito": max(p for _, _, p, _ in notas) - min(p for _, _, p, _ in notas),
    }


def comparar(perfiles_gen: list[dict], perfiles_ref: list[dict]) -> dict:
    """Distancias JS de histogramas agregados + medias de escalares (gen vs ref)."""
    def agg(ps, k):
        c = Counter()
        for p in ps:
            c.update(p[k])
        return c

    def media(ps, k):
        v = [p[k] for p in ps if p]
        return sum(v) / len(v) if v else float("nan")

    g = [p for p in perfiles_gen if p]
    r = [p for p in perfiles_ref if p]
    out = {"js_clases_altura": _js(agg(g, "pcs"), agg(r, "pcs")),
           "js_duraciones": _js(agg(g, "dur"), agg(r, "dur"))}
    for k in ("en_escala", "polifonia", "repeticion", "notas_por_compas", "ambito"):
        out[f"{k}_gen"] = media(g, k)
        out[f"{k}_ref"] = media(r, k)
    return out


def copia_ngramas(notas_gen_por_pieza, indice_train: set, n: int = 4) -> float:
    """Fracción de ventanas de n compases (contenido completo, normalizado a la 1.ª nota) presentes en train."""
    tot = hit = 0
    for segs in notas_gen_por_pieza:
        if not segs:
            continue
        b0, b1 = min(b for b, _ in segs), max(b for b, _ in segs)
        for i in range(b0, b1 - n + 2):          # compases consecutivos
            clave = _clave_ventana(segs, list(range(i, i + n)))
            if clave is None:
                continue
            tot += 1
            hit += clave in indice_train
    return hit / tot if tot else 0.0


def _clave_ventana(segs, bloque):
    filas = []
    ref = None
    for j, b in enumerate(bloque):
        for (bb, p), s in sorted(segs.items()):
            if bb == b:
                if ref is None and s:
                    ref = s[0][1]
                filas.append((j, p, s))
    if ref is None or sum(len(s) for *_, s in filas) < 8:
        return None
    return hash(tuple((j, tuple((x[0], x[1] - ref, x[2]) for x in s)) for j, p, s in filas))


def indice_ngramas(segs_por_pieza, n: int = 4) -> set:
    idx = set()
    for segs in segs_por_pieza:
        if not segs:
            continue
        b0, b1 = min(b for b, _ in segs), max(b for b, _ in segs)
        for i in range(b0, b1 - n + 2):
            k = _clave_ventana(segs, list(range(i, i + n)))
            if k is not None:
                idx.add(k)
    return idx
