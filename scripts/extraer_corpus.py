"""Extrae el corpus de dominio público de music21 a JSONL de notas cuantizadas.

Uso: python scripts/extraer_corpus.py --out data/corpus_pd.jsonl [--procs 2] [--limite N]
Cada línea: {id, fuente, tpq, compases[], tonica, modo, tempo, partes, notas[[on,dur,pitch,parte]], avisos[]}.
`avisos` registra lo que antes se tragaba en silencio (stripTies fallido, tonalidad no analizada, modo forzado).
El orden de salida es el de `corpus.getCorePaths()` (imap ordenado): determinista con cualquier --procs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

TPQ = 12  # ticks por negra: cubre semicorcheas y tresillos de corchea


def _q(x: float) -> int:
    return int(round(float(x) * TPQ))


def _score_a_pieza(s, pid: str, fuente: str) -> dict | None:
    from music21 import tempo as m21tempo

    avisos = []
    try:
        s = s.stripTies()
    except Exception as e:  # se anota en la pieza, no se oculta
        avisos.append(f"stripTies: {type(e).__name__}")
    partes = list(s.parts) or [s]
    notas = []
    for i, p in enumerate(partes):
        for n in p.flatten().notes:
            on = _q(n.offset)
            dur = max(1, _q(n.duration.quarterLength))
            if n.duration.isGrace:
                continue
            pitches = [x.midi for x in n.pitches] if n.isChord else [n.pitch.midi]
            for m in pitches:
                if 0 <= m <= 127:
                    notas.append((on, dur, int(m), i))
    if len(notas) < 8:
        return None
    notas.sort()
    medidas = partes[0].getElementsByClass("Measure")
    compases = sorted({_q(m.offset) for m in medidas})
    if len(compases) < 2:
        ts = s.recurse().getElementsByClass("TimeSignature").first()
        largo = _q(ts.barDuration.quarterLength) if ts else 4 * TPQ
        fin = max(o + d for o, d, _, _ in notas)
        compases = list(range(0, fin + largo, largo))
    try:
        k = s.analyze("key")
        tonica, modo = int(k.tonic.pitchClass), str(k.mode)
    except Exception as e:
        tonica, modo = 0, "major"
        avisos.append(f"tonalidad no analizada ({type(e).__name__}): C major por defecto")
    if modo not in ("major", "minor"):
        avisos.append(f"modo {modo} forzado a major")
    mm = s.recurse().getElementsByClass(m21tempo.MetronomeMark).first()
    bpm = float(mm.number) if (mm is not None and mm.number) else 90.0
    return {
        "id": pid,
        "fuente": fuente,
        "tpq": TPQ,
        "compases": compases,
        "tonica": tonica,
        "modo": modo if modo in ("major", "minor") else "major",
        "tempo": bpm,
        "partes": len(partes),
        "notas": notas,
        "avisos": avisos,
    }


def _fuente_y_nombre(ruta: str) -> tuple[str, str]:
    """('bach', 'bwv1.6.mxl') a partir de .../corpus/bach/bwv1.6.mxl, con separadores de cualquier sistema."""
    partes = pathlib.PureWindowsPath(ruta).parts   # parte por \ y por /, también en Linux
    i = max(j for j, x in enumerate(partes) if x == "corpus")
    return partes[i + 1], partes[-1]


def _procesar(ruta: str) -> tuple[str, list[dict], str | None]:
    from music21 import converter, stream

    try:
        fuente, nombre = _fuente_y_nombre(ruta)
        obj = converter.parse(ruta)
        scores = list(obj.scores) if isinstance(obj, stream.Opus) else [obj]
        out = []
        for j, s in enumerate(scores):
            pz = _score_a_pieza(s, f"{fuente}/{nombre}#{j}", fuente)
            if pz:
                out.append(pz)
        return ruta, out, None
    except Exception as e:  # pieza ilegible: se reporta, no se oculta
        return ruta, [], f"{type(e).__name__}: {e}"[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    from music21 import corpus

    rutas = [str(p) for p in corpus.getCorePaths() if p.suffix in (".mxl", ".xml", ".krn", ".abc")]
    if a.limite:
        rutas = rutas[: a.limite]
    t0, n_piezas, errores = time.time(), 0, []
    with open(a.out, "w") as f, mp.Pool(a.procs, maxtasksperchild=50) as pool:
        for k, (ruta, piezas, err) in enumerate(pool.imap(_procesar, rutas, chunksize=4)):
            if err:
                errores.append((ruta, err))
            for pz in piezas:
                f.write(json.dumps(pz, separators=(",", ":")) + "\n")
                n_piezas += 1
            if k % 200 == 0:
                print(f"[{k}/{len(rutas)}] piezas={n_piezas} errores={len(errores)} t={time.time()-t0:.0f}s", flush=True)
    print(f"FIN archivos={len(rutas)} piezas={n_piezas} errores={len(errores)} t={time.time()-t0:.0f}s")
    for r, e in errores[:20]:
        print("ERR", r, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
