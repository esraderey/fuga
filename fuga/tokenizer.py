"""Representación FUGA: compás → partes (orden por parte) → notas; con tokens ECO opcionales.

Segmento (compás b, parte p) se codifica como:
  literal : PART_p (POS_x PITCH_m DUR_d | PITCH_m DUR_d)*
  ECO     : PART_p ECO  DIST_d FROM_q (TRC_t | TRD_k)   copia exacta transpuesta (cromática o diatónica)
  ECOR    : PART_p ECOR DIST_d FROM_q PITCH_m×n        copia el ritmo de la fuente, alturas nuevas
DIST_0 permite referir partes anteriores del mismo compás (doblajes).
"""
from __future__ import annotations

import bisect
import math
import warnings
from dataclasses import dataclass, field

MAX_POS = 192          # ticks dentro de un compás (16 negras a 12 tpq)
MAX_PARTES = 16
MAX_DIST = 32
DUR_BINS = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 168, 192, 240, 288, 384]
ESCALAS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
TEMPO_BINS = 16


def bin_dur(d: int) -> int:
    i = bisect.bisect_left(DUR_BINS, d)
    if i == 0:
        return 0
    if i == len(DUR_BINS):
        return len(DUR_BINS) - 1
    return i if DUR_BINS[i] - d < d - DUR_BINS[i - 1] else i - 1


def bin_tempo(bpm: float) -> int:
    x = (math.log(max(40.0, min(200.0, bpm))) - math.log(40)) / (math.log(200) - math.log(40))
    return min(TEMPO_BINS - 1, int(x * TEMPO_BINS))


def tempo_de_bin(b: int) -> float:
    return math.exp(math.log(40) + (b + 0.5) / TEMPO_BINS * (math.log(200) - math.log(40)))


class Vocab:
    def __init__(self, estilos: list[str]):
        t = ["<pad>", "<bos>", "<eos>", "BAR", "ECO", "ECOR"]
        t += [f"STY_{e}" for e in estilos]
        t += [f"KEY_{k}_{m}" for m in ("major", "minor") for k in range(12)]
        t += [f"TEMPO_{i}" for i in range(TEMPO_BINS)]
        t += [f"BL_{i}" for i in range(1, MAX_POS + 1)]
        t += [f"PART_{i}" for i in range(MAX_PARTES)]
        t += [f"POS_{i}" for i in range(MAX_POS)]
        t += [f"PITCH_{i}" for i in range(128)]
        t += [f"DUR_{i}" for i in range(len(DUR_BINS))]
        t += [f"DIST_{i}" for i in range(MAX_DIST + 1)]
        t += [f"FROM_{i}" for i in range(MAX_PARTES)]
        t += [f"TRC_{i}" for i in range(-12, 13)]
        t += [f"TRD_{i}" for i in range(-7, 8)]
        self.itos = t
        self.stoi = {s: i for i, s in enumerate(t)}
        self.estilos = estilos

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, s: str) -> int:
        return self.stoi[s]

    def rango(self, pref: str) -> tuple[int, int]:
        ids = [i for i, s in enumerate(self.itos) if s.startswith(pref)]
        return ids[0], ids[-1] + 1


# ---------------------------------------------------------------- utilidades musicales

def grado(pitch: int, tonica: int, modo: str):
    """Grado diatónico global o None si la nota no pertenece a la escala."""
    rel = pitch - tonica
    pc = rel % 12
    esc = ESCALAS[modo]
    if pc not in esc:
        return None
    return (rel // 12) * 7 + esc.index(pc)


def de_grado(g: int, tonica: int, modo: str) -> int:
    return tonica + (g // 7) * 12 + ESCALAS[modo][g % 7]


def transformar(fuente, tipo: str, k: int, tonica: int, modo: str):
    """Aplica TRC/TRD a [(pos, pitch, dur)]; None si sale de rango o no es diatónica."""
    out = []
    for pos, p, d in fuente:
        if tipo == "TRC":
            q = p + k
        else:
            g = grado(p, tonica, modo)
            if g is None:
                return None
            q = de_grado(g + k, tonica, modo)
        if not 0 <= q <= 127:
            return None
        out.append((pos, q, d))
    return out


def len_literal(seg) -> int:
    n, ult = 0, None
    for pos, _, _ in seg:
        n += 2 + (pos != ult)
        ult = pos
    return n


# ---------------------------------------------------------------- pieza → compases

def compases_de(pieza: dict) -> tuple[list[int], list[int]]:
    """Inicios y largos de compás, extendidos hasta cubrir todas las notas; compases > MAX_POS se parten.

    Si el primer compás declarado no empieza en 0 (anacrusa o inicio tardío) se antepone un compás parcial [0, ini[0])
    para que ninguna nota se pierda y la ida y vuelta conserve los ticks absolutos."""
    ini = sorted(set(pieza["compases"]))
    fin = max(o + 1 for o, _, _, _ in pieza["notas"])
    paso = (ini[-1] - ini[-2]) if len(ini) > 1 else 48
    paso = max(1, paso)
    if ini[0] > 0:
        ini.insert(0, 0)
    while ini[-1] + paso <= fin:
        ini.append(ini[-1] + paso)
    largos = [b - a for a, b in zip(ini, ini[1:], strict=False)] + [paso]
    inis2, lar2 = [], []
    for a, lar in zip(ini, largos, strict=True):
        while lar > MAX_POS:
            inis2.append(a)
            lar2.append(MAX_POS)
            a, lar = a + MAX_POS, lar - MAX_POS
        if lar > 0:
            inis2.append(a)
            lar2.append(lar)
    return inis2, lar2


def desplazamiento_tonica(tonica: int) -> int:
    """Tónica como desplazamiento mínimo respecto a Do, en [-6, 5] (marco de MARCO)."""
    return tonica if tonica <= 5 else tonica - 12


def segmentar(pieza: dict, t: int = 0, relativo: bool = False):
    """Devuelve (largos, segs) con segs[b][p] = [(pos, pitch, dur_cuantizada)] ordenado.

    relativo=True: alturas relativas a la tónica (pitch − desplazamiento), sin pérdida."""
    ini, largos = compases_de(pieza)
    segs = [dict() for _ in ini]
    fuera = 0
    for on, dur, pitch, parte in pieza["notas"]:
        if parte >= MAX_PARTES or on < ini[0]:
            continue
        b = bisect.bisect_right(ini, on) - 1
        pos = on - ini[b]
        if pos >= largos[b]:
            continue
        q = pitch + t - (desplazamiento_tonica((pieza["tonica"] + t) % 12) if relativo else 0)
        if 0 <= q <= 127:
            segs[b].setdefault(parte, []).append((pos, q, DUR_BINS[bin_dur(dur)]))
        else:
            fuera += 1
    if fuera:
        warnings.warn(f"{pieza.get('id', '?')}: {fuera} notas fuera de 0..127 descartadas (t={t}, relativo={relativo})",
                      stacklevel=2)
    for s in segs:
        for p in s:
            unico = {}
            for pos, pitch, dur in s[p]:            # misma (pos, altura): se queda la más larga
                unico[(pos, pitch)] = max(dur, unico.get((pos, pitch), 0))
            s[p] = sorted((pos, pitch, dur) for (pos, pitch), dur in unico.items())
    return largos, segs


# ---------------------------------------------------------------- codificación

@dataclass
class Codigo:
    ids: list[int] = field(default_factory=list)
    compas: list[int] = field(default_factory=list)   # índice de compás por token (eje musical de RoPE)
    perdida: list[int] = field(default_factory=list)  # 0 = token fantasma (determinista, sin pérdida)
    notas: int = 0
    stats: dict = field(default_factory=lambda: {"lit": 0, "eco": 0, "ecor": 0})


def literal(voc: Vocab, seg) -> list[int]:
    """Tokens literales de un segmento: (POS PITCH DUR | PITCH DUR)*."""
    out, ult = [], None
    for pos, pitch, dur in seg:
        if pos != ult:
            out.append(voc[f"POS_{pos}"])
            ult = pos
        out += [voc[f"PITCH_{pitch}"], voc[f"DUR_{bin_dur(dur)}"]]
    return out


def _buscar_eco(segs, b, p, tonica, modo, b0):
    """Mejor referencia para segs[b][p]: ('ECO', d, q, tipo, k) | ('ECOR', d, q) | None."""
    seg = segs[b][p]
    ritmo = [(x[0], x[2]) for x in seg]
    lit = len_literal(seg)
    mejor_r = None
    for d in range(0, min(MAX_DIST, b - b0) + 1):
        fuente_b = segs[b - d]
        cand = sorted(fuente_b) if d else [q for q in sorted(fuente_b) if q < p]
        cand.sort(key=lambda q: (q != p, q))
        for q in cand:
            src = fuente_b[q]
            if len(src) != len(seg) or [(x[0], x[2]) for x in src] != ritmo:
                continue
            difs = {s[1] - x[1] for s, x in zip(seg, src, strict=True)}
            if len(difs) == 1 and abs(next(iter(difs))) <= 12 and lit > 4:
                return ("ECO", d, q, "TRC", next(iter(difs)))
            gs = [grado(x[1], tonica, modo) for x in src]
            gd = [grado(x[1], tonica, modo) for x in seg]
            if None not in gs and None not in gd:
                kk = {a - c for a, c in zip(gd, gs, strict=True)}
                if len(kk) == 1 and abs(next(iter(kk))) <= 7 and lit > 4:
                    return ("ECO", d, q, "TRD", next(iter(kk)))
            if mejor_r is None and 3 + len(seg) < lit:
                mejor_r = ("ECOR", d, q)
    return mejor_r


def codificar(pieza: dict, voc: Vocab, t: int = 0, eco: bool = True, b0: int = 0,
              max_tokens: int | None = None, segmentado=None, b1: int | None = None,
              relativo: bool = False, fantasma: bool = False) -> tuple[Codigo, int]:
    """Codifica compases [b0, b1) o hasta llenar max_tokens (cortando en límite de compás).

    fantasma=True: tras cada ECO/ECOR se materializa el contenido referido como tokens literales sin pérdida.
    Returns (Codigo, compás siguiente no incluido)."""
    largos, segs = segmentado or segmentar(pieza, t, relativo)
    n_total = len(segs)
    if b1 is not None:
        segs, largos = segs[:b1], largos[:b1]
    tonica_abs, modo = (pieza["tonica"] + t) % 12, pieza["modo"]
    tonica = 0 if relativo else tonica_abs   # en modo relativo la escala se calcula sobre Do
    c = Codigo()
    sty = f"STY_{pieza['fuente']}"
    if sty not in voc.stoi:
        raise ValueError(f"estilo {pieza['fuente']!r} fuera del vocabulario (estilos: {voc.estilos})")
    cab = [voc["<bos>"], voc[sty], voc[f"KEY_{tonica_abs}_{modo}"], voc[f"TEMPO_{bin_tempo(pieza['tempo'])}"]]
    c.ids += cab
    c.compas += [0] * len(cab)
    c.perdida += [1] * len(cab)
    bl_prev = None
    b = b0
    while b < len(segs):
        bi = b - b0 + 1  # compás 0 = cabecera
        tok = [voc["BAR"]]
        if largos[b] != bl_prev:
            tok.append(voc[f"BL_{largos[b]}"])
        msk = [1] * len(tok)
        st = {"lit": 0, "eco": 0, "ecor": 0}
        n_notas = 0
        for p in sorted(segs[b]):
            seg = segs[b][p]
            n_notas += len(seg)
            ref = _buscar_eco(segs, b, p, tonica, modo, b0) if eco else None
            if ref and ref[0] == "ECO":
                _, d, q, tipo, k = ref
                nuevos = [voc[f"PART_{p}"], voc["ECO"], voc[f"DIST_{d}"], voc[f"FROM_{q}"], voc[f"{tipo}_{k}"]]
                fant = literal(voc, seg) if fantasma else []
                st["eco"] += 1
                tok += nuevos + fant
                msk += [1] * len(nuevos) + [0] * len(fant)
            elif ref:
                _, d, q = ref
                cab_r = [voc[f"PART_{p}"], voc["ECOR"], voc[f"DIST_{d}"], voc[f"FROM_{q}"]]
                fant = literal(voc, segs[b - d][q]) if fantasma else []
                alt = [voc[f"PITCH_{x[1]}"] for x in seg]
                tok += cab_r + fant + alt
                msk += [1] * len(cab_r) + [0] * len(fant) + [1] * len(alt)
                st["ecor"] += 1
            else:
                nuevos = [voc[f"PART_{p}"]] + literal(voc, seg)
                tok += nuevos
                msk += [1] * len(nuevos)
                st["lit"] += 1
        if max_tokens and len(c.ids) + len(tok) + 1 > max_tokens:
            if b == b0:  # un compás que no cabe: se trunca
                n = max_tokens - len(c.ids) - 1
                c.ids += tok[:n]
                c.compas += [bi] * len(tok[:n])
                c.perdida += msk[:n]
                b += 1
            break
        c.ids += tok
        c.compas += [bi] * len(tok)
        c.perdida += msk
        c.notas += n_notas
        for k_, v in st.items():
            c.stats[k_] += v
        bl_prev = largos[b]
        b += 1
    if b >= n_total and (not max_tokens or len(c.ids) < max_tokens):
        c.ids.append(voc["<eos>"])
        c.compas.append(c.compas[-1])
        c.perdida.append(1)
    return c, b


# ---------------------------------------------------------------- decodificador incremental + gramática

class Decodificador:
    """Consume tokens uno a uno; mantiene el contenido resuelto y la gramática (tokens permitidos)."""

    def __init__(self, voc: Vocab, eco: bool = True, relativo: bool = False, fantasma: bool = False):
        self.v, self.eco, self.relativo, self.fantasma = voc, eco, relativo, fantasma
        self.cola: list[int] = []
        self.tras_cola = None
        self.desp = 0
        self.tonica_abs = 0
        self.fase = "bos"
        self.tonica, self.modo, self.tempo = 0, "major", 90.0
        self.bar = -1
        self.bl = 48
        self.inicio = 0          # tick absoluto del compás actual
        self.inicios: list[int] = []
        self.segs: list[dict] = []
        self.p = -1
        self.modo_seg = None     # 'lit' | 'eco' | 'ecor'
        self.ult_pos = -1
        self.ult_pitch = -1
        self.pend = None         # estado de ECO/ECOR en curso
        self.fin = False
        r = voc.rango
        self.R = {k: r(k) for k in ("STY_", "KEY_", "TEMPO_", "BL_", "PART_", "POS_", "PITCH_", "DUR_",
                                    "DIST_", "FROM_", "TRC_", "TRD_")}

    def _ids(self, pref, lo=None, hi=None):
        a, b = self.R[pref]
        return list(range(a + (lo or 0), a + (hi if hi is not None else b - a)))

    # -- consulta
    def _alturas(self, desde: int = 0) -> list[int]:
        """PITCH permitidos; en modo relativo, solo los que decodifican dentro de 0..127."""
        lo, hi = max(desde, -self.desp), min(128, 128 - self.desp)
        a = self.R["PITCH_"][0]
        return list(range(a + lo, a + hi))

    def _transf_ok(self, src, tipo, k) -> bool:
        seg = transformar(src, tipo, k, self.tonica, self.modo)
        return bool(seg) and all(0 <= x[1] + self.desp <= 127 for x in seg)

    def permitidos(self) -> list[int]:
        v = self.v
        f = self.fase
        if self.fin:
            return []
        if f == "bos":
            return [v["<bos>"]]
        if f == "sty":
            return self._ids("STY_")
        if f == "key":
            return self._ids("KEY_")
        if f == "tempo":
            return self._ids("TEMPO_")
        cierre = [v["BAR"]] + ([v["<eos>"]] if self.bar >= 0 else [])
        partes = self._ids("PART_", self.p + 1) if self.bar >= 0 else []
        if f == "post_bar":
            return self._ids("BL_") + partes + cierre
        if f == "post_bl":
            return partes + cierre
        if f == "part":
            out = [v[f"POS_{i}"] for i in range(self.bl)]
            if self.eco and self._fuentes_validas(None):
                out += [v["ECO"], v["ECOR"]]
            return out
        if f == "pos":
            return self._alturas()
        if f == "pitch":
            return self._ids("DUR_")
        if f == "dur":
            out = [v[f"POS_{i}"] for i in range(self.ult_pos + 1, self.bl)]
            out += self._alturas(self.ult_pitch + 1)
            return out + partes + cierre
        if f == "eco_dist":
            return [v[f"DIST_{d}"] for d in self._dists_validas()]
        if f == "eco_from":
            d = self.pend["d"]
            return [v[f"FROM_{q}"] for q in self._fuentes_validas(d)]
        if f == "eco_tr":
            src = self._fuente()
            out = [v[f"TRC_{k}"] for k in range(-12, 13) if self._transf_ok(src, "TRC", k)]
            out += [v[f"TRD_{k}"] for k in range(-7, 8) if self._transf_ok(src, "TRD", k)]
            return out
        if f == "ecor_pitch":
            return self._alturas()
        if f == "fantasma":
            return [self.cola[0]]
        if f == "seg_fin":
            return partes + cierre
        raise RuntimeError(f)

    def _fuentes_validas(self, d):
        ds = [d] if d is not None else range(0, min(MAX_DIST, self.bar) + 1)
        out = []
        for dd in ds:
            if self.bar - dd < 0:
                continue
            s = self.segs[self.bar - dd]
            out += [q for q in s if s[q] and (dd > 0 or q < self.p)]
            if d is None and out:
                return out
        return out

    def _dists_validas(self):
        return [d for d in range(0, min(MAX_DIST, self.bar) + 1) if self._fuentes_validas(d)]

    def _fuente(self):
        return self.segs[self.bar - self.pend["d"]][self.pend["q"]]

    # -- avance
    def paso(self, i: int):
        s = self.v.itos[i]
        f = self.fase
        if f == "fantasma":
            if i != self.cola[0]:
                raise ValueError(f"fantasma esperado {self.v.itos[self.cola[0]]}, llegó {s}")
            self.cola.pop(0)
            if not self.cola:
                self.fase = self.tras_cola
        elif f == "bos":
            self.fase = "sty"
        elif f == "sty":
            self.fase = "key"
        elif f == "key":
            _, k, m = s.split("_")
            self.tonica, self.modo = int(k), m
            self.tonica_abs = int(k)
            if self.relativo:
                self.desp = desplazamiento_tonica(self.tonica)
                self.tonica = 0
            self.fase = "tempo"
        elif f == "tempo":
            self.tempo = tempo_de_bin(int(s.split("_")[1]))
            self.fase = "post_bl"
        elif s == "<eos>":
            self.fin = True
        elif s == "BAR":
            if self.bar >= 0:
                self.inicio += self.bl
            self.bar += 1
            self.inicios.append(self.inicio)
            self.segs.append({})
            self.p, self.fase = -1, "post_bar"
        elif s.startswith("BL_"):
            self.bl = int(s[3:])
            self.fase = "post_bl"
        elif s.startswith("PART_"):
            self.p = int(s[5:])
            self.segs[self.bar][self.p] = []
            self.ult_pos, self.ult_pitch, self.fase = -1, -1, "part"
        elif s.startswith("POS_"):
            self.ult_pos, self.ult_pitch, self.fase = int(s[4:]), -1, "pos"
        elif s.startswith("PITCH_") and f in ("pos", "dur"):
            self.ult_pitch, self.fase = int(s[6:]), "pitch"
        elif s.startswith("DUR_"):
            self.segs[self.bar][self.p].append((self.ult_pos, self.ult_pitch, DUR_BINS[int(s[4:])]))
            self.fase = "dur"
        elif s in ("ECO", "ECOR"):
            self.pend = {"tipo": s}
            self.fase = "eco_dist"
        elif s.startswith("DIST_"):
            self.pend["d"] = int(s[5:])
            self.fase = "eco_from"
        elif s.startswith("FROM_"):
            self.pend["q"] = int(s[5:])
            if self.pend["tipo"] == "ECO":
                self.fase = "eco_tr"
            else:
                self.pend["ritmo"] = [(x[0], x[2]) for x in self._fuente()]
                self.pend["k"] = 0
                self.fase = "ecor_pitch"
                if self.fantasma:
                    self.cola, self.tras_cola, self.fase = literal(self.v, self._fuente()), "ecor_pitch", "fantasma"
        elif s.startswith("TRC_") or s.startswith("TRD_"):
            seg = transformar(self._fuente(), s[:3], int(s[4:]), self.tonica, self.modo)
            self.segs[self.bar][self.p] = list(seg or [])
            self.fase = "seg_fin"
            if self.fantasma and seg:
                self.cola, self.tras_cola, self.fase = literal(self.v, seg), "seg_fin", "fantasma"
        elif s.startswith("PITCH_") and f == "ecor_pitch":
            pos, dur = self.pend["ritmo"][self.pend["k"]]
            self.segs[self.bar][self.p].append((pos, int(s[6:]), dur))
            self.pend["k"] += 1
            if self.pend["k"] == len(self.pend["ritmo"]):
                self.segs[self.bar][self.p].sort()   # forma canónica (acordes en orden de altura)
                self.fase = "seg_fin"
        else:
            raise ValueError(f"token {s} inválido en fase {f}")

    def notas(self) -> list[tuple[int, int, int, int]]:
        out = []
        for b, segs in enumerate(self.segs):
            for p, seg in segs.items():
                for pos, pitch, dur in seg:
                    out.append((self.inicios[b] + pos, dur, pitch + self.desp, p))
        return sorted(out)


def decodificar(ids: list[int], voc: Vocab, eco: bool = True, estricto: bool = True,
                relativo: bool = False, fantasma: bool = False) -> Decodificador:
    d = Decodificador(voc, eco, relativo, fantasma)
    for i in ids:
        if estricto and i not in set(d.permitidos()):
            raise ValueError(f"token {voc.itos[i]} no permitido en fase {d.fase}")
        d.paso(i)
        if d.fin:
            break
    return d
