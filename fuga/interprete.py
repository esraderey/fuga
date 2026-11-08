"""Intérprete FUGA-I, capa determinista: partitura cuantizada → ejecución expresiva → MIDI y WAV.

Entrada: notas [(on, dur, pitch, parte)] en ticks (`tpq` por negra), inicios y largos de compás en ticks (el primer
compás empieza en 0; `tokenizer.compases_de` antepone el compás parcial de una anacrusa) y tempo.
Salida: eventos [(t0_s, dur_s, pitch, vel, parte)] en segundos, con dinámica, alineados uno a uno con `origen`.

Reglas de la familia KTH (acento métrico, arco de frase en dinámica y tempo, high-loud, ritardando final,
articulación por voz, adelanto de la voz principal, microtiempo) parametrizadas por `Estilo`. Todo es determinista
dado (notas, estilo, semilla). Invariantes: no añade ni quita notas distintas (dos notas de la misma voz con igual
ataque y altura se fusionan en una con la duración mayor y se cuentan en `fusionadas`); |t0 − mapa de tempo| ≤
Estilo.max_desvio_s; el orden de ataques dentro de cada voz se conserva y las notas de un acorde comparten ataque; una
altura no sigue sonando cuando la misma voz la vuelve a atacar; el estilo «plano» reproduce la partitura tal cual
(vel 80).

Uso: python -m fuga.interprete --datos data/corpus_pd.jsonl --indice 0 --estilo neutro --out muestras/ [--wav]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Estilo:
    nombre: str = "neutro"
    vel_base: int = 72
    acento_compas: int = 14      # +vel en el primer tiempo del compás
    acento_pulso: int = 6        # +vel en los demás pulsos
    arco_vel: int = 10           # ±vel a lo largo de la frase (máximo en el centro)
    arco_tempo: float = 0.03     # ±fracción de la duración del pulso: más lento al empezar y al cerrar la frase
    frase_compases: int = 4
    high_loud: float = 6.0       # +vel por octava sobre el do central
    voz_principal: int = 6       # +vel de la voz principal (la más aguda en media)
    acomp: int = -6              # +vel de las demás voces
    rit_final: float = 0.35      # alargamiento máximo del pulso en el tramo final
    rit_pulsos: int = 4
    legato: float = 1.0          # razón de duración de la voz principal
    legato_acomp: float = 0.9
    adelanto_s: float = 0.012    # la voz principal ataca antes
    jitter_s: float = 0.006      # desviación típica del microtiempo por ataque
    max_desvio_s: float = 0.05   # tope duro de |t0 − mapa de tempo|
    dur_min_s: float = 0.04      # duración mínima de un evento (0 = exacta); el recorte por repetición de la misma
                                 # altura puede dejarla por debajo


PLANO = Estilo("plano", vel_base=80, acento_compas=0, acento_pulso=0, arco_vel=0, arco_tempo=0.0, high_loud=0.0,
               voz_principal=0, acomp=0, rit_final=0.0, legato=1.0, legato_acomp=1.0, adelanto_s=0.0, jitter_s=0.0,
               dur_min_s=0.0)
ESTILOS: dict[str, Estilo] = {
    "plano": PLANO,
    "neutro": Estilo(),
    "romantico": Estilo("romantico", arco_vel=16, arco_tempo=0.06, rit_final=0.6, rit_pulsos=8, jitter_s=0.009,
                        legato=1.05),
    "mecanico": replace(PLANO, nombre="mecanico", acento_compas=8, acento_pulso=4, legato=0.85, legato_acomp=0.85),
}


@dataclass
class Interpretacion:
    tiempos_pulso: list[float]                          # segundo de inicio de cada pulso, más el final
    tpq: int
    voz_principal: int
    estilo: str
    semilla: int
    eventos: list[tuple[float, float, int, int, int]] = field(default_factory=list)   # (t0_s, dur_s, pitch, vel, parte)
    origen: list[tuple[int, int, int, int]] = field(default_factory=list)             # la nota de partitura de cada evento
    fusionadas: int = 0                                 # notas de entrada fusionadas por repetir (parte, ataque, altura)

    def segundos(self, tick: float) -> float:
        """Mapa de tempo: tick → segundo (lineal dentro de cada pulso)."""
        k = min(max(int(tick // self.tpq), 0), len(self.tiempos_pulso) - 2)
        d = self.tiempos_pulso[k + 1] - self.tiempos_pulso[k]
        return self.tiempos_pulso[k] + (tick - k * self.tpq) / self.tpq * d

    @property
    def duracion_s(self) -> float:
        return max((t + d for t, d, *_ in self.eventos), default=0.0)


def _fase_frase(tick: int, inicios, largos, frase: int, anacrusa: bool = False) -> float:
    """Posición continua dentro de la frase, en [0, 1). Con anacrusa, el compás parcial cierra la frase anterior."""
    b = max(0, bisect.bisect_right(inicios, tick) - 1)
    dentro = (tick - inicios[b]) / max(1, largos[b])
    bf = b - 1 if anacrusa else b
    return ((bf % frase) + min(dentro, 0.999)) / frase


def mapa_tempo(total_ticks: int, inicios, largos, tempo_bpm: float, tpq: int, e: Estilo,
               anacrusa: bool = False) -> list[float]:
    """Segundo de inicio de cada pulso (n+1 valores). Arco de frase y ritardando final sobre la duración del pulso."""
    n = max(1, math.ceil(total_ticks / tpq))
    base = 60.0 / tempo_bpm
    t, out = 0.0, [0.0]
    for k in range(n):
        fase = _fase_frase(k * tpq, inicios, largos, e.frase_compases, anacrusa)
        mult = 1.0 + e.arco_tempo * math.cos(2 * math.pi * fase)
        if e.rit_final and k >= n - e.rit_pulsos:
            x = (k - (n - e.rit_pulsos) + 1) / e.rit_pulsos
            mult *= 1.0 + e.rit_final * x * x
        t += base * mult
        out.append(t)
    return out


def voz_principal_de(notas) -> int:
    """Parte con mayor altura media (skyline); -1 si no hay notas."""
    suma, cnt = {}, {}
    for _, _, p, parte in notas:
        suma[parte] = suma.get(parte, 0) + p
        cnt[parte] = cnt.get(parte, 0) + 1
    return max(suma, key=lambda q: (suma[q] / cnt[q], -q)) if suma else -1


def notas_unicas(notas) -> list[tuple[int, int, int, int]]:
    """Fusiona las notas con igual (parte, ataque, altura) dejando la duración mayor. Orden (parte, ataque, altura)."""
    unico: dict[tuple[int, int, int], int] = {}
    for on, dur, pitch, parte in notas:
        k = (int(parte), int(on), int(pitch))
        unico[k] = max(int(dur), unico.get(k, 0))
    return [(on, dur, pitch, parte) for (parte, on, pitch), dur in sorted(unico.items())]


def _validar(notas, inicios, largos, tpq):
    inicios, largos = list(inicios), list(largos)
    if not inicios and not largos:
        inicios, largos = [0], [4 * tpq]
    if len(inicios) != len(largos):
        raise ValueError(f"inicios y largos deben tener la misma longitud ({len(inicios)} y {len(largos)})")
    if inicios[0] != 0 or inicios != sorted(inicios) or any(x <= 0 for x in largos):
        raise ValueError("inicios debe empezar en 0 e ir ordenado, y largos ser positivo (usa tokenizer.compases_de)")
    if any(on < 0 for on, *_ in notas):
        raise ValueError("hay ataques en ticks negativos")
    return inicios, largos


def interpretar(notas, inicios, largos, tempo_bpm: float = 90.0, tpq: int = 12, estilo: Estilo = ESTILOS["neutro"],
                semilla: int = 0) -> Interpretacion:
    """Aplica las reglas del estilo a la partitura. notas: [(on, dur, pitch, parte)] en ticks.

    Los eventos salen en orden (parte, ataque, altura) y `origen[i]` es la nota de la que sale `eventos[i]`."""
    e = estilo
    inicios, largos = _validar(notas, inicios, largos, tpq)
    unicas = notas_unicas(notas)
    anacrusa = len(largos) > 1 and largos[0] < largos[1]        # compás parcial inicial: es levare, no primer tiempo
    total = max([inicios[-1] + largos[-1]] + [on + d for on, d, _, _ in unicas])
    pulsos = mapa_tempo(total, inicios, largos, tempo_bpm, tpq, e, anacrusa)
    it = Interpretacion(pulsos, tpq, voz_principal_de(unicas), e.nombre, semilla, fusionadas=len(notas) - len(unicas))
    rng = random.Random(semilla)
    ataque: dict[tuple[int, int], float] = {}       # t0 por (parte, on): las notas de un acorde comparten ataque
    ultimo: dict[int, tuple[int, float]] = {}       # por parte: (on, t0) del ataque anterior
    for on, dur, pitch, parte in unicas:
        b = max(0, bisect.bisect_right(inicios, on) - 1)
        pos = on - inicios[b]
        fase = _fase_frase(on, inicios, largos, e.frase_compases, anacrusa)
        principal = parte == it.voz_principal
        primer_tiempo = pos == 0 and not (anacrusa and b == 0)
        vel = e.vel_base
        vel += e.acento_compas if primer_tiempo else (e.acento_pulso if pos % tpq == 0 else 0)
        vel += e.arco_vel * (2 * math.sin(math.pi * fase) - 1)
        vel += e.high_loud * (pitch - 60) / 12
        vel += e.voz_principal if principal else e.acomp
        vel = int(min(127, max(1, round(vel))))
        t_nom = it.segundos(on)
        if (parte, on) not in ataque:
            desvio = (rng.gauss(0.0, e.jitter_s) if e.jitter_s else 0.0) - (e.adelanto_s if principal else 0.0)
            desvio = max(-e.max_desvio_s, min(e.max_desvio_s, desvio))
            t0 = t_nom + desvio
            if parte in ultimo:                     # el orden de ataques dentro de la voz se conserva...
                on_prev, t_prev = ultimo[parte]
                eps = min(1e-3, 0.5 * (t_nom - it.segundos(on_prev)))
                t0 = max(t0, t_prev + eps)         # ...sin salirse del tope: eps ≤ mitad de la separación nominal
            ataque[(parte, on)] = max(0.0, t0)
            ultimo[parte] = (on, ataque[(parte, on)])
        t0 = ataque[(parte, on)]
        dur_s = (it.segundos(on + dur) - t_nom) * (e.legato if principal else e.legato_acomp)
        it.eventos.append((t0, max(e.dur_min_s, dur_s), pitch, vel, parte))
        it.origen.append((on, dur, pitch, parte))
    _cortar_repeticiones(it)
    return it


def _cortar_repeticiones(it: Interpretacion) -> None:
    """Una altura deja de sonar, como muy tarde, cuando la misma voz la vuelve a atacar (si no, el MIDI la cortaría).

    La adyacencia exacta se respeta: a igual tick, el MIDI pone el off antes que el on. Tras `notas_unicas` dos ataques
    de la misma tecla en la misma voz nunca comparten t0, así que la duración recortada es siempre positiva."""
    por_tecla: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, p, _, q) in enumerate(it.eventos):
        por_tecla.setdefault((q, p), []).append(i)
    for idx in por_tecla.values():
        idx.sort(key=lambda i: it.eventos[i][0])
        for i, j in zip(idx, idx[1:], strict=False):
            t0, d, p, v, q = it.eventos[i]
            limite = it.eventos[j][0]
            if t0 + d > limite + 1e-12 and limite > t0:
                it.eventos[i] = (t0, limite - t0, p, v, q)


def main(argv=None) -> dict:
    from .data import cargar
    from .render import a_audio_eventos, a_midi_eventos, a_wav
    from .tokenizer import compases_de

    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--indice", type=int, default=0)
    ap.add_argument("--id", default="", help="id exacto de la pieza (tiene prioridad sobre --indice)")
    ap.add_argument("--estilo", default="neutro", choices=sorted(ESTILOS))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    piezas = cargar(a.datos)
    pz = next(p for p in piezas if p["id"] == a.id) if a.id else piezas[a.indice]
    inicios, largos = compases_de(pz)
    os.makedirs(a.out, exist_ok=True)
    nombre = pz["id"].replace("/", "_").replace("#", "_")
    resumen = {"id": pz["id"], "notas": len(pz["notas"]), "tempo": pz["tempo"], "salidas": []}
    for est in dict.fromkeys(("plano", a.estilo)):
        it = interpretar(pz["notas"], inicios, largos, tempo_bpm=pz["tempo"], tpq=pz["tpq"], estilo=ESTILOS[est],
                         semilla=a.semilla)
        base = os.path.join(a.out, f"{nombre}_{est}")
        a_midi_eventos(it.eventos, base + ".mid")
        if a.wav:
            a_wav(a_audio_eventos(it.eventos, semilla=a.semilla), base + ".wav")
        resumen["salidas"].append({"estilo": est, "duracion_s": round(it.duracion_s, 2), "fusionadas": it.fusionadas,
                                   "voz_principal": it.voz_principal, "archivo": base})
    print(json.dumps(resumen, ensure_ascii=False), flush=True)
    return resumen


if __name__ == "__main__":
    main()
