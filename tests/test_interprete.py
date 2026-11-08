import bisect
from dataclasses import replace

import numpy as np
import pytest
from test_fuga import pieza_sintetica

from fuga.interprete import ESTILOS, interpretar, notas_unicas, voz_principal_de
from fuga.render import a_audio_eventos, a_midi, a_midi_eventos
from fuga.tokenizer import compases_de

TPS = 960 * 120 / 60.0          # ticks por segundo del MIDI de eventos (960 tpq a 120 bpm)


def _preparar(semilla=0, compases=12, partes=2):
    pz = pieza_sintetica(semilla, compases=compases, partes=partes)
    inicios, largos = compases_de(pz)
    return pz, inicios, largos


def _invariantes(it, notas, e):
    """Comprueba los invariantes declarados del intérprete sobre una interpretación."""
    assert sorted(it.origen) == sorted(notas_unicas(notas))                    # ni una nota distinta más ni menos
    assert it.fusionadas == len(notas) - len(notas_unicas(notas))
    for (t0, d, p, v, q), (on, _dur, p0, q0) in zip(it.eventos, it.origen, strict=True):
        assert (p, q) == (p0, q0)
        assert abs(t0 - it.segundos(on)) <= e.max_desvio_s + 1e-9              # tope duro de desvío
        assert 1 <= v <= 127 and d > 0
    por_parte = {}
    for (t0, _d, _p, _v, q), (on, *_r) in zip(it.eventos, it.origen, strict=True):
        por_parte.setdefault(q, []).append((on, t0))
    for pares in por_parte.values():                                            # orden y cohesión por voz
        ts = [t for _, t in pares]
        assert ts == sorted(ts)
        por_on = {}
        for on, t in pares:
            por_on.setdefault(on, set()).add(t)
        assert all(len(s) == 1 for s in por_on.values())                        # un acorde comparte ataque
    por_tecla = {}
    for t0, d, p, _v, q in it.eventos:
        por_tecla.setdefault((q, p), []).append((t0, d))
    for ev in por_tecla.values():                                               # sin solapes de la misma altura
        ev.sort()
        for (t0, d), (t1, _) in zip(ev, ev[1:], strict=False):
            assert t0 + d <= t1 + 1e-12


def _ticks_midi(ruta):
    """[(track, tick_abs, tipo, nota, vel, canal)] en orden de archivo."""
    import mido
    out = []
    for k, tr in enumerate(mido.MidiFile(ruta).tracks):
        t = 0
        for m in tr:
            t += m.time
            if m.type in ("note_on", "note_off"):
                out.append((k, t, m.type, m.note, m.velocity, m.channel))
    return out


def _teclas_sin_solape(msgs):
    """True si en cada canal ninguna tecla recibe un segundo note_on antes de su note_off (off antes que on a igual tick)."""
    sonando = set()
    for _k, _t, tipo, nota, vel, ch in msgs:
        if tipo == "note_on" and vel > 0:
            if (ch, nota) in sonando:
                return False
            sonando.add((ch, nota))
        else:
            sonando.discard((ch, nota))
    return True


def test_plano_reproduce_la_partitura():
    pz, inicios, largos = _preparar()
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=ESTILOS["plano"])
    seg_tick = 60.0 / 100.0 / 12
    assert len(it.eventos) == len(pz["notas"]) == len(it.origen) and it.fusionadas == 0
    esperado = sorted((on * seg_tick, d * seg_tick, p, 80, parte) for on, d, p, parte in pz["notas"])
    for (t0, d, p, v, q), (t1, d1, p1, v1, q1) in zip(sorted(it.eventos), esperado, strict=True):
        assert abs(t0 - t1) < 1e-6 and abs(d - d1) < 1e-6 and (p, v, q) == (p1, v1, q1)


def test_plano_conserva_notas_mas_cortas_de_40_ms():
    it = interpretar([(0, 1, 60, 0), (12, 12, 62, 0)], [0], [48], tempo_bpm=200.0, estilo=ESTILOS["plano"])
    assert abs(it.eventos[0][1] - 0.025) < 1e-9                                 # 1 tick a 200 bpm, sin mínimo


@pytest.mark.parametrize("estilo", ["neutro", "romantico", "mecanico"])
@pytest.mark.parametrize("semilla", [0, 1])
def test_invariantes_en_pieza_sintetica(estilo, semilla):
    pz, inicios, largos = _preparar(semilla)
    e = ESTILOS[estilo]
    _invariantes(interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=e, semilla=semilla), pz["notas"], e)


@pytest.mark.parametrize("estilo,semilla", [("neutro", 15), ("romantico", 0), ("neutro", 3)])
def test_acordes_con_ataques_a_un_tick(estilo, semilla):
    # ataques separados 1 tick (25 ms a 200 bpm) y acordes en la misma voz: fuerza la pinza de orden
    notas = [(k, 1, p, 0) for k in range(60) for p in (60, 64)]
    e = ESTILOS[estilo]
    it = interpretar(notas, [0], [48], tempo_bpm=200.0, estilo=e, semilla=semilla)
    _invariantes(it, notas, e)


def test_tope_de_desvio_actua_con_jitter_grande():
    pz, inicios, largos = _preparar()
    apretado = replace(ESTILOS["neutro"], jitter_s=0.2, max_desvio_s=0.05)
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=apretado, semilla=1)
    _invariantes(it, pz["notas"], apretado)
    suelto = replace(apretado, max_desvio_s=0.5)
    it2 = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=suelto, semilla=1)
    assert max(abs(t0 - it2.segundos(on)) for (t0, *_), (on, *_r) in zip(it2.eventos, it2.origen, strict=True)) > 0.05


@pytest.mark.parametrize("alturas", ["distintas", "misma"])
def test_tope_con_tpq_alto_y_ataques_densos(alturas):
    notas = [(k, 1, 60 + (k % 12 if alturas == "distintas" else 0), 0) for k in range(300)]
    e = ESTILOS["neutro"]
    _invariantes(interpretar(notas, [0], [3840], tempo_bpm=200.0, tpq=960, estilo=e), notas, e)


def test_repeticion_de_la_misma_altura_no_se_solapa():
    notas = [(k, 1, 60, 0) for k in range(0, 48)]
    e = ESTILOS["romantico"]                                                      # legato 1,05 > 1
    it = interpretar(notas, [0], [48], tempo_bpm=120.0, estilo=e, semilla=0)
    _invariantes(it, notas, e)
    sin_cortar = interpretar([(0, 12, 60, 0), (24, 12, 60, 0)], [0], [48], tempo_bpm=120.0, estilo=e, semilla=0)
    assert abs(sin_cortar.eventos[0][1] / sin_cortar.segundos(12) - 1.05) < 1e-9   # lejos, el legato se respeta


def test_duplicados_de_misma_voz_ataque_y_altura_se_fusionan(tmp_path):
    notas = [(0, 6, 60, 0), (0, 24, 60, 0), (24, 12, 62, 0)]                       # patrón real del corpus
    for nombre in ("neutro", "plano"):
        e = ESTILOS[nombre]
        it = interpretar(notas, [0], [48], tempo_bpm=100.0, estilo=e)
        _invariantes(it, notas, e)
        assert it.fusionadas == 1 and (0, 24, 60, 0) in it.origen and (0, 6, 60, 0) not in it.origen
        ruta = tmp_path / f"{nombre}.mid"
        a_midi_eventos(it.eventos, str(ruta))
        msgs = _ticks_midi(str(ruta))
        assert _teclas_sin_solape(msgs) and sum(m[2] == "note_on" for m in msgs) == 2
    assert notas_unicas([(0, 6, 60, 0), (0, 24, 60, 1)]) == [(0, 6, 60, 0), (0, 24, 60, 1)]   # voces distintas no


def test_determinismo_y_dependencia_de_la_semilla():
    pz, inicios, largos = _preparar()
    a = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, semilla=3)
    b = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, semilla=3)
    c = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, semilla=4)
    assert a.eventos == b.eventos
    assert a.eventos != c.eventos
    assert a.tiempos_pulso == c.tiempos_pulso          # el mapa de tempo no depende de la semilla


def test_arco_de_tempo_y_ritardando_final():
    pz, inicios, largos = _preparar()
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=ESTILOS["neutro"])
    pulsos = np.diff(it.tiempos_pulso)
    assert abs(pulsos[0] - 0.6 * 1.03) < 1e-9 and abs(pulsos[8] - 0.6 * 0.97) < 1e-9   # arco: lento, rápido
    assert pulsos[-1] > 1.2 * pulsos[0] and pulsos[-1] > pulsos[-ESTILOS["neutro"].rit_pulsos]
    plano = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=ESTILOS["plano"])
    assert np.allclose(np.diff(plano.tiempos_pulso), 0.6)


def test_acento_metrico_y_voz_principal():
    pz, inicios, largos = _preparar()
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=ESTILOS["neutro"])
    assert it.voz_principal == voz_principal_de(pz["notas"]) == 1     # la parte 1 va una octava arriba
    fuertes, debiles = [], []
    for (_t0, _d, _p, v, _q), (on, *_resto) in zip(it.eventos, it.origen, strict=True):
        b = bisect.bisect_right(inicios, on) - 1
        (fuertes if on - inicios[b] == 0 else debiles).append(v)
    assert np.mean(fuertes) > np.mean(debiles) + 5


def test_high_loud_legato_y_adelanto():
    e = ESTILOS["neutro"]
    it = interpretar([(12, 12, 60, 0), (12, 12, 72, 0)], [0], [48], tempo_bpm=100.0, estilo=e)
    assert it.eventos[1][3] - it.eventos[0][3] == round(e.high_loud)             # +1 octava → +high_loud
    for nombre, razon in (("romantico", 1.05), ("plano", 1.0), ("mecanico", 0.85)):
        it = interpretar([(0, 12, 60, 0)], [0], [48], tempo_bpm=100.0, estilo=ESTILOS[nombre])
        assert abs(it.eventos[0][1] / it.segundos(12) - razon) < 1e-9
    quieto = replace(e, jitter_s=0.0)
    it = interpretar([(48, 12, 60, 0), (48, 12, 72, 1)], [0, 48], [48, 48], tempo_bpm=100.0, estilo=quieto)
    assert it.voz_principal == 1
    assert abs((it.eventos[0][0] - it.eventos[1][0]) - e.adelanto_s) < 1e-9        # la principal ataca antes


def test_anacrusa_no_recibe_acento_de_primer_tiempo():
    pz = {"compases": [6, 54, 102], "notas": [(0, 6, 60, 0), (6, 12, 62, 0), (18, 12, 64, 0), (54, 12, 65, 0)]}
    inicios, largos = compases_de(pz)
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, estilo=ESTILOS["mecanico"])
    vel = {on: v for (_t, _d, _p, v, _q), (on, *_r) in zip(it.eventos, it.origen, strict=True)}
    assert vel[0] < vel[6] == vel[54]


def test_validacion_de_entradas():
    with pytest.raises(ValueError, match="misma longitud"):
        interpretar([(0, 1, 60, 0)], [], [48])
    with pytest.raises(ValueError, match="empezar en 0"):
        interpretar([(12, 1, 60, 0)], [12], [48])
    with pytest.raises(ValueError, match="empezar en 0"):
        interpretar([(0, 1, 60, 0)], [0, 96, 48], [48, 48, 48])
    with pytest.raises(ValueError, match="negativos"):
        interpretar([(-6, 12, 60, 0)], [0], [48])


def test_render_de_eventos_wav_y_midi(tmp_path):
    pz, inicios, largos = _preparar(compases=4)
    it = interpretar(pz["notas"], inicios, largos, tempo_bpm=100.0, semilla=0)
    y = a_audio_eventos(it.eventos, sr=8000, semilla=0)
    assert y.dtype == np.float32 and 0.89 < np.abs(y).max() <= 0.9 + 1e-6
    assert abs(len(y) / 8000 - (it.duracion_s + 1.5)) < 0.01
    assert np.array_equal(y, a_audio_eventos(it.eventos, sr=8000, semilla=0))
    ruta = tmp_path / "it.mid"
    a_midi_eventos(it.eventos, str(ruta))
    msgs = _ticks_midi(str(ruta))
    ons = [m for m in msgs if m[2] == "note_on" and m[4] > 0]
    assert len(ons) == len(it.eventos) and _teclas_sin_solape(msgs)
    assert {m[4] for m in ons} == {v for *_, v, _ in it.eventos}
    assert {m[5] for m in msgs} == {0, 1}                                         # una parte por canal
    for k, parte in ((1, 0), (2, 1)):                                             # ticks = round(t0 · 1920)
        primero = min(t0 for t0, *_r, q in it.eventos if q == parte)
        assert min(m[1] for m in ons if m[0] == k) == round(primero * TPS)


def test_midi_off_antes_que_on_y_sin_corte_de_repeticiones(tmp_path):
    ruta = tmp_path / "rep.mid"
    a_midi_eventos([(0.0, 0.6, 60, 80, 0), (0.5, 0.5, 60, 80, 0), (1.0, 0.5, 60, 80, 0)], str(ruta))
    msgs = [m for m in _ticks_midi(str(ruta)) if m[0] == 1]
    assert [(m[1], m[2]) for m in msgs] == [(0, "note_on"), (960, "note_off"), (960, "note_on"),
                                           (1920, "note_off"), (1920, "note_on"), (2880, "note_off")]
    ruta2 = tmp_path / "dup.mid"                                                  # misma tecla en el mismo tick
    a_midi_eventos([(0.0, 0.5, 60, 80, 0), (0.0002, 1.0, 60, 90, 0), (1.5, 0.2, 60, 70, 0)], str(ruta2))
    msgs2 = [m for m in _ticks_midi(str(ruta2)) if m[0] == 1]
    assert _teclas_sin_solape(msgs2)
    assert [(m[1], m[2], m[4]) for m in msgs2] == [(0, "note_on", 90), (1920, "note_off", 0),
                                                  (2880, "note_on", 70), (3264, "note_off", 0)]
    ruta3 = tmp_path / "dup2.mid"                                                 # el primero es el largo y fuerte
    a_midi_eventos([(0.0, 1.0, 60, 90, 0), (0.0002, 0.5, 60, 80, 0)], str(ruta3))
    msgs3 = [m for m in _ticks_midi(str(ruta3)) if m[0] == 1]
    assert [(m[1], m[2], m[4]) for m in msgs3] == [(0, "note_on", 90), (1920, "note_off", 0)]


def test_midi_plano_no_corta_notas_repetidas(tmp_path):
    # el render plano de fuga.sample: una nota larga seguida de la misma altura antes de que acabe
    notas = [(0, 24, 60, 0), (12, 12, 60, 0), (12, 12, 64, 0), (36, 12, 60, 0)]
    ruta = tmp_path / "plano.mid"
    a_midi(notas, str(ruta), tempo_bpm=100.0)
    msgs = [m for m in _ticks_midi(str(ruta)) if m[0] == 1]
    assert _teclas_sin_solape(msgs)
    assert [(m[1], m[2], m[3]) for m in msgs if m[3] == 60] == [(0, "note_on", 60), (480, "note_off", 60),
                                                                (480, "note_on", 60), (959, "note_off", 60),
                                                                (1440, "note_on", 60), (1919, "note_off", 60)]
    assert sum(m[2] == "note_on" for m in msgs) == 4 and {m[4] for m in msgs if m[2] == "note_on"} == {80}


def test_audio_respeta_dinamica_y_tiempo():
    sr = 8000
    y = a_audio_eventos([(0.0, 0.5, 60, 40, 0), (2.0, 0.5, 60, 120, 0)], sr=sr, reverb=0.0)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
    assert rms(y[int(2.0 * sr):int(2.5 * sr)]) > 2 * rms(y[: int(0.5 * sr)])
    assert np.abs(y[int(0.9 * sr):int(1.9 * sr)]).max() < 1e-6                      # silencio entre notas
    tarde = a_audio_eventos([(1.0, 0.5, 60, 80, 0)], sr=sr, reverb=0.0)
    assert np.abs(tarde[: int(0.95 * sr)]).max() == 0.0


def test_sin_notas_ni_compases():
    it = interpretar([], [], [], tempo_bpm=90.0)
    assert it.eventos == [] and it.duracion_s == 0.0 and it.voz_principal == -1 and it.fusionadas == 0
    assert len(a_audio_eventos([], sr=4000)) == 4000
