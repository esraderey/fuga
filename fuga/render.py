"""Salida: MIDI (mido) y WAV por síntesis aditiva determinista (numpy), sin soundfonts.

Dos familias de entrada: notas cuantizadas [(on, dur, pitch, parte)] en ticks (render plano) y eventos interpretados
[(t0_s, dur_s, pitch, vel, parte)] en segundos con dinámica (salida de fuga.interprete).
En los dos MIDI una tecla no vuelve a atacar mientras suena: dos ataques de la misma tecla en el mismo tick se fusionan
y el off de una nota no pasa del siguiente on de esa tecla (a igual tick, el off va antes que el on)."""
from __future__ import annotations

import wave

import numpy as np

SR = 22050


def _canal(i: int, parte: int, programas) -> int:
    """Canal MIDI de la i-ésima parte: salta el 9 (percusión) salvo programa -1."""
    if programas and programas.get(parte) == -1:
        return 9
    return i % 9 if i < 9 else i + 1 if i < 15 else 15


def _sin_solapes(notas: list[list[int]]) -> list[list[int]]:
    """notas: [[on_tick, off_tick, pitch, vel]] de una misma parte. Fusiona los ataques simultáneos de la misma tecla
    (quedan el off y la velocidad mayores) y recorta cada off al siguiente on de esa tecla. Orden por on estable."""
    por_tecla: dict[tuple[int, int], list[int]] = {}
    for a, b, pitch, vel in notas:
        k = (a, pitch)
        if k in por_tecla:
            por_tecla[k][1] = max(por_tecla[k][1], b)
            por_tecla[k][3] = max(por_tecla[k][3], vel)
        else:
            por_tecla[k] = [a, b, pitch, vel]
    out = sorted(por_tecla.values(), key=lambda n: n[0])
    siguiente: dict[int, int] = {}
    for nota in reversed(out):
        a, b, pitch, _ = nota
        if pitch in siguiente:
            nota[1] = max(a + 1, min(b, siguiente[pitch]))
        siguiente[pitch] = a
    return out


def _escribir_pista(tr, notas: list[list[int]], ch: int) -> None:
    """Añade a la pista los note_on/note_off de `notas` ([[on, off, pitch, vel]]) con tiempos delta."""
    import mido

    ev = []
    for a, b, pitch, vel in _sin_solapes(notas):
        ev.append((a, 1, pitch, vel))
        ev.append((b, 0, pitch, 0))
    ev.sort(key=lambda e: (e[0], e[1]))             # a igual tick, el off va antes que el on
    t = 0
    for tt, es_on, pitch, vel in ev:
        tr.append(mido.Message("note_on" if es_on else "note_off", note=int(pitch), velocity=int(vel), channel=ch,
                               time=int(tt - t)))
        t = tt


def a_midi(notas, ruta: str, tpq: int = 12, tempo_bpm: float = 90.0, programas=None):
    """notas: [(on, dur, pitch, parte)] en ticks de `tpq`. Escribe un MIDI tipo 1 (480 ticks por negra, vel 80)."""
    import mido

    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    esc = 480 // tpq
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    mid.tracks.append(meta)
    partes = sorted({p for *_, p in notas})
    for i, p in enumerate(partes):
        tr = mido.MidiTrack()
        ch = _canal(i, p, programas)
        prog = (programas or {}).get(p, 0)
        if ch != 9:
            tr.append(mido.Message("program_change", program=max(0, prog), channel=ch, time=0))
        _escribir_pista(tr, [[on * esc, (on + max(1, dur)) * esc - 1, int(pitch), 80]
                             for on, dur, pitch, pp in notas if pp == p], ch)
        mid.tracks.append(tr)
    mid.save(ruta)


def a_midi_eventos(eventos, ruta: str, tpq: int = 960, tempo_bpm: float = 120.0, programas=None):
    """eventos: [(t0_s, dur_s, pitch, vel, parte)] en segundos. Tempo fijo y 960 ticks por negra: el tiempo
    expresivo va codificado en los ticks (resolución ≈0.5 ms)."""
    import mido

    mid = mido.MidiFile(type=1, ticks_per_beat=tpq)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    mid.tracks.append(meta)
    k = tpq * tempo_bpm / 60.0                      # ticks por segundo
    partes = sorted({e[4] for e in eventos})
    for i, p in enumerate(partes):
        tr = mido.MidiTrack()
        ch = _canal(i, p, programas)
        prog = (programas or {}).get(p, 0)
        if ch != 9:
            tr.append(mido.Message("program_change", program=max(0, prog), channel=ch, time=0))
        notas = []
        for t0, dur, pitch, vel, pp in eventos:
            if pp == p:
                a = int(round(t0 * k))
                notas.append([a, max(a + 1, int(round((t0 + dur) * k))), int(pitch), int(vel)])
        _escribir_pista(tr, notas, ch)
        mid.tracks.append(tr)
    mid.save(ruta)


_TIMBRES = [  # (pesos de armónicos, ataque s, caída exp.)
    ([1.0, 0.5, 0.25, 0.12], 0.02, 1.2),
    ([1.0, 0.3, 0.6, 0.1, 0.2], 0.03, 0.9),
    ([1.0, 0.7, 0.4, 0.3, 0.2, 0.1], 0.01, 2.0),
    ([1.0, 0.2, 0.1], 0.04, 0.7),
]


def a_audio(notas, tpq: int = 12, tempo_bpm: float = 90.0, sr: int = SR) -> np.ndarray:
    """Síntesis aditiva con envolvente ADSR simple. Returns float32 mono normalizado."""
    seg_tick = 60.0 / tempo_bpm / tpq
    fin = max(on + d for on, d, _, _ in notas) * seg_tick + 1.0
    y = np.zeros(int(fin * sr) + 1, dtype=np.float32)
    for on, dur, pitch, parte in notas:
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        t0, dt = on * seg_tick, max(dur * seg_tick, 0.06)
        n = int((dt + 0.15) * sr)
        t = np.arange(n) / sr
        arm, ata, cai = _TIMBRES[parte % len(_TIMBRES)]
        s = sum(w * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, w in enumerate(arm) if f0 * (k + 1) < sr / 2)
        env = np.minimum(1.0, t / ata) * np.exp(-cai * t)
        env *= np.clip((dt + 0.15 - t) / 0.15, 0.0, 1.0)  # liberación
        i0 = int(t0 * sr)
        y[i0:i0 + n] += (0.15 * s * env).astype(np.float32)[: len(y) - i0]
    pico = np.abs(y).max()
    return y / pico * 0.9 if pico > 0 else y


def a_audio_eventos(eventos, sr: int = SR, reverb: float = 0.12, semilla: int = 0) -> np.ndarray:
    """eventos: [(t0_s, dur_s, pitch, vel, parte)]. Síntesis aditiva con dinámica (vel → amplitud y brillo) y una
    reverberación corta determinista. Returns float32 mono normalizado a 0.9."""
    if not eventos:
        return np.zeros(sr, dtype=np.float32)
    fin = max(t0 + d for t0, d, *_ in eventos) + 1.5
    y = np.zeros(int(fin * sr) + 1, dtype=np.float32)
    for t0, dur, pitch, vel, parte in eventos:
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        dt = max(dur, 0.06)
        n = int((dt + 0.25) * sr)
        t = np.arange(n) / sr
        arm, ata, cai = _TIMBRES[parte % len(_TIMBRES)]
        g = max(1, min(127, vel)) / 127.0
        brillo = 0.4 + 0.6 * g                      # con más fuerza, más armónicos
        s = sum(w * brillo ** k * np.sin(2 * np.pi * f0 * (k + 1) * t)
                for k, w in enumerate(arm) if f0 * (k + 1) < sr / 2)
        env = np.minimum(1.0, t / ata) * np.exp(-cai * t)
        env *= np.clip((dt + 0.25 - t) / 0.25, 0.0, 1.0)      # liberación
        i0 = int(max(0.0, t0) * sr)
        y[i0:i0 + n] += (0.12 * g ** 1.5 * s * env).astype(np.float32)[: len(y) - i0]
    if reverb > 0:
        rng = np.random.default_rng(semilla)
        n_ir = int(0.6 * sr)
        ir = rng.standard_normal(n_ir).astype(np.float32) * np.exp(-6.0 * np.arange(n_ir) / n_ir)
        ir /= np.sqrt((ir ** 2).sum())
        n_fft = len(y) + n_ir
        cola = np.fft.irfft(np.fft.rfft(y, n_fft) * np.fft.rfft(ir, n_fft))[: len(y)]
        y = y + reverb * cola.astype(np.float32)
    pico = np.abs(y).max()
    return (y / pico * 0.9).astype(np.float32) if pico > 0 else y


def a_wav(y: np.ndarray, ruta: str, sr: int = SR):
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
