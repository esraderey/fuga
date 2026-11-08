"""E1 (RFC-FUGA-2 § 5): fábrica de pares render↔audio real sobre 300 clips FMA CC BY/CC0 sin voz declarada.

Pre-registro con todas las definiciones: ensayo/e1_prerregistro.md. Corre en el venv aparte (.venv_e1, Python 3.10,
Basic Pitch 0.4.0) y con -I: los datos descargados son no confiables y el script no importa nada del árbol.

    .venv_e1/Scripts/python.exe -I scripts/e1_pares.py licencias
    .venv_e1/Scripts/python.exe -I scripts/e1_pares.py descargar --n 300
    .venv_e1/Scripts/python.exe -I scripts/e1_pares.py medir
    .venv_e1/Scripts/python.exe -I scripts/e1_pares.py veredicto
    .venv_e1/Scripts/python.exe -I scripts/e1_pares.py todo        # los cuatro pasos seguidos

Entradas: data/fma_e1/meta/fma_metadata/{tracks,genres,echonest}.csv (fma_metadata.zip, Defferrard et al.),
data/fma_e1/tools/fluidsynth/.../fluidsynth.exe y data/fma_e1/tools/soundfont/MuseScore_General.sf3.
Salidas: data/fma_e1/{clips,trabajo}/ (intermedias) y ensayo/e1_pares.json (resultado).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
URL_LARGE = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
SHA1_METADATA = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"      # README oficial de mdeff/fma
SEMILLA = 20261025
SR = 22050
HOP_ONSET = 128                 # 5,8 ms
HOP_CROMA = 512                 # 23,2 ms; 43 frames por ventana de 1 s
SIGMA_MS = 20.0
MAX_LAG_MS = 250.0
TAU = 0.6
TOL_C_S = 0.050
UMBRAL_MS = 23.0
MIN_DUR_CLIP = 25.0

# --- licencias -------------------------------------------------------------------------------------------------------

def clasificar_licencia(texto) -> str:
    """Clase fina de una cadena de `track.license` (ver pre-registro)."""
    if texto is None or (isinstance(texto, float) and math.isnan(texto)) or str(texto).strip() == "":
        return "sin licencia"
    s = str(texto).lower()
    n = re.sub(r"[\s\-_]", "", s)
    if "cc0" in n or "publicdomain" in n:
        return "CC0/PD"
    if "freeartlicense" in n or "licenceartlibre" in n or "artlibre" in n:
        return "FAL"
    if "fmalimited" in n:
        return "FMA-Limited"
    by, nc, nd, sa = "attribution" in n, "noncommercial" in n, "noderiv" in n, "sharealike" in n
    if by or nc or nd or sa:
        if nc and nd:
            return "CC BY-NC-ND"
        if nc and sa:
            return "CC BY-NC-SA"
        if nc:
            return "CC BY-NC"
        if nd:
            return "CC BY-ND"
        if sa:
            return "CC BY-SA"
        return "CC BY"
    return "otras"


CUBO = {"CC BY": "CC BY", "CC0/PD": "CC0", "CC BY-NC": "NC", "CC BY-NC-SA": "NC", "CC BY-NC-ND": "NC",
        "CC BY-ND": "ND"}


def cubo(clase: str) -> str:
    return CUBO.get(clase, "otras")


GENEROS_VOZ_RAIZ = {20, 21}                                 # Spoken, Hip-Hop (y todo lo que cuelga de ellos)
GENEROS_VOZ = {103, 443, 441, 440, 37, 567, 170, 186}       # Singer-Songwriter, Choral, Opera, Rock Opera, Jazz: Vocal,
                                                            # Gospel, Easy Listening: Vocal, Sound Poetry
PALABRAS_VOZ = re.compile(r"\b(vocals?|vocalists?|voices?|singing|singers?|sung|rap|rappers?|rapping|spoken|lyrics?|"
                          r"lyrical|acapella|a cappella|choir|chants?|chanting|voice-over)\b")


def _lista(s) -> list:
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else []
        return list(v) if isinstance(v, (list, tuple, set)) else []
    except Exception:
        return []


def paso_licencias(datos: Path, salida: Path) -> dict:
    import pandas as pd

    meta = datos / "meta" / "fma_metadata"
    zip_meta = datos / "meta" / "fma_metadata.zip"
    sha1_meta = sha1_archivo(zip_meta) if zip_meta.exists() else None
    tracks = pd.read_csv(meta / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    generos = pd.read_csv(meta / "genres.csv", index_col=0)
    echo = pd.read_csv(meta / "echonest.csv", index_col=0, header=[0, 1, 2], low_memory=False)
    instr = echo[("echonest", "audio_features", "instrumentalness")].astype(float)
    tr = tracks["track"]
    n = len(tr)
    dur = tr["duration"].astype(float).fillna(0.0)

    clase = tr["license"].map(clasificar_licencia)
    cubos = clase.map(cubo)
    fino = {k: int(v) for k, v in clase.value_counts().items()}
    grueso = {k: int(v) for k, v in cubos.value_counts().items()}
    horas_fino = {k: float(dur[clase == k].sum() / 3600) for k in fino}
    horas_grueso = {k: float(dur[cubos == k].sum() / 3600) for k in grueso}
    textos = {}                                                     # trazabilidad: cada texto → clase
    for texto, cl in zip(tr["license"].fillna("<NaN>"), clase, strict=False):
        textos.setdefault(cl, {})
        textos[cl][texto] = textos[cl].get(texto, 0) + 1

    # --- filtro «sin voz declarada» ---
    top = generos["top_level"].to_dict()
    def genero_voz(gs) -> bool:
        return any((g in GENEROS_VOZ) or (top.get(g) in GENEROS_VOZ_RAIZ) for g in gs)
    def tag_voz(ts) -> bool:
        return any(PALABRAS_VOZ.search(str(t).lower()) for t in ts)
    f_idioma = tr["language_code"].notna()
    f_letrista = tr["lyricist"].notna()
    f_genero = tr["genres_all"].map(lambda s: genero_voz(_lista(s)))
    f_tag = tr["tags"].map(lambda s: tag_voz(_lista(s)))
    inst_al = instr.reindex(tr.index)
    f_echo = inst_al.notna() & (inst_al < 0.5)
    voz = f_idioma | f_letrista | f_genero | f_tag | f_echo

    libre = cubos.isin(["CC BY", "CC0"])
    libre_sa = libre | (clase == "CC BY-SA")
    E = libre & ~voz
    E_sa = libre_sa & ~voz
    F = E & (dur >= 30.0)
    embudo = {
        "total": int(n),
        "cc_by_o_cc0": int(libre.sum()),
        "tras_idioma": int((libre & ~f_idioma).sum()),
        "tras_letrista": int((libre & ~f_idioma & ~f_letrista).sum()),
        "tras_generos": int((libre & ~f_idioma & ~f_letrista & ~f_genero).sum()),
        "tras_tags": int((libre & ~f_idioma & ~f_letrista & ~f_genero & ~f_tag).sum()),
        "tras_echonest_E": int(E.sum()),
        "marco_F_dur_ge_30s": int(F.sum()),
        "E_con_by_sa": int(E_sa.sum()),
    }
    horas = {
        "fma_full_total": float(dur.sum() / 3600),
        "cc_by_o_cc0": float(dur[libre].sum() / 3600),
        "E_pistas_completas": float(dur[E].sum() / 3600),
        "E_clips_30s": float(E.sum() * 30 / 3600),
        "E_con_by_sa_pistas_completas": float(dur[E_sa].sum() / 3600),
    }
    generos_E = {str(k): int(v) for k, v in tr.loc[E, "genre_top"].fillna("<sin genre_top>").value_counts().items()}
    subset_E = {str(k): int(v) for k, v in tracks.loc[E, ("set", "subset")].value_counts().items()}

    salida.mkdir(parents=True, exist_ok=True)
    eleg = pd.DataFrame({"track_id": tr.index[E], "duration": dur[E].values, "clase": clase[E].values,
                         "genre_top": tr.loc[E, "genre_top"].values, "subset": tracks.loc[E, ("set", "subset")].values})
    eleg.to_csv(salida / "elegibles.csv", index=False)
    rng = np.random.default_rng(SEMILLA)
    ids_F = np.array(sorted(tr.index[F].tolist()))
    orden = ids_F[rng.permutation(len(ids_F))]
    pd.DataFrame({"orden": np.arange(len(orden)), "track_id": orden}).to_csv(salida / "muestra_orden.csv", index=False)

    res = {
        "fuente": {"archivo": "fma_metadata.zip", "sha1": sha1_meta, "sha1_esperado": SHA1_METADATA,
                   "sha1_ok": sha1_meta == SHA1_METADATA, "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
                   "cita": "Defferrard et al., FMA: A Dataset for Music Analysis, ISMIR 2017 (arXiv:1612.01840)"},
        "n_pistas": int(n),
        "clases_finas": {"conteo": fino, "fraccion": {k: v / n for k, v in fino.items()}, "horas_fma_full": horas_fino},
        "cubos": {"conteo": grueso, "fraccion": {k: v / n for k, v in grueso.items()}, "horas_fma_full": horas_grueso},
        "textos_por_clase": textos,
        "filtro_voz": {"idioma": int(f_idioma.sum()), "letrista": int(f_letrista.sum()), "generos": int(f_genero.sum()),
                       "tags": int(f_tag.sum()), "echonest_instrumentalness_lt_0.5": int(f_echo.sum()),
                       "cualquiera": int(voz.sum())},
        "embudo_elegibles": embudo,
        "horas": horas,
        "elegibles_por_genre_top": generos_E,
        "elegibles_por_subset": subset_E,
        "semilla": SEMILLA,
    }
    (salida / "licencias.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[licencias] pistas {n}; CC BY {grueso.get('CC BY', 0)} ({grueso.get('CC BY', 0) / n:.1%}); "
          f"CC0 {grueso.get('CC0', 0)} ({grueso.get('CC0', 0) / n:.1%}); NC {grueso.get('NC', 0) / n:.1%}; "
          f"ND {grueso.get('ND', 0) / n:.1%}; otras {grueso.get('otras', 0) / n:.1%}")
    print(f"[licencias] elegibles E = {embudo['tras_echonest_E']} pistas, {horas['E_pistas_completas']:.0f} h "
          f"completas / {horas['E_clips_30s']:.0f} h en clips; marco F = {embudo['marco_F_dur_ge_30s']}")
    return res


# --- descarga remota -------------------------------------------------------------------------------------------------

class ArchivoHTTP(io.RawIOBase):
    """Archivo remoto de solo lectura por peticiones Range; suficiente para zipfile (zip64)."""

    def __init__(self, url: str, minimo: int = 1 << 18):
        import requests
        self.url, self.s, self.pos, self.minimo, self.peticiones = url, requests.Session(), 0, minimo, 0
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.etag = r.headers.get("ETag")
        self.cache = (0, b"")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, a: int, b: int) -> bytes:
        for intento in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
                if r.status_code != 206:
                    raise IOError(f"status {r.status_code}")
                self.peticiones += 1
                return r.content
            except Exception:
                if intento == 5:
                    raise
                time.sleep(2 ** intento)
        raise IOError("inalcanzable")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        a, b = self.pos, min(self.size, self.pos + n)
        ca, cd = self.cache
        if not (ca <= a and b <= ca + len(cd)):
            cd = self._fetch(a, min(self.size, max(b, a + self.minimo)))
            ca = a
            self.cache = (ca, cd)
        self.pos = b
        return cd[a - ca:b - ca]

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_archivo(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def sha256_archivo(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def paso_descargar(datos: Path, trabajo: Path, n_obj: int) -> dict:
    import pandas as pd
    import soundfile as sf

    clips = datos / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    orden = pd.read_csv(trabajo / "muestra_orden.csv")["track_id"].tolist()
    ruta_reg = trabajo / "descarga.json"
    reg = json.loads(ruta_reg.read_text(encoding="utf-8")) if ruta_reg.exists() else {"ok": [], "saltados": []}
    hechos = {c["track_id"] for c in reg["ok"]} | {c["track_id"] for c in reg["saltados"]}
    t0 = time.time()
    f = ArchivoHTTP(URL_LARGE)
    z = zipfile.ZipFile(f)
    nombres = set(z.namelist())
    sumas = {}
    for linea in z.read("fma_large/checksums").decode("utf-8", "replace").splitlines():
        partes = linea.split()
        if len(partes) >= 2:
            sumas[partes[-1].lstrip("./")] = partes[0]
    reg["zip"] = {"url": URL_LARGE, "bytes": f.size, "etag": f.etag, "entradas": len(nombres), "checksums": len(sumas)}
    print(f"[descargar] índice remoto: {len(nombres)} entradas, {f.peticiones} peticiones, {time.time() - t0:.1f} s")
    for tid in orden:
        if len(reg["ok"]) >= n_obj:
            break
        if tid in hechos:
            continue
        nombre = f"fma_large/{tid // 1000:03d}/{tid:06d}.mp3"
        destino = clips / f"{tid:06d}.mp3"
        if nombre not in nombres:
            reg["saltados"].append({"track_id": int(tid), "motivo": "no_en_zip"})
            continue
        try:
            datos_mp3 = z.read(nombre)
        except Exception as e:                       # noqa: BLE001
            reg["saltados"].append({"track_id": int(tid), "motivo": f"lectura: {type(e).__name__}"})
            continue
        s1 = sha1_bytes(datos_mp3)
        esperado = sumas.get(nombre.split("/", 1)[1])
        if esperado is not None and esperado != s1:
            reg["saltados"].append({"track_id": int(tid), "motivo": "sha1_distinto"})
            continue
        destino.write_bytes(datos_mp3)
        try:
            info = sf.info(str(destino))
            dur = info.frames / info.samplerate
        except Exception as e:                       # noqa: BLE001
            destino.unlink(missing_ok=True)
            reg["saltados"].append({"track_id": int(tid), "motivo": f"decodificacion: {type(e).__name__}"})
            continue
        if dur < MIN_DUR_CLIP:
            destino.unlink(missing_ok=True)
            reg["saltados"].append({"track_id": int(tid), "motivo": f"corto_{dur:.1f}s"})
            continue
        reg["ok"].append({"track_id": int(tid), "archivo": destino.name, "bytes": len(datos_mp3), "sha1": s1,
                          "sha1_verificado": esperado is not None, "dur_s": round(dur, 3), "sr": info.samplerate,
                          "canales": info.channels})
        if len(reg["ok"]) % 25 == 0:
            print(f"[descargar] {len(reg['ok'])}/{n_obj} ok, {len(reg['saltados'])} saltados, "
                  f"{f.peticiones} peticiones, {time.time() - t0:.0f} s")
            ruta_reg.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    reg["n_ok"] = len(reg["ok"])
    reg["bytes_total"] = int(sum(c["bytes"] for c in reg["ok"]))
    reg["peticiones_http"] = f.peticiones
    reg["segundos"] = round(time.time() - t0, 1)
    reg["fecha"] = dt.datetime.now().isoformat(timespec="seconds")
    ruta_reg.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(f"[descargar] {reg['n_ok']} clips, {reg['bytes_total'] / 1e6:.1f} MB, {len(reg['saltados'])} saltados, "
          f"{reg['peticiones_http']} peticiones HTTP, {reg['segundos']} s")
    return reg


# --- medición --------------------------------------------------------------------------------------------------------

def localizar_fluidsynth(datos: Path) -> Path | None:
    c = sorted((datos / "tools" / "fluidsynth").glob("**/fluidsynth.exe")) + \
        sorted((datos / "tools" / "fluidsynth").glob("**/fluidsynth"))
    return c[0] if c else None


def eventos_a_midi(eventos, ruta: Path, programa: int = 0, tpq: int = 960, bpm: float = 120.0) -> int:
    """eventos de Basic Pitch: (inicio_s, fin_s, pitch, amplitud, bends). Devuelve número de notas escritas."""
    import mido
    mid = mido.MidiFile(type=1, ticks_per_beat=tpq)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    mid.tracks.append(meta)
    tr = mido.MidiTrack()
    tr.append(mido.Message("program_change", program=programa, channel=0, time=0))
    k = tpq * bpm / 60.0
    ev = []
    for e in eventos:
        t0, t1, pitch, amp = float(e[0]), float(e[1]), int(e[2]), float(e[3])
        a = int(round(t0 * k))
        b = max(a + 1, int(round(t1 * k)))
        vel = int(np.clip(round(127 * amp), 1, 127))
        ev.append((a, 1, pitch, vel))
        ev.append((b, 0, pitch, 0))
    ev.sort(key=lambda x: (x[0], x[1]))
    t = 0
    for tt, on, pitch, vel in ev:
        tr.append(mido.Message("note_on" if on else "note_off", note=pitch, velocity=vel, channel=0, time=tt - t))
        t = tt
    mid.tracks.append(tr)
    mid.save(str(ruta))
    return len(eventos)


def renderizar(fs: Path, sf2: Path, midi: Path, wav: Path) -> None:
    cmd = [str(fs), "-ni", "-q", "-F", str(wav), "-r", str(SR), "-O", "s16", "-T", "wav", "-g", "0.8",
           "-o", "synth.reverb.active=0", "-o", "synth.chorus.active=0", str(sf2), str(midi)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError(f"fluidsynth rc={r.returncode}: {r.stderr[-400:]}")


def a_mono_sr(x: np.ndarray, sr_in: int) -> np.ndarray:
    import librosa
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr_in != SR:
        x = librosa.resample(x, orig_sr=sr_in, target_sr=SR, res_type="soxr_hq")
    return x


def desfase_y_correlacion(env_y: np.ndarray, env_r: np.ndarray) -> tuple[float, float]:
    """(desfase_ms, R). desfase > 0: r va tarde respecto a y. NaN/0 si alguna envolvente es constante."""
    from scipy.ndimage import gaussian_filter1d
    sig = SIGMA_MS / 1000 * SR / HOP_ONSET
    ey, er = gaussian_filter1d(env_y, sig), gaussian_filter1d(env_r, sig)
    if ey.std() < 1e-9 or er.std() < 1e-9:
        return float("nan"), 0.0
    ey = (ey - ey.mean()) / ey.std()
    er = (er - er.mean()) / er.std()
    n = len(ey)
    L = int(round(MAX_LAG_MS / 1000 * SR / HOP_ONSET))
    rho = np.empty(2 * L + 1)
    for i, tau in enumerate(range(-L, L + 1)):
        if tau >= 0:
            a, b = ey[: n - tau], er[tau:]
        else:
            a, b = ey[-tau:], er[: n + tau]
        rho[i] = float(np.dot(a, b) / len(a))
    i = int(np.argmax(rho))
    delta = 0.0
    if 0 < i < len(rho) - 1:
        den = rho[i - 1] - 2 * rho[i] + rho[i + 1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5 * (rho[i - 1] - rho[i + 1]) / den, -1, 1))
    lag = (i - L + delta) * HOP_ONSET / SR * 1000.0
    return float(lag), float(rho[i])


def similitud_croma(y: np.ndarray, r: np.ndarray) -> tuple[float, int, list[float]]:
    import librosa
    cy = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_CROMA)
    cr = librosa.feature.chroma_cqt(y=r, sr=SR, hop_length=HOP_CROMA)
    por_v = int(round(SR / HOP_CROMA))
    T = min(cy.shape[1], cr.shape[1])
    sims = []
    for w in range(T // por_v):
        a = cy[:, w * por_v:(w + 1) * por_v].mean(axis=1)
        b = cr[:, w * por_v:(w + 1) * por_v].mean(axis=1)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        sims.append(float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0)
    return (float(np.median(sims)) if sims else 0.0), len(sims), sims


def completitud(env_y: np.ndarray, onsets_notas: np.ndarray) -> tuple[float, int]:
    import librosa
    t_on = librosa.onset.onset_detect(onset_envelope=env_y, sr=SR, hop_length=HOP_ONSET, units="time")
    if len(t_on) == 0:
        return float("nan"), 0
    if len(onsets_notas) == 0:
        return 0.0, int(len(t_on))
    d = np.abs(t_on[:, None] - onsets_notas[None, :]).min(axis=1)
    return float((d <= TOL_C_S).mean()), int(len(t_on))


def paso_medir(datos: Path, trabajo: Path, limite: int | None = None, usar_fuga: bool = False) -> dict:
    import librosa
    import soundfile as sf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    reg = json.loads((trabajo / "descarga.json").read_text(encoding="utf-8"))
    fs = None if usar_fuga else localizar_fluidsynth(datos)
    sf2 = datos / "tools" / "soundfont" / "MuseScore_General.sf3"
    if fs is None and not usar_fuga:
        raise SystemExit("FluidSynth no encontrado en data/fma_e1/tools/fluidsynth (o pasa --fuga para la síntesis aditiva)")
    if usar_fuga:
        sys.path.insert(0, str(RAIZ))
    dir_midi, dir_ren = trabajo / "midi", trabajo / "render"
    dir_midi.mkdir(exist_ok=True)
    dir_ren.mkdir(exist_ok=True)
    ruta_med = trabajo / "medidas.jsonl"
    hechas = set()
    if ruta_med.exists():
        for linea in ruta_med.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                hechas.add(json.loads(linea)["track_id"])
    modelo = Model(ICASSP_2022_MODEL_PATH)
    pendientes = [c for c in reg["ok"] if c["track_id"] not in hechas]
    if limite:
        pendientes = pendientes[:limite]
    print(f"[medir] {len(pendientes)} clips pendientes ({len(hechas)} ya medidos); render = "
          f"{'fuga.render.a_audio_eventos' if usar_fuga else fs.name}")
    t_ini = time.time()
    with open(ruta_med, "a", encoding="utf-8") as salida:
        for k, c in enumerate(pendientes, 1):
            tid = c["track_id"]
            ruta_mp3 = datos / "clips" / c["archivo"]
            t0 = time.time()
            x, sr_in = sf.read(str(ruta_mp3), dtype="float32", always_2d=True)
            y = a_mono_sr(x, sr_in)
            t_carga = time.time() - t0
            t0 = time.time()
            _, _, eventos = predict(str(ruta_mp3), modelo)
            t_amt = time.time() - t0
            eventos = sorted(eventos, key=lambda e: e[0])
            onsets_notas = np.array([float(e[0]) for e in eventos])
            ruta_mid = dir_midi / f"{tid:06d}.mid"
            n_notas = eventos_a_midi(eventos, ruta_mid)
            t0 = time.time()
            if usar_fuga:
                from fuga.render import a_audio_eventos
                ev = [(float(e[0]), float(e[1]) - float(e[0]), int(e[2]), int(np.clip(round(127 * e[3]), 1, 127)), 0)
                      for e in eventos]
                r = a_audio_eventos(ev, sr=SR, reverb=0.0) if ev else np.zeros(len(y), dtype=np.float32)
            else:
                ruta_wav = dir_ren / f"{tid:06d}.wav"
                if n_notas:
                    renderizar(fs, sf2, ruta_mid, ruta_wav)
                    xr, sr_r = sf.read(str(ruta_wav), dtype="float32", always_2d=True)
                    r = a_mono_sr(xr, sr_r)
                else:
                    r = np.zeros(len(y), dtype=np.float32)
            t_render = time.time() - t0
            if len(r) < len(y):
                r = np.pad(r, (0, len(y) - len(r)))
            r = r[: len(y)]
            t0 = time.time()
            env_y = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_ONSET)
            env_r = librosa.onset.onset_strength(y=r, sr=SR, hop_length=HOP_ONSET)
            desfase, R = desfase_y_correlacion(env_y, env_r) if n_notas else (float("nan"), 0.0)
            C, n_vent, sims = similitud_croma(y, r) if n_notas else (0.0, 0, [])
            c_idx, n_on = completitud(env_y, onsets_notas)
            S = 0.5 * (C + R)
            t_met = time.time() - t0
            fila = {
                "track_id": tid, "dur_s": round(len(y) / SR, 3), "n_notas": int(n_notas),
                "n_onsets_y": n_on, "notas_por_s": round(n_notas / (len(y) / SR), 3),
                "pitch_min": int(min(e[2] for e in eventos)) if eventos else None,
                "pitch_max": int(max(e[2] for e in eventos)) if eventos else None,
                "desfase_ms": None if math.isnan(desfase) else round(desfase, 2),
                "R_onsets": round(R, 4), "C_croma": round(C, 4), "n_ventanas_croma": n_vent,
                "c_completitud": None if math.isnan(c_idx) else round(c_idx, 4),
                "S": round(S, 4), "pasa": bool(n_notas > 0 and S >= TAU),
                "t_s": {"carga": round(t_carga, 2), "amt": round(t_amt, 2), "render": round(t_render, 2),
                        "metricas": round(t_met, 2)},
            }
            salida.write(json.dumps(fila) + "\n")
            salida.flush()
            if k % 10 == 0 or k == len(pendientes):
                print(f"[medir] {k}/{len(pendientes)} ({(time.time() - t_ini) / k:.1f} s/clip) último: {tid} "
                      f"notas={n_notas} desfase={fila['desfase_ms']} R={R:.2f} C={C:.2f} S={S:.2f} pasa={fila['pasa']}")
    return {"medidos": len(hechas) + len(pendientes)}


# --- veredicto -------------------------------------------------------------------------------------------------------

def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def _ic_fraccion(k: int, n: int) -> list[float]:
    """IC95 de Wilson para una fracción."""
    if n == 0:
        return [None, None]
    z, p = 1.959964, k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    semi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centro - semi, 4), round(centro + semi, 4)]


def _ic_mediana_boot(v: np.ndarray, n_boot: int = 2000) -> list[float]:
    if len(v) == 0:
        return [None, None]
    rng = np.random.default_rng(SEMILLA)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return [round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)]


def versiones(datos: Path) -> dict:
    v = {"python": sys.version.split()[0], "plataforma": platform.platform()}
    from importlib.metadata import version as _v
    for dist in ("basic-pitch", "onnxruntime", "librosa", "soundfile", "numpy", "scipy", "mido", "pandas", "requests",
                 "pretty_midi", "resampy", "soxr"):
        try:
            v[dist] = _v(dist)
        except Exception:                           # noqa: BLE001
            v[dist] = None
    try:
        import soundfile as sf
        v["libsndfile"] = sf.__libsndfile_version__
    except Exception:                               # noqa: BLE001
        pass
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        v["basic_pitch_modelo"] = {"ruta": Path(ICASSP_2022_MODEL_PATH).name,
                                   "sha256": sha256_archivo(Path(ICASSP_2022_MODEL_PATH))}
    except Exception:                               # noqa: BLE001
        pass
    fs = localizar_fluidsynth(datos)
    if fs:
        r = subprocess.run([str(fs), "--version"], capture_output=True, text=True)
        v["fluidsynth"] = (r.stdout.splitlines() or ["?"])[0]
    sf2 = datos / "tools" / "soundfont" / "MuseScore_General.sf3"
    if sf2.exists():
        ver = (datos / "tools" / "soundfont" / "VERSION")
        v["soundfont"] = {"archivo": sf2.name, "version": ver.read_text().strip() if ver.exists() else None,
                          "sha256": sha256_archivo(sf2), "licencia": "MIT (MuseScore_General_License.md)",
                          "url": "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/"}
    return v


def paso_veredicto(datos: Path, trabajo: Path, salida_json: Path) -> dict:
    lic = json.loads((trabajo / "licencias.json").read_text(encoding="utf-8"))
    desc = json.loads((trabajo / "descarga.json").read_text(encoding="utf-8"))
    filas = [json.loads(ln) for ln in (trabajo / "medidas.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids_ok = {c["track_id"] for c in desc["ok"]}
    filas = [f for f in filas if f["track_id"] in ids_ok]
    n = len(filas)
    con_notas = [f for f in filas if f["n_notas"] > 0]
    des = np.array([f["desfase_ms"] for f in con_notas if f["desfase_ms"] is not None])
    abs_des = np.abs(des)
    R = np.array([f["R_onsets"] for f in filas])
    C = np.array([f["C_croma"] for f in filas])
    S = np.array([f["S"] for f in filas])
    cc = np.array([f["c_completitud"] for f in filas if f["c_completitud"] is not None])
    pasan = int(sum(f["pasa"] for f in filas))
    frac = pasan / n if n else float("nan")
    h_E = lic["horas"]["E_pistas_completas"]
    proy = {
        "primaria_E_pistas_completas_h": h_E * frac,
        "secundaria_E_clips_30s_h": lic["horas"]["E_clips_30s"] * frac,
        "secundaria_E_con_by_sa_pistas_completas_h": lic["horas"]["E_con_by_sa_pistas_completas"] * frac,
    }
    med_abs = float(np.median(abs_des)) if len(abs_des) else None
    D1 = (med_abs is None) or (med_abs > UMBRAL_MS)
    D2 = frac < 0.30
    D3 = proy["primaria_E_pistas_completas_h"] < 300
    veredicto = "SOBREVIVE" if not (D1 or D2 or D3) else "MUERE (" + ", ".join(
        k for k, v in (("D1", D1), ("D2", D2), ("D3", D3)) if v) + ")"
    res = {
        "experimento": "E1 (RFC-FUGA-2 § 5): fábrica de pares render↔audio real sobre clips FMA CC BY/CC0",
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "prerregistro": "ensayo/e1_prerregistro.md",
        "script": "scripts/e1_pares.py",
        "versiones": versiones(datos),
        "parametros": {"sr": SR, "hop_onset": HOP_ONSET, "hop_croma": HOP_CROMA, "sigma_ms": SIGMA_MS,
                       "max_lag_ms": MAX_LAG_MS, "tau": TAU, "tol_completitud_s": TOL_C_S, "umbral_ms": UMBRAL_MS,
                       "semilla": SEMILLA, "basic_pitch": "por defecto (onset 0.5, frame 0.3, nota mínima 127.7 ms)",
                       "fluidsynth": "-ni -F -r 22050 -O s16 -T wav -g 0.8 reverb=0 chorus=0, GM programa 0"},
        "a_licencias": {k: lic[k] for k in ("fuente", "n_pistas", "clases_finas", "cubos", "filtro_voz",
                                              "embudo_elegibles", "horas", "elegibles_por_genre_top",
                                              "elegibles_por_subset")},
        "descarga": {"n_ok": desc["n_ok"], "bytes_total": desc["bytes_total"], "saltados": desc["saltados"],
                     "zip": desc.get("zip"), "peticiones_http": desc.get("peticiones_http"),
                     "segundos": desc.get("segundos")},
        "b_desfase": {
            "n_clips_con_notas": len(con_notas), "n_sin_notas": n - len(con_notas),
            "mediana_abs_ms": med_abs, "ic95_boot_mediana_abs_ms": _ic_mediana_boot(abs_des),
            "p25_abs_ms": _q(abs_des, 25), "p75_abs_ms": _q(abs_des, 75), "p90_abs_ms": _q(abs_des, 90),
            "mediana_con_signo_ms": float(np.median(des)) if len(des) else None,
            "media_con_signo_ms": float(np.mean(des)) if len(des) else None,
            "fraccion_abs_le_23ms": float((abs_des <= UMBRAL_MS).mean()) if len(abs_des) else None,
            "fraccion_abs_le_11_6ms": float((abs_des <= 11.6).mean()) if len(abs_des) else None,
            "fraccion_en_tope_250ms": float((abs_des >= MAX_LAG_MS - 6).mean()) if len(abs_des) else None,
        },
        "c_croma": {"mediana_C": float(np.median(C)), "p25": _q(C, 25), "p75": _q(C, 75),
                    "mediana_R_onsets": float(np.median(R)), "R_p25": _q(R, 25), "R_p75": _q(R, 75)},
        "completitud": {"mediana_c": float(np.median(cc)) if len(cc) else None, "p25": _q(cc, 25), "p75": _q(cc, 75),
                        "n": int(len(cc))},
        "d_compuerta": {
            "tau": TAU, "n": n, "pasan": pasan, "fraccion": frac, "ic95_wilson": _ic_fraccion(pasan, n),
            "mediana_S": float(np.median(S)), "S_p25": _q(S, 25), "S_p75": _q(S, 75),
            "secundarias": {"fraccion_C_ge_tau": float((C >= TAU).mean()), "fraccion_R_ge_tau": float((R >= TAU).mean()),
                            "fraccion_min_ge_tau": float((np.minimum(C, R) >= TAU).mean())},
        },
        "horas_proyectadas": proy,
        "criterios_de_muerte": {
            "D1_mediana_abs_desfase_gt_23ms": {"valor_ms": med_abs, "umbral_ms": UMBRAL_MS, "dispara": bool(D1)},
            "D2_pasan_lt_30pct": {"valor": frac, "umbral": 0.30, "dispara": bool(D2)},
            "D3_horas_utiles_lt_300": {"valor_h": proy["primaria_E_pistas_completas_h"], "umbral_h": 300,
                                       "dispara": bool(D3)},
        },
        "veredicto": veredicto,
        "clips": filas,
    }
    salida_json.parent.mkdir(parents=True, exist_ok=True)
    salida_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[veredicto] n={n} | mediana |desfase| = {med_abs} ms | pasan {pasan}/{n} = {frac:.1%} | "
          f"horas proyectadas {proy['primaria_E_pistas_completas_h']:.0f} h | {veredicto}")
    return res


# --- cli -------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["licencias", "descargar", "medir", "veredicto", "todo"])
    ap.add_argument("--datos", default=str(RAIZ / "data" / "fma_e1"))
    ap.add_argument("--n", type=int, default=300, help="clips a descargar")
    ap.add_argument("--limite", type=int, default=None, help="medir solo los primeros k pendientes (pruebas)")
    ap.add_argument("--fuga", action="store_true", help="renderizar con fuga.render.a_audio_eventos en vez de FluidSynth")
    ap.add_argument("--salida", default=str(RAIZ / "ensayo" / "e1_pares.json"))
    a = ap.parse_args(argv)
    datos = Path(a.datos)
    trabajo = datos / "trabajo"
    trabajo.mkdir(parents=True, exist_ok=True)
    if a.paso in ("licencias", "todo"):
        paso_licencias(datos, trabajo)
    if a.paso in ("descargar", "todo"):
        paso_descargar(datos, trabajo, a.n)
    if a.paso in ("medir", "todo"):
        paso_medir(datos, trabajo, a.limite, a.fuga)
    if a.paso in ("veredicto", "todo"):
        paso_veredicto(datos, trabajo, Path(a.salida))


if __name__ == "__main__":
    main()
