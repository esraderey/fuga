"""Evaluación del ensayo FUGA (criterios pre-registrados en ensayo/prerregistro.md).

Uso: python scripts/ensayo.py --datos data/corpus_pd.jsonl --runs runs/base runs/eco runs/fant runs/eco_rel \
        --out ensayo/resultados.json --audio_dir ensayo/audio
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from fuga.data import cargar, colar, particion, trozos_eval  # noqa: E402
from fuga.metrics import _js, copia_ngramas, indice_ngramas  # noqa: E402
from fuga.render import a_audio, a_midi, a_wav  # noqa: E402
from fuga.sample import cargar_modelo, generar  # noqa: E402
from fuga.tokenizer import ESCALAS, codificar, segmentar  # noqa: E402

KK_MAY = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MEN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
LAGS = (1, 2, 4, 8)


def tonalidad_kk(notas) -> tuple[int, str]:
    """Krumhansl–Kessler ponderado por duración (estimador para transposición post hoc)."""
    h = np.zeros(12)
    for _, d, p, _ in notas:
        h[p % 12] += d
    mejor = (-2.0, 0, "major")
    if h.std() == 0:
        return 0, "major"
    for modo, perfil in (("major", KK_MAY), ("minor", KK_MEN)):
        for t in range(12):
            r = np.corrcoef(h, np.roll(perfil, t))[0, 1]
            if r > mejor[0]:
                mejor = (r, t, modo)
    return mejor[1], mejor[2]


def tonalidad_juez(notas) -> tuple[int, str]:
    """Juez independiente del etiquetador del corpus: Temperley–Kostka–Payne (music21)."""
    from music21 import note, stream
    s = stream.Stream()
    for on, d, p, _ in notas:
        n = note.Note(int(p))
        n.quarterLength = max(d, 1) / 12
        s.insert(on / 12, n)
    k = s.analyze("TemperleyKostkaPayne")
    return int(k.tonic.pitchClass), str(k.mode)


def bootstrap(vals, n=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [float("nan")] * 3
    m = rng.choice(v, (n, len(v))).mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


@torch.no_grad()
def nll_por_trozo(model, trozos, lote=16):
    out = []
    dev = next(model.parameters()).device
    seqs = [(a, c, p) for a, c, p, _ in trozos]
    for i in range(0, len(seqs), lote):
        idx = list(range(i, min(i + lote, len(seqs))))
        ids, pos, m = (x.to(dev) for x in colar(seqs, idx))
        lg = model(ids, pos)[:, :-1]
        per = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view(len(idx), -1)
        out += ((per * m[:, 1:]).sum(1) / math.log(2)).tolist()
    return out


def tasas_lag(segs, b_ini, b_fin, min_notas: int = 3):
    """Por lag, sobre segmentos de ≥min_notas: P_exact (idéntico), P_trans (igual salvo transposición),
    P_near (fuente de ≥min_notas que difiere en 1–2 notas)."""
    res = {}
    for lag in LAGS:
        tot = ex = tr = ne = 0
        for b in range(b_ini, min(b_fin, len(segs))):
            for p, s in segs[b].items():
                if len(s) < min_notas or b - lag < 0:
                    continue
                tot += 1
                f = segs[b - lag].get(p) or []
                if list(f) == list(s):
                    ex += 1
                elif len(f) >= min_notas and 1 <= len(set(f) ^ set(s)) <= 2:
                    ne += 1
                if f and len(f) == len(s) and [(x[0], x[2]) for x in s] == [(y[0], y[2]) for y in f] and \
                        len({x[1] - y[1] for x, y in zip(s, f, strict=True)}) == 1:
                    tr += 1
        res[lag] = (ex / tot if tot else 0.0, tr / tot if tot else 0.0, ne / tot if tot else 0.0)
    return res


def dist_estructura(a, b):
    return sum(abs(a[L][k] - b[L][k]) for L in LAGS for k in range(3))


def en_escala(notas, tonica, modo):
    if not notas:
        return float("nan")
    esc = ESCALAS[modo]
    return sum(((p - tonica) % 12) in esc for _, _, p, _ in notas) / len(notas)


def segs_dict(segs):
    return {(b, p): tuple(s) for b, sb in enumerate(segs) for p, s in sb.items() if s}


def contrastes(nll_por_brazo, d_por_brazo) -> tuple[dict, dict]:
    """Contrastes del pre-registro con su signo (A − B < 0 favorece a A)."""
    pares_b, pares_d = {}, {}
    for a_, b_ in [("eco", "base"), ("fant", "base"), ("fant", "eco"), ("eco_rel", "eco")]:
        if a_ in nll_por_brazo and b_ in nll_por_brazo:
            A, B = nll_por_brazo[a_], nll_por_brazo[b_]
            comunes = sorted(set(A) & set(B))
            pares_b[f"{a_} - {b_}"] = {
                "media_por_pieza_IC95": bootstrap([(A[k][0] - B[k][0]) / A[k][1] for k in comunes if A[k][1]]),
                "agregada": (sum(A[k][0] for k in comunes) - sum(B[k][0] for k in comunes))
                            / max(1, sum(A[k][1] for k in comunes))}
            DA, DB = d_por_brazo.get(a_, {}), d_por_brazo.get(b_, {})
            cd = sorted(set(DA) & set(DB))
            pares_d[f"{a_} - {b_}"] = bootstrap([DA[k] - DB[k] for k in cd])
    return pares_b, pares_d


def fusionar(rutas, out):
    res = None
    for r in rutas:
        with open(r) as f:
            x = json.load(f)
        if res is None:
            res = x
        else:
            res["brazos"].update(x["brazos"])
            for k in ("nll", "d"):
                res["crudo"][k].update(x["crudo"][k])
    d = {a: {int(k): v for k, v in m.items()} for a, m in res["crudo"]["d"].items()}
    res["dif_bits_por_nota_pareada"], res["dif_D_estructura_pareada_por_prompt"] = contrastes(res["crudo"]["nll"], d)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"bits": res["dif_bits_por_nota_pareada"], "D": res["dif_D_estructura_pareada_por_prompt"]}, indent=1))


def main():
    if sys.argv[1:2] == ["--fusionar"]:
        return fusionar(sys.argv[2:-1], sys.argv[-1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=60)
    ap.add_argument("--por_tonalidad", type=int, default=10)
    ap.add_argument("--hilos", type=int, default=1)
    ap.add_argument("--audio_dir", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    t0 = time.time()
    piezas = cargar(a.datos)
    tr = [p for p in piezas if particion(p) == "train"]
    te = [p for p in piezas if particion(p) == "test"]
    rng = random.Random(0)
    prompts = [p for p in te if len(segmentar(p)[1]) >= 16]
    rng.shuffle(prompts)
    prompts = prompts[: a.n_prompts]
    idx_train = indice_ngramas([segs_dict(segmentar(p)[1]) for p in tr])
    real_segs = [segmentar(p)[1] for p in prompts]
    real_lag = [tasas_lag(sg, 4, 16) for sg in real_segs]
    ruido = [dist_estructura(real_lag[i], real_lag[i + 1]) for i in range(0, len(real_lag) - 1, 2)]
    # techo de los jueces de tonalidad sobre música real (8 compases, contra la etiqueta del corpus)
    techo = {"kk": [0, 0], "tkp": [0, 0]}
    for p in te[:200]:
        largos = segmentar(p)[0]
        nr = [n for n in p["notas"] if n[0] < sum(largos[:8])]
        if not nr:
            continue
        for nombre, f in (("kk", tonalidad_kk), ("tkp", tonalidad_juez)):
            t_, m_ = f(nr)
            techo[nombre][0] += (t_ == p["tonica"] and m_ == p["modo"])
            techo[nombre][1] += 1
    res = {"piezas": {"train": len(tr), "test": len(te), "prompts": len(prompts)},
           "copia_4compases_test_real_vs_train": copia_ngramas([segs_dict(segmentar(p)[1]) for p in te], idx_train),
           "D_ruido_real_vs_real": bootstrap(ruido),
           "techo_juez_tonalidad_8compases": {k: v[0] / max(1, v[1]) for k, v in techo.items()},
           "brazos": {}}
    print("base común", json.dumps(res), flush=True)
    nll_por_brazo, d_por_brazo = {}, {}
    for run in a.runs:
        model, voc, op = cargar_modelo(f"{run}/modelo.pt")
        rep = "fant" if op["fantasma"] else "eco" if op["eco"] else "base"
        rel, eco, fan = op["relativo"], op["eco"], op["fantasma"]
        nombre = run.rstrip("/").split("/")[-1]
        ctx = model.c.ctx
        R = {"representacion": rep, "relativo": rel}
        te_b = [p for p in te if f"STY_{p['fuente']}" in voc.stoi]      # un modelo con --max_piezas no conoce todos
        R["piezas_test_omitidas_por_estilo"] = len(te) - len(te_b)
        if not te_b:
            print(nombre, "sin piezas de test de un estilo conocido por el checkpoint: brazo omitido", flush=True)
            res["brazos"][nombre] = R | {"omitido": True}
            continue
        # (a) bits por nota en test (mismos compases en todos los brazos)
        porpieza = defaultdict(lambda: [0.0, 0])
        trozos_p = []
        for p in te_b:
            trozos_p += [(p["id"], t) for t in trozos_eval([p], voc, ctx, rel)[rep]]
        nll = nll_por_trozo(model, [t for _, t in trozos_p])
        for (pid, t), v in zip(trozos_p, nll, strict=True):
            porpieza[pid][0] += v
            porpieza[pid][1] += t[3]
        nll_por_brazo[nombre] = dict(porpieza)
        tot_n = sum(v[1] for v in porpieza.values())
        R["test_bits_por_nota"] = sum(v[0] for v in porpieza.values()) / tot_n
        R["test_tokens_predichos_por_nota"] = sum(
            int((t[2] if t[2] is not None else np.ones(len(t[0]), bool))[1:].sum()) for _, t in trozos_p) / tot_n
        R["test_tokens_contexto_por_nota"] = sum(len(t[0]) for _, t in trozos_p) / tot_n
        # (b) continuaciones: 4 compases de prompt → compases 4..15
        pares, esc_g, esc_r, gen_segs, fallos, comp_gen = {}, [], [], [], [], []
        pcs_g, pcs_r = Counter(), Counter()
        for k, p in enumerate(prompts):
            if f"STY_{p['fuente']}" not in voc.stoi:
                continue
            seg = segmentar(p, 0, rel)
            c, _ = codificar(p, voc, eco=eco, fantasma=fan, b0=0, b1=4, segmentado=seg, relativo=rel)
            try:
                dec = generar(model, voc, eco, prefijo_ids=c.ids, max_tokens=ctx, max_compases=16,
                              semilla=k, relativo=rel, fantasma=fan)
            except Exception as e:  # se registra con su tipo; no anula el resto del brazo
                fallos.append(f"{k}:{type(e).__name__}:{str(e)[:80]}")
                continue
            b_fin = min(16, dec.bar + 1 if dec.fin else dec.bar)   # solo compases completos
            comp_gen.append(b_fin)
            pares[k] = (tasas_lag(dec.segs, 4, b_fin), tasas_lag(real_segs[k], 4, b_fin))
            i4 = dec.inicios[min(4, len(dec.inicios) - 1)]
            ng = [n for n in dec.notas() if n[0] >= i4]
            largos = segmentar(p)[0]
            nr = [n for n in p["notas"] if sum(largos[:4]) <= n[0] < sum(largos[:b_fin])]
            esc_g.append(en_escala(ng, dec.tonica_abs, dec.modo))
            esc_r.append(en_escala(nr, p["tonica"], p["modo"]))
            pcs_g.update((x[2] - dec.tonica_abs) % 12 for x in ng)
            pcs_r.update((x[2] - p["tonica"]) % 12 for x in nr)
            gen_segs.append(segs_dict(dec.segs))
        d_por_brazo[nombre] = {k: dist_estructura(g, r) for k, (g, r) in pares.items()}
        R["estructura"] = {
            "D_gen_vs_real": bootstrap(list(d_por_brazo[nombre].values())),
            "compases_completos_media": float(np.mean(comp_gen)) if comp_gen else 0.0,
            **{f"P_{n}_{w}": {L: float(np.mean([x[i][L][j] for x in pares.values()])) for L in LAGS}
               for j, n in enumerate(("exact", "trans", "near")) for i, w in enumerate(("gen", "real"))},
            "en_escala_gen": float(np.nanmean(esc_g)), "en_escala_real": float(np.nanmean(esc_r)),
            "js_clases_altura": _js(pcs_g, pcs_r), "fallos_decodificacion": fallos,
            "copia_4compases_vs_train": copia_ngramas(gen_segs, idx_train)}
        print(nombre, "estructura", json.dumps(R["estructura"]["D_gen_vs_real"]), flush=True)
        # (c) adherencia tonal: 24 tonalidades × por_tonalidad, 8 compases; juez TKP independiente
        cuenta = Counter()
        for t in range(12):
            for modo in ("major", "minor"):
                for s in range(a.por_tonalidad):
                    dec = generar(model, voc, eco, max_tokens=ctx, max_compases=8, semilla=1000 + s,
                                  relativo=rel, fantasma=fan, forzar={"key": f"KEY_{t}_{modo}"})
                    notas = dec.notas()
                    cuenta["n"] += 1
                    if not notas:            # generación vacía = fallo de adherencia
                        cuenta["vacias"] += 1
                        continue
                    jt, jm = tonalidad_juez(notas)
                    cuenta["tonica"] += jt == t
                    cuenta["modo"] += jm == modo
                    cuenta["exacta"] += (jt == t and jm == modo)
                    et, _ = tonalidad_kk(notas)            # transposición post hoc a la tónica pedida
                    desp = ((t - et + 6) % 12) - 6
                    jt2, jm2 = tonalidad_juez([(o, d, p + desp, q) for o, d, p, q in notas])
                    cuenta["posthoc_exacta"] += (jt2 == t and jm2 == modo)
                    cuenta["posthoc_tonica"] += jt2 == t
        n = max(1, cuenta["n"])
        R["adherencia_tonal_juez_tkp"] = {k: cuenta[k] / n for k in ("exacta", "tonica", "modo", "posthoc_exacta",
                                                                      "posthoc_tonica", "vacias")} | {"n": n}
        print(nombre, "adherencia", json.dumps(R["adherencia_tonal_juez_tkp"]), flush=True)
        # (d) perilla de forma
        if eco:
            perilla = {}
            for sesgo in (-3.0, 0.0, 3.0):
                vals = []
                for s in range(16):
                    dec = generar(model, voc, eco, max_tokens=ctx, max_compases=16, semilla=2000 + s,
                                  relativo=rel, fantasma=fan, sesgo_eco=sesgo)
                    lt = tasas_lag(dec.segs, 1, 16)
                    vals.append(max(lt[L][1] for L in LAGS))
                perilla[str(sesgo)] = bootstrap(vals)
            R["perilla_forma_max_P_trans"] = perilla
        # (e) audio y MIDI de muestra
        if a.audio_dir:
            for s in range(2):
                dec = generar(model, voc, eco, max_tokens=ctx, max_compases=16, semilla=3000 + s,
                              relativo=rel, fantasma=fan)
                notas = dec.notas()
                if notas:
                    base = f"{a.audio_dir}/{nombre}_libre_{s}"
                    a_midi(notas, base + ".mid", tempo_bpm=dec.tempo)
                    a_wav(a_audio(notas, tempo_bpm=dec.tempo), base + ".wav")
        res["brazos"][nombre] = R
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    res["dif_bits_por_nota_pareada"], res["dif_D_estructura_pareada_por_prompt"] = contrastes(nll_por_brazo, d_por_brazo)
    res["crudo"] = {"nll": nll_por_brazo, "d": d_por_brazo}
    res["segundos"] = round(time.time() - t0)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("OK", a.out, res["segundos"], "s", flush=True)


if __name__ == "__main__":
    main()
