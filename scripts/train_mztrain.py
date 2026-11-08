"""Ensayo pre-registrado: FUGA-M factorizado con MZTrain frente a FUGA-S denso, con el mismo bucle de FUGA.

Ambos brazos comparten datos (caché de fuga.train), lotes por presupuesto de tokens, máscara de pérdida, calendario
WSD, recorte de gradiente, bf16 y evaluación. Solo cambian el modelo (FUGA-S denso | FUGA-M con sus nn.Linear
sustituidas por ZFactorizedLinear de rango fijo) y el optimizador (AdamW | ZCompressedAdam con estados INT8).

Uso:
  python scripts/train_mztrain.py --brazo m_mztrain --rango 512 --probar --cache cache/eco_rel_1024.pkl --tokens_por_lote 8192
  python scripts/train_mztrain.py --brazo s_denso  --cache cache/eco_rel_1024.pkl --pasos 1200 --tokens_por_lote 8192 --semilla 0 --out runs/mz/s_denso_0
  python scripts/train_mztrain.py --brazo m_mztrain --rango 512 --cache ... --pasos 1200 --tokens_por_lote 8192 --semilla 0 --out runs/mz/m_mz_0
  python scripts/train_mztrain.py --contrastar runs/mz/*/log.json --out ensayo/mztrain_resultados.json
El brazo m_mztrain necesita `mztrain` importable (PYTHONPATH a la copia aislada de la versión bajo prueba).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from fuga.data import cargar, colar, lotes, particion, trozos_eval  # noqa: E402
from fuga.model import Transformer, config, config_guardable, contar_parametros  # noqa: E402
from fuga.tokenizer import Vocab  # noqa: E402
from fuga.train import bits_por_nota, lr_wsd  # noqa: E402

PRESET = {"s_denso": "fuga-s", "m_mztrain": "fuga-m", "m_denso": "fuga-m", "nano_denso": "nano", "nano_mztrain": "nano"}


def construir(brazo: str, vocab: int, ctx: int, rango: int, dropout: float, ckpt: bool, dev: torch.device):
    """Modelo del brazo en el dispositivo. Para los brazos MZTrain devuelve además el motor (para exportar a denso)."""
    c = config(PRESET[brazo], vocab=vocab, ctx=ctx, dropout=dropout, ckpt=ckpt)
    model = Transformer(c).to(dev)          # en GPU: las SVD de la factorización corren en cuSOLVER, no en CPU
    if not brazo.endswith("_mztrain"):
        return model, c, None, {}
    import mztrain
    from mztrain import ZTrainConfig, ZTrainEngine
    from mztrain.layers import ZFactorizedLinear

    zc = ZTrainConfig(initial_rank=rango, max_rank=rango, checkpoint_activations=False, use_amp=False,
                      compress_optimizer_states=True)
    motor = ZTrainEngine(model, zc, device=dev)          # sustituye nn.Linear ≥ 4096 params por ZFactorizedLinear
    fact = [m for m in motor.model.modules() if isinstance(m, ZFactorizedLinear)]
    info = {"mztrain_version": getattr(mztrain, "__version__", "?"), "mztrain_archivo": mztrain.__file__,
            "capas_factorizadas": len(fact), "rango": rango,
            "rangos_efectivos": sorted({m.rank for m in fact}),
            "parametros_densos_equivalentes": contar_parametros(model)}
    del model                                # libera la copia densa; el motor conserva la factorizada
    torch.cuda.empty_cache()
    return motor.model, c, motor, info


def optimizador(brazo: str, model, lr: float, wd: float):
    """Misma regla que fuga.train: decaimiento solo en parámetros de dimensión ≥ 2 (U, V, embeddings; no S ni normas)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    grupos = [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}]
    if brazo.endswith("_mztrain"):
        from mztrain.optimizer import ZCompressedAdam
        return ZCompressedAdam(grupos, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd, compress_states=True)
    return torch.optim.AdamW(grupos, lr=lr, betas=(0.9, 0.95))


def paso_entrenamiento(model, opt, ids, pos, m, bf16: bool):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        lg = model(ids, pos)[:, :-1]
    obj, mm = ids[:, 1:], m[:, 1:]
    per = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), obj.reshape(-1), reduction="none").view_as(obj)
    loss = (per * mm).sum() / mm.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss, int(mm.sum())


def memoria(dev) -> dict:
    return {"cuda_pico_asignado_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 3),
            "cuda_pico_reservado_gb": round(torch.cuda.max_memory_reserved(dev) / 1e9, 3)}


def probar(a, model, opt, dev, vocab):
    """Tres pasos con lotes sintéticos: pico de memoria y segundos por paso. No toca datos reales."""
    B = max(1, a.tokens_por_lote // a.ctx)
    ids = torch.randint(0, vocab, (B, a.ctx), device=dev)
    t = torch.arange(a.ctx, device=dev)
    pos = torch.stack([t, t // 48]).expand(B, 2, a.ctx)
    m = torch.ones((B, a.ctx), dtype=torch.bool, device=dev)
    for g in opt.param_groups:
        g["lr"] = a.lr
    tiempos = []
    for _ in range(3):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        paso_entrenamiento(model, opt, ids, pos, m, a.precision == "bf16")
        torch.cuda.synchronize(dev)
        tiempos.append(time.time() - t0)
    return {"lote_secuencias": B, "tokens_por_paso": B * a.ctx, "seg_por_paso": round(float(np.median(tiempos)), 3),
            "tokens_por_seg": round(B * a.ctx / float(np.median(tiempos))), **memoria(dev)}


def trozos_test(a, voc: Vocab, relativo: bool, rep: str):
    """Trozos de test por pieza (cacheados): [(id, (ids, compas, perdida, notas))]."""
    ruta = a.cache_test
    if ruta and os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)
    piezas = [p for p in cargar(a.datos) if particion(p) == "test" and f"STY_{p['fuente']}" in voc.stoi]
    out = []
    for p in piezas:
        out += [(p["id"], t) for t in trozos_eval([p], voc, a.ctx, relativo)[rep]]
    if ruta:
        with open(ruta, "wb") as f:
            pickle.dump(out, f)
    return out


@torch.no_grad()
def nll_por_pieza(model, trozos_p, lote: int = 8) -> dict:
    """{id: [nll_bits, notas]} sobre todos los trozos de test, en fp32 (TF32 si está activo)."""
    model.eval()
    dev = next(model.parameters()).device
    acum: dict[str, list] = {}
    seqs = [(t[0], t[1], t[2]) for _, t in trozos_p]
    for i in range(0, len(seqs), lote):
        idx = list(range(i, min(i + lote, len(seqs))))
        ids, pos, m = (x.to(dev) for x in colar(seqs, idx))
        lg = model(ids, pos)[:, :-1]
        per = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1),
                              reduction="none").view(len(idx), -1)
        nll = ((per * m[:, 1:]).sum(1) / math.log(2)).tolist()
        for j, k in enumerate(idx):
            pid, t = trozos_p[k]
            acc = acum.setdefault(pid, [0.0, 0])
            acc[0] += nll[j]
            acc[1] += t[3]
    model.train()
    return acum


def bootstrap(vals, n=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [float("nan")] * 3
    med = rng.choice(v, (n, len(v))).mean(1)
    return [float(v.mean()), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def contrastar(rutas: list[str], out: str):
    """Contraste pre-registrado: IC95 pareado por pieza de bits/nota (m_mztrain − s_denso), por semilla y agregado."""
    logs = [json.load(open(r, encoding="utf-8")) for r in rutas]
    por = {}
    for lg in logs:
        por.setdefault(lg["args"]["brazo"], {})[lg["args"]["semilla"]] = lg
    res = {"brazos": {b: {s: {"test_bits_por_nota": lg["test"]["bits_por_nota"], "val_bits_por_nota": lg["final"]["bits_por_nota"],
                           "parametros": lg["parametros"], "segundos": lg["final"]["segundos"],
                           "tokens_vistos": lg["final"]["tokens_vistos"], **lg["memoria"]}
                       for s, lg in sem.items()} for b, sem in por.items()},
           "contrastes": {}}
    pares = [("m_mztrain", "s_denso"), ("nano_mztrain", "nano_denso"), ("m_denso", "s_denso")]
    for A, B in pares:
        if A not in por or B not in por:
            continue
        difs_todas = []
        por_semilla = {}
        for s in sorted(set(por[A]) & set(por[B])):
            a_, b_ = por[A][s]["test"]["por_pieza"], por[B][s]["test"]["por_pieza"]
            comunes = sorted(set(a_) & set(b_))
            difs = [(a_[k][0] - b_[k][0]) / a_[k][1] for k in comunes if a_[k][1]]
            por_semilla[s] = {"n_piezas": len(comunes), "IC95_pareado": bootstrap(difs),
                              "agregada": (sum(a_[k][0] for k in comunes) - sum(b_[k][0] for k in comunes))
                                          / max(1, sum(a_[k][1] for k in comunes))}
            difs_todas += difs
        res["contrastes"][f"{A} - {B}"] = {"por_semilla": por_semilla, "IC95_pareado_todas_las_semillas": bootstrap(difs_todas),
                                           "veredicto_K1": "pasa (IC95 < 0)" if bootstrap(difs_todas)[2] < 0 else "falla"}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res["contrastes"], indent=1, ensure_ascii=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brazo", choices=sorted(PRESET), default="s_denso")
    ap.add_argument("--rango", type=int, default=512)
    ap.add_argument("--datos", default="data/corpus_pd.jsonl")
    ap.add_argument("--cache", default="", help="pickle (seqs, ev, estilos) de fuga.train --solo_preparar")
    ap.add_argument("--cache_test", default="", help="pickle de trozos de test por pieza; se crea si no existe")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens_por_lote", type=int, default=8192)
    ap.add_argument("--pasos", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--eval_cada", type=int, default=200)
    ap.add_argument("--probar", action="store_true", help="3 pasos sintéticos: memoria y velocidad, sin datos")
    ap.add_argument("--contrastar", nargs="*", default=None, help="logs de los brazos; escribe el contraste en --out")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.contrastar is not None:
        rutas = [r for pat in a.contrastar for r in sorted(glob.glob(pat))]
        return contrastar(rutas, a.out)
    assert torch.cuda.is_available(), "este ensayo es de GPU"
    dev = torch.device("cuda")
    torch.manual_seed(a.semilla)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bf16 = a.precision == "bf16"

    if a.probar:
        vocab = 723
        model, c, motor, info = construir(a.brazo, vocab, a.ctx, a.rango, a.dropout, a.ckpt, dev)
        opt = optimizador(a.brazo, model, a.lr, a.wd)
        torch.cuda.reset_peak_memory_stats(dev)
        r = {"brazo": a.brazo, "parametros_entrenables": contar_parametros(model), **info, **probar(a, model, opt, dev, vocab)}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        return r

    assert a.pasos and a.cache and a.out, "fija --pasos, --cache y --out"
    with open(a.cache, "rb") as fc:
        seqs, ev, estilos = pickle.load(fc)
    voc = Vocab(estilos)
    relativo, rep = True, "eco"            # la caché del ensayo es eco_rel
    model, c, motor, info = construir(a.brazo, len(voc), a.ctx, a.rango, a.dropout, a.ckpt, dev)
    opt = optimizador(a.brazo, model, a.lr, a.wd)
    n_par = contar_parametros(model)
    print(f"brazo={a.brazo} parámetros entrenables={n_par/1e6:.1f}M {info}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.semilla)
    log = {"args": vars(a), "parametros": n_par, "info": info, "vocab": len(voc), "curva": [], "val": []}
    torch.cuda.reset_peak_memory_stats(dev)
    paso, tok_vistos, t_ini = 0, 0, time.time()
    while paso < a.pasos:
        for idx in lotes(seqs, a.tokens_por_lote, rng):
            if paso >= a.pasos:
                break
            for g in opt.param_groups:
                g["lr"] = lr_wsd(paso / a.pasos, a.lr)
            ids, pos, m = (x.to(dev, non_blocking=True) for x in colar(seqs, idx))
            loss, n_tok = paso_entrenamiento(model, opt, ids, pos, m, bf16)
            paso += 1
            tok_vistos += n_tok
            if paso % 25 == 0:
                log["curva"].append({"paso": paso, "t": round(time.time() - t_ini, 1), "tokens": tok_vistos,
                                     "loss": round(loss.item(), 4)})
                print(f"paso {paso} f={paso/a.pasos:.2f} loss={loss.item():.4f} tok={tok_vistos} "
                      f"t={time.time()-t_ini:.0f}s", flush=True)
            if a.eval_cada and paso % a.eval_cada == 0:
                v = bits_por_nota(model, ev[:200])
                log["val"].append({"paso": paso, **v})
                print(f"  val(sub) bpn={v['bits_por_nota']:.3f}", flush=True)
    seg = round(time.time() - t_ini)
    v = bits_por_nota(model, ev)
    log["final"] = {"paso": paso, "tokens_vistos": tok_vistos, "segundos": seg, **v}
    log["memoria"] = memoria(dev)
    print("FINAL", json.dumps(log["final"]), log["memoria"], flush=True)
    tp = trozos_test(a, voc, relativo, rep)
    por_pieza = nll_por_pieza(model, tp)
    tot = sum(x[1] for x in por_pieza.values())
    log["test"] = {"bits_por_nota": sum(x[0] for x in por_pieza.values()) / max(1, tot), "notas": tot,
                   "piezas": len(por_pieza), "por_pieza": por_pieza}
    print("TEST bits/nota", round(log["test"]["bits_por_nota"], 4), "piezas", len(por_pieza), flush=True)
    denso = motor.export_full_model() if motor is not None else model      # FUGA carga el resultado como denso
    estado = {k: v_.detach().cpu() for k, v_ in denso.state_dict().items()}
    torch.save({"modelo": estado, "config": config_guardable(c), "estilos": estilos, "modo": "eco_rel"},
               os.path.join(a.out, "modelo.pt"))
    with open(os.path.join(a.out, "log.json"), "w", encoding="utf-8") as fo:
        json.dump(log, fo, indent=1, ensure_ascii=False)
    return log


if __name__ == "__main__":
    main()
