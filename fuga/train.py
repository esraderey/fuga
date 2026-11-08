"""Entrenamiento con presupuesto de tiempo o pasos, calendario WSD y evaluación en bits por nota.

Uso: python -m fuga.train --datos data/corpus_pd.jsonl --modo eco --preset nano --pasos 360 --out runs/eco
     --device auto usa CUDA si existe (bf16 + TF32); --pasos es reproducible, --minutos depende de la máquina.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F

from .data import cargar, colar, lotes, particion, secuencias, trozos_eval
from .model import Transformer, config, config_guardable, contar_parametros
from .tokenizer import Vocab


def lr_wsd(f: float, pico: float, calent: float = 0.05, decae: float = 0.2) -> float:
    """Warmup-Stable-Decay sobre la fracción f∈[0,1] del presupuesto."""
    if f < calent:
        return pico * f / calent
    if f < 1 - decae:
        return pico
    return pico * max(0.0, (1 - f) / decae)


def resolver_device(nombre: str) -> torch.device:
    """'auto' → cuda si existe, si no cpu."""
    if nombre == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(nombre)


def guardar_modelo(model, c, estilos, modo: str, ruta: str) -> None:
    """Checkpoint portable: pesos en CPU, config sin flags de ejecucion, estilos y modo."""
    estado = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"modelo": estado, "config": config_guardable(c), "estilos": estilos, "modo": modo}, ruta)


@torch.no_grad()
def bits_por_nota(model, trozos, lote: int = 16) -> dict:
    """Σ NLL (bits) de todos los tokens tras <bos> / Σ notas, en fp32 (TF32 si está activo) en el dispositivo del modelo."""
    model.eval()
    dev = next(model.parameters()).device
    nll, notas, toks = 0.0, 0, 0
    seqs = [(a, c, p) for a, c, p, _ in trozos]
    for i in range(0, len(seqs), lote):
        idx = list(range(i, min(i + lote, len(seqs))))
        ids, pos, m = (x.to(dev) for x in colar(seqs, idx))
        lg = model(ids, pos)[:, :-1]
        obj = ids[:, 1:]
        mm = m[:, 1:]
        per = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), obj.reshape(-1), reduction="none").view_as(obj)
        nll += (per * mm).sum().item() / math.log(2)
        toks += mm.sum().item()
    notas = sum(t[3] for t in trozos)
    model.train()
    return {"bits_por_nota": nll / max(1, notas), "bits_por_token": nll / max(1, toks),
            "tokens_por_nota": toks / max(1, notas), "notas": notas}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--modo", choices=["base", "eco", "fant", "base_rel", "eco_rel", "fant_rel"], default="eco")
    ap.add_argument("--n_transp", type=int, default=2)
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--kv_compas", type=int, default=None)
    ap.add_argument("--tokens_por_lote", type=int, default=8192)
    ap.add_argument("--minutos", type=float, default=0)
    ap.add_argument("--pasos", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="bf16 solo aplica en CUDA")
    ap.add_argument("--ckpt", action="store_true", help="checkpointing de activaciones (menos memoria, más lento)")
    ap.add_argument("--eval_cada", type=int, default=400, help="pasos entre evaluaciones parciales (0 = solo final)")
    ap.add_argument("--max_piezas", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="", help="pickle con (seqs, ev, estilos); se crea si no existe")
    ap.add_argument("--solo_preparar", action="store_true")
    ap.add_argument("--guardar_mejor", action="store_true",
                    help="guarda modelo_mejor.pt cada vez que la evaluacion parcial mejora")
    ap.add_argument("--guardar_cada", type=int, default=0,
                    help="pasos entre checkpoints periodicos en un almacen Z-Space (0 = no); requiere z-space-core")
    ap.add_argument("--zspace", default="", help="raiz del almacen Z-Space (por defecto <out>/zspace)")
    ap.add_argument("--zspace_dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--inicial", default="", help="checkpoint .pt del que partir (arranque en caliente; el optimizador empieza de cero)")
    a = ap.parse_args(argv)
    assert a.minutos or a.pasos, "fija --minutos o --pasos"
    torch.manual_seed(a.semilla)
    torch.set_num_threads(a.hilos)
    dev = resolver_device(a.device)
    bf16 = dev.type == "cuda" and a.precision == "bf16"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(a.semilla)
    os.makedirs(a.out, exist_ok=True)

    relativo = a.modo.endswith("_rel")
    repr_ = a.modo.removesuffix("_rel")
    t0 = time.time()
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fc:
            seqs, ev, estilos = pickle.load(fc)
        voc = Vocab(estilos)
    else:
        piezas = cargar(a.datos)
        if a.max_piezas:
            piezas = piezas[: a.max_piezas]
        estilos = sorted({p["fuente"] for p in piezas})
        voc = Vocab(estilos)
        tr = [p for p in piezas if particion(p) == "train"]
        va = [p for p in piezas if particion(p) == "val"]
        seqs = secuencias(tr, voc, repr_, a.ctx, n_transp=a.n_transp, relativo=relativo)
        ev = trozos_eval(va, voc, a.ctx, relativo)[repr_]
        del piezas, tr, va
        if a.cache:
            with open(a.cache, "wb") as fc:
                pickle.dump((seqs, ev, estilos), fc)
    print(f"secuencias={len(seqs)} tokens={sum(len(s[0]) for s in seqs)} val_trozos={len(ev)} "
          f"prep={time.time()-t0:.0f}s", flush=True)
    if a.solo_preparar:
        return {}

    kw = {"ctx": a.ctx, "dropout": a.dropout, "ckpt": a.ckpt}
    if a.kv_compas is not None:
        kw["kv_compas"] = a.kv_compas
    c = config(a.preset, vocab=len(voc), **kw)
    model = Transformer(c).to(dev)
    if a.inicial:
        ini = torch.load(a.inicial, map_location="cpu", weights_only=False)
        assert ini["config"] == config_guardable(c), "el checkpoint inicial no coincide con la configuracion"
        model.load_state_dict({k: v.to(torch.float32) for k, v in ini["modelo"].items()})
        print(f"pesos iniciales de {a.inicial}", flush=True)
    n_par = contar_parametros(model)
    print(f"parámetros={n_par/1e6:.2f}M vocab={len(voc)} device={dev} precision={'bf16' if bf16 else 'fp32'}",
          flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    log = {"args": vars(a), "parametros": n_par, "vocab": len(voc), "device": str(dev),
           "precision": "bf16" if bf16 else "fp32", "curva": [], "val": []}
    almacen = None
    if a.guardar_cada:
        from .zcheckpoints import AlmacenZ
        almacen = AlmacenZ(a.zspace or os.path.join(a.out, "zspace"), dtype=a.zspace_dtype)
        log["zspace"] = {"raiz": almacen.raiz, "dtype": almacen.dtype, "guardar_cada": a.guardar_cada, "versiones": almacen.versiones}
    paso, tok_vistos, t_ini = 0, 0, time.time()
    presupuesto = a.minutos * 60
    hecho, mejor_bpn = False, float("inf")
    while not hecho:
        for idx in lotes(seqs, a.tokens_por_lote, rng):
            f = (time.time() - t_ini) / presupuesto if a.minutos else paso / a.pasos
            if f >= 1:
                hecho = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_wsd(f, a.lr)
            ids, pos, m = (x.to(dev, non_blocking=True) for x in colar(seqs, idx))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                lg = model(ids, pos)[:, :-1]
            obj, mm = ids[:, 1:], m[:, 1:]
            per = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), obj.reshape(-1), reduction="none").view_as(obj)
            loss = (per * mm).sum() / mm.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            paso += 1
            tok_vistos += int(mm.sum())
            if paso % 25 == 0:
                log["curva"].append({"paso": paso, "t": round(time.time() - t_ini, 1),
                                     "tokens": tok_vistos, "loss": round(loss.item(), 4)})
                print(f"paso {paso} f={f:.2f} loss={loss.item():.4f} tok={tok_vistos}", flush=True)
            if a.eval_cada and paso % a.eval_cada == 0:
                v = bits_por_nota(model, ev[:200])
                log["val"].append({"paso": paso, **v})
                print(f"  val(sub) bpn={v['bits_por_nota']:.3f}", flush=True)
                if a.guardar_mejor and v["bits_por_nota"] < mejor_bpn:
                    mejor_bpn = v["bits_por_nota"]
                    log["mejor"] = {"paso": paso, **v}
                    guardar_modelo(model, c, estilos, a.modo, os.path.join(a.out, "modelo_mejor.pt"))
            if almacen is not None and paso % a.guardar_cada == 0:
                t_z = time.time()
                v_z = almacen.guardar(model.state_dict(), paso,
                                      {"config": config_guardable(c), "estilos": estilos, "modo": a.modo})
                print(f"  zspace v{v_z['version']} paso {paso}: +{v_z['bytes_version']/1e6:.1f} MB, "
                      f"tienda {v_z['bytes_tienda']/1e6:.1f} MB, {time.time()-t_z:.1f}s", flush=True)
    v = bits_por_nota(model, ev)
    log["final"] = {"paso": paso, "tokens_vistos": tok_vistos, "segundos": round(time.time() - t_ini), **v}
    print("FINAL", json.dumps(log["final"]), flush=True)
    guardar_modelo(model, c, estilos, a.modo, os.path.join(a.out, "modelo.pt"))
    if almacen is not None:
        if paso % a.guardar_cada:          # ultima version = modelo final, aunque el paso no sea multiplo
            almacen.guardar(model.state_dict(), paso, {"config": config_guardable(c), "estilos": estilos, "modo": a.modo})
        almacen.cerrar()
    with open(os.path.join(a.out, "log.json"), "w") as fo:
        json.dump(log, fo, indent=1)
    return log


if __name__ == "__main__":
    main()
