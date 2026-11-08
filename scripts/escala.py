"""Prueba de escala: instancia FUGA-S/M, mide un paso fwd+bwd y estima la memoria de entrenamiento.

Uso: python scripts/escala.py [--tokens 512] [--device auto|cpu|cuda] [--precision fp32|bf16] [--out ensayo/escala.json]
Portable: el pico de memoria residente sale de psapi en Windows y de getrusage en POSIX.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from fuga.model import Transformer, config, contar_parametros  # noqa: E402

VOCAB = 760


def rss_max_gb() -> float:
    """Pico de memoria residente del proceso en GB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        campos = ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                  "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in campos]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE          # sin esto el handle se trunca a 32 bits y falla
        fn = psapi.GetProcessMemoryInfo
        fn.argtypes, fn.restype = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD], wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo")
        return pmc.PeakWorkingSetSize / 1e9
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e9 if sys.platform == "darwin" else ru / 1e6   # macOS en bytes, Linux en KB


def memoria_entreno_gb(n: int, capas: int, d: int, tokens: int, ckpt: bool = True) -> dict:
    """Estimación: AdamW mixto (16 B/param) + activaciones bf16 (~34·d B/token/capa sin ckpt, ~2·d con ckpt)."""
    estado = 16 * n
    act = tokens * capas * d * (2 if ckpt else 34)
    return {"estado_gb": round(estado / 1e9, 2), "activaciones_gb": round(act / 1e9, 2),
            "total_gb": round((estado + act) / 1e9, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32", help="bf16 solo aplica en CUDA")
    ap.add_argument("--presets", nargs="+", default=["fuga-s", "fuga-m"])
    ap.add_argument("--out", default="ensayo/escala.json")
    a = ap.parse_args()
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    bf16 = dev.type == "cuda" and a.precision == "bf16"
    torch.set_num_threads(2)
    res = {"device": str(dev), "precision": "bf16" if bf16 else "fp32", "tokens": a.tokens}
    if dev.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(dev)
    for p in a.presets:
        torch.manual_seed(0)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        m = Transformer(config(p, vocab=VOCAB)).to(dev)
        n = contar_parametros(m)
        ids = torch.randint(0, VOCAB, (1, a.tokens), device=dev)
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            lg = m(ids)[:, :-1]
        loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        loss.backward()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dt = time.time() - t0
        c = m.c
        r = {"parametros": n, "d": c.d, "capas": c.capas, "cabezas": c.cabezas, "kv": c.cabezas_kv,
             "ffn": c.ffn, "ctx": c.ctx, "loss_inicial": round(loss.item(), 3),
             "ln_vocab": round(float(torch.log(torch.tensor(float(VOCAB)))), 3),
             "seg_fwd_bwd": round(dt, 2), "rss_max_gb": round(rss_max_gb(), 2),
             "grad_finito": all(torch.isfinite(q.grad).all().item() for q in m.parameters() if q.grad is not None),
             "estimacion_ctx4096_lote8_ckpt": memoria_entreno_gb(n, c.capas, c.d, 8 * 4096)}
        if dev.type == "cuda":
            r["cuda_pico_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
        res[p] = r
        print(json.dumps({p: r}), flush=True)
        del m, lg, loss
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
