"""Generación con gramática (100% decodificable), KV-cache y perilla de forma (sesgo ECO).

Uso: python -m fuga.sample --ckpt runs/eco/modelo.pt --n 4 --compases 32 --out muestras/
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .interprete import ESTILOS, interpretar
from .model import ConfigModelo, Transformer
from .render import a_audio, a_audio_eventos, a_midi, a_midi_eventos, a_wav
from .tokenizer import Decodificador, Vocab


def _leer_checkpoint(ruta: str) -> dict:
    """Checkpoint .pt de torch.save, o directorio/archivo safetensors exportado por scripts/exportar_hf.py."""
    if os.path.isdir(ruta) or ruta.endswith(".safetensors"):
        from safetensors.torch import load_file  # dependencia opcional: pip install safetensors
        carpeta = ruta if os.path.isdir(ruta) else (os.path.dirname(ruta) or ".")
        pesos = load_file(os.path.join(carpeta, "model.safetensors") if os.path.isdir(ruta) else ruta)
        with open(os.path.join(carpeta, "config.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return {"modelo": pesos, "config": meta["config"], "estilos": meta["estilos"], "modo": meta["modo"]}
    return torch.load(ruta, map_location="cpu", weights_only=False)


def cargar_modelo(ruta: str, device: str = "cpu"):
    ck = _leer_checkpoint(ruta)
    c = ConfigModelo(**ck["config"])
    m = Transformer(c)
    m.load_state_dict(ck["modelo"])
    m.to(torch.device(device)).eval()
    r = ck["modo"].removesuffix("_rel")
    return m, Vocab(ck["estilos"]), {"eco": r != "base", "fantasma": r == "fant", "relativo": ck["modo"].endswith("_rel")}


@torch.no_grad()
def generar(model, voc: Vocab, eco: bool, prefijo: list[str] | None = None, max_tokens: int = 1024,
            max_compases: int = 64, temperatura: float = 1.0, top_p: float = 0.95,
            sesgo_eco: float = 0.0, semilla: int = 0, relativo: bool = False,
            prefijo_ids: list[int] | None = None, forzar: dict[str, str] | None = None,
            fantasma: bool = False) -> Decodificador:
    """Muestrea token a token bajo la gramática.

    prefijo/prefijo_ids: tokens forzados al inicio; forzar: {fase: token} (p. ej. {'key': 'KEY_2_minor'});
    sesgo_eco: suma a los logits de ECO/ECOR (perilla de forma)."""
    g = torch.Generator().manual_seed(semilla)
    dev = next(model.parameters()).device
    dec = Decodificador(voc, eco, relativo, fantasma)
    caches = [{} for _ in model.bloques]
    forzados = list(prefijo_ids or []) + [voc[s] for s in (prefijo or [])]
    forzar = {k: voc[v] for k, v in (forzar or {}).items()}
    ids_eco = {voc["ECO"], voc["ECOR"]}
    t = 0
    ult = None
    lg = None
    while not dec.fin and t < max_tokens:
        perm = dec.permitidos()
        if not perm:
            break
        if forzados:
            i = forzados.pop(0)
            if i not in perm:
                raise ValueError(f"prefijo inválido en {voc.itos[i]}")
        elif dec.fase in forzar:
            i = forzar[dec.fase]
        elif len(perm) == 1:
            i = perm[0]
        elif t == 0:
            i = voc["<bos>"]
        else:
            x = lg / max(temperatura, 1e-4)
            mask = torch.full_like(x, float("-inf"))
            mask[perm] = 0.0
            x = x + mask
            if sesgo_eco:
                for j in ids_eco & set(perm):
                    x[j] += sesgo_eco
            p = torch.softmax(x, -1)
            if top_p < 1.0:
                ps, orden = p.sort(descending=True)
                corte = (ps.cumsum(-1) - ps) > top_p
                ps[corte] = 0
                p = torch.zeros_like(p).scatter(0, orden, ps)
            i = int(torch.multinomial(p / p.sum(), 1, generator=g))
            if i == voc["BAR"] and dec.bar + 1 >= max_compases:
                i = voc["<eos>"]          # tope: cerrar tras completar el último compás
        dec.paso(i)
        ult = i
        pos = torch.tensor([[[t], [dec.bar + 1]]], device=dev)
        lg = model(torch.tensor([[ult]], device=dev), pos, caches)[0, -1].float().cpu()
        t += 1
    return dec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="modelo.pt, o directorio con model.safetensors + config.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--compases", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=0, help="0 = contexto de entrenamiento del modelo")
    ap.add_argument("--temperatura", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--sesgo_eco", type=float, default=0.0)
    ap.add_argument("--prefijo", default="", help="tokens separados por espacio, p. ej. '<bos> STY_bach KEY_7_major TEMPO_6'")
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu | cuda; con modelos nano la CPU es más rápida token a token")
    ap.add_argument("--interpretar", default="", choices=["", *ESTILOS],
                    help="además del render plano, escribe la ejecución expresiva con ese estilo (fuga.interprete)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, voc, op = cargar_modelo(a.ckpt, a.device)
    os.makedirs(a.out, exist_ok=True)
    resumen = []
    for k in range(a.n):
        dec = generar(model, voc, op["eco"], a.prefijo.split() or None, a.max_tokens or model.c.ctx, a.compases,
                      a.temperatura, a.top_p, a.sesgo_eco, semilla=k, relativo=op["relativo"],
                      fantasma=op["fantasma"])
        notas = dec.notas()
        base = os.path.join(a.out, f"muestra_{k:02d}")
        if notas:
            a_midi(notas, base + ".mid", tempo_bpm=dec.tempo)
            if a.wav:
                a_wav(a_audio(notas, tempo_bpm=dec.tempo), base + ".wav")
            if a.interpretar:
                inicios = list(dec.inicios)
                largos = [y - x for x, y in zip(inicios, inicios[1:], strict=False)] + [dec.bl]
                it = interpretar(notas, inicios, largos, tempo_bpm=dec.tempo, estilo=ESTILOS[a.interpretar], semilla=k)
                a_midi_eventos(it.eventos, f"{base}_{a.interpretar}.mid")
                if a.wav:
                    a_wav(a_audio_eventos(it.eventos, semilla=k), f"{base}_{a.interpretar}.wav")
        resumen.append({"muestra": k, "notas": len(notas), "compases": dec.bar + 1, "tonica": dec.tonica_abs,
                        "modo": dec.modo, "tempo": round(dec.tempo)})
        print(resumen[-1], flush=True)
    with open(os.path.join(a.out, "resumen.json"), "w") as f:
        json.dump(resumen, f, indent=1)


if __name__ == "__main__":
    main()
