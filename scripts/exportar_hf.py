"""Exporta un checkpoint de FUGA a una carpeta lista para subir a Hugging Face.

Escribe model.safetensors, config.json, una ficha de modelo (README.md) generada desde los registros de la corrida,
una copia del paquete `fuga/` para poder cargar y generar sin este repositorio, requirements.txt y muestras opcionales.

Uso:
  python scripts/exportar_hf.py --ckpt runs/fuga_s_6h/modelo_mejor.pt --run runs/fuga_s_6h --out hf/fuga-s-eco-rel \
      --muestras muestras/fuga_s_6h/romantico12/muestra_06_romantico.wav muestras/fuga_s_6h/romantico12/muestra_06_romantico.mid
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuga.model import ConfigModelo, Transformer, contar_parametros  # noqa: E402

ARQUITECTURA = ("Transformer solo decodificador: RMSNorm, RoPE con un eje por índice de compás en una cabeza kv, GQA, "
                "QK-norm, SwiGLU y embeddings atados. Tokenizador por compás y parte con operadores ECO/ECOR y marco tonal "
                "relativo (modo eco_rel). Decodificación restringida por gramática.")


def _leer_json(ruta: str) -> dict:
    if not ruta or not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, dec=3):
    return "—" if x is None else f"{x:.{dec}f}".replace(".", ",")


def leer_run(run: str, ckpt: str) -> dict:
    """Métricas de la corrida que acompañan al checkpoint (todas opcionales)."""
    log = _leer_json(os.path.join(run, "log.json")) if run else {}
    eleccion = _leer_json(os.path.join(run, "eleccion.json")) if run else {}
    analisis = _leer_json(os.path.join(run, "analisis_muestras.json")) if run else {}
    nombre = os.path.basename(ckpt)
    evals = eleccion.get("evaluaciones", {})
    propia = next((v for k, v in evals.items() if os.path.basename(k) == nombre), None)
    if propia is None and nombre in eleccion.get("val_completo", {}):   # formato de ensayo/fuga_s_3h
        propia = {"val": eleccion["val_completo"][nombre], "test": eleccion.get("test_completo", {}).get(nombre, {})}
        eleccion.setdefault("test_trozos", eleccion.get("test_completo", {}).get("trozos"))
    m = {"args": log.get("args", {}), "parametros": log.get("parametros"), "final": log.get("final", {}),
         "mejor": log.get("mejor", {}), "val": propia["val"]["bits_por_nota"] if propia else None,
         "test": propia["test"]["bits_por_nota"] if propia else None,
         "val_trozos": eleccion.get("val_trozos"), "test_trozos": eleccion.get("test_trozos"),
         "device": log.get("device"), "precision": log.get("precision"),
         "paso": log.get("mejor", {}).get("paso") if nombre == "modelo_mejor.pt" else log.get("final", {}).get("paso")}
    if analisis:
        mu = analisis["muestras"]
        m["muestras"] = {"n": len(mu), "copia4": max(x["copia4"] for x in mu), "copia2": max(x["copia2"] for x in mu),
                         "en_escala": sum(x["en_escala"] for x in mu) / len(mu),
                         "repeticion": sum(x["repeticion"] for x in mu) / len(mu),
                         "mayores": sum(x["modo"] == "major" for x in mu), "base": analisis["base_real"]}
    return m


def fuentes_corpus(datos: str) -> list[tuple[str, int]]:
    if not datos or not os.path.exists(datos):
        return []
    c: collections.Counter = collections.Counter()
    with open(datos, encoding="utf-8") as f:
        for ln in f:
            c[json.loads(ln)["fuente"]] += 1
    return c.most_common()


def parrafo_checkpoint(ckpt: str, m: dict) -> str:
    """Qué es este checkpoint dentro de su corrida: el modelo final o el mínimo de validación parcial."""
    corrida = os.path.basename(os.path.dirname(ckpt)) or "—"
    total = m["final"].get("paso")
    if os.path.basename(ckpt) == "modelo.pt" or (m["paso"] is not None and m["paso"] == total):
        return (f"Este checkpoint es el modelo final de la corrida `{corrida}` ({total or '—'} pasos con calentamiento, "
                "meseta y decaimiento completo de la tasa de aprendizaje).")
    return (f"Este checkpoint es `{os.path.basename(ckpt)}` de la corrida `{corrida}`: el punto de menor bits por nota en "
            f"validación parcial durante el entrenamiento (paso {m['paso'] or '—'} de {total or '—'}), no el modelo final.")


def bloque_seleccion(sel: dict | None) -> str:
    """Tabla de candidatos y regla de elección, si la cadena final dejó un seleccion.json."""
    if not sel:
        return ""
    filas = "\n".join(f"| `{x['corrida']}` | `{x['archivo']}` | {x['paso']} de {x['pasos_corrida']} | {_fmt(x['val'])} | {_fmt(x['test'])} | "
                      f"{'**publicado**' if x['ckpt'] == sel['elegido'] else ''} |" for x in sel["candidatos"])
    txt = ["## Selección del modelo", "",
           f"Regla fijada antes de entrenar: {sel['regla']}. El test no interviene en la elección.", "",
           "| Corrida | Checkpoint | Paso | Val bits/nota | Test bits/nota | |", "|---|---|---|---|---|---|", filas, ""]
    k1 = sel.get("KF1")
    if k1:
        ic = k1["IC95_pareado"]
        txt += [f"Contraste pre-registrado entre el mejor presupuesto corto y el mejor checkpoint de la corrida larga, pareado "
                f"por pieza en test ({k1['n_piezas']} piezas): {_fmt(ic[0], 4)} [{_fmt(ic[1], 4)}; {_fmt(ic[2], 4)}] bits por nota "
                f"({k1['veredicto']}).", ""]
    return "\n".join(txt)


def ficha(nombre: str, licencia: str, c: ConfigModelo, n_par: int, modo: str, m: dict, fuentes: list, sha: str,
          ckpt: str, muestras: list[str], autor: str, seleccion: dict | None = None) -> str:
    a = m["args"]
    tokens = m["final"].get("tokens_vistos")
    front = "\n".join(["---", f"license: {licencia.lower()}", "library_name: pytorch",
                       "tags:", "  - music", "  - symbolic-music", "  - midi", "  - music-generation", "  - transformer",
                       "  - fuga", "---"])
    tabla_res = "\n".join(["| Conjunto | Bits por nota |", "|---|---|",
                           f"| Validación completa ({m['val_trozos'] or '—'} trozos) | {_fmt(m['val'])} |",
                           f"| Test completo ({m['test_trozos'] or '—'} trozos) | {_fmt(m['test'])} |"])
    filas_ent = [("Corpus", f"{sum(n for _, n in fuentes) or '—'} piezas de dominio público (music21), partición por libro"),
                 ("Modo", f"`{modo}`"), ("Parámetros", f"{n_par/1e6:.1f}M".replace(".", ",")),
                 ("Contexto", f"{c.ctx} tokens"), ("Tokens vistos", f"{tokens/1e6:.1f}M".replace(".", ",") if tokens else "—"),
                 ("Paso del checkpoint", str(m["paso"] or "—")),
                 ("Presupuesto", f"{a.get('minutos'):g} min" if a.get("minutos") else f"{a.get('pasos') or '—'} pasos"),
                 ("Optimizador", f"AdamW, lr {a.get('lr', '—')}, decaimiento {a.get('wd', '—')}, dropout {a.get('dropout', '—')}, WSD"),
                 ("Hardware", f"{m.get('device') or '—'}, {m.get('precision') or '—'}")]
    tabla_ent = "\n".join(["| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in filas_ent]])
    tabla_fuentes = "\n".join(["| Fuente (music21) | Piezas |", "|---|---|", *[f"| {k} | {n} |" for k, n in fuentes]]) if fuentes else "_(sin corpus a mano al exportar)_"
    mu = m.get("muestras")
    bloque_mu = ""
    if mu:
        b = mu["base"]
        bloque_mu = "\n".join([
            f"Sobre {mu['n']} piezas de 16 compases generadas con semillas 0 a {mu['n']-1} (temperatura 1,0, top-p 0,95):", "",
            "| Medida | Generado | Real (test frente a train) |", "|---|---|---|",
            f"| Copia de ventanas de 4 compases presentes en train | {_fmt(mu['copia4'])} (máximo por pieza) | {_fmt(b['test_real_copia4'])} |",
            f"| Copia de ventanas de 2 compases | {_fmt(mu['copia2'])} | {_fmt(b['test_real_copia2'])} |",
            f"| Notas en la escala declarada | {_fmt(mu['en_escala'])} | {_fmt(b['test_real_en_escala'])} |",
            f"| Repetición de segmentos (ventana de 32 compases) | {_fmt(mu['repeticion'])} | {_fmt(b['test_real_repeticion'])} |",
            f"| Piezas en modo mayor | {mu['mayores']} de {mu['n']} | 73 % del corpus |"])
    audios = "\n".join(f'- `{os.path.basename(x)}`' + (f' <audio controls src="muestras/{os.path.basename(x)}"></audio>' if x.endswith(".wav") else "")
                       for x in muestras) or "_(sin muestras adjuntas)_"
    return f"""{front}

# {nombre}

**English summary.** FUGA is a small symbolic-music language model (decoder-only transformer, {n_par/1e6:.0f}M parameters)
trained on a public-domain corpus of folk songs and Renaissance/Baroque polyphony. It writes 8–16 bar tonal pieces as
quantized notes, with explicit repetition operators (ECO), and ships with a deterministic expressive performer
(FUGA-I) that renders MIDI and WAV. Weights are in safetensors; the `fuga/` package in this repo loads and samples them (source and experiments:
https://github.com/esraderey/fuga).
Metrics, training budget, data sources and limitations are documented below in Spanish.

## Qué es

{ARQUITECTURA}

{parrafo_checkpoint(ckpt, m)}

## Resultados

{tabla_res}

Bits por nota es la pérdida total en bits de todos los tokens de una pieza dividida por su número de notas; menor es
mejor. Referencias del proyecto sobre los mismos conjuntos: FUGA-S a 700 pasos 4,49 (val) / 4,87 (test); nano (2,5M)
5,25 (val).

{bloque_seleccion(seleccion)}Los registros de entrenamiento, evaluación y selección, cuando acompañan a este export, están en `corrida/`.

## Entrenamiento

{tabla_ent}

## Muestras

{bloque_mu}

Archivos adjuntos en `muestras/`:

{audios}

## Uso

```bash
pip install torch numpy mido safetensors
python -m fuga.sample --ckpt . --n 4 --compases 16 --wav --interpretar romantico --out salida/
```

`fuga.sample` escribe por pieza un MIDI y un WAV planos y, con `--interpretar`, la ejecución expresiva (estilos `plano`,
`neutro`, `romantico`, `mecanico`). Para cargar desde Python:

```python
from fuga.sample import cargar_modelo, generar
modelo, voc, op = cargar_modelo(".")          # carpeta con model.safetensors y config.json
dec = generar(modelo, voc, op["eco"], None, modelo.c.ctx, 16, 1.0, 0.95, 0.0, semilla=0,
              relativo=op["relativo"], fantasma=op["fantasma"])
notas = dec.notas()                            # [(inicio, duración, altura MIDI, parte)] en ticks de 12 por negra
```

## Datos y licencias

Corpus extraído del corpus de music21 (obras de dominio público) con `scripts/extraer_corpus.py` del proyecto FUGA,
cuantizado a 12 ticks por negra; partición train/val/test estable por libro de origen.

{tabla_fuentes}

Pesos y código bajo licencia {licencia}. Las piezas generadas no reproducen ventanas de 2 ni 4 compases del corpus de
entrenamiento según el detector del proyecto (tabla de muestras), pero el modelo aprende de obras concretas: comprueba
las fuentes antes de usos que exijan originalidad garantizada.

## Limitaciones

- Corpus pequeño (unos 2 millones de notas, {sum(n for _, n in fuentes) or '—'} piezas): el modelo se sobreajusta en pocas épocas; el
  checkpoint publicado se eligió por validación para evitarlo.
- Estilos limitados a lo que hay en el corpus: canción popular europea y polifonía renacentista y barroca; nada fuera de eso.
- Piezas cortas (8 a 16 compases); sin condicionamiento por texto; sin audio producido (el WAV es un render sintético).
- Una sola semilla de entrenamiento; las métricas no llevan intervalo entre semillas.

## Procedencia

| | |
|---|---|
| Archivo de origen | `{ckpt}` |
| SHA-256 de `model.safetensors` | `{sha}` |
| Exportado | {dt.date.today().isoformat()} con `scripts/exportar_hf.py` |
| Proyecto | FUGA (serie RFC-FUGA, firma OMRI): código, RFC, pre-registros e informes en https://github.com/esraderey/fuga (carpeta `ensayo/`). |
| Autoría | {autor} |
"""


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description="Exporta un checkpoint de FUGA a una carpeta lista para Hugging Face")
    ap.add_argument("--ckpt", required=True, help="modelo.pt o modelo_mejor.pt")
    ap.add_argument("--run", default="", help="carpeta de la corrida con log.json, eleccion.json y analisis_muestras.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nombre", default="FUGA-S eco_rel")
    ap.add_argument("--licencia", default="MIT")
    ap.add_argument("--autor", default="Raul Cruz Acosta")
    ap.add_argument("--datos", default="data/corpus_pd.jsonl", help="para contar las fuentes del corpus en la ficha")
    ap.add_argument("--muestras", nargs="*", default=[], help="archivos .wav/.mid que se copian a <out>/muestras/")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--seleccion", default="", help="seleccion.json de scripts/cadena_final.py (tabla de candidatos en la ficha)")
    a = ap.parse_args(argv)
    from safetensors.torch import save_file

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ConfigModelo(**ck["config"])
    modelo = Transformer(c)
    modelo.load_state_dict(ck["modelo"])
    n_par = contar_parametros(modelo)
    os.makedirs(a.out, exist_ok=True)
    pesos = {k: v.detach().to(getattr(torch, a.dtype)).contiguous() for k, v in ck["modelo"].items()}
    ruta_pesos = os.path.join(a.out, "model.safetensors")
    save_file(pesos, ruta_pesos, metadata={"format": "pt", "arquitectura": "fuga", "modo": ck["modo"]})
    sha = hashlib.sha256(open(ruta_pesos, "rb").read()).hexdigest()
    m = leer_run(a.run, a.ckpt)
    config = {"arquitectura": ARQUITECTURA, "config": ck["config"], "estilos": ck["estilos"], "modo": ck["modo"],
              "parametros": n_par, "dtype": a.dtype,
              "origen": {"checkpoint": a.ckpt, "paso": m["paso"], "exportado": dt.date.today().isoformat()},
              "metricas": {"val_bits_por_nota": m["val"], "test_bits_por_nota": m["test"]},
              "entrenamiento": {k: m["args"].get(k) for k in ("lr", "wd", "dropout", "tokens_por_lote", "pasos", "minutos", "semilla")},
              "sha256_model_safetensors": sha}
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1, ensure_ascii=False)
    destino = os.path.join(a.out, "fuga")
    os.makedirs(destino, exist_ok=True)
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for src in glob.glob(os.path.join(raiz, "fuga", "*.py")):
        shutil.copy2(src, destino)
    with open(os.path.join(a.out, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("torch>=2.1\nnumpy\nmido\nsafetensors\n")
    if os.path.exists(os.path.join(raiz, "LICENSE")):
        shutil.copy2(os.path.join(raiz, "LICENSE"), os.path.join(a.out, "LICENSE"))
    copiadas = []
    if a.muestras:
        os.makedirs(os.path.join(a.out, "muestras"), exist_ok=True)
        for x in a.muestras:
            shutil.copy2(x, os.path.join(a.out, "muestras", os.path.basename(x)))
            copiadas.append(x)
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(ficha(a.nombre, a.licencia, c, n_par, ck["modo"], m, fuentes_corpus(a.datos), sha, a.ckpt, copiadas, a.autor,
                      _leer_json(a.seleccion) or None))
    total = sum(os.path.getsize(os.path.join(d, x)) for d, _, xs in os.walk(a.out) for x in xs)
    print(f"exportado a {a.out}: {n_par/1e6:.1f}M parámetros, {a.dtype}, {total/1e6:.1f} MB en total, sha256 {sha[:12]}…", flush=True)
    return a.out


if __name__ == "__main__":
    main()
