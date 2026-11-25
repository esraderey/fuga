"""Datos que necesita el entrenamiento fuera de esta máquina: la caché tokenizada de train/val, la caché de test y el
corpus (para el análisis de copia). Viven en un dataset PRIVADO de Hugging Face y se verifican contra el manifiesto
SHA-256 versionado junto a este script, de modo que el pod entrena exactamente con los datos de los ensayos locales.

Uso:
  python scripts/runpod/datos.py manifiesto          # recalcula scripts/runpod/datos.sha256.json desde los archivos locales
  python scripts/runpod/datos.py subir               # sube los archivos al dataset privado (requiere sesión con escritura)
  python scripts/runpod/datos.py bajar               # descarga lo que falte y verifica los hashes (HF_TOKEN o sesión)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFIESTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos.sha256.json")
ARCHIVOS = ("cache/eco_rel_1024.pkl", "cache/test_eco_rel_1024.pkl", "data/corpus_pd.jsonl")
REPO = "esraderey/fuga-datos"
FICHA = """---
license: other
pretty_name: FUGA - datos de entrenamiento (privado)
---

Datos de trabajo del proyecto FUGA (https://github.com/esraderey/fuga): corpus de dominio público extraído del corpus
de music21 y cuantizado a 12 ticks por negra (`data/corpus_pd.jsonl`), y sus cachés tokenizadas en modo `eco_rel` con
contexto 1024 (`cache/`). Repositorio privado: existe para que un pod de entrenamiento use exactamente los mismos datos
que los ensayos locales (hashes en `scripts/runpod/datos.sha256.json` del proyecto). `corridas/` guarda copias de los
registros de entrenamiento.
"""


def sha256(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def leer_manifiesto() -> dict:
    with open(MANIFIESTO, encoding="utf-8") as f:
        return json.load(f)


def estado(base: str = RAIZ) -> dict[str, str]:
    """'ok', 'falta' o 'distinto' por archivo, frente al manifiesto."""
    man = leer_manifiesto()
    out = {}
    for rel in ARCHIVOS:
        ruta = os.path.join(base, rel)
        out[rel] = "falta" if not os.path.exists(ruta) else "ok" if sha256(ruta) == man[rel]["sha256"] else "distinto"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Datos de entrenamiento de FUGA en un dataset privado de Hugging Face")
    ap.add_argument("accion", choices=["manifiesto", "subir", "bajar", "estado"])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--base", default=RAIZ, help="raíz donde están o se dejan cache/ y data/")
    a = ap.parse_args(argv)
    if a.accion == "manifiesto":
        man = {rel: {"sha256": sha256(os.path.join(a.base, rel)), "bytes": os.path.getsize(os.path.join(a.base, rel))}
               for rel in ARCHIVOS}
        with open(MANIFIESTO, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1)
        print(json.dumps(man, indent=1))
        return 0
    est = estado(a.base)
    if a.accion == "estado":
        print(json.dumps(est, indent=1))
        return 0 if all(v == "ok" for v in est.values()) else 1
    from huggingface_hub import HfApi, hf_hub_download
    if a.accion == "subir":
        malos = {k: v for k, v in est.items() if v != "ok"}
        if malos:
            print("los archivos locales no coinciden con el manifiesto:", malos, file=sys.stderr)
            return 2
        api = HfApi()
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=FICHA.encode("utf-8"), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset",
                        commit_message="ficha")
        for rel in ARCHIVOS:
            api.upload_file(path_or_fileobj=os.path.join(a.base, rel), path_in_repo=rel, repo_id=a.repo, repo_type="dataset",
                            commit_message=f"datos: {rel}")
            print("subido", rel, flush=True)
        info = api.repo_info(a.repo, repo_type="dataset")
        print("dataset:", a.repo, "| privado:", info.private)
        return 0
    for rel, v in est.items():                      # bajar
        if v == "ok":
            print("ok (ya estaba)", rel, flush=True)
            continue
        hf_hub_download(a.repo, rel, repo_type="dataset", local_dir=a.base)
        print("descargado", rel, flush=True)
    est = estado(a.base)
    malos = {k: v for k, v in est.items() if v != "ok"}
    if malos:
        print("los datos no coinciden con el manifiesto:", malos, file=sys.stderr)
        return 2
    print("datos verificados contra el manifiesto SHA-256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
