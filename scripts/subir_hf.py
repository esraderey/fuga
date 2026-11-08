"""Sube una carpeta exportada por scripts/exportar_hf.py a un repositorio de modelos de Hugging Face.

Sin `--confirmar` solo lista lo que subiría y comprueba la sesión. Requiere un token con permiso de escritura, en la
variable HF_TOKEN o guardado con `hf auth login`.

Uso:
  python scripts/subir_hf.py --dir hf/fuga-s-eco-rel --repo usuario/fuga-s-eco-rel            # ensayo en seco
  python scripts/subir_hf.py --dir hf/fuga-s-eco-rel --repo usuario/fuga-s-eco-rel --confirmar
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sube una carpeta de modelo FUGA a Hugging Face")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True, help="usuario/nombre del repositorio de modelos")
    ap.add_argument("--privado", action="store_true", help="crea el repositorio como privado")
    ap.add_argument("--mensaje", default="FUGA: export de pesos, ficha y código")
    ap.add_argument("--confirmar", action="store_true", help="sin esta opción solo se lista lo que se subiría")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi

    obligatorios = ("model.safetensors", "config.json", "README.md", "LICENSE", "fuga/sample.py")
    faltan = [x for x in obligatorios if not os.path.exists(os.path.join(a.dir, x))]
    if faltan:
        print("faltan archivos en la carpeta:", ", ".join(faltan), file=sys.stderr)
        return 2
    archivos = [(os.path.relpath(os.path.join(d, x), a.dir).replace(os.sep, "/"), os.path.getsize(os.path.join(d, x)))
                for d, _, xs in os.walk(a.dir) for x in xs if "__pycache__" not in d]
    total = sum(t for _, t in archivos)
    for ruta, t in sorted(archivos):
        print(f"{t/1e6:9.1f} MB  {ruta}")
    print(f"{total/1e6:9.1f} MB  total en {len(archivos)} archivos -> {a.repo} ({'privado' if a.privado else 'público'})")
    api = HfApi()
    try:
        quien = api.whoami()
        print("sesión de Hugging Face:", quien.get("name"))
    except Exception as e:  # sin token o token inválido
        print("sin sesión de Hugging Face:", type(e).__name__, "-> exporta HF_TOKEN o ejecuta `hf auth login`", file=sys.stderr)
        return 3
    if not a.confirmar:
        print("ensayo en seco: nada subido (añade --confirmar para subir)")
        return 0
    url = api.create_repo(a.repo, repo_type="model", private=a.privado, exist_ok=True)
    info = api.upload_folder(folder_path=a.dir, repo_id=a.repo, repo_type="model", commit_message=a.mensaje,
                             ignore_patterns=["__pycache__/*", "*.pyc"])
    print("subido:", url, "| commit:", getattr(info, "oid", info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
