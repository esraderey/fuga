"""Ancla del export a Hugging Face: safetensors + config.json cargan igual que el .pt, y la ficha lleva lo esencial."""
import importlib.util
import json
import pathlib

import pytest
import torch

pytest.importorskip("safetensors")

from fuga.sample import cargar_modelo  # noqa: E402
from fuga.train import main as entrenar  # noqa: E402
from test_train import _args, _corpus  # noqa: E402

RUTA = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "exportar_hf.py"


@pytest.fixture(scope="module")
def exportar():
    spec = importlib.util.spec_from_file_location("exportar_hf", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_export_carga_igual_y_ficha_completa(tmp_path, exportar):
    datos = _corpus(tmp_path)
    log = entrenar(_args(datos, tmp_path / "run", "--guardar_mejor"))
    muestra = tmp_path / "pieza.wav"
    muestra.write_bytes(b"RIFF....WAVE")
    out = exportar.main(["--ckpt", str(tmp_path / "run" / "modelo_mejor.pt"), "--run", str(tmp_path / "run"),
                         "--out", str(tmp_path / "hf"), "--datos", datos, "--muestras", str(muestra), "--nombre", "FUGA nano prueba"])
    pt = torch.load(tmp_path / "run" / "modelo_mejor.pt", map_location="cpu", weights_only=False)
    m, voc, op = cargar_modelo(out)                                     # directorio con safetensors + config.json
    sd = m.state_dict()
    assert sd.keys() == pt["modelo"].keys() and all(torch.equal(sd[k], pt["modelo"][k]) for k in sd)
    assert op == {"eco": True, "fantasma": False, "relativo": True} and len(voc) == pt["modelo"]["emb.weight"].shape[0]
    m2 = cargar_modelo(str(tmp_path / "hf" / "model.safetensors"))[0]    # también por archivo
    assert torch.equal(m2.state_dict()["emb.weight"], sd["emb.weight"])
    cfg = json.load(open(tmp_path / "hf" / "config.json", encoding="utf-8"))
    assert cfg["config"] == pt["config"] and cfg["estilos"] == pt["estilos"] and cfg["modo"] == "eco_rel"
    assert cfg["parametros"] == log["parametros"] and cfg["origen"]["paso"] == log["mejor"]["paso"]
    assert len(cfg["sha256_model_safetensors"]) == 64
    ficha = (tmp_path / "hf" / "README.md").read_text(encoding="utf-8")
    assert ficha.startswith("---\nlicense: mit\n") and "# FUGA nano prueba" in ficha
    assert "## Limitaciones" in ficha and "## Procedencia" in ficha and "pieza.wav" in ficha
    assert (tmp_path / "hf" / "fuga" / "sample.py").exists() and (tmp_path / "hf" / "LICENSE").exists()
    assert (tmp_path / "hf" / "muestras" / "pieza.wav").exists() and (tmp_path / "hf" / "requirements.txt").exists()
