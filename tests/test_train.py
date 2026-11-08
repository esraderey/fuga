"""Ancla de fuga.train --guardar_mejor: guarda el checkpoint de la mejor evaluacion parcial y lo registra en el log."""
import json

import torch

from fuga.data import particion
from fuga.sample import cargar_modelo
from fuga.train import main
from test_fuga import pieza_sintetica  # pytest anade tests/ al sys.path


def _corpus(tmp_path):
    piezas = []
    for k in range(40):
        p = pieza_sintetica(semilla=k, compases=8)
        p["id"] = f"lib{k}#0"               # prefijo distinto por pieza: reparte train/val/test
        piezas.append(p)
    partes = {particion(p) for p in piezas}
    assert {"train", "val"} <= partes, partes
    ruta = tmp_path / "corpus.jsonl"
    with open(ruta, "w", encoding="utf-8") as f:
        for p in piezas:
            f.write(json.dumps(p) + "\n")
    return str(ruta)


def _args(datos, out, *extra):
    return ["--datos", datos, "--modo", "eco_rel", "--preset", "nano", "--ctx", "256", "--pasos", "4",
            "--eval_cada", "2", "--device", "cpu", "--tokens_por_lote", "512", "--hilos", "1", "--lr", "1e-3",
            "--out", str(out), *extra]


def _iguales(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_guardar_mejor_guarda_el_minimo_de_val(tmp_path):
    datos = _corpus(tmp_path)
    log = main(_args(datos, tmp_path / "run", "--guardar_mejor"))
    assert [e["paso"] for e in log["val"]] == [2, 4]
    minimo = min(log["val"], key=lambda e: e["bits_por_nota"])
    assert log["mejor"] == minimo
    final = torch.load(tmp_path / "run" / "modelo.pt", map_location="cpu")
    mejor = torch.load(tmp_path / "run" / "modelo_mejor.pt", map_location="cpu")
    assert final["config"] == mejor["config"] and final["modo"] == mejor["modo"] == "eco_rel"
    # el mejor coincide con el final exactamente cuando la ultima evaluacion fue la mejor
    assert _iguales(final["modelo"], mejor["modelo"]) == (minimo["paso"] == log["final"]["paso"])
    m, voc, modo = cargar_modelo(str(tmp_path / "run" / "modelo_mejor.pt"))[:3]
    assert sum(p.numel() for p in m.parameters()) == log["parametros"]


def test_sin_guardar_mejor_no_hay_fichero(tmp_path):
    datos = _corpus(tmp_path)
    log = main(_args(datos, tmp_path / "run2"))
    assert "mejor" not in log
    assert (tmp_path / "run2" / "modelo.pt").exists()
    assert not (tmp_path / "run2" / "modelo_mejor.pt").exists()
