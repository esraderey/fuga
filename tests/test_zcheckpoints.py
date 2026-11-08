"""Anclas del almacén Z-Space de FUGA: versiones exactas, reapertura, export y su uso desde fuga.train."""
import json

import pytest
import torch

pytest.importorskip("z_space_core")

from fuga.model import Transformer, config  # noqa: E402
from fuga.sample import cargar_modelo  # noqa: E402
from fuga.train import main  # noqa: E402
from fuga.zcheckpoints import AlmacenZ  # noqa: E402
from test_train import _args, _corpus  # noqa: E402


def _bf16(sd):
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}


def _iguales(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_versiones_exactas_reapertura_y_export(tmp_path):
    torch.manual_seed(0)
    m = Transformer(config("nano", vocab=40))
    estados = []
    alm = AlmacenZ(str(tmp_path / "z"), full_every=2)
    for paso in (10, 20, 30):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        estados.append(_bf16(m.state_dict()))
        e = alm.guardar(m.state_dict(), paso, {"config": config("nano", vocab=40).__dict__ | {}, "estilos": ["sint"], "modo": "eco_rel"})
        assert e["paso"] == paso and e["bytes_tienda"] > 0
    assert [v["completo"] for v in alm.versiones] == [True, False, True]   # base, delta, completo periódico
    for i, sd in enumerate(estados):
        assert _iguales(alm.cargar(version=i), sd)
    assert _iguales(alm.cargar(paso=20), estados[1])
    alm.cerrar()
    re = AlmacenZ(str(tmp_path / "z"))                                      # reapertura desde disco
    assert [v["paso"] for v in re.versiones] == [10, 20, 30]
    assert _iguales(re.cargar(), estados[2]) and _iguales(re.cargar(version=1), estados[1])
    re.cerrar()


def test_train_guardar_cada_e_inicial(tmp_path):
    datos = _corpus(tmp_path)
    log = main(_args(datos, tmp_path / "run", "--guardar_cada", "2", "--pasos", "5"))
    assert [v["paso"] for v in log["zspace"]["versiones"]] == [2, 4, 5]        # la final se guarda aunque 5 no sea multiplo
    meta = json.load(open(tmp_path / "run" / "zspace" / "versiones.json", encoding="utf-8"))
    assert meta["extra"]["modo"] == "eco_rel" and meta["dtype"] == "bf16"
    final = torch.load(tmp_path / "run" / "modelo.pt", map_location="cpu", weights_only=False)
    alm = AlmacenZ(str(tmp_path / "run" / "zspace"))
    assert _iguales(alm.cargar(paso=5), _bf16(final["modelo"]))             # la última versión es el modelo final en bf16
    ruta = alm.exportar(str(tmp_path / "paso5.pt"), paso=5)
    alm.cerrar()
    m, voc, op = cargar_modelo(ruta)
    assert op == {"eco": True, "fantasma": False, "relativo": True}
    assert m.c.ctx == 256 and len(voc) == final["modelo"]["emb.weight"].shape[0] if "emb.weight" in final["modelo"] else True
    esperado = {k: v.to(torch.bfloat16).to(torch.float32) for k, v in final["modelo"].items()}
    assert _iguales(m.state_dict(), esperado)                             # el export es el final redondeado a bf16
    # arranque en caliente con lr 0 durante 1 paso: los pesos deben salir idénticos al checkpoint inicial
    base = _args(datos, tmp_path / "run2")[:-4]                              # sin --lr ni --out
    log2 = main([*base, "--pasos", "1", "--eval_cada", "0", "--lr", "0", "--inicial", str(tmp_path / "run" / "modelo.pt"),
                 "--out", str(tmp_path / "run2")])
    assert log2["final"]["paso"] == 1
    salida = torch.load(tmp_path / "run2" / "modelo.pt", map_location="cpu", weights_only=False)
    assert _iguales(salida["modelo"], final["modelo"])
