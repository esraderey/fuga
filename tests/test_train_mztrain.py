"""Tests del script del ensayo MZTrain que no necesitan GPU ni mztrain: regla de grupos del optimizador y contraste."""
import importlib.util
import json
import pathlib

import pytest
import torch

RUTA = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_mztrain.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("train_mztrain", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_optimizador_denso_decae_solo_matrices(mod):
    from fuga.model import Transformer, config
    m = Transformer(config("nano", vocab=40))
    opt = mod.optimizador("s_denso", m, lr=1e-3, wd=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    con, sin = opt.param_groups
    assert con["weight_decay"] == 0.1 and sin["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in con["params"]) and all(p.dim() < 2 for p in sin["params"])
    assert len(con["params"]) + len(sin["params"]) == sum(1 for _ in m.parameters())
    assert opt.defaults["betas"] == (0.9, 0.95)


def _log(brazo, semilla, por_pieza, bpn):
    return {"args": {"brazo": brazo, "semilla": semilla}, "parametros": 1, "final": {"bits_por_nota": bpn, "segundos": 1,
            "tokens_vistos": 10}, "memoria": {"cuda_pico_asignado_gb": 1.0, "cuda_pico_reservado_gb": 1.5},
            "test": {"bits_por_nota": bpn, "por_pieza": por_pieza}}


def test_contrastar_signo_y_pareo_por_pieza(mod, tmp_path):
    # m_mztrain es 0,5 bits/nota PEOR en cada pieza común; una pieza solo existe en un brazo y debe ignorarse
    base = {f"p{i}": [10.0 * (i + 1), i + 1] for i in range(30)}
    peor = {k: [v[0] + 0.5 * v[1], v[1]] for k, v in base.items()}
    peor["extra"] = [5.0, 1]
    rutas = []
    for brazo, pp in (("s_denso", base), ("m_mztrain", peor)):
        for s in (0, 1):
            r = tmp_path / f"{brazo}_{s}.json"
            r.write_text(json.dumps(_log(brazo, s, pp, 5.0)), encoding="utf-8")
            rutas.append(str(r))
    res = mod.contrastar(rutas, str(tmp_path / "out.json"))
    c = res["contrastes"]["m_mztrain - s_denso"]
    assert c["por_semilla"][0]["n_piezas"] == 30
    media, lo, hi = c["IC95_pareado_todas_las_semillas"]
    assert abs(media - 0.5) < 1e-9 and lo > 0 and hi > 0            # IC entero por encima de 0
    assert c["veredicto_K1"] == "falla"
    mejor = {k: [v[0] - 0.3 * v[1], v[1]] for k, v in base.items()}
    for s in (0, 1):
        (tmp_path / f"m_mztrain_{s}.json").write_text(json.dumps(_log("m_mztrain", s, mejor, 4.7)), encoding="utf-8")
    res2 = mod.contrastar(rutas, str(tmp_path / "out2.json"))
    assert res2["contrastes"]["m_mztrain - s_denso"]["veredicto_K1"] == "pasa (IC95 < 0)"


def test_bootstrap_vacio_y_constante(mod):
    assert all(x != x for x in mod.bootstrap([]))                      # nan, nan, nan
    assert mod.bootstrap([0.2] * 10) == [pytest.approx(0.2)] * 3
