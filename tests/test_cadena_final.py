"""Anclas de la cadena final: reglas puras (elección, contraste, muestras, proyección) y un ensayo de extremo a extremo
en miniatura sobre un corpus sintético (entrenar, Z-Space, evaluar, elegir, generar, analizar y exportar)."""
import importlib.util
import json
import pathlib
import sys

import pytest
import torch

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cf():
    return _cargar("cadena_final")


def _ev(val, por_pieza=None):
    return {"val": {"bits_por_nota": val}, "test": {"bits_por_nota": val + 0.3}, "por_pieza": por_pieza or {}}


def test_elegir_minimo_de_validacion(cf):
    evals = {"r/completo/modelo.pt": _ev(6.8), "r/completo/modelo_mejor.pt": _ev(3.43), "r/wsd_6000/modelo.pt": _ev(3.31),
             "r/wsd_6000/modelo_mejor.pt": _ev(3.31)}
    assert cf.elegir(evals) == "r/wsd_6000/modelo.pt"              # empate: orden alfabético
    assert cf.elegir({k: v for k, v in evals.items() if "completo" in k}) == "r/completo/modelo_mejor.pt"


def test_contraste_pareado_signo_y_piezas_comunes(cf):
    mz = _cargar("evaluar_ckpts")._mz()
    base = {f"p{i}": [10.0 * (i + 1), i + 1] for i in range(40)}
    mejor = {k: [v[0] - 0.2 * v[1], v[1]] for k, v in base.items()}
    mejor["solo_en_a"] = [1.0, 1]
    evals = {"a": _ev(3.3, mejor), "b": _ev(3.4, base)}
    c = cf.contraste(evals, "a", "b", mz.bootstrap, n=500)
    assert c["n_piezas"] == 40 and abs(c["IC95_pareado"][0] + 0.2) < 1e-9 and c["veredicto"] == "pasa (IC95 < 0)"
    assert cf.contraste(evals, "b", "a", mz.bootstrap, n=500)["veredicto"] == "al revés (IC95 > 0)"
    assert cf.contraste({"a": _ev(1), "b": _ev(1)}, "a", "b", mz.bootstrap)["veredicto"] == "sin piezas comunes"


def test_elegir_muestras_y_proyeccion(cf):
    def m(i, **kw):
        return {"muestra": i, "notas": 120, "compases": 16, "partes": 1, "copia4": 0.0, "repeticion": 0.1, **kw}
    an = {"compases": 16, "base_real": {"test_real_repeticion": 0.24},
          "muestras": [m(0, repeticion=0.05), m(1, repeticion=0.25, copia4=0.2), m(2, repeticion=0.22, partes=4),
                       m(3, repeticion=0.26, compases=9), m(4, repeticion=0.26), m(5, notas=0)]}
    assert cf.elegir_muestras(an) == [2, 4]                         # 1 copia, 3 incompleta, 5 vacía
    solo_malas = {"compases": 16, "base_real": {"test_real_repeticion": 0.24}, "muestras": [m(0, compases=5, notas=20), m(1, compases=4, notas=30)]}
    assert cf.elegir_muestras(solo_malas) == [1, 0]                 # sin candidatas: las más largas
    p = cf.proyectar(0.15, 36880, 37, s_version=12.0, fijo_min=20.0, precio_hora=3.0)
    assert abs(p["minutos_entrenamiento"] - 92.2) < 0.01 and abs(p["minutos_totales"] - 119.6) < 0.01
    assert abs(p["coste"] - 5.98) < 0.01


def test_cadena_en_miniatura_de_extremo_a_extremo(tmp_path, cf):
    pytest.importorskip("z_space_core")
    pytest.importorskip("safetensors")
    sys.path.insert(0, str(RAIZ / "tests"))
    from test_fuga import pieza_sintetica

    from fuga.data import particion
    piezas = []
    for k in range(80):
        p = pieza_sintetica(semilla=k, compases=8)
        p["id"] = f"lib{k}#0"
        piezas.append(p)
    assert {"train", "val", "test"} <= {particion(p) for p in piezas}
    datos = tmp_path / "corpus.jsonl"
    datos.write_text("\n".join(json.dumps(p) for p in piezas) + "\n", encoding="utf-8")
    raiz = tmp_path / "final"
    rc = cf.main(["--prueba", "--raiz", str(raiz), "--device", "cpu", "--datos", str(datos),
                  "--cache", str(tmp_path / "cache.pkl"), "--cache_test", str(tmp_path / "test.pkl")])
    registro = (raiz / "cadena.log").read_text(encoding="utf-8")
    assert rc == 0, registro[-3000:]
    assert "PROYECCIÓN" in registro and "CADENA FINAL TERMINADA" in registro
    sel = json.loads((raiz / "seleccion.json").read_text(encoding="utf-8"))
    assert {c["corrida"] for c in sel["candidatos"]} == {"completo", "wsd_30", "wsd_40"} and len(sel["candidatos"]) == 6
    assert sel["elegido"] == min(sel["candidatos"], key=lambda c: (c["val"], c["ckpt"]))["ckpt"]
    assert "KF1" in sel and sel["KF1"]["n_piezas"] == sel["test_piezas"] > 0
    z = json.loads((raiz / "completo" / "zspace_verificacion.json").read_text(encoding="utf-8"))
    assert z["versiones"] == 3 and z["exacto_modelo.pt"] is True
    hf = raiz / "hf"
    ficha = (hf / "README.md").read_text(encoding="utf-8")
    assert "## Selección del modelo" in ficha and "**publicado**" in ficha
    assert (hf / "corrida" / "seleccion.json").exists() and (hf / "corrida" / "completo" / "log.json").exists()
    assert "RESUMEN" in (hf / "corrida" / "RESUMEN.md").read_text(encoding="utf-8").upper()
    from fuga.sample import cargar_modelo
    m = cargar_modelo(str(hf))[0]                                   # el export carga y es el checkpoint elegido
    ref = torch.load(sel["elegido"], map_location="cpu", weights_only=False)["modelo"]
    assert all(torch.equal(v, ref[k]) for k, v in m.state_dict().items())
    assert cf.main(["--prueba", "--raiz", str(raiz), "--device", "cpu", "--datos", str(datos),
                    "--cache", str(tmp_path / "cache.pkl"), "--cache_test", str(tmp_path / "test.pkl")]) == 0   # reanudable
