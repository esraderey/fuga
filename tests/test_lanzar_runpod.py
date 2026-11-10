"""Anclas del lanzador de RunPod, sin red: elección de GPU, cuerpo de la petición sin secretos, cliente y vigilancia."""
import importlib.util
import json
import pathlib
import re

import pytest

RUTA = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "lanzar.py"


@pytest.fixture(scope="module")
def lz():
    spec = importlib.util.spec_from_file_location("lanzar", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gql(precios):
    """Respuesta de gpuTypes como la de RunPod: {alias: [{id, seguro:{...}, comunidad:{...}}]}."""
    def lp(x):
        return None if x is None else {"stockStatus": x[1], "uninterruptablePrice": x[0]}
    return {f"g{i}": [{"id": g, "displayName": g, "seguro": lp(s), "comunidad": lp(c)}] for i, (g, s, c) in enumerate(precios)}


def test_eleccion_prefiere_nube_segura_y_respeta_el_tope(lz):
    ofs = lz.ofertas(_gql([("NVIDIA H100 PCIe", (2.39, "Low"), (1.99, "High")),
                           ("NVIDIA H100 80GB HBM3", (2.69, "High"), None),
                           ("NVIDIA H100 NVL", (3.19, "High"), (2.59, None))]))
    assert len(ofs) == 5
    o = lz.elegir_oferta(ofs, 3.0)
    assert o["nube"] == "SECURE" and o["gpus"] == ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]     # NVL supera el tope
    assert o["precio"] == 2.39 and o["precio_peor"] == 2.69
    o2 = lz.elegir_oferta(ofs, 2.2)                                   # nada seguro bajo 2,20: cae a la comunitaria con stock
    assert o2["nube"] == "COMMUNITY" and o2["gpus"] == ["NVIDIA H100 PCIe"]
    assert lz.elegir_oferta(ofs, 1.5) is None
    sin_stock = lz.ofertas(_gql([("NVIDIA H100 PCIe", (2.39, None), (1.99, "None"))]))
    assert lz.elegir_oferta(sin_stock, 3.0) is None


def test_especificacion_sin_secretos_y_con_seguros(lz):
    oferta = {"nube": "SECURE", "gpus": ["NVIDIA H100 PCIe"], "precio": 2.39, "precio_peor": 2.39, "stock": "High"}
    spec = lz.especificacion(oferta, "a" * 40, plan="final", max_horas=5, secreto="hf_token", precio_hora=2.39)
    txt = json.dumps(spec)
    assert spec["env"]["HF_TOKEN"] == "{{ RUNPOD_SECRET_hf_token }}"          # referencia al secreto, nunca el valor
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", txt) and "api_key" not in txt.lower()
    assert spec["interruptible"] is False and spec["gpuCount"] == 1 and spec["cloudType"] == "SECURE"
    assert spec["gpuTypePriority"] == "custom" and spec["imageName"] == lz.IMAGEN and spec["volumeMountPath"] == "/workspace"
    assert spec["env"]["FUGA_COMMIT"] == "a" * 40 and spec["env"]["FUGA_APAGAR"] == "1" and spec["env"]["FUGA_MAX_HORAS"] == "5"
    cmd = spec["dockerStartCmd"]
    assert cmd[:2] == ["bash", "-c"] and 'git checkout -q "$FUGA_COMMIT"' in cmd[2] and "scripts/runpod/arranque.sh" in cmd[2]
    assert cmd[2].startswith("(sleep $(( ${FUGA_MAX_HORAS:-5} * 3600 )); runpodctl pod stop")      # temporizador de seguridad
    assert '|| runpodctl pod stop "$RUNPOD_POD_ID" ) &' in cmd[2]                                    # si el arranque falla, se apaga
    sin_precios = lz.especificacion(None, "b" * 40)
    assert sin_precios["gpuTypeIds"] == list(lz.H100) and sin_precios["gpuTypePriority"] == "availability"


def test_cliente_autentica_por_cabecera_y_propaga_errores(lz):
    llamadas = []

    def enviar(metodo, url, clave, cuerpo=None, timeout=60):
        llamadas.append((metodo, url, cuerpo))
        if url.endswith("/graphql"):
            return 200, {"data": _gql([("NVIDIA H100 PCIe", (2.39, "High"), None)])}
        if metodo == "POST" and url.endswith("/pods"):
            return 201, {"id": "pod123", "costPerHr": 2.39, "desiredStatus": "RUNNING"}
        if metodo == "GET" and url.endswith("/pods"):
            return 200, [{"id": "pod123", "name": lz.NOMBRE}]
        return 401, {"error": "unauthorized"}
    cli = lz.Cliente("CLAVE-SECRETA", enviar)
    assert cli.precios()[0] == {"gpu": "NVIDIA H100 PCIe", "nube": "SECURE", "precio": 2.39, "stock": "High"}
    assert cli.crear({"name": "x"})["id"] == "pod123" and cli.listar()[0]["id"] == "pod123"
    with pytest.raises(RuntimeError, match="401"):
        cli.detener("pod123")
    assert all("CLAVE-SECRETA" not in u and "api_key" not in u for _, u, _ in llamadas)        # la clave no va en la URL
    assert lz.Cliente("k", lambda *a, **k: (500, "caído")).precios() == []                      # sin precios: lista vacía


def _vigilar(lz, estados, registros, max_horas=5.0, sin_latido_min=35.0):
    """Simula la vigilancia con un reloj que avanza 120 s por vuelta; devuelve (resultado, detenciones, mensajes)."""
    reloj = {"t": 0.0}
    detenido, dicho = [], []

    class Cli:
        def pod(self, pid):
            i = min(int(reloj["t"] // 120), len(estados) - 1)
            return {"desiredStatus": estados[i], "costPerHr": 2.5}

        def detener(self, pid):
            detenido.append(reloj["t"])

    def leer():
        i = min(int(reloj["t"] // 120), len(registros) - 1)
        return {"cadena.log": registros[i], "arranque.log": ""}

    def dormir(s):
        reloj["t"] += s
    r = lz.vigilar(Cli(), "pod123", 0.0, max_horas, 120, sin_latido_min, leer, dormir, lambda: reloj["t"], dicho.append)
    return r, detenido, dicho


def test_vigilancia_casos(lz):
    hito = "### 2026-11-10 10:00:00 entrenar_completo: python -m fuga.train\npaso 500 f=0.03 loss=1.2 tok=4000000\n"
    fin = hito + f"### 2026-11-10 12:00:00 {lz.FIN_OK} en 110 min\n"
    r, det, dicho = _vigilar(lz, ["RUNNING", "RUNNING", "EXITED"], ["", hito, fin])
    assert r == "ok" and det == [] and any("completo: paso 500" in d for d in dicho)       # el pod se apagó solo
    r, det, _ = _vigilar(lz, ["RUNNING"], ["", hito, fin])
    assert r == "ok" and len(det) == 1                                                      # terminó pero no se apagó: se detiene
    r, det, _ = _vigilar(lz, ["RUNNING"], [""], sin_latido_min=10)
    assert r == "sin_latido" and len(det) == 1 and det[0] > 600
    r, det, _ = _vigilar(lz, ["RUNNING"], [hito], max_horas=1)
    assert r == "tope" and len(det) == 1 and det[0] > 1.25 * 3600 - 1
    r, det, _ = _vigilar(lz, ["RUNNING", "EXITED"], ["", hito + f"### x {lz.FIN_MAL}: RuntimeError\n"])
    assert r == "fallo" and det == []
    assert _vigilar(lz, ["EXITED"], [""])[0] == "parado"


def test_clave_de_entorno_o_archivo_y_sin_clave(lz, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lz, "ARCHIVO_CLAVE", str(tmp_path / "clave"))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert lz.leer_clave() == "" and lz.main(["simulacro"]) == 2
    assert "lanzar.py clave" in capsys.readouterr().err
    (tmp_path / "clave").write_text("  abc123\n", encoding="utf-8")
    assert lz.leer_clave() == "abc123"
    monkeypatch.setenv("RUNPOD_API_KEY", "del-entorno")
    assert lz.leer_clave() == "del-entorno"
