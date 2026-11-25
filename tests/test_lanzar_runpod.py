"""Anclas del lanzador de RunPod (API v2), sin red: catálogo y elección de GPU, cuerpo del pod validado contra el esquema
oficial, cliente (rutas, paginación, errores), creación con salto de GPU sin capacidad, registros y vigilancia."""
import importlib.util
import json
import pathlib
import re

import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
RUTA = RAIZ / "scripts" / "runpod" / "lanzar.py"
ESQUEMA = RAIZ / "tests" / "datos" / "runpod_v2_crear_pod.json"


@pytest.fixture(scope="module")
def lz():
    spec = importlib.util.spec_from_file_location("lanzar", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _catalogo(filas):
    """Respuesta de GET /v2/catalog/gpus como la de RunPod: filas (id, seguro, comunidad, disponibilidad)."""
    return {"gpus": [{"id": g, "name": g, "secure": seg > 0, "community": com > 0, "availability": disp,
                      "price": {"secure": seg, "community": com}} for g, seg, com, disp in filas]}


# precios y disponibilidad leídos del catálogo real el 2026-11-25
SEGURO = _catalogo([("NVIDIA H100 PCIe", 2.89, 1.99, "NONE"), ("NVIDIA H100 NVL", 3.19, 2.59, "NONE"),
                    ("NVIDIA H100 80GB HBM3", 4.39, 2.69, "LOW"), ("NVIDIA A100-SXM4-80GB", 2.19, 1.39, "LOW")])
COMUNIDAD = _catalogo([("NVIDIA H100 PCIe", 2.89, 1.99, "NONE"), ("NVIDIA H100 NVL", 3.19, 2.59, "NONE"),
                       ("NVIDIA H100 80GB HBM3", 4.39, 2.69, "NONE"), ("NVIDIA A100 80GB PCIe", 2.09, 1.19, "LOW")])


def test_ofertas_y_eleccion_con_el_catalogo_real(lz):
    ofs = lz.ofertas(SEGURO, "SECURE") + lz.ofertas(COMUNIDAD, "COMMUNITY")
    assert {(o["gpu"], o["nube"], o["precio"]) for o in ofs if o["gpu"] == "NVIDIA H100 NVL"} == \
        {("NVIDIA H100 NVL", "SECURE", 3.19), ("NVIDIA H100 NVL", "COMMUNITY", 2.59)}
    assert all("A100" not in o["gpu"] for o in ofs)                       # solo la familia pedida
    assert lz.elegir_oferta(ofs, 3.2) is None                              # hoy no hay H100 bajo 3,20
    o = lz.elegir_oferta(ofs, 4.5)
    assert o == {"nube": "SECURE", "gpus": ["NVIDIA H100 80GB HBM3"], "precio": 4.39, "precio_peor": 4.39, "stock": "LOW"}
    hay_nvl = lz.ofertas(_catalogo([("NVIDIA H100 NVL", 3.19, 2.59, "LOW"), ("NVIDIA H100 80GB HBM3", 4.39, 2.69, "LOW")]), "SECURE")
    assert lz.elegir_oferta(hay_nvl, 4.5)["gpus"] == ["NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3"]   # la más barata primero
    a100 = lz.ofertas(SEGURO, "SECURE", lz.A100) + lz.ofertas(COMUNIDAD, "COMMUNITY", lz.A100)
    assert lz.elegir_oferta(a100, 3.0)["gpus"] == ["NVIDIA A100-SXM4-80GB"]                    # segura antes que comunitaria
    assert lz.elegir_oferta(a100, 1.5) == {"nube": "COMMUNITY", "gpus": ["NVIDIA A100 80GB PCIe"], "precio": 1.19,
                                           "precio_peor": 1.19, "stock": "LOW"}
    assert lz.ofertas(_catalogo([("NVIDIA A100-SXM4-40GB", 0, 1.0, "LOW")]), "SECURE", ("NVIDIA A100-SXM4-40GB",)) == []


def test_especificacion_valida_contra_el_esquema_oficial_v2(lz):
    jsonschema = pytest.importorskip("jsonschema")
    validador = jsonschema.Draft202012Validator(json.loads(ESQUEMA.read_text(encoding="utf-8")))
    spec = lz.especificacion("NVIDIA H100 NVL", "SECURE", "a" * 40, plan="final", max_horas=3, secreto="hf_token", precio_hora=3.19)
    errores = [e.message for e in validador.iter_errors(spec)]
    assert errores == []
    for campo_v1 in ({"gpuTypeIds": ["x"]}, {"interruptible": False}, {"imageName": "x"}, {"dockerStartCmd": ["x"]}):
        assert list(validador.iter_errors({**spec, **campo_v1})), campo_v1        # la v2 rechaza los campos de la v1
    assert list(validador.iter_errors({**spec, "gpu": {**spec["gpu"], "allowedCudaVersions": ["12.4"], "extra": 1}}))
    txt = json.dumps(spec)
    assert spec["env"]["HF_TOKEN"] == "{{ RUNPOD_SECRET_hf_token }}"          # referencia al secreto, nunca el valor
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{10,}", txt)                  # ningún valor de token
    assert "api_key" not in txt.replace("$RUNPOD_API_KEY", "").lower()     # solo la variable de la clave propia del pod
    assert spec["gpu"] == {"id": "NVIDIA H100 NVL", "count": 1, "minCudaVersion": "12.4"} and spec["cloud"] == "SECURE"
    assert spec["mounts"] == {"persistent": {"size": 40, "path": "/workspace"}} and spec["image"] == lz.IMAGEN
    assert spec["env"]["FUGA_COMMIT"] == "a" * 40 and spec["env"]["FUGA_APAGAR"] == "1" and spec["env"]["FUGA_MAX_HORAS"] == "3"
    cmd = spec["cmd"]
    assert cmd[:2] == ["bash", "-c"] and cmd[2].startswith("parar() { runpodctl pod stop")
    assert "(sleep $(( ${FUGA_MAX_HORAS:-3} * 3600 )); parar)" in cmd[2]                  # temporizador de seguridad
    assert 'git checkout -q "$FUGA_COMMIT"' in cmd[2] and "|| parar ) &" in cmd[2]          # si el arranque falla, se apaga
    assert '"https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action"' in cmd[2] and "$RUNPOD_API_KEY" in cmd[2]


class _Falso:
    """Servidor v2 de mentira: registra las llamadas y responde según la ruta."""

    def __init__(self, crear=None):
        self.llamadas, self.crear = [], list(crear or [])

    def __call__(self, metodo, url, clave, cuerpo=None, timeout=60):
        self.llamadas.append((metodo, url, cuerpo))
        ruta = url.split("api.runpod.io")[-1]
        if ruta.startswith("/v2/catalog/gpus?"):
            return 200, SEGURO if "cloud=SECURE" in ruta else COMUNIDAD
        if ruta == "/v2/pods":
            if metodo == "POST":
                return self.crear.pop(0)
            return 200, {"pods": [{"id": "p1", "name": "fuga-final"}], "pagination": {"hasNextPage": True, "nextCursor": "c/2"}}
        if ruta == "/v2/pods?cursor=c%2F2":
            return 200, {"pods": [{"id": "p2", "name": "otro"}], "pagination": {"hasNextPage": False, "nextCursor": None}}
        if ruta == "/v2/account/secrets":
            return 200, {"secrets": [{"id": "s1", "name": "hf_token"}]}
        if ruta == "/v2/pods/p1/action":
            return 200, {"id": "p1", "status": "EXITED"}
        if ruta == "/v2/pods/p1" and metodo == "DELETE":
            return 204, None
        return 401, {"title": "Unauthorized", "status": 401, "detail": "API key inválida"}


def test_cliente_v2_rutas_paginacion_y_errores(lz):
    f = _Falso()
    cli = lz.Cliente("CLAVE-SECRETA", f)
    assert [p["id"] for p in cli.listar()] == ["p1", "p2"]                 # sigue la paginación por cursor
    assert cli.secretos() == ["hf_token"]
    ofs = cli.precios()
    assert {o["nube"] for o in ofs} == {"SECURE", "COMMUNITY"} and all("minCudaVersion=12.4" in u for m, u, _ in f.llamadas if "catalog" in u)
    assert cli.detener("p1")["status"] == "EXITED"
    assert ("POST", "https://api.runpod.io/v2/pods/p1/action", {"action": "stop"}) in f.llamadas
    assert cli.eliminar("p1") is None
    with pytest.raises(RuntimeError, match="401.*API key inválida"):
        cli.pod("p9")
    assert all("CLAVE-SECRETA" not in u and "api_key" not in u for _, u, _ in f.llamadas)     # la clave no va en la URL
    with pytest.raises(RuntimeError, match="respondió 0.*sin conexión"):
        lz.Cliente("k", lambda *a, **k: (0, "sin conexión con api.runpod.io: getaddrinfo failed")).listar()
    cod, motivo = lz.http("GET", "https://fuga-prueba.invalid/v2/pods", "k", timeout=10)      # .invalid nunca resuelve
    assert cod == 0 and "sin conexión" in motivo


def test_crear_salta_las_gpu_sin_capacidad(lz):
    sin_cap = (400, {"title": "Bad Request", "status": 400, "detail": "There are no longer any instances available with the requested specifications."})
    f = _Falso(crear=[sin_cap, (409, {"title": "Conflict", "status": 409}), (201, {"id": "p1", "cost": 4.39, "status": "PROVISIONING"})])
    construidos = []

    def construir(gpu):
        construidos.append(gpu)
        return {"gpu": {"id": gpu}}
    pod, intentos = lz.Cliente("k", f).crear_primero(["G1", "G2", "G3"], construir)
    assert pod["id"] == "p1" and construidos == ["G1", "G2", "G3"] and intentos[-1] == "G3: creado"
    malo = _Falso(crear=[(422, {"title": "Unprocessable", "status": 422, "errors": ["additional properties 'x' not allowed"]})])
    with pytest.raises(RuntimeError, match="422.*additional properties"):
        lz.Cliente("k", malo).crear_primero(["G1", "G2"], construir)          # un error nuestro no se salta
    nada = _Falso(crear=[sin_cap, sin_cap])
    assert lz.Cliente("k", nada).crear_primero(["G1", "G2"], construir) == (None, ["G1: sin capacidad (400)", "G2: sin capacidad (400)"])
    assert lz.sin_capacidad(503, None) and not lz.sin_capacidad(401, {"detail": "no instances"})


def test_parsear_sse(lz):
    flujo = [b"id: 2026-11-25T10:00:00Z\n", b'data: {"source": "container", "line": "### arranque de FUGA", "ts": "t"}\n', b"\n",
             b"data: no-json\n", 'data: {"source": "system", "line": "pulling image"}\n', b": comentario\n"]
    assert lz.parsear_sse(flujo) == ["### arranque de FUGA", "pulling image"]


def _vigilar(lz, estados, registros, max_horas=5.0, sin_latido_min=35.0):
    """Simula la vigilancia con un reloj que avanza 120 s por vuelta; devuelve (resultado, detenciones, mensajes)."""
    reloj = {"t": 0.0}
    detenido, dicho = [], []

    class Cli:
        def pod(self, pid):
            i = min(int(reloj["t"] // 120), len(estados) - 1)
            return {"status": estados[i], "cost": 3.19}

        def detener(self, pid):
            detenido.append(reloj["t"])

    def leer():
        i = min(int(reloj["t"] // 120), len(registros) - 1)
        return {"cadena.log": registros[i], "arranque.log": ""}

    def dormir(s):
        reloj["t"] += s
    r = lz.vigilar(Cli(), "p1", 0.0, max_horas, 120, sin_latido_min, leer, dormir, lambda: reloj["t"], dicho.append,
                   ver_registros=lambda: ["linea del contenedor"])
    return r, detenido, dicho


def test_vigilancia_casos(lz):
    hito = "### 2026-11-25 10:00:00 entrenar_completo: python -m fuga.train\npaso 500 f=0.03 loss=1.2 tok=4000000\n"
    fin = hito + f"### 2026-11-25 12:00:00 {lz.FIN_OK} en 110 min\n"
    r, det, dicho = _vigilar(lz, ["PROVISIONING", "RUNNING", "EXITED"], ["", hito, fin])
    assert r == "ok" and det == [] and any("completo: paso 500" in d for d in dicho)       # el pod se apagó solo
    r, det, _ = _vigilar(lz, ["RUNNING"], ["", hito, fin])
    assert r == "ok" and len(det) == 1                                                      # terminó pero no se apagó
    r, det, dicho = _vigilar(lz, ["STARTING", "RUNNING"], [""], sin_latido_min=10)
    assert r == "sin_latido" and len(det) == 1 and det[0] > 600 and "  | linea del contenedor" in dicho
    r, det, _ = _vigilar(lz, ["RUNNING"], [hito], max_horas=1)
    assert r == "tope" and len(det) == 1 and det[0] > 1.25 * 3600 - 1
    r, det, _ = _vigilar(lz, ["RUNNING", "EXITED"], ["", hito + f"### x {lz.FIN_MAL}: RuntimeError\n"])
    assert r == "fallo" and det == []
    r, det, _ = _vigilar(lz, ["RUNNING", "ERROR"], ["", hito])
    assert r == "error" and len(det) == 1
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
