"""Lanza, vigila y detiene desde esta máquina el pod de RunPod del entrenamiento final de FUGA (API REST v2).

La API key de RunPod la guarda el usuario (`clave`, con entrada oculta) o la variable RUNPOD_API_KEY; este script nunca
la imprime ni la pone en una URL. El token de Hugging Face no pasa por aquí: el pod lo recibe de un secreto de RunPod.

  python scripts/runpod/lanzar.py clave                         # (tú, en tu terminal) guarda la API key de RunPod
  python scripts/runpod/lanzar.py simulacro --precio_max 3.2    # clave, secreto, precios y stock en vivo; no crea nada
  python scripts/runpod/lanzar.py crear --confirmar --precio_max 3.2   # crea el pod (gasta saldo de RunPod)
  python scripts/runpod/lanzar.py vigilar                       # sigue el progreso; detiene el pod al terminar o al pasar el tope
  python scripts/runpod/lanzar.py registros                     # últimas líneas del contenedor (diagnóstico)
  python scripts/runpod/lanzar.py estado                        # pods de este proyecto y su coste
  python scripts/runpod/lanzar.py detener                       # detiene el pod (conserva el volumen)
  python scripts/runpod/lanzar.py eliminar --confirmar          # elimina el pod y su volumen
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = "https://api.runpod.io/v2"          # la v1 (rest.runpod.io) se retiró el 2026-11-15
H100 = ("NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3")
A100 = ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe")
# Todas compatibles con la imagen (torch 2.4, CUDA 12.4: sm_80 y sm_90). Una RTX 5090 (sm_120) necesitaría otra imagen.
FAMILIAS = {"h100": H100, "a100": A100}
IMAGEN = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CUDA_MIN = "12.4"                         # el host debe tener un driver con CUDA >= 12.4 (13.x incluido)
NOMBRE = "fuga-final"
REPO_GIT = "https://github.com/esraderey/fuga"
ARCHIVO_CLAVE = os.path.join(os.path.expanduser("~"), ".runpod", "fuga_api_key")
ESTADO = os.path.join(RAIZ, "runs", "runpod_pod.json")
REGISTRO_HF = "esraderey/fuga-datos"
RUTA_LATIDO = "corridas/final_h100"
FIN_OK, FIN_MAL = "CADENA FINAL TERMINADA", "CADENA FINAL FALLÓ"
VIVOS = ("PROVISIONING", "STARTING", "RUNNING")
# El pod se detiene a sí mismo con runpodctl o, si no lo hay, con la API v2 y su clave propia (RUNPOD_API_KEY).
PARAR = ('runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1 || curl -fsS -X POST '
         '"https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" -H "Authorization: Bearer $RUNPOD_API_KEY" '
         '-H "Content-Type: application/json" -d \'{"action":"stop"}\'')
# Arranque del contenedor: temporizador de seguridad, clon del commit exacto, arranque de FUGA y servicios de la imagen.
ARRANQUE = (
    'parar() { ' + PARAR + '; }; '
    '(sleep $(( ${FUGA_MAX_HORAS:-3} * 3600 )); parar) >/dev/null 2>&1 & '
    '( (mkdir -p /workspace && cd /workspace && { [ -d fuga/.git ] || git clone -q ' + REPO_GIT + '; } && cd fuga '
    '&& git fetch -q origin && git checkout -q "$FUGA_COMMIT" && bash scripts/runpod/arranque.sh) '
    '> /workspace/arranque.salida.log 2>&1 || parar ) & '
    'if [ -x /start.sh ]; then exec /start.sh; else exec sleep infinity; fi'
)


def leer_clave() -> str:
    clave = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not clave and os.path.exists(ARCHIVO_CLAVE):
        with open(ARCHIVO_CLAVE, encoding="utf-8") as f:
            clave = f.read().strip()
    return clave


def http(metodo: str, url: str, clave: str, cuerpo: dict | None = None, timeout: int = 60):
    """(código, respuesta) de una llamada JSON autenticada por cabecera; sin conexión devuelve (0, motivo)."""
    datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
    req = urllib.request.Request(url, data=datos, method=metodo, headers={
        "Authorization": f"Bearer {clave}", "Content-Type": "application/json", "User-Agent": "fuga-lanzar/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt, cod = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        txt, cod = e.read().decode("utf-8", "replace"), e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"sin conexión con {urllib.parse.urlsplit(url).netloc}: {getattr(e, 'reason', e)}"
    try:
        return cod, json.loads(txt) if txt else None
    except json.JSONDecodeError:
        return cod, txt


def motivo(resp) -> str:
    """Texto legible de un error de la v2 (formato RFC 9457: title, status, detail, errors)."""
    if isinstance(resp, dict):
        partes = [str(resp.get(k)) for k in ("title", "detail") if resp.get(k)] + [str(e) for e in resp.get("errors") or []]
        return " | ".join(partes) or json.dumps(resp)[:300]
    return str(resp)[:300]


def sin_capacidad(cod: int, resp) -> bool:
    """¿El error al crear es falta de GPU libre (probar otra) y no un fallo nuestro o de credenciales?"""
    if cod in (409, 503):
        return True
    return cod == 400 and bool(re.search(r"capacit|availab|no longer|not enough|insufficient|no (free )?gpu|instances",
                                         motivo(resp), re.I))


def ofertas(resp: dict, nube: str, gpus=H100) -> list[dict]:
    """[{gpu, nube, precio, stock}] de la respuesta de GET /v2/catalog/gpus pedida para una nube."""
    campo = "secure" if nube == "SECURE" else "community"
    out = []
    for g in (resp or {}).get("gpus", []):
        precio = float((g.get("price") or {}).get(campo) or 0)
        if g.get("id") in gpus and precio > 0 and g.get(campo, True):
            out.append({"gpu": g["id"], "nube": nube, "precio": precio, "stock": g.get("availability")})
    return out


def elegir_oferta(ofs: list[dict], precio_max: float) -> dict | None:
    """Nube y GPUs a probar, de la más barata a la más cara, entre las que tienen stock y precio dentro del tope; se
    prefiere la nube segura (centros de datos de RunPod) y solo si no hay nada en ella se usa la comunitaria."""
    validas = [o for o in ofs if o["precio"] <= precio_max and str(o["stock"] or "none").lower() != "none"]
    for nube in ("SECURE", "COMMUNITY"):
        suyas = sorted((o for o in validas if o["nube"] == nube), key=lambda o: (o["precio"], o["gpu"]))
        if suyas:
            return {"nube": nube, "gpus": [o["gpu"] for o in suyas], "precio": suyas[0]["precio"],
                    "precio_peor": suyas[-1]["precio"], "stock": suyas[0]["stock"]}
    return None


def especificacion(gpu: str, nube: str, commit: str, plan: str = "final", repo_hf: str = "esraderey/fuga-s-eco-rel",
                   max_horas: int = 3, secreto: str = "hf_token", precio_hora: float = 3.0, volumen_gb: int = 40) -> dict:
    """Cuerpo de POST /v2/pods (una sola GPU por pod en la v2). El token de Hugging Face va como referencia a un
    secreto de RunPod, nunca como valor."""
    return {"name": NOMBRE, "image": IMAGEN, "cloud": nube,
            "gpu": {"id": gpu, "count": 1, "minCudaVersion": CUDA_MIN},
            "disk": 20, "mounts": {"persistent": {"size": volumen_gb, "path": "/workspace"}}, "ports": ["22/tcp"],
            "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_" + secreto + " }}", "FUGA_COMMIT": commit, "FUGA_PLAN": plan,
                    "FUGA_REPO_HF": repo_hf, "FUGA_APAGAR": "1", "FUGA_MAX_HORAS": str(int(max_horas)),
                    "FUGA_PRECIO_HORA": f"{precio_hora:.2f}"},
            "cmd": ["bash", "-c", ARRANQUE]}


def parsear_sse(lineas) -> list[str]:
    """Líneas de registro de un flujo Server-Sent Events de /v2/pods/{id}/logs (`data: {"line": ...}`)."""
    out = []
    for raw in lineas:
        ln = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        ln = ln.strip()
        if not ln.startswith("data:"):
            continue
        try:
            ev = json.loads(ln[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and "line" in ev:
            out.append(str(ev["line"]))
    return out


class Cliente:
    def __init__(self, clave: str, enviar=http):
        self.clave, self.enviar = clave, enviar

    def _ok(self, cod: int, resp, que: str):
        if not 200 <= cod < 300:
            raise RuntimeError(f"RunPod respondió {cod} al {que}: {motivo(resp)}")
        return resp

    def listar(self) -> list:
        pods, cursor = [], None
        while True:
            url = f"{API}/pods" + (f"?{urllib.parse.urlencode({'cursor': cursor})}" if cursor else "")
            r = self._ok(*self.enviar("GET", url, self.clave), "listar pods") or {}
            pods += r.get("pods", [])
            pag = r.get("pagination") or {}
            if not pag.get("hasNextPage") or not pag.get("nextCursor"):
                return pods
            cursor = pag["nextCursor"]

    def pod(self, pid: str) -> dict:
        return self._ok(*self.enviar("GET", f"{API}/pods/{pid}", self.clave), "leer el pod")

    def accion(self, pid: str, accion: str):
        return self._ok(*self.enviar("POST", f"{API}/pods/{pid}/action", self.clave, {"action": accion}), f"{accion} del pod")

    def detener(self, pid: str):
        return self.accion(pid, "stop")

    def eliminar(self, pid: str):
        return self._ok(*self.enviar("DELETE", f"{API}/pods/{pid}", self.clave), "eliminar el pod")

    def secretos(self) -> list[str]:
        """Nombres de los secretos de la cuenta (la API no devuelve sus valores)."""
        r = self._ok(*self.enviar("GET", f"{API}/account/secrets", self.clave), "listar secretos") or {}
        return [s.get("name") for s in r.get("secrets", [])]

    def precios(self, gpus=H100) -> list[dict]:
        """Ofertas de las GPU pedidas en las dos nubes, con disponibilidad para hosts con CUDA >= CUDA_MIN."""
        out = []
        for nube in ("SECURE", "COMMUNITY"):
            q = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD", "cloud": nube, "minCudaVersion": CUDA_MIN})
            cod, resp = self.enviar("GET", f"{API}/catalog/gpus?{q}", self.clave)
            if 200 <= cod < 300 and isinstance(resp, dict):
                out += ofertas(resp, nube, gpus)
        return out

    def crear_primero(self, gpus: list[str], construir) -> tuple[dict | None, list[str]]:
        """Crea el pod con la primera GPU de la lista que tenga capacidad. Devuelve (pod o None, intentos)."""
        intentos = []
        for gpu in gpus:
            cod, resp = self.enviar("POST", f"{API}/pods", self.clave, construir(gpu))
            if 200 <= cod < 300:
                intentos.append(f"{gpu}: creado")
                return resp, intentos
            if not sin_capacidad(cod, resp):
                raise RuntimeError(f"RunPod respondió {cod} al crear el pod con {gpu}: {motivo(resp)}")
            intentos.append(f"{gpu}: sin capacidad ({cod})")
        return None, intentos

    def registros(self, pid: str, cola: int = 80, segundos: float = 8.0) -> list[str]:
        """Últimas líneas del contenedor y del sistema del pod (lee el flujo unos segundos y lo cierra)."""
        req = urllib.request.Request(f"{API}/pods/{pid}/logs?{urllib.parse.urlencode({'tail': cola})}", headers={
            "Authorization": f"Bearer {self.clave}", "Accept": "text/event-stream", "User-Agent": "fuga-lanzar/2.0"})
        lineas, t0 = [], time.time()
        try:
            with urllib.request.urlopen(req, timeout=segundos) as r:
                for raw in r:
                    lineas.append(raw)
                    if time.time() - t0 > segundos:
                        break
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        return parsear_sse(lineas)


def commit_publicado() -> str:
    """SHA del commit actual, exigiendo que esté en el remoto y que no haya cambios sin confirmar en archivos versionados."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD")
    if not sha:
        raise RuntimeError("no se pudo leer el commit actual")
    git("fetch", "-q", "origin")
    if not git("branch", "-r", "--contains", sha):
        raise RuntimeError(f"el commit {sha[:10]} no está en GitHub: haz push antes de lanzar")
    sucios = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if not ln.endswith("cadena.log")]
    if sucios:
        raise RuntimeError("hay cambios sin confirmar en archivos versionados: " + "; ".join(sucios[:5]))
    return sha


def hitos(texto: str) -> list[str]:
    return [ln for ln in texto.splitlines() if ln.startswith("### ")]


def progreso(texto: str) -> str:
    pasos = [ln for ln in texto.splitlines() if ln.startswith("paso ")]
    corr = [ln for ln in hitos(texto) if " entrenar_" in ln and "exit=" not in ln]
    nombre = corr[-1].split(" entrenar_")[1].split(":")[0] if corr else "?"
    return f"{nombre}: {pasos[-1]}" if pasos else ""


def leer_latido(repo: str = REGISTRO_HF, ruta: str = RUTA_LATIDO) -> dict[str, str]:
    """Texto de cadena.log y arranque.log tal como los dejó el último latido del pod ('' si aún no existen)."""
    from huggingface_hub import hf_hub_download
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for nombre in ("cadena.log", "arranque.log"):
            try:
                p = hf_hub_download(repo, f"{ruta}/{nombre}", repo_type="dataset", local_dir=tmp, force_download=True)
                with open(p, encoding="utf-8", errors="replace") as f:
                    out[nombre] = f.read()
            except Exception:
                out[nombre] = ""
    return out


def _estado() -> dict:
    if not os.path.exists(ESTADO):
        raise RuntimeError(f"no hay pod registrado en {ESTADO}")
    with open(ESTADO, encoding="utf-8") as f:
        return json.load(f)


def vigilar(cli: Cliente, pid: str, creado: float, max_horas: float, cada_s: int = 120, sin_latido_min: float = 35.0,
            leer=leer_latido, dormir=time.sleep, ahora=time.time, decir=print, ver_registros=None) -> str:
    """Bucle de vigilancia. Devuelve 'ok', 'fallo', 'parado', 'error', 'sin_latido' o 'tope'; detiene el pod si sigue
    encendido cuando no debería."""
    vistos: set[str] = set()
    ultimo = ""
    fin_visto = None

    def volcar_registros():
        if ver_registros is not None:
            for ln in ver_registros()[-15:]:
                decir("  | " + ln[:240])

    while True:
        t = ahora()
        pod = cli.pod(pid)
        st = pod.get("status")
        reg = leer()
        for ln in hitos(reg["cadena.log"]) + hitos(reg["arranque.log"]):
            if ln not in vistos:
                vistos.add(ln)
                decir(ln[:260])
        pr = progreso(reg["cadena.log"])
        if pr and pr != ultimo:
            ultimo = pr
            decir(f"[{time.strftime('%H:%M')}] {pr} | pod {st} | {(t - creado)/60:.0f} min | ~{(t - creado)/3600*float(pod.get('cost') or 0):.2f} USD")
        todo = reg["cadena.log"] + reg["arranque.log"]
        resultado = "ok" if FIN_OK in todo else "fallo" if (FIN_MAL in todo or "ARRANQUE FALLÓ" in todo) else None
        if st == "ERROR":
            decir("el pod está en ERROR: lo detengo")
            volcar_registros()
            cli.detener(pid)
            return "error"
        if st not in VIVOS:
            decir(f"pod {st}: fin de la vigilancia ({resultado or 'parado'})")
            return resultado or "parado"
        if resultado:
            fin_visto = fin_visto or t
            if t - fin_visto > 360:          # 6 min de gracia para que el pod se detenga solo
                decir("el pod no se detuvo solo: lo detengo")
                cli.detener(pid)
                return resultado
        elif not todo and t - creado > sin_latido_min * 60:
            decir(f"sin señales del pod tras {sin_latido_min:.0f} min: lo detengo")
            volcar_registros()
            cli.detener(pid)
            return "sin_latido"
        elif t - creado > (max_horas + 0.25) * 3600:
            decir(f"superado el tope de {max_horas} h: lo detengo")
            cli.detener(pid)
            return "tope"
        dormir(cada_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pod de RunPod del entrenamiento final de FUGA")
    ap.add_argument("accion", choices=["clave", "simulacro", "crear", "vigilar", "registros", "estado", "detener", "eliminar"])
    ap.add_argument("--plan", choices=["final", "completo"], default="final")
    ap.add_argument("--repo_hf", default="esraderey/fuga-s-eco-rel")
    ap.add_argument("--gpu", choices=sorted(FAMILIAS), default="h100", help="familia de GPU a alquilar")
    ap.add_argument("--precio_max", type=float, default=3.0, help="USD por hora que no se debe superar")
    ap.add_argument("--max_horas", type=int, default=3, help="tope de vida del pod; pasado ese tiempo se detiene")
    ap.add_argument("--secreto", default="hf_token", help="nombre del secreto de RunPod con el token de Hugging Face")
    ap.add_argument("--pod", default="", help="id del pod (por defecto, el registrado en runs/runpod_pod.json)")
    ap.add_argument("--confirmar", action="store_true")
    a = ap.parse_args(argv)
    try:
        return _ejecutar(a)
    except RuntimeError as e:
        print("ERROR:", e, file=sys.stderr)
        return 1


def _ejecutar(a) -> int:
    if a.accion == "clave":
        clave = getpass.getpass("Pega tu API key de RunPod (no se muestra): ").strip()
        if not clave:
            print("no se guardó nada")
            return 1
        os.makedirs(os.path.dirname(ARCHIVO_CLAVE), exist_ok=True)
        with open(ARCHIVO_CLAVE, "w", encoding="utf-8") as f:
            f.write(clave)
        try:
            os.chmod(ARCHIVO_CLAVE, 0o600)
        except OSError:
            pass
        n = len(Cliente(clave).listar())
        print(f"clave guardada en {ARCHIVO_CLAVE} y aceptada por RunPod ({n} pods en la cuenta)")
        return 0
    clave = leer_clave()
    if not clave:
        print("falta la API key de RunPod: ejecuta `python scripts/runpod/lanzar.py clave` en tu terminal", file=sys.stderr)
        return 2
    cli = Cliente(clave)
    if a.accion in ("simulacro", "crear"):
        commit = commit_publicado()
        secretos = cli.secretos()
        if a.secreto not in secretos:
            print(f"falta el secreto «{a.secreto}» en la cuenta de esta API key (secretos: {secretos or 'ninguno'}). "
                  "Créalo en la consola de RunPod → Secrets con tu token de Hugging Face.")
            return 5
        familia = FAMILIAS[a.gpu]
        ofs = cli.precios(familia)
        for o in sorted(ofs, key=lambda o: (o["nube"], o["precio"])):
            print(f"  {o['gpu']:<24} {o['nube']:<9} {o['precio']:.2f} USD/h  disponibilidad {o['stock']}")
        oferta = elegir_oferta(ofs, a.precio_max)
        if not oferta:
            print(f"ninguna {a.gpu.upper()} con disponibilidad por debajo de {a.precio_max:.2f} USD/h; no se crea nada")
            return 3
        precio = oferta["precio_peor"]
        print(f"se probaría, en este orden: {oferta['gpus']} en {oferta['nube']} (secreto «{a.secreto}» presente)")
        print(f"imagen {IMAGEN} | CUDA del host >= {CUDA_MIN} | volumen 40 GB en /workspace | commit {commit[:10]} | plan {a.plan}")
        print(f"precio: hasta {precio:.2f} USD/h | estimado 1,2 a 1,9 h | tope {a.max_horas} h = {a.max_horas * precio:.2f} USD como máximo")
        activos = [p for p in cli.listar() if p.get("name") == NOMBRE and p.get("status") in VIVOS]
        if activos:
            print(f"ya hay un pod {NOMBRE} encendido ({activos[0]['id']}); no se crea otro")
            return 4
        if a.accion == "simulacro" or not a.confirmar:
            print("simulacro: no se creó nada" + ("" if a.accion == "simulacro" else " (falta --confirmar)"))
            return 0

        def construir(gpu):
            return especificacion(gpu, oferta["nube"], commit, a.plan, a.repo_hf, a.max_horas, a.secreto, precio)
        pod, intentos = cli.crear_primero(oferta["gpus"], construir)
        for i in intentos:
            print("  " + i)
        if not pod:
            print("ninguna GPU de la lista tenía capacidad al crear; no se creó nada")
            return 3
        coste = float(pod.get("cost") or 0)
        if coste > a.precio_max + 0.05:
            cli.eliminar(pod["id"])
            print(f"el pod salía a {coste:.2f} USD/h, por encima del tope; eliminado")
            return 3
        os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
        with open(ESTADO, "w", encoding="utf-8") as f:
            json.dump({"id": pod["id"], "creado": time.time(), "creado_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "cost": coste, "gpu": (pod.get("gpu") or {}).get("id"), "nube": pod.get("cloud") or oferta["nube"],
                       "commit": commit, "plan": a.plan, "max_horas": a.max_horas}, f, indent=1)
        print(f"POD CREADO {pod['id']} ({(pod.get('gpu') or {}).get('id')}) a {coste:.2f} USD/h. "
              "Siguiente: python scripts/runpod/lanzar.py vigilar")
        return 0
    if a.accion == "estado":
        for p in cli.listar():
            if p.get("name") == NOMBRE or p.get("id") == a.pod:
                print(json.dumps({k: p.get(k) for k in ("id", "name", "status", "cost", "cloud", "createdAt", "startedAt")},
                                 ensure_ascii=False))
        return 0
    est = {"id": a.pod} if a.pod else _estado()
    if a.accion == "registros":
        for ln in cli.registros(est["id"]):
            print(ln)
        return 0
    if a.accion == "vigilar":
        r = vigilar(cli, est["id"], est.get("creado", time.time()), est.get("max_horas", a.max_horas),
                    ver_registros=lambda: cli.registros(est["id"]))
        print("RESULTADO DE LA VIGILANCIA:", r)
        return 0 if r == "ok" else 1
    if a.accion == "detener":
        cli.detener(est["id"])
        print("pod detenido:", est["id"])
        return 0
    if not a.confirmar:
        print("eliminar borra el pod y su volumen (checkpoints y almacén Z-Space incluidos); añade --confirmar")
        return 1
    cli.eliminar(est["id"])
    print("pod eliminado:", est["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
