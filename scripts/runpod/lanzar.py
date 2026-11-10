"""Lanza, vigila y detiene desde esta máquina el pod de RunPod del entrenamiento final de FUGA.

La API key de RunPod la guarda el usuario (`clave`, con entrada oculta) o la variable RUNPOD_API_KEY; este script nunca
la imprime ni la pone en una URL. El token de Hugging Face no pasa por aquí: el pod lo recibe de un secreto de RunPod.

  python scripts/runpod/lanzar.py clave                 # (tú, en tu terminal) guarda la API key de RunPod
  python scripts/runpod/lanzar.py simulacro             # precios y stock de H100, pod que se crearía y coste; no crea nada
  python scripts/runpod/lanzar.py crear --confirmar     # crea el pod (gasta saldo de RunPod)
  python scripts/runpod/lanzar.py vigilar               # sigue el progreso; detiene el pod al terminar o al pasar el tope
  python scripts/runpod/lanzar.py estado                # pods de este proyecto y su coste
  python scripts/runpod/lanzar.py detener               # detiene el pod (conserva el volumen)
  python scripts/runpod/lanzar.py eliminar --confirmar  # elimina el pod y su volumen
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
H100 = ("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL")
IMAGEN = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CUDA = ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0"]
NOMBRE = "fuga-final"
REPO_GIT = "https://github.com/esraderey/fuga"
ARCHIVO_CLAVE = os.path.join(os.path.expanduser("~"), ".runpod", "fuga_api_key")
ESTADO = os.path.join(RAIZ, "runs", "runpod_pod.json")
REGISTRO_HF = "esraderey/fuga-datos"
RUTA_LATIDO = "corridas/final_h100"
FIN_OK, FIN_MAL = "CADENA FINAL TERMINADA", "CADENA FINAL FALLÓ"
# Arranque del contenedor: temporizador de seguridad, clon del commit exacto, arranque de FUGA y servicios de la imagen.
ARRANQUE = (
    '(sleep $(( ${FUGA_MAX_HORAS:-5} * 3600 )); runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID") '
    '>/dev/null 2>&1 & '
    '( (mkdir -p /workspace && cd /workspace && { [ -d fuga/.git ] || git clone -q ' + REPO_GIT + '; } && cd fuga '
    '&& git fetch -q origin && git checkout -q "$FUGA_COMMIT" && bash scripts/runpod/arranque.sh) '
    '> /workspace/arranque.salida.log 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" ) & '
    'if [ -x /start.sh ]; then exec /start.sh; else exec sleep infinity; fi'
)


def leer_clave() -> str:
    clave = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not clave and os.path.exists(ARCHIVO_CLAVE):
        with open(ARCHIVO_CLAVE, encoding="utf-8") as f:
            clave = f.read().strip()
    return clave


def http(metodo: str, url: str, clave: str, cuerpo: dict | None = None, timeout: int = 60):
    """(código, respuesta) de una llamada JSON autenticada por cabecera; nunca lanza por un código de error HTTP."""
    datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
    req = urllib.request.Request(url, data=datos, method=metodo, headers={
        "Authorization": f"Bearer {clave}", "Content-Type": "application/json", "User-Agent": "fuga-lanzar/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt, cod = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        txt, cod = e.read().decode("utf-8", "replace"), e.code
    try:
        return cod, json.loads(txt) if txt else None
    except json.JSONDecodeError:
        return cod, txt


class Cliente:
    def __init__(self, clave: str, enviar=http):
        self.clave, self.enviar = clave, enviar

    def _ok(self, cod: int, resp, que: str):
        if not 200 <= cod < 300:
            raise RuntimeError(f"RunPod respondió {cod} al {que}: {str(resp)[:400]}")
        return resp

    def listar(self) -> list:
        return self._ok(*self.enviar("GET", f"{REST}/pods", self.clave), "listar pods") or []

    def pod(self, pid: str) -> dict:
        return self._ok(*self.enviar("GET", f"{REST}/pods/{pid}", self.clave), "leer el pod")

    def crear(self, spec: dict) -> dict:
        return self._ok(*self.enviar("POST", f"{REST}/pods", self.clave, spec), "crear el pod")

    def detener(self, pid: str):
        return self._ok(*self.enviar("POST", f"{REST}/pods/{pid}/stop", self.clave), "detener el pod")

    def eliminar(self, pid: str):
        return self._ok(*self.enviar("DELETE", f"{REST}/pods/{pid}", self.clave), "eliminar el pod")

    def precios(self, gpus=H100) -> list[dict]:
        """Ofertas [{gpu, nube, precio, stock}] de GPU bajo demanda; lista vacía si la consulta de precios no responde."""
        campos = ("id displayName seguro: lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus uninterruptablePrice } "
                  "comunidad: lowestPrice(input: {gpuCount: 1, secureCloud: false}) { stockStatus uninterruptablePrice }")
        consulta = "query { " + " ".join(f'g{i}: gpuTypes(input: {{id: "{g}"}}) {{ {campos} }}' for i, g in enumerate(gpus)) + " }"
        cod, resp = self.enviar("POST", GQL, self.clave, {"query": consulta})
        if not 200 <= cod < 300 or not isinstance(resp, dict) or not resp.get("data"):
            return []
        return ofertas(resp["data"])


def ofertas(data: dict) -> list[dict]:
    out = []
    for lista in data.values():
        for g in lista or []:
            for campo, nube in (("seguro", "SECURE"), ("comunidad", "COMMUNITY")):
                lp = g.get(campo) or {}
                if lp.get("uninterruptablePrice"):
                    out.append({"gpu": g["id"], "nube": nube, "precio": float(lp["uninterruptablePrice"]),
                                "stock": lp.get("stockStatus")})
    return out


def elegir_oferta(ofs: list[dict], precio_max: float) -> dict | None:
    """Nube y GPUs a pedir: las H100 con stock y precio dentro del tope, de la más barata a la más cara; se prefiere la
    nube segura (centros de datos de RunPod) y solo si no hay nada en ella se usa la comunitaria."""
    validas = [o for o in ofs if o["precio"] <= precio_max and str(o["stock"] or "none").lower() != "none"]
    for nube in ("SECURE", "COMMUNITY"):
        suyas = sorted((o for o in validas if o["nube"] == nube), key=lambda o: (o["precio"], o["gpu"]))
        if suyas:
            return {"nube": nube, "gpus": [o["gpu"] for o in suyas], "precio": suyas[0]["precio"],
                    "precio_peor": suyas[-1]["precio"], "stock": suyas[0]["stock"]}
    return None


def especificacion(oferta: dict | None, commit: str, plan: str = "final", repo_hf: str = "esraderey/fuga-s-eco-rel",
                   max_horas: int = 5, secreto: str = "hf_token", precio_hora: float = 3.0, volumen_gb: int = 40) -> dict:
    """Cuerpo de POST /pods. El token de Hugging Face va como referencia a un secreto de RunPod, nunca como valor."""
    gpus = oferta["gpus"] if oferta else list(H100)
    return {"name": NOMBRE, "imageName": IMAGEN, "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": gpus,
            "gpuTypePriority": "custom" if oferta else "availability", "cloudType": oferta["nube"] if oferta else "SECURE",
            "interruptible": False, "allowedCudaVersions": CUDA, "containerDiskInGb": 20, "volumeInGb": volumen_gb,
            "volumeMountPath": "/workspace", "ports": ["22/tcp"],
            "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_" + secreto + " }}", "FUGA_COMMIT": commit, "FUGA_PLAN": plan,
                    "FUGA_REPO_HF": repo_hf, "FUGA_APAGAR": "1", "FUGA_MAX_HORAS": str(int(max_horas)),
                    "FUGA_PRECIO_HORA": f"{precio_hora:.2f}"},
            "dockerStartCmd": ["bash", "-c", ARRANQUE]}


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
            leer=leer_latido, dormir=time.sleep, ahora=time.time, decir=print) -> str:
    """Bucle de vigilancia. Devuelve 'ok', 'fallo', 'parado', 'sin_latido' o 'tope'; detiene el pod si sigue encendido."""
    vistos: set[str] = set()
    ultimo = ""
    fin_visto = None
    while True:
        t = ahora()
        pod = cli.pod(pid)
        st = pod.get("desiredStatus")
        reg = leer()
        for ln in hitos(reg["cadena.log"]) + hitos(reg["arranque.log"]):
            if ln not in vistos:
                vistos.add(ln)
                decir(ln[:260])
        pr = progreso(reg["cadena.log"])
        if pr and pr != ultimo:
            ultimo = pr
            decir(f"[{time.strftime('%H:%M')}] {pr} | pod {st} | {(t - creado)/60:.0f} min | ~{(t - creado)/3600*float(pod.get('costPerHr') or 0):.2f} USD")
        todo = reg["cadena.log"] + reg["arranque.log"]
        resultado = "ok" if FIN_OK in todo else "fallo" if (FIN_MAL in todo or "ARRANQUE FALLÓ" in todo) else None
        if st != "RUNNING":
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
            cli.detener(pid)
            return "sin_latido"
        elif t - creado > (max_horas + 0.25) * 3600:
            decir(f"superado el tope de {max_horas} h: lo detengo")
            cli.detener(pid)
            return "tope"
        dormir(cada_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pod de RunPod del entrenamiento final de FUGA")
    ap.add_argument("accion", choices=["clave", "simulacro", "crear", "vigilar", "estado", "detener", "eliminar"])
    ap.add_argument("--plan", choices=["final", "completo"], default="final")
    ap.add_argument("--repo_hf", default="esraderey/fuga-s-eco-rel")
    ap.add_argument("--precio_max", type=float, default=3.0, help="USD por hora que no se debe superar")
    ap.add_argument("--max_horas", type=int, default=5, help="tope de vida del pod; pasado ese tiempo se detiene")
    ap.add_argument("--secreto", default="hf_token", help="nombre del secreto de RunPod con el token de Hugging Face")
    ap.add_argument("--pod", default="", help="id del pod (por defecto, el registrado en runs/runpod_pod.json)")
    ap.add_argument("--confirmar", action="store_true")
    a = ap.parse_args(argv)
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
        ofs = cli.precios()
        oferta = elegir_oferta(ofs, a.precio_max)
        for o in sorted(ofs, key=lambda o: (o["nube"], o["precio"])):
            print(f"  {o['gpu']:<24} {o['nube']:<9} {o['precio']:.2f} USD/h  stock {o['stock']}")
        if ofs and not oferta:
            print(f"ninguna H100 con stock por debajo de {a.precio_max:.2f} USD/h; no se crea nada")
            return 3
        precio = oferta["precio_peor"] if oferta else a.precio_max
        spec = especificacion(oferta, commit, a.plan, a.repo_hf, a.max_horas, a.secreto, precio)
        print(f"pod: {spec['gpuTypeIds']} en {spec['cloudType']}, imagen {IMAGEN}, volumen {spec['volumeInGb']} GB")
        print(f"código: commit {commit[:10]} | plan {a.plan} | publica en {a.repo_hf} | secreto HF: {a.secreto}")
        print(f"precio: {'hasta ' + format(precio, '.2f') if oferta else 'sin consulta previa; tope ' + format(a.precio_max, '.2f')} USD/h | "
              f"estimado 1,5 a 2,5 h | tope {a.max_horas} h = {a.max_horas * precio:.2f} USD como máximo")
        activos = [p for p in cli.listar() if p.get("name") == NOMBRE and p.get("desiredStatus") == "RUNNING"]
        if activos:
            print(f"ya hay un pod {NOMBRE} encendido ({activos[0]['id']}); no se crea otro")
            return 4
        if a.accion == "simulacro" or not a.confirmar:
            print("simulacro: no se creó nada" + ("" if a.accion == "simulacro" else " (falta --confirmar)"))
            return 0
        pod = cli.crear(spec)
        coste = float(pod.get("costPerHr") or 0)
        if coste > a.precio_max + 0.05:
            cli.eliminar(pod["id"])
            print(f"el pod salía a {coste:.2f} USD/h, por encima del tope; eliminado")
            return 3
        os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
        with open(ESTADO, "w", encoding="utf-8") as f:
            json.dump({"id": pod["id"], "creado": time.time(), "creado_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "costPerHr": coste, "commit": commit, "plan": a.plan, "max_horas": a.max_horas,
                       "gpu": (pod.get("machine") or {}).get("gpuTypeId") or spec["gpuTypeIds"], "nube": spec["cloudType"]}, f, indent=1)
        print(f"POD CREADO {pod['id']} a {coste:.2f} USD/h. Siguiente: python scripts/runpod/lanzar.py vigilar")
        return 0
    if a.accion == "estado":
        for p in cli.listar():
            if p.get("name") == NOMBRE or p.get("id") == a.pod:
                print(json.dumps({k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "lastStartedAt", "lastStatusChange")},
                                 ensure_ascii=False))
        return 0
    est = {"id": a.pod} if a.pod else _estado()
    if a.accion == "vigilar":
        r = vigilar(cli, est["id"], est.get("creado", time.time()), est.get("max_horas", a.max_horas))
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
