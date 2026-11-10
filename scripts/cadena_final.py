"""Cadena del entrenamiento final de FUGA (ensayo/prerregistro_final.md).

1. Sonda de velocidad y proyección de tiempo y coste.
2. Corrida completa (equivalente a 6 h de la RTX 4060) con checkpoints periódicos en Z-Space.
3. Candidatos de presupuesto corto con decaimiento completo de la tasa (plan `final`).
4. Evaluación de todos los checkpoints en validación y test completos; elección por validación.
5. Muestras con el elegido y análisis de copia, escala y repetición.
6. Export a Hugging Face (safetensors + ficha) y, con --subir, publicación.
7. Con --apagar, detiene el pod de RunPod al terminar, también si algo falla.

Reanudable: cada paso se salta si su salida ya existe.

Uso:
  python scripts/cadena_final.py --plan final --raiz runs/final_h100 --device cuda --subir usuario/fuga-s-eco-rel --apagar
  python scripts/cadena_final.py --prueba --raiz runs/_prueba --device cpu          # ensayo en miniatura
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REGLA = "menor bits por nota en validación completa entre modelo_mejor.pt y modelo.pt de todas las corridas"


def _leer(ruta: str) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _escribir(ruta: str, obj) -> None:
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def _corrida(clave: str) -> str:
    return os.path.basename(os.path.dirname(clave))


def elegir(evals: dict) -> str:
    """Checkpoint con menor bits por nota en validación completa (empates: orden alfabético de la ruta)."""
    return min(sorted(evals), key=lambda k: evals[k]["val"]["bits_por_nota"])


def contraste(evals: dict, a: str, b: str, bootstrap, n: int = 10000) -> dict:
    """IC95 pareado por pieza de bits por nota en test (a − b); negativo favorece a `a`."""
    pa, pb = evals[a].get("por_pieza", {}), evals[b].get("por_pieza", {})
    comunes = sorted(k for k in set(pa) & set(pb) if pa[k][1] > 0 and pb[k][1] > 0)
    d = [pa[k][0] / pa[k][1] - pb[k][0] / pb[k][1] for k in comunes]
    ic = bootstrap(d, n=n)
    if not d:
        veredicto = "sin piezas comunes"
    else:
        veredicto = "pasa (IC95 < 0)" if ic[2] < 0 else "al revés (IC95 > 0)" if ic[1] > 0 else "sin efecto detectable"
    return {"a": a, "b": b, "n_piezas": len(d), "IC95_pareado": ic, "veredicto": veredicto}


def elegir_muestras(analisis: dict, k: int = 2) -> list[int]:
    """Índices de las piezas que ilustran la ficha: completas, sin copia de 4 compases y con la repetición más
    parecida a la de la música real (desempate: más voces, más notas). Si ninguna cumple, las más largas."""
    real = analisis["base_real"]["test_real_repeticion"]
    mu = [m for m in analisis["muestras"] if m["notas"] > 0]
    buenas = [m for m in mu if m["compases"] >= analisis.get("compases", 16) and m["copia4"] == 0 and m["notas"] >= 40]
    orden = sorted(buenas, key=lambda m: (abs(m["repeticion"] - real), -m["partes"], -m["notas"]))
    if not orden:
        orden = sorted(mu, key=lambda m: -m["notas"])
    return [m["muestra"] for m in orden[:k]]


def proyectar(s_paso: float, pasos: int, versiones: int, s_version: float = 12.0, fijo_min: float = 20.0,
              precio_hora: float = 3.0) -> dict:
    """Minutos y coste previstos: entrenamiento + guardados Z-Space + margen fijo (evaluar, muestras, export, subida)."""
    entreno = pasos * s_paso / 60
    total = entreno + versiones * s_version / 60 + fijo_min
    return {"s_por_paso": s_paso, "pasos": pasos, "minutos_entrenamiento": entreno, "minutos_totales": total,
            "precio_hora": precio_hora, "coste": total / 60 * precio_hora}


def apagar_pod(registrar) -> None:
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        registrar("apagar: RUNPOD_POD_ID no está definido; no se apaga nada")
        return
    for cmd in (["runpodctl", "pod", "stop", pod], ["runpodctl", "stop", "pod", pod]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            registrar("apagar: runpodctl no está en esta imagen; DETÉN EL POD DESDE LA CONSOLA")
            return
        except subprocess.TimeoutExpired:
            continue
        registrar(f"apagar: {' '.join(cmd[:3])} -> exit {r.returncode} {(r.stdout or r.stderr).strip()[:200]}")
        if r.returncode == 0:
            return
    registrar("apagar: no se pudo detener el pod; DETENLO DESDE LA CONSOLA")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento final de FUGA: entrenar, seleccionar, publicar")
    ap.add_argument("--raiz", default="runs/final")
    ap.add_argument("--plan", choices=["final", "completo"], default="final",
                    help="completo = solo la corrida larga; final = además los candidatos cortos y la selección entre todos")
    ap.add_argument("--preset", default="fuga-s")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--pasos_completo", type=int, default=18380, help="360 min de la RTX 4060 a 1,1752 s/paso")
    ap.add_argument("--candidatos", type=int, nargs="*", default=[4500, 6000, 8000])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tokens_por_lote", type=int, default=8192)
    ap.add_argument("--eval_cada", type=int, default=250)
    ap.add_argument("--guardar_cada", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt_activaciones", action="store_true", help="checkpointing de activaciones (GPUs de poca memoria)")
    ap.add_argument("--datos", default="data/corpus_pd.jsonl")
    ap.add_argument("--cache", default="cache/eco_rel_1024.pkl")
    ap.add_argument("--cache_test", default="cache/test_eco_rel_1024.pkl")
    ap.add_argument("--n_muestras", type=int, default=12)
    ap.add_argument("--nombre", default="FUGA-S eco_rel")
    ap.add_argument("--hf_dir", default="", help="carpeta del export (por defecto <raiz>/hf)")
    ap.add_argument("--subir", default="", help="repositorio de modelos de Hugging Face donde publicar (usuario/nombre)")
    ap.add_argument("--privado", action="store_true")
    ap.add_argument("--registro_hf", default="", help="dataset de Hugging Face donde dejar copia de los registros (opcional)")
    ap.add_argument("--latido_min", type=float, default=5.0,
                    help="minutos entre copias de los registros a --registro_hf mientras la cadena corre (0 = solo al final)")
    ap.add_argument("--precio_hora", type=float, default=3.0)
    ap.add_argument("--apagar", action="store_true", help="detiene el pod de RunPod al terminar (también si falla)")
    ap.add_argument("--prueba", action="store_true", help="miniatura: nano, pocos pasos, 2 muestras")
    a = ap.parse_args(argv)
    if a.prueba:
        a.preset, a.ctx, a.pasos_completo, a.candidatos = "nano", 256, 60, [30, 40]
        a.eval_cada, a.guardar_cada, a.n_muestras, a.tokens_por_lote = 10, 20, 2, 1024
    raiz = a.raiz
    os.makedirs(raiz, exist_ok=True)
    hf_dir = a.hf_dir or os.path.join(raiz, "hf")
    ruta_log = os.path.join(raiz, "cadena.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=RAIZ + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t_inicio = time.time()
    tiempos: dict[str, float] = {}

    def registrar(msg: str) -> None:
        linea = f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(linea, flush=True)
        with open(ruta_log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")

    def paso(nombre: str, cmd: list[str], obligatorio: bool = True) -> int:
        t0 = time.time()
        registrar(f"{nombre}: {' '.join(cmd)}")
        with open(ruta_log, "a", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=RAIZ, env=env, stdout=f, stderr=subprocess.STDOUT)
        tiempos[nombre] = time.time() - t0
        registrar(f"{nombre} exit={r.returncode} segundos={tiempos[nombre]:.0f}")
        if r.returncode and obligatorio:
            raise RuntimeError(f"el paso «{nombre}» falló con código {r.returncode}; ver {ruta_log}")
        return r.returncode

    def entrenar(out: str, pasos: int, guardar_cada: int, eval_cada: int, mejor: bool) -> list[str]:
        c = [PY, "-m", "fuga.train", "--datos", a.datos, "--modo", "eco_rel", "--preset", a.preset, "--ctx", str(a.ctx),
             "--cache", a.cache, "--device", a.device, "--tokens_por_lote", str(a.tokens_por_lote), "--pasos", str(pasos),
             "--lr", str(a.lr), "--wd", str(a.wd), "--dropout", str(a.dropout), "--eval_cada", str(eval_cada),
             "--semilla", str(a.semilla), "--out", out]
        if mejor:
            c.append("--guardar_mejor")
        if guardar_cada:
            c += ["--guardar_cada", str(guardar_cada)]
        if a.ckpt_activaciones:
            c.append("--ckpt")
        return c

    def subir_registros(mensaje: str) -> None:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=raiz, repo_id=a.registro_hf, repo_type="dataset",
                              path_in_repo=f"corridas/{os.path.basename(os.path.normpath(raiz))}",
                              allow_patterns=["cadena.log", "RESUMEN.md", "*.json", "*/zspace_listado.txt"],
                              ignore_patterns=["hf/*", "sonda/*", "muestras/*"], commit_message=mensaje)

    parar_latido = threading.Event()

    def latir() -> None:      # copia periódica de los registros: permite seguir la corrida sin entrar al pod
        while not parar_latido.is_set():
            try:
                subir_registros("latido")
            except Exception as e:
                print(f"latido: {type(e).__name__}: {str(e)[:200]}", flush=True)
            parar_latido.wait(a.latido_min * 60)

    corridas = [("completo", a.pasos_completo, a.guardar_cada)]
    if a.plan == "final":
        corridas += [(f"wsd_{t}", t, 0) for t in a.candidatos]
    estado = "fallo"
    hilo_latido = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    try:
        registrar(f"cadena final: commit={commit[:12] or 'desconocido'} plan={a.plan} preset={a.preset} "
                  f"corridas={[(n, p) for n, p, _ in corridas]} dropout={a.dropout} device={a.device} raiz={raiz}")
        if a.registro_hf and a.latido_min > 0:
            hilo_latido = threading.Thread(target=latir, daemon=True)
            hilo_latido.start()
        # 1. sonda
        ruta_sonda = os.path.join(raiz, "sonda.json")
        if not os.path.exists(ruta_sonda):
            n_sonda = 20 if a.prueba else 100
            d_sonda = os.path.join(raiz, "sonda")
            paso("sonda", entrenar(d_sonda, n_sonda, 0, 0, False))
            lg = _leer(os.path.join(d_sonda, "log.json"))
            cv = lg["curva"]
            s_paso = (cv[-1]["t"] - cv[0]["t"]) / (cv[-1]["paso"] - cv[0]["paso"]) if len(cv) >= 2 else \
                max(lg["final"]["segundos"], 1) / n_sonda
            pr = proyectar(s_paso, sum(p for _, p, _ in corridas), math.ceil(a.pasos_completo / max(1, a.guardar_cada)),
                           precio_hora=a.precio_hora)
            pr["device"] = lg.get("device")
            _escribir(ruta_sonda, pr)
            shutil.rmtree(d_sonda, ignore_errors=True)
        pr = _leer(ruta_sonda)
        registrar(f"PROYECCIÓN: {pr['s_por_paso']:.3f} s/paso -> entrenamiento {pr['minutos_entrenamiento']:.0f} min, "
                  f"total {pr['minutos_totales']:.0f} min, coste ~{pr['coste']:.2f} USD a {pr['precio_hora']:.2f} USD/h")
        # 2 y 3. corridas
        for nombre, pasos, gc in corridas:
            out = os.path.join(raiz, nombre)
            if os.path.exists(os.path.join(out, "log.json")):
                registrar(f"[saltado] {nombre}: ya tiene log.json")
                continue
            paso(f"entrenar_{nombre}", entrenar(out, pasos, gc, a.eval_cada, True))
        # Z-Space de la corrida completa
        d_comp = os.path.join(raiz, "completo")
        if a.guardar_cada and os.path.isdir(os.path.join(d_comp, "zspace")):
            if not os.path.exists(os.path.join(d_comp, "zspace_verificacion.json")):
                paso("verificar_zspace", [PY, os.path.join("scripts", "verificar_zspace.py"), "--run", d_comp], obligatorio=False)
            with open(os.path.join(d_comp, "zspace_listado.txt"), "w", encoding="utf-8") as f:
                subprocess.run([PY, "-m", "fuga.zcheckpoints", "--raiz", os.path.join(d_comp, "zspace")], cwd=RAIZ, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
        # 4. evaluación y elección
        ruta_eval = os.path.join(raiz, "evaluacion.json")
        ckpts = [os.path.join(raiz, n, x) for n, _, _ in corridas for x in ("modelo_mejor.pt", "modelo.pt")
                 if os.path.exists(os.path.join(raiz, n, x))]
        if not os.path.exists(ruta_eval):
            paso("evaluar", [PY, os.path.join("scripts", "evaluar_ckpts.py"), "--ckpts", *ckpts, "--cache", a.cache,
                             "--cache_test", a.cache_test, "--datos", a.datos, "--device", a.device, "--por_pieza",
                             "--out", ruta_eval])
        ev = _leer(ruta_eval)
        evals = ev["evaluaciones"]
        elegido = elegir(evals)
        logs = {n: _leer(os.path.join(raiz, n, "log.json")) for n, _, _ in corridas}
        candidatos = []
        for k in sorted(evals):
            lg = logs[_corrida(k)]
            es_mejor = os.path.basename(k) == "modelo_mejor.pt"
            candidatos.append({"ckpt": k, "corrida": _corrida(k), "archivo": os.path.basename(k),
                               "paso": lg.get("mejor", {}).get("paso") if es_mejor else lg["final"]["paso"],
                               "pasos_corrida": lg["final"]["paso"], "val": evals[k]["val"]["bits_por_nota"],
                               "test": evals[k]["test"]["bits_por_nota"], "segundos_corrida": lg["final"]["segundos"]})
        sel = {"regla": REGLA, "plan": a.plan, "candidatos": candidatos, "elegido": elegido,
               "val_trozos": ev["val_trozos"], "test_trozos": ev.get("test_trozos"), "test_piezas": ev.get("test_piezas")}
        cortos = [k for k in evals if _corrida(k).startswith("wsd_")]
        largos = [k for k in evals if _corrida(k) == "completo"]
        if cortos and largos:
            sys.path.insert(0, os.path.join(RAIZ, "scripts"))
            from evaluar_ckpts import _mz
            sel["KF1"] = contraste(evals, elegir({k: evals[k] for k in cortos}), elegir({k: evals[k] for k in largos}),
                                   _mz().bootstrap)
        _escribir(os.path.join(raiz, "seleccion.json"), sel)
        registrar(f"ELEGIDO {elegido}: val {evals[elegido]['val']['bits_por_nota']:.4f} "
                  f"test {evals[elegido]['test']['bits_por_nota']:.4f}" + (f" | KF1 {sel['KF1']['IC95_pareado']} {sel['KF1']['veredicto']}" if "KF1" in sel else ""))
        # 5. muestras y análisis
        d_mu = os.path.join(raiz, "muestras")
        if not os.path.exists(os.path.join(d_mu, "resumen.json")):
            paso("generar", [PY, "-m", "fuga.sample", "--ckpt", elegido, "--n", str(a.n_muestras), "--compases", "16", "--wav",
                             "--interpretar", "romantico", "--device", "cuda" if a.device.startswith("cuda") else "cpu",
                             "--out", d_mu])
        ruta_an = os.path.join(raiz, "analisis_muestras.json")
        if not os.path.exists(ruta_an):
            paso("analizar", [PY, os.path.join("scripts", "analizar_muestras.py"), "--ckpt", elegido, "--n", str(a.n_muestras),
                              "--datos", a.datos, "--device", a.device, "--out", ruta_an])
        an = _leer(ruta_an)
        ilustran = elegir_muestras(an)
        archivos = [f for i in ilustran for f in (os.path.join(d_mu, f"muestra_{i:02d}_romantico.wav"),
                                                    os.path.join(d_mu, f"muestra_{i:02d}_romantico.mid")) if os.path.exists(f)]
        # 6. export y publicación
        d_run = os.path.dirname(elegido)
        _escribir(os.path.join(d_run, "eleccion.json"),
                  {"evaluaciones": {k: {"val": v["val"], "test": v["test"]} for k, v in evals.items() if os.path.dirname(k) == d_run},
                   "elegido": elegido, "val_trozos": ev["val_trozos"], "test_trozos": ev.get("test_trozos")})
        shutil.copyfile(ruta_an, os.path.join(d_run, "analisis_muestras.json"))
        paso("exportar", [PY, os.path.join("scripts", "exportar_hf.py"), "--ckpt", elegido, "--run", d_run, "--out", hf_dir,
                          "--nombre", a.nombre, "--datos", a.datos, "--seleccion", os.path.join(raiz, "seleccion.json"),
                          *(["--commit", commit] if commit else []), "--muestras", *archivos])
        # resumen y registros dentro del export
        dur = time.time() - t_inicio
        z = _leer(os.path.join(d_comp, "zspace_verificacion.json")) if os.path.exists(os.path.join(d_comp, "zspace_verificacion.json")) else None
        filas = "\n".join(f"| {c['corrida']} | {c['archivo']} | {c['paso']} / {c['pasos_corrida']} | {c['val']:.3f} | {c['test']:.3f} | "
                          f"{'**elegido**' if c['ckpt'] == elegido else ''} |" for c in candidatos)
        resumen = [f"# Entrenamiento final de FUGA: resumen ({time.strftime('%Y-%m-%d %H:%M')})", "",
                   f"- Plan `{a.plan}`, preset `{a.preset}`, dropout {a.dropout}, dispositivo {pr.get('device')}.",
                   f"- Sonda: {pr['s_por_paso']:.3f} s/paso. Duración real de la cadena hasta aquí: {dur/60:.0f} min "
                   f"(~{dur/3600*a.precio_hora:.2f} USD a {a.precio_hora:.2f} USD/h).",
                   f"- Regla de elección: {REGLA}.", "",
                   "| Corrida | Checkpoint | Paso | Val bits/nota | Test bits/nota | |", "|---|---|---|---|---|---|", filas, ""]
        if "KF1" in sel:
            k1 = sel["KF1"]
            resumen += [f"- KF1 (mejor corto − mejor largo, pareado por pieza en test, {k1['n_piezas']} piezas): "
                        f"{k1['IC95_pareado'][0]:+.4f} [{k1['IC95_pareado'][1]:+.4f}; {k1['IC95_pareado'][2]:+.4f}] → {k1['veredicto']}."]
        if z:
            sv = z.get("segundos_por_version", {})
            resumen += [f"- Z-Space ({z['dtype']}): {z['versiones']} versiones, {z['bytes_tienda']/1e9:.2f} GB = "
                        f"{100*z['ratio_vs_mismo_dtype']:.1f} % del equivalente {z['dtype']} y {100*z['ratio_vs_fp32']:.1f} % del fp32; "
                        f"exacto final: {z.get('exacto_modelo.pt')}, exacto mejor: {z.get('exacto_modelo_mejor.pt')}; "
                        f"guardado medio {sv.get('media', float('nan')):.1f} s, suma {sv.get('suma', float('nan')):.0f} s."]
        mu = [m for m in an["muestras"] if m["notas"] > 0]
        if mu:
            resumen += [f"- Muestras ({len(mu)} con notas): copia de 4 compases máx. {max(m['copia4'] for m in mu):.3f}, "
                        f"repetición media {sum(m['repeticion'] for m in mu)/len(mu):.3f} (real {an['base_real']['test_real_repeticion']:.3f}); "
                        f"ilustran la ficha: {ilustran}."]
        with open(os.path.join(raiz, "RESUMEN.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(resumen) + "\n")
        d_reg = os.path.join(hf_dir, "corrida")
        os.makedirs(d_reg, exist_ok=True)
        for patron in ("cadena.log", "RESUMEN.md", "sonda.json", "seleccion.json", "analisis_muestras.json", "*/log.json",
                       "completo/zspace_verificacion.json", "completo/zspace_listado.txt", "completo/zspace/versiones.json"):
            for src in glob.glob(os.path.join(raiz, patron)):
                rel = os.path.relpath(src, raiz)
                if rel.split(os.sep)[0] == os.path.basename(hf_dir):
                    continue
                os.makedirs(os.path.dirname(os.path.join(d_reg, rel)), exist_ok=True)
                shutil.copyfile(src, os.path.join(d_reg, rel))
        if a.subir:
            paso("subir", [PY, os.path.join("scripts", "subir_hf.py"), "--dir", hf_dir, "--repo", a.subir, "--confirmar",
                           *(["--privado"] if a.privado else [])])
            registrar(f"PUBLICADO en https://huggingface.co/{a.subir}")
        estado = "ok"
        registrar(f"CADENA FINAL TERMINADA en {(time.time() - t_inicio)/60:.0f} min")
        return 0
    except Exception as e:  # se registra y se sigue al cierre (registros y apagado)
        registrar(f"CADENA FINAL FALLÓ: {type(e).__name__}: {e}")
        with open(ruta_log, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1
    finally:
        parar_latido.set()
        if hilo_latido is not None:
            hilo_latido.join(timeout=180)
        if a.registro_hf:
            try:
                registrar(f"copiando los registros a {a.registro_hf} (estado: {estado})")
                subir_registros(f"registros de la cadena final ({estado})")
            except Exception as e:
                registrar(f"no se pudieron copiar los registros a {a.registro_hf}: {type(e).__name__}: {e}")
        if a.apagar:
            apagar_pod(registrar)


if __name__ == "__main__":
    sys.exit(main())
