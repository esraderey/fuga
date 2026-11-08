import random

import pytest
import torch

from fuga.data import colar, secuencias, trozos_eval
from fuga.metrics import perfil, repeticion, segmentos
from fuga.model import Transformer, config, contar_parametros
from fuga.sample import generar
from fuga.tokenizer import Vocab, codificar, decodificar, segmentar

TPQ = 12


def pieza_sintetica(semilla=0, compases=12, partes=2):
    rnd = random.Random(semilla)
    notas = []
    motivo = [(0, 60, 12), (12, 62, 12), (24, 64, 6), (30, 65, 6), (36, 67, 12)]
    for b in range(compases):
        base = b * 48
        for p in range(partes):
            if b % 4 == 1:          # copia exacta del compás anterior
                fuente = [(o, pi + 12 * p, d) for o, pi, d in motivo]
            elif b % 4 == 2:        # transposición cromática +2
                fuente = [(o, pi + 2 + 12 * p, d) for o, pi, d in motivo]
            elif b % 4 == 3:        # mismo ritmo, alturas nuevas
                fuente = [(o, rnd.choice([55, 57, 59, 60, 69]) + 12 * p, d) for o, _, d in motivo]
            else:
                fuente = [(o, pi + 12 * p, d) for o, pi, d in motivo]
            notas += [(base + o, d, pi, p) for o, pi, d in fuente]
    return {"id": f"sint#{semilla}", "fuente": "sint", "tpq": TPQ, "compases": [i * 48 for i in range(compases)],
            "tonica": 0, "modo": "major", "tempo": 100.0, "partes": partes, "notas": sorted(notas)}


@pytest.fixture
def voc():
    return Vocab(["sint"])


def contenido(segs):
    return sorted((b, p, x) for b, s in enumerate(segs) for p, sg in s.items() for x in sg)


@pytest.mark.parametrize("eco", [False, True])
def test_ida_y_vuelta(voc, eco):
    for s in range(5):
        pz = pieza_sintetica(s)
        _, segs = segmentar(pz)
        c, _ = codificar(pz, voc, eco=eco)
        d = decodificar(c.ids, voc, eco=eco)
        assert contenido(d.segs) == contenido(segs)
        assert d.fin


def test_eco_comprime_y_usa_todas_las_formas(voc):
    pz = pieza_sintetica(1)
    cb, _ = codificar(pz, voc, eco=False)
    ce, _ = codificar(pz, voc, eco=True)
    assert len(ce.ids) < 0.75 * len(cb.ids)
    toks = {voc.itos[i] for i in ce.ids}
    assert {"ECO", "ECOR", "TRC_0", "TRC_2", "DIST_0"} <= toks


def test_transposicion_diatonica(voc):
    # do-re-mi → re-mi-fa (no es cromática constante: +2,+2,+1) debe salir TRD_1
    notas = []
    for b, desp in ((0, 0), (1, 1)):
        esc = [60, 62, 64, 65, 67]
        for k in range(3):
            notas.append((b * 48 + k * 12, 12, esc[k + desp], 0))
    pz = {"id": "d#0", "fuente": "sint", "tpq": 12, "compases": [0, 48], "tonica": 0, "modo": "major",
          "tempo": 90.0, "partes": 1, "notas": notas}
    ce, _ = codificar(pz, voc, eco=True)
    assert "TRD_1" in {voc.itos[i] for i in ce.ids}
    d = decodificar(ce.ids, voc)
    assert contenido(d.segs) == contenido(segmentar(pz)[1])


def test_trozo_acotado_sin_eos_y_mismas_notas(voc):
    pz = pieza_sintetica(2, compases=40)
    t = trozos_eval([pz], voc, ctx=200)
    assert len(t["base"]) == len(t["eco"]) == len(t["fant"]) > 1
    assert [x[3] for x in t["base"]] == [x[3] for x in t["eco"]] == [x[3] for x in t["fant"]]
    assert all(len(x[0]) <= 200 for r in t for x in t[r])
    ids0 = t["eco"][0][0].tolist()
    assert voc["<eos>"] not in ids0
    assert t["eco"][-1][0].tolist()[-1] == voc["<eos>"]


def test_gramatica_acepta_todo_lo_codificado(voc):
    pz = pieza_sintetica(3)
    for r in ("base", "eco", "fant"):
        for a, _, _ in secuencias([pz], voc, r, ctx=150, n_transp=3):
            decodificar(a.tolist(), voc, eco=r != "base", estricto=True, fantasma=r == "fant")


def test_presets_en_rango():
    for p, lo, hi in (("fuga-s", 150e6, 160e6), ("fuga-m", 370e6, 400e6)):
        with torch.device("meta"):
            m = Transformer(config(p, vocab=700))
        assert lo <= contar_parametros(m) <= hi


def test_kv_cache_y_eje_compas():
    torch.manual_seed(0)
    m = Transformer(config("nano", vocab=100)).eval()
    ids = torch.randint(0, 100, (1, 30))
    tok = torch.arange(30)[None]
    pos_a = torch.stack([tok, tok // 5], 1)
    pos_b = torch.stack([tok, tok // 3], 1)
    with torch.no_grad():
        ya, yb = m(ids, pos_a), m(ids, pos_b)
        assert (ya - yb).abs().max() > 1e-4          # el eje compás influye
        caches = [{} for _ in m.bloques]
        paso = [m(ids[:, t:t + 1], pos_a[:, :, t:t + 1], caches)[0, -1] for t in range(30)]
        assert torch.allclose(torch.stack(paso), ya[0], atol=1e-4)
    m0 = Transformer(config("nano", vocab=100, kv_compas=0)).eval()
    with torch.no_grad():
        assert torch.allclose(m0(ids, pos_a), m0(ids, pos_b))


@pytest.mark.parametrize("eco", [False, True])
def test_generacion_siempre_decodificable(voc, eco):
    torch.manual_seed(0)
    m = Transformer(config("nano", vocab=len(voc))).eval()
    for s in range(3):
        dec = generar(m, voc, eco, max_tokens=300, max_compases=6, semilla=s, sesgo_eco=2.0 if eco else 0.0)
        notas = dec.notas()
        assert all(0 <= p <= 127 and d > 0 for _, d, p, _ in notas)


def test_colar_y_metricas(voc):
    pz = pieza_sintetica(4)
    seqs = secuencias([pz], voc, "fant", ctx=256, relativo=True)
    ids, pos, mm = colar(seqs, [0])
    assert ids.shape == mm.shape and pos.shape == (1, 2, ids.shape[1])
    assert 0 < (~mm).sum() < mm.numel()  # hay fantasmas sin pérdida
    ini = pz["compases"]
    pf = perfil(pz["notas"], ini, 0, "major")
    assert 0.5 < pf["repeticion"] <= 1.0
    assert repeticion(segmentos([], ini)) == 0.0


@pytest.mark.parametrize("eco", [False, True])
def test_marco_relativo_sin_perdida(voc, eco):
    for t in (-5, 0, 3, 6):
        pz = pieza_sintetica(5)
        pz = {**pz, "tonica": (2 + t) % 12}
        c, _ = codificar(pz, voc, eco=eco, relativo=True)
        d = decodificar(c.ids, voc, eco=eco, relativo=True)
        assert d.tonica_abs == pz["tonica"]
        orig = sorted((on, pi) for on, _, pi, _ in pz["notas"])
        assert sorted((on, pi) for on, _, pi, _ in d.notas()) == orig


def test_marco_hace_invariante_la_transposicion(voc):
    pz = pieza_sintetica(6)
    c0, _ = codificar(pz, voc, t=0, relativo=True)
    c3, _ = codificar(pz, voc, t=3, relativo=True)
    dif = [i for i, (a, b) in enumerate(zip(c0.ids, c3.ids, strict=True)) if a != b]
    assert dif == [2]  # solo cambia el token de marco (KEY)


def test_fantasma_materializa_y_no_cuesta(voc):
    pz = pieza_sintetica(7)
    ce, _ = codificar(pz, voc, eco=True)
    cf, _ = codificar(pz, voc, eco=True, fantasma=True)
    assert len(cf.ids) == len(cf.perdida) == len(cf.compas)
    # los tokens con pérdida del fantasma coinciden exactamente con la secuencia ECO
    assert [i for i, p in zip(cf.ids, cf.perdida, strict=True) if p] == ce.ids
    d = decodificar(cf.ids, voc, eco=True, fantasma=True)
    assert contenido(d.segs) == contenido(segmentar(pz)[1])


def test_generacion_con_fantasma(voc):
    torch.manual_seed(1)
    m = Transformer(config("nano", vocab=len(voc))).eval()
    dec = generar(m, voc, True, max_tokens=400, max_compases=6, semilla=0, sesgo_eco=4.0, fantasma=True)
    assert dec.notas()


def test_misma_altura_y_ataque_con_duraciones_distintas(voc):
    pz = {"id": "dup#0", "fuente": "sint", "tpq": 12, "compases": [0, 48], "tonica": 0, "modo": "major",
          "tempo": 90.0, "partes": 1, "notas": [(0, 12, 60, 0), (0, 24, 60, 0), (12, 12, 62, 0)]}
    for eco in (False, True):
        c, _ = codificar(pz, voc, eco=eco)
        decodificar(c.ids, voc, eco=eco, estricto=True)


def test_relativo_nunca_decodifica_fuera_de_rango(voc):
    from fuga.tokenizer import Decodificador
    for clave in ("KEY_3_major", "KEY_9_minor"):
        d = Decodificador(voc, eco=True, relativo=True)
        for s in ("<bos>", "STY_sint", clave, "TEMPO_5", "BAR", "PART_0", "POS_0"):
            d.paso(voc[s])
        alturas = [int(voc.itos[i][6:]) + d.desp for i in d.permitidos()]
        assert 0 <= min(alturas) and max(alturas) <= 127
        assert len(alturas) == 128 - abs(d.desp)


def test_tope_de_compases_cierra_completos(voc):
    torch.manual_seed(2)
    m = Transformer(config("nano", vocab=len(voc))).eval()
    for s in range(3):
        dec = generar(m, voc, True, max_tokens=600, max_compases=4, semilla=s)
        assert dec.bar + 1 <= 4


def test_anacrusa_ida_y_vuelta(voc):
    pz = {"id": "ana#0", "fuente": "sint", "tpq": 12, "compases": [12, 60, 108], "tonica": 0, "modo": "major",
          "tempo": 90.0, "partes": 1, "notas": [(0, 6, 60, 0), (6, 6, 62, 0), (12, 12, 64, 0), (60, 12, 65, 0)]}
    for eco in (False, True):
        c, _ = codificar(pz, voc, eco=eco)
        d = decodificar(c.ids, voc, eco=eco)
        assert sorted((on, pi) for on, _, pi, _ in d.notas()) == sorted((on, pi) for on, _, pi, _ in pz["notas"])
        assert d.fin


def test_estilo_fuera_del_vocabulario_falla_claro(voc):
    pz = {**pieza_sintetica(0), "fuente": "desconocido"}
    with pytest.raises(ValueError, match="fuera del vocabulario"):
        codificar(pz, voc)


def test_relativo_avisa_si_descarta_notas(voc):
    pz = {"id": "ext#0", "fuente": "sint", "tpq": 12, "compases": [0, 48], "tonica": 6, "modo": "major",
          "tempo": 90.0, "partes": 1, "notas": [(0, 12, 125, 0), (12, 12, 60, 0)]}
    with pytest.warns(UserWarning, match="fuera de 0..127"):
        _, segs = segmentar(pz, relativo=True)
    assert sum(len(s) for seg in segs for s in seg.values()) == 1


def test_checkpoint_de_activaciones_da_los_mismos_gradientes():
    torch.manual_seed(0)
    m1 = Transformer(config("nano", vocab=50))
    m2 = Transformer(config("nano", vocab=50, ckpt=True))
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 50, (2, 20))
    for m in (m1, m2):
        m.train()
        m(ids).float().logsumexp(-1).sum().backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
def test_cuda_coincide_con_cpu_y_genera(voc):
    torch.manual_seed(0)
    m = Transformer(config("nano", vocab=len(voc))).eval()
    ids = torch.randint(0, len(voc), (1, 24))
    with torch.no_grad():
        y_cpu = m(ids)
        mg = m.to("cuda")
        y_gpu = mg(ids.cuda()).cpu()
        assert torch.allclose(y_cpu, y_gpu, atol=2e-3, rtol=1e-3)
        caches = [{} for _ in mg.bloques]
        tok = torch.arange(24, device="cuda")
        pos = torch.stack([tok, tok // 4], 0)[None]
        paso = [mg(ids[:, t:t + 1].cuda(), pos[:, :, t:t + 1], caches)[0, -1] for t in range(24)]
        assert torch.allclose(torch.stack(paso), mg(ids.cuda(), pos)[0], atol=2e-3)
    dec = generar(mg, voc, True, max_tokens=150, max_compases=3, semilla=0, sesgo_eco=2.0)
    assert dec.bar + 1 <= 3 and all(0 <= p <= 127 for _, _, p, _ in dec.notas())


def test_config_guardable_no_lleva_ckpt_y_carga_en_ambas_direcciones():
    from fuga.model import ConfigModelo, config_guardable
    c = config("nano", vocab=10, ckpt=True)
    d = config_guardable(c)
    assert "ckpt" not in d and d["d"] == 192
    assert ConfigModelo(**d).ckpt is False          # el checkpoint no arrastra la bandera de entrenamiento


def test_checkpoint_con_dropout_misma_semilla_mismo_resultado():
    torch.manual_seed(0)
    m1 = Transformer(config("nano", vocab=50, dropout=0.1))
    m2 = Transformer(config("nano", vocab=50, dropout=0.1, ckpt=True))
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 50, (2, 20))
    salidas = []
    for m in (m1, m2):
        m.train()
        torch.manual_seed(7)
        y = m(ids)
        y.float().logsumexp(-1).sum().backward()
        salidas.append(y.detach())
    assert torch.allclose(salidas[0], salidas[1], atol=1e-5)
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):      # la recomputación repite la máscara
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4)


def test_fuente_y_nombre_con_rutas_de_windows_y_posix():
    import importlib.util
    import pathlib
    ruta = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "extraer_corpus.py"
    spec = importlib.util.spec_from_file_location("extraer_corpus", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._fuente_y_nombre(r"C:\Python\Lib\site-packages\music21\corpus\bach\bwv1.6.mxl") == ("bach", "bwv1.6.mxl")
    assert mod._fuente_y_nombre("/usr/lib/python3/site-packages/music21/corpus/essenFolksong/altdeu10.abc") == \
        ("essenFolksong", "altdeu10.abc")
    with pytest.raises(ValueError):
        mod._fuente_y_nombre("/tmp/sin_corpus/a.mxl")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
def test_bits_por_nota_en_cuda_coincide_con_cpu(voc):
    from fuga.train import bits_por_nota
    torch.manual_seed(0)
    m = Transformer(config("nano", vocab=len(voc))).eval()
    trozos = trozos_eval([pieza_sintetica(8, compases=20)], voc, ctx=256)["eco"]
    cpu = bits_por_nota(m, trozos)
    gpu = bits_por_nota(m.to("cuda"), trozos)
    assert abs(cpu["bits_por_nota"] - gpu["bits_por_nota"]) < 1e-3 and cpu["notas"] == gpu["notas"]
