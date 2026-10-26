"""
Variação no tempo dos 6 elementos orbitais (a, e, i, Ω, ω, M) das luas de
Júpiter, de Metis até Callisto, calculada com SPICE (efemérides oficiais dos
kernels), para comparar com a integração N-corpos.

1. Carrega os kernels (meta-kernel da JUICE ou lista de arquivos genéricos)
2. Confere se cada lua tem cobertura SPK no período pedido
3. Lê o estado (posição, velocidade) jovicêntrico de cada lua com spkezr
4. Gira para o frame escolhido (equador de Júpiter ou J2000)
5. Converte estado -> elementos osculadores com oscltx
6. Salva CSV + gráfico por lua e um CSV com os estados iniciais
   (para usar como condição inicial do N-corpos)

Kernels necessários (todos em https://naif.jpl.nasa.gov/pub/naif/generic_kernels/):
  lsk/naif0012.tls                 tempo (UTC <-> TDB)
  pck/pck00011.tpc                 orientação do polo de Júpiter (IAU_JUPITER)
  pck/gm_de440.tpc                 GM de Júpiter e das galileanas
  spk/satellites/jup365.bsp        Io, Europa, Ganimedes, Callisto, Amalthea,
                                   Thebe, Adrastea, Metis e Júpiter (599)
O meta-kernel da JUICE já carrega equivalentes desses arquivos.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # só salva PNG, sem abrir janela (não precisa de Qt)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import spiceypy as spice

# ======================= CONFIGURAÇÃO (edite aqui) =======================
# Pode ser um meta-kernel (.tm) ou uma lista de arquivos de kernel.
KERNELS = [r"/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"]
# KERNELS = [
#     r"/caminho/kernels/naif0012.tls",
#     r"/caminho/kernels/pck00011.tpc",
#     r"/caminho/kernels/gm_de440.tpc",
#     r"/caminho/kernels/jup365.bsp",
# ]
INICIO = "2030-07-01"    # início do período (UTC)
FIM = "2040-07-01"       # fim do período (UTC)
PASSO_S = 600            # passo em segundos (Metis tem período de ~7 h)

# "EQUADOR_JUPITER": frame inercial alinhado ao equador de Júpiter na época
#                    INICIO (o mais usual para N-corpos jovicêntrico)
# "J2000":           equador terrestre J2000 (ICRF)
FRAME = "EQUADOR_JUPITER"

# Soma o GM da lua ao de Júpiter (problema de 2 corpos Júpiter+lua). Deixe
# igual ao que o seu N-corpos usa para converter estado -> elementos.
SOMAR_GM_LUA = True

PASTA_SAIDA = "saida_elementos"
LUAS = {                 # nome SPICE : código NAIF (de dentro para fora)
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
    "IO": 501,
    "EUROPA": 502,
    "GANYMEDE": 503,
    "CALLISTO": 504,
}
# =========================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]


PASTA_DE_CARGA = {}  # nome usado no furnsh -> pasta de onde foi carregado


def carrega_kernels(kernels):
    """furnsh de cada kernel; meta-kernels são carregados da própria pasta,
    pois os caminhos dentro deles (PATH_VALUES = '..') são relativos."""
    pasta_atual = os.getcwd()
    for k in kernels:
        k = os.path.abspath(k)
        os.chdir(os.path.dirname(k))
        try:
            spice.furnsh(os.path.basename(k))
        finally:
            os.chdir(pasta_atual)
        PASTA_DE_CARGA[os.path.basename(k)] = os.path.dirname(k)


def caminho_absoluto(arquivo, origem):
    """Caminho real de um kernel listado por kdata (que pode ser relativo à
    pasta do meta-kernel que o carregou)."""
    pasta = PASTA_DE_CARGA.get(origem or arquivo, os.getcwd())
    return os.path.join(pasta, arquivo)


def cobertura_spk(id_corpo):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(20000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, origem, _ = spice.kdata(i, "SPK")
        arquivo = caminho_absoluto(arquivo, origem)
        ids = spice.spkobj(arquivo)
        if id_corpo in [ids[j] for j in range(spice.card(ids))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


def matriz_equador_jupiter(et):
    """Matriz J2000 -> frame inercial com z no polo de Júpiter e x no nó
    ascendente do equador de Júpiter sobre o equador J2000, na época 'et'."""
    polo = spice.pxform("IAU_JUPITER", "J2000", et)[:, 2].copy()  # contíguo p/ spiceypy
    nodo = spice.ucrss([0.0, 0.0, 1.0], polo)
    y = spice.ucrss(polo, nodo)
    return np.array([nodo, y, polo])


def gm_corpo(nome):
    """GM (km³/s²) do kernel de constantes, ou 0 se não houver."""
    codigo = spice.bods2c(nome)
    if not spice.bodfnd(codigo, "GM"):
        return 0.0
    return spice.bodvrd(nome, "GM", 1)[1][0]


def formata_eixo(eixo):
    """Eixo x começando e terminando nos dados e eixo y sem notação
    científica (nem o '1e-7+4.217e5' em cima do eixo)."""
    eixo.margins(x=0)
    if eixo.get_yscale() == "log":
        eixo.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        eixo.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: np.format_float_positional(float(f"{v:.6g}"), trim="-")))
        eixo.yaxis.set_minor_formatter(NullFormatter())
    else:
        eixo.ticklabel_format(axis="y", style="plain", useOffset=False)


carrega_kernels(KERNELS)
os.makedirs(PASTA_SAIDA, exist_ok=True)

et0, et1 = spice.str2et(INICIO), spice.str2et(FIM)
janela = spice.cell_double(2)
spice.wninsd(et0, et1, janela)

if FRAME == "EQUADOR_JUPITER":
    R = matriz_equador_jupiter(et0)  # fixa na época inicial (frame inercial)
elif FRAME == "J2000":
    R = np.eye(3)
else:
    raise ValueError(f"FRAME desconhecido: {FRAME}")

GM_JUP = gm_corpo("JUPITER")
if GM_JUP == 0.0:
    raise RuntimeError("GM de Júpiter não encontrado: carregue gm_de440.tpc")
print(f"GM Júpiter = {GM_JUP:.6f} km³/s²   frame = {FRAME}")

estados_iniciais = []  # (nome, código, GM, x, y, z, vx, vy, vz)
dados = {}
for nome, id_lua in LUAS.items():
    periodo = spice.wnintd(cobertura_spk(id_lua), janela)
    if spice.wncard(periodo) == 0:
        print(f"{nome}: SEM cobertura SPK no período, pulando")
        continue
    if spice.wncard(periodo) > 1 or spice.wnfetd(periodo, 0) != (et0, et1):
        print(f"{nome}: cobertura parcial no período, usando só o que existe")

    gm_lua = gm_corpo(nome)
    mu = GM_JUP + (gm_lua if SOMAR_GM_LUA else 0.0)

    ets, elems = [], []
    for k in range(spice.wncard(periodo)):
        a, b = spice.wnfetd(periodo, k)
        ets_k = np.arange(a, b, PASSO_S)
        estados, _ = spice.spkezr(str(id_lua), ets_k, "J2000", "NONE", "JUPITER")
        for et, s in zip(ets_k, np.asarray(estados)):
            s_rot = np.concatenate([R @ s[:3], R @ s[3:]])
            elems.append(spice.oscltx(s_rot, et, mu))
        ets.extend(ets_k)
        if k == 0:
            s0 = np.asarray(estados)[0]
            estados_iniciais.append(
                (nome, id_lua, gm_lua, *(R @ s0[:3]), *(R @ s0[3:])))

    ets = np.array(ets)
    elems = np.array(elems)
    # oscltx: [rp, ecc, inc, lnode, argp, m0, t0, mu, nu, a, tau, ...]
    tabela = np.column_stack([
        elems[:, 9],                            # a  [km]
        elems[:, 1],                            # e
        np.degrees(elems[:, 2]),                # i  [°]
        np.degrees(elems[:, 3]) % 360.0,        # Ω  [°]
        np.degrees(elems[:, 4]) % 360.0,        # ω  [°]
        np.degrees(elems[:, 5]) % 360.0,        # M  [°]
    ])
    dados[nome] = (ets, tabela)

    datas_utc = [spice.et2utc(et, "ISOC", 3) for et in ets]
    with open(os.path.join(PASTA_SAIDA, f"elementos_{nome.lower()}.csv"), "w") as f:
        f.write("data_utc,et_tdb_s,a_km,e,i_deg,Omega_deg,omega_deg,M_deg\n")
        for d, et, linha in zip(datas_utc, ets, tabela):
            f.write(f"{d},{et:.3f}," + ",".join(f"{v:.10g}" for v in linha) + "\n")

    medias = tabela[:, :3].mean(axis=0)
    print(f"{nome:9s} GM={gm_lua:10.4f}  <a>={medias[0]:12.1f} km  "
          f"<e>={medias[1]:.5f}  <i>={medias[2]:.4f}°")

    datas = spice.et2datetime(ets)
    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        if j >= 3:  # ângulos 0–360: pontos, para não ligar os saltos 360 -> 0
            eixo.plot(datas, tabela[:, j], ".", ms=0.8)
            eixo.set_ylim(0, 360)
        else:
            eixo.plot(datas, tabela[:, j], lw=0.6)
        eixo.set_ylabel(rot)
        eixo.grid(alpha=0.3)
        formata_eixo(eixo)
    eixos[0].set_title(f"{nome} – elementos osculadores jovicêntricos (SPICE, {FRAME})")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_SAIDA, f"elementos_{nome.lower()}.png"), dpi=150)
    plt.close(fig)

# Estados iniciais (km, km/s) no mesmo frame, para alimentar o N-corpos
with open(os.path.join(PASTA_SAIDA, "estados_iniciais.csv"), "w") as f:
    f.write(f"# epoca_utc={spice.et2utc(et0, 'ISOC', 3)} et_tdb_s={et0:.3f} "
            f"frame={FRAME} centro=JUPITER GM_JUPITER={GM_JUP:.10g}\n")
    f.write("nome,naif_id,GM_km3_s2,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n")
    for linha in estados_iniciais:
        f.write(",".join(str(v) if isinstance(v, (str, int)) else f"{v:.15g}"
                         for v in linha) + "\n")

# Resumo: a, e, i de todas as luas juntas (a normalizado pela média)
fig, eixos = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for nome, (ets, tabela) in dados.items():
    datas = spice.et2datetime(ets)
    eixos[0].plot(datas, tabela[:, 0] / tabela[:, 0].mean(), lw=0.5, label=nome)
    eixos[1].plot(datas, tabela[:, 1], lw=0.5)
    eixos[2].plot(datas, tabela[:, 2], lw=0.5)
for eixo, rot in zip(eixos, ["a / <a>", "e", "i [°]"]):
    eixo.set_ylabel(rot)
    eixo.grid(alpha=0.3)
eixos[1].set_yscale("log")
for eixo in eixos:
    formata_eixo(eixo)
eixos[0].legend(ncol=4, fontsize=8)
eixos[-1].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA_SAIDA, "resumo_a_e_i.png"), dpi=150)
plt.close(fig)

print(f"\nArquivos salvos em {os.path.abspath(PASTA_SAIDA)}")
