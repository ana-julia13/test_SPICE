"""
Longitude sub-S/C (ponto da lua logo "embaixo" da JUICE) × tempo
para as 4 luas internas: Metis, Adrastea, Amalthea e Thebe.

longitude sub-S/C = longitude planetocêntrica da JUICE vista do centro da lua,
no referencial fixo da lua (IAU_<LUA>, precisa do PCK).
Convenção: LESTE positivo, 0–360° (0° ≈ ponto sub-Júpiter, convenção IAU).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice

PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL = "juice_plan.tm"
INICIO = "2031-03-01"    # None = começo da trajetória da JUICE
FIM = "2034-05-01"       # None = fim da trajetória

PASSO_S = 1800           # 1 ponto a cada 30 min (zoom: use <= 600)
CORRECAO = "LT+S"        # correção de tempo-luz/aberração

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
# =========================================================================


def carrega_meta_kernel(pasta_kernels, nome_mk):
    pasta_mk = os.path.join(pasta_kernels, "mk")
    caminho = os.path.join(pasta_mk, nome_mk)
    if not os.path.isfile(caminho):
        sys.exit(f"ERRO: meta-kernel não encontrado: {caminho}")
    pasta_original = os.getcwd()
    os.chdir(pasta_mk)
    try:
        spice.furnsh(nome_mk)
        arquivos = []
        for i in range(spice.ktotal("ALL")):
            arquivo, tipo, _, _ = spice.kdata(i, "ALL")
            if tipo != "META":
                arquivos.append(os.path.abspath(arquivo))
    finally:
        os.chdir(pasta_original)
    # recarrega com caminho absoluto (assim spkcov acha os arquivos)
    spice.kclear()
    for arquivo in arquivos:
        spice.furnsh(arquivo)
    print(f"Meta-kernel carregado: {caminho} ({len(arquivos)} kernels)")


def cobertura_spk(id_corpo):
    cobertura = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        objetos = spice.spkobj(arquivo)
        if id_corpo in [objetos[k] for k in range(spice.card(objetos))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)

# período: cobertura da JUICE, cortada em INICIO/FIM
cob_juice = cobertura_spk(JUICE_ID)
if spice.wncard(cob_juice) == 0:
    sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
if INICIO or FIM:
    n = spice.wncard(cob_juice)
    et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
    et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
    janela = spice.cell_double(2)
    spice.wninsd(et0, et1, janela)
    cob_juice = spice.wnintd(cob_juice, janela)

dados = {}   # nome -> (datas, longitudes)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, lons = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # vetor JUICE -> lua no referencial fixo da lua; o sinal "-" vira lua -> JUICE
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        v = -np.asarray(pos)
        lon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360   # leste, 0–360
        datas.extend(spice.et2datetime(ets))
        lons.extend(lon)

    dados[nome] = (datas, np.array(lons))
    print(f"{nome}: {len(lons)} pontos")

spice.kclear()

# ---- longitude sub-S/C × tempo ----
# pontos em vez de linha: a longitude dá a volta 360 -> 0 e a linha riscaria o painel
n = len(dados)
fig, eixos = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, lons)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, lons, ".", ms=1)
    eixo.set_ylim(0, 360)
    eixo.set_yticks(range(0, 361, 90))
    eixo.set_title(f"{nome.capitalize()} ({LUAS[nome]})")
    eixo.set_ylabel("lon. sub-S/C [°E]")
    eixo.grid(True, alpha=0.3)
    eixo.margins(x=0)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.suptitle(f"Longitude sub-S/C, leste positivo ({META_KERNEL}, {CORRECAO})", fontsize=10)
fig.tight_layout()
fig.savefig("longitude_subsc_tempo.png", dpi=150)
print("gráfico salvo: longitude_subsc_tempo.png")
