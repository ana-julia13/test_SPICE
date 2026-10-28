"""
Latitude sub-S/C (ponto da lua logo "embaixo" da JUICE) × tempo
para as 4 luas internas: Metis, Adrastea, Amalthea e Thebe.

Plot P4 da observação A: faixa "quase equatorial" (|b| <= 10°, a confirmar)
e linhas em ±20° (começo do tipo B).

latitude sub-S/C = latitude planetocêntrica da JUICE vista do centro da lua,
no referencial fixo da lua (IAU_<LUA>, precisa do PCK).
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
INICIO = "2031-03-01"            # None = começo da trajetória da JUICE; ou "2031-07-21"
FIM = "2034-05-01"               # None = fim da trajetória; ou "2034-12-01"

PASSO_S = 1800           # 1 ponto a cada 30 min (zoom: use <= 600)
CORRECAO = "LT+S"        # correção de tempo-luz/aberração

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
FAIXA_EQ = 10            # |b| <= 10° = "quase equatorial" (a confirmar com o TD)
LIMITE_B = 20            # |b| >= 20° = tipo B
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

dados = {}   # nome -> (datas, latitudes)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, lats = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # vetor JUICE -> lua no referencial fixo da lua; o sinal "-" vira lua -> JUICE
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        v = -np.asarray(pos)
        lat = np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))
        datas.extend(spice.et2datetime(ets))
        lats.extend(lat)
        datas.append(datas[-1])          # NaN separa lacunas no gráfico
        lats.append(np.nan)

    lats = np.array(lats)
    dados[nome] = (datas, lats)
    perto_eq = np.mean(np.abs(lats[np.isfinite(lats)]) <= FAIXA_EQ) * 100
    print(f"{nome}: b de {np.nanmin(lats):.1f}° a {np.nanmax(lats):.1f}° | "
          f"{perto_eq:.0f}% do tempo com |b| <= {FAIXA_EQ}°")

spice.kclear()

# ---- latitude sub-S/C × tempo ----
n = len(dados)
fig, eixos = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, lats)) in zip(eixos[:, 0], dados.items()):
    eixo.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15,
                 label=f"|b| ≤ {FAIXA_EQ}° (quase equatorial)")
    for y in (-LIMITE_B, LIMITE_B):
        eixo.axhline(y, color="gray", ls=":", lw=0.8)
    eixo.plot(datas, lats, lw=0.5)
    eixo.set_title(f"{nome.capitalize()} ({LUAS[nome]})")
    eixo.set_ylabel("lat. sub-S/C [°]")
    eixo.grid(True, alpha=0.3)
eixos[0, 0].legend(loc="upper right", fontsize=8)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.suptitle(f"Latitude sub-S/C ({META_KERNEL}, {CORRECAO})", fontsize=10)
fig.tight_layout()
fig.savefig("latitude_subsc_tempo.png", dpi=150)
print("gráfico salvo: latitude_subsc_tempo.png")
