"""
Ângulo de fase (Sol – lua – JUICE, vértice na lua) × tempo
para as 4 luas internas: Metis, Adrastea, Amalthea e Thebe.

Observação A: α <= 70°; campanhas em α ≈ 10° (fase baixa) e α ≈ 50° (fase média).
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
LIMITE_FASE = 70         # α <= 70°
CAMPANHAS = [10, 50]     # fase baixa e fase média (faixa de ±10° em volta)
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

dados = {}   # nome -> (datas, fases)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, fases = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # phaseq: ângulo Sol–lua–JUICE (vértice na lua), em radianos
        fase = [spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO) for et in ets]
        datas.extend(spice.et2datetime(ets))
        fases.extend(np.degrees(fase))
        datas.append(datas[-1])          # NaN separa lacunas no gráfico
        fases.append(np.nan)

    fases = np.array(fases)
    dados[nome] = (datas, fases)
    ok = np.mean(fases[np.isfinite(fases)] <= LIMITE_FASE) * 100
    print(f"{nome}: α de {np.nanmin(fases):.1f}° a {np.nanmax(fases):.1f}° | "
          f"{ok:.0f}% do tempo com α <= {LIMITE_FASE}°")

spice.kclear()

# ---- ângulo de fase × tempo ----
n = len(dados)
fig, eixos = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, fases)) in zip(eixos[:, 0], dados.items()):
    for c in CAMPANHAS:
        eixo.axhspan(c - 10, c + 10, color="tab:blue", alpha=0.12)
    eixo.axhline(LIMITE_FASE, color="red", ls="--", lw=0.8)
    eixo.plot(datas, fases, lw=0.5, color="tab:purple")
    eixo.set_ylim(0, 180)
    eixo.set_title(f"{nome.capitalize()} ({LUAS[nome]})")
    eixo.set_ylabel("fase α [°]")
    eixo.grid(True, alpha=0.3)
    eixo.margins(x=0)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.suptitle(f"Ângulo de fase ({META_KERNEL}, {CORRECAO})", fontsize=10)
fig.tight_layout()
fig.savefig("fase_tempo.png", dpi=150)
print("gráfico salvo: fase_tempo.png")
