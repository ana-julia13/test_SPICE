"""
Created on Mon Oct 12 16:03:15 2026

@author: mend_aj

Pixels que cada lua interna ocupa na JANUS ao longo do tempo.

Usa os mesmos kernels do distancia_juice_luas.py (meta-kernel):
distâncias dos SPKs, raios das luas do PCK e IFOV da JANUS do IK.
pixels = D / (d × IFOV)
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
INICIO = "2031-03-01"    # início do período de interesse (UTC)
FIM = "2034-05-01"       # fim do período de interesse (UTC)

PASSO_GRAFICO_S = 1800   # 1 ponto a cada 30 min no gráfico
PASSO_BUSCA_S = 600      # passo da busca da distância mínima (10 min)
BLOCO_BUSCA_S = 30 * 86400  # a busca do mínimo é feita em blocos de 30 dias

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
JANUS_ID = -28200    # nome do instrumento no IK/FK
NIVEIS_PX = [1, 10] # linhas de referência nos gráficos de pixels
# =========================================================================

def carrega_meta_kernel(pasta_kernels, nome_mk):
    pasta_mk = os.path.join(pasta_kernels, "mk")
    caminho = os.path.join(pasta_mk, nome_mk)
    if not os.path.isfile(caminho):
        print(f"ERRO: meta-kernel não encontrado: {caminho}")
        if os.path.isdir(pasta_mk):
            print("Meta-kernels disponíveis em", pasta_mk)
            for f in sorted(os.listdir(pasta_mk)):
                if f.endswith(".tm"):
                    print("   ", f)
        sys.exit(1)

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

    spice.kclear()
    for arquivo in arquivos:
        spice.furnsh(arquivo)
    print(f"Meta-kernel carregado: {caminho}")
    print(f"  ({len(arquivos)} kernels no total)")


def cobertura_spk(id_corpo):
    cobertura = spice.cell_double(200000)
    arquivos = []
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        objetos = spice.spkobj(arquivo)
        if id_corpo in [objetos[k] for k in range(spice.card(objetos))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
            arquivos.append(arquivo)
    return cobertura, arquivos


def imprime_janela(titulo, janela, max_linhas=10):
    n = spice.wncard(janela)
    print(f"  {titulo}: {n} intervalo(s)")
    for k in range(min(n, max_linhas)):
        a, b = spice.wnfetd(janela, k)
        print(f"    {spice.et2utc(a, 'C', 0)}  ->  {spice.et2utc(b, 'C', 0)}")
    if n > max_linhas:
        print(f"    ... (+{n - max_linhas} intervalos)")

def ifov_janus():
    """IFOV (rad/pixel) lido direto do IK da JANUS."""
    return spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]

carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
IFOV = ifov_janus()
print(f"{spice.ktotal('ALL')} KERNELS CARREGADOS ; IFOV da JANUS (IK): {IFOV * 1e6:.2f} µrad" )

#periodo utilizado:
cob_juice, _ = cobertura_spk(JUICE_ID)
if INICIO or FIM:
    n = spice.wncard(cob_juice)
    et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
    et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
    janela = spice.cell_double(2)
    spice.wninsd(et0, et1, janela)
    cob_juice = spice.wnintd(cob_juice, janela)

dados = {}   # nome -> (datas, pixels, diâmetro)
for nome, id_lua in LUAS.items():
    cob_lua, _ = cobertura_spk(id_lua)
    comum = spice.wnintd(cob_juice, cob_lua)
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    _, raios = spice.bodvrd(nome, "RADII", 3)
    D = 2 * np.mean(raios)                       # diâmetro médio (km)

    datas, pixels = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_BUSCA_S)
        pos, _ = spice.spkpos(nome, ets, "J2000", "NONE", "JUICE")
        datas.extend(spice.et2datetime(ets))
        pixels.extend(D / (np.linalg.norm(pos, axis=1) * IFOV))
        datas.append(datas[-1])                   # NaN separa lacunas no gráfico
        pixels.append(np.nan)

    pixels = np.array(pixels)
    dados[nome] = (datas, pixels, D)

    i = np.nanargmax(pixels)
    print(f"{nome}: D = {D:.1f} km | máx. {pixels[i]:.1f} px "
          f"em {datas[i]:%Y-%m-%d %H:%M} UTC")

spice.kclear()

n = len(dados)

# ---- pixels × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, pixels, D)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        eixo.axhline(p, color="gray", ls="--", lw=0.8)
    eixo.set_title(f"{nome} (D ≈ {D:.0f} km)")
    eixo.set_ylabel("pixels através da lua")
    eixo.grid(True, which="both", alpha=0.3)
    eixo.margins(x=0)                    # sem espaço antes/depois dos dados
    eixo.set_ylim(bottom=0)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("pixels_tempo.png", dpi=150)
