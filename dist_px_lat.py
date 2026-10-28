"""
Uma imagem por lua interna (Metis, Adrastea, Amalthea, Thebe), com 3 painéis
contra o tempo:
  1) distância JUICE–lua [km]
  2) pixels que a lua ocupa na JANUS = D / (d × IFOV)
  3) latitude sub-S/C [°]
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
INICIO = "2032-06-01"    # None = começo da trajetória da JUICE
FIM = "2032-09-01"       # None = fim da trajetória

PASSO_S = 1800           # 1 ponto a cada 30 min
CORRECAO = "LT+S"

JUICE_ID = -28
JANUS_ID = -28200
LUAS = {                 # nome SPICE : (código NAIF, pixels mínimos da obs. A)
    "METIS": (516, 3),
    "ADRASTEA": (515, 2),
    "AMALTHEA": (505, 10),
    "THEBE": (514, 6),
}
FAIXA_EQ = 10            # |b| <= 10° = "quase equatorial" (a confirmar com o TD)
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
IFOV = spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]     # rad/pixel, do IK
print(f"IFOV da JANUS: {IFOV * 1e6:.2f} µrad")

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

for nome, (id_lua, px_min) in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    _, raios = spice.bodvrd(nome, "RADII", 3)
    D = 2 * np.mean(raios)                       # diâmetro médio (km)

    datas, dists, lats = [], [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # vetor JUICE -> lua no referencial fixo da lua
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        v = -np.asarray(pos)                     # lua -> JUICE
        d = np.linalg.norm(v, axis=1)
        datas.extend(spice.et2datetime(ets))
        dists.extend(d)
        lats.extend(np.degrees(np.arcsin(v[:, 2] / d)))
        datas.append(datas[-1])                  # NaN separa lacunas no gráfico
        dists.append(np.nan)
        lats.append(np.nan)

    dists = np.array(dists)
    lats = np.array(lats)
    pixels = D / (dists * IFOV)

    fig, (ax_d, ax_p, ax_l) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_ylabel("distância [km]")
    ax_d.set_title(f"{nome.capitalize()} ({id_lua}) – D ≈ {D:.0f} km")
    ax_d.grid(True, which="both", alpha=0.3)

    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    ax_p.axhline(px_min, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.grid(True, alpha=0.3)

    ax_l.plot(datas, lats, lw=0.6, color="tab:green")
    ax_l.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15)
    ax_l.set_ylabel("lat. sub-S/C [°]")
    ax_l.set_xlabel("data (UTC)")
    ax_l.grid(True, alpha=0.3)

    fig.tight_layout()
    arq = f"dist_px_lat_{nome.lower()}.png"
    fig.savefig(arq, dpi=150)
    plt.close(fig)
    print(f"{nome}: gráfico salvo em {arq}")

spice.kclear()
