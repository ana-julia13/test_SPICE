"""
Uma imagem por lua interna (Metis, Adrastea, Amalthea, Thebe), com 5 painéis
contra o tempo:
  1) distância JUICE–lua [km]
  2) pixels que a lua ocupa na JANUS = D / (d × IFOV)
  3) latitude sub-S/C [°]
  4) longitude sub-S/C [°], leste positivo, 0–360
  5) ângulo de fase α [°] (Sol – lua – JUICE)

Versão para ZOOM (poucos dias em torno de um perijove): passo curto e
eixo x com dia e hora.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import spiceypy as spice

PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL = "juice_plan.tm"
INICIO = "2032-03-10 00:00"   # janela do zoom (UTC), ex.: ~4 dias
FIM = "2032-03-14 00:00"

PASSO_S = 60             # 1 ponto por minuto (bem menor que os 10 min de uma janela)
CORRECAO = "LT+S"        # correção de tempo-luz/aberração

JUICE_ID = -28
JANUS_ID = -28200
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
NIVEIS_PX = [1, 10]      # linhas de referência no painel de pixels
FAIXA_EQ = 10            # |b| <= 10° = "quase equatorial" (a confirmar com o TD)
LIMITE_B = 20            # |b| >= 20° = tipo B
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
    arquivos = []
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        objetos = spice.spkobj(arquivo)
        if id_corpo in [objetos[k] for k in range(spice.card(objetos))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
            arquivos.append(arquivo)
    return cobertura, arquivos


carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
IFOV = spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]     # rad/pixel, do IK
print(f"IFOV da JANUS (IK): {IFOV * 1e6:.2f} µrad")

# período: cobertura da JUICE, cortada em INICIO/FIM
cob_juice, arq_juice = cobertura_spk(JUICE_ID)
if not arq_juice:
    sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
if INICIO or FIM:
    n = spice.wncard(cob_juice)
    et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
    et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
    janela = spice.cell_double(2)
    spice.wninsd(et0, et1, janela)
    cob_juice = spice.wnintd(cob_juice, janela)

for nome, id_lua in LUAS.items():
    cob_lua, _ = cobertura_spk(id_lua)
    comum = spice.wnintd(cob_juice, cob_lua)
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    _, raios = spice.bodvrd(nome, "RADII", 3)
    D = 2 * np.mean(raios)                       # diâmetro médio (km)

    datas, dists, lats, lons, fases = [], [], [], [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # vetor JUICE -> lua no referencial fixo da lua; o "-" vira lua -> JUICE
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        v = -np.asarray(pos)
        d = np.linalg.norm(v, axis=1)
        datas.extend(spice.et2datetime(ets))
        dists.extend(d)
        lats.extend(np.degrees(np.arcsin(v[:, 2] / d)))
        lons.extend(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360)   # leste, 0–360
        # phaseq: ângulo Sol–lua–JUICE (vértice na lua), em radianos
        fases.extend(np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO)
                                 for et in ets]))
        datas.append(datas[-1])                  # NaN separa lacunas no gráfico
        for lista in (dists, lats, lons, fases):
            lista.append(np.nan)

    dists = np.array(dists)
    lats = np.array(lats)
    lons = np.array(lons)
    fases = np.array(fases)
    pixels = D / (dists * IFOV)

    i = np.nanargmin(dists)
    print(f"{nome}: mín. {dists[i]:,.0f} km = {pixels[i]:.1f} px, "
          f"b = {lats[i]:.1f}°, λ = {lons[i]:.0f}°E, α = {fases[i]:.0f}° "
          f"em {datas[i]:%Y-%m-%d %H:%M} UTC")

    fig, (ax_d, ax_p, ax_l, ax_lon, ax_f) = plt.subplots(5, 1, figsize=(12, 13),
                                                         sharex=True)

    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_title(f"{nome.capitalize()} ({id_lua}) – D ≈ {D:.0f} km")
    ax_d.set_ylabel("distância [km] (log)")
    ax_d.grid(True, which="both", axis="y", alpha=0.3)   # linhas menores só no y (log)

    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        ax_p.axhline(p, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.set_ylim(bottom=0)
    ax_p.grid(True, alpha=0.3)

    ax_l.plot(datas, lats, lw=0.5, color="tab:green")
    ax_l.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15)
    for y in (-LIMITE_B, LIMITE_B):
        ax_l.axhline(y, color="gray", ls=":", lw=0.8)
    ax_l.set_ylabel("lat. sub-S/C [°]")
    ax_l.grid(True, alpha=0.3)

    # pontos em vez de linha: a longitude dá a volta 360 -> 0 e a linha riscaria o painel
    ax_lon.plot(datas, lons, ".", ms=1.5, color="tab:brown")
    ax_lon.set_ylim(0, 360)
    ax_lon.set_yticks(range(0, 361, 90))
    ax_lon.set_ylabel("lon. sub-S/C [°E]")
    ax_lon.grid(True, alpha=0.3)

    for c in CAMPANHAS:
        ax_f.axhspan(c - 10, c + 10, color="tab:blue", alpha=0.12)
    ax_f.axhline(LIMITE_FASE, color="red", ls="--", lw=0.8)
    ax_f.plot(datas, fases, lw=0.6, color="tab:purple")
    ax_f.set_ylim(0, 180)
    ax_f.set_ylabel("fase α [°]")
    ax_f.set_xlabel("data (UTC)")
    ax_f.grid(True, alpha=0.3)
    ax_f.margins(x=0)                            # sem espaço antes/depois dos dados
    ax_f.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%Hh"))
    ax_f.xaxis.set_minor_locator(mdates.HourLocator())      # tracinho a cada hora
    for ax in (ax_d, ax_p, ax_l, ax_lon, ax_f):
        ax.grid(True, which="major", axis="x", alpha=0.6)

    fig.tight_layout()
    arq = f"dist_px_lat_zoom_{nome.lower()}.png"
    fig.savefig(arq, dpi=150)
    plt.close(fig)
    print(f"  gráfico salvo: {arq}")

spice.kclear()
