"""
Janelas de observação: uma imagem por lua, de cima para baixo:
  1) distância JUICE–lua [km]
  2) pixels na JANUS = D / (d × IFOV)
  3) latitude sub-S/C [°]
  4) longitude sub-S/C [°], leste positivo, 0–360
  5) ângulo de fase α [°], com o limite em vermelho
  6) barras = o PROBLEMA acontece (ocultada, eclipse, perto de Júpiter, ...)
  7) JANELA = nenhum problema por >= DURACAO_MIN_S
A distância NÃO entra na janela (só aparece nos painéis 1 e 2).
As janelas também vão para um CSV.

Período: INICIO/FIM abaixo, ou na linha de comando:
    python janelas.py "2032-03-10 00:00" "2032-03-14 00:00"

Fase: o limite depende do hemisfério que a JUICE vê (sinal da latitude sub-S/C).
  70° sul / 110° norte = observação B (Parametrization v02, Matrix V03 B-1/B-2).
  Observação A é α <= 70° sempre -> coloque FASE_MAX_NORTE = 70.
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
INICIO = "2032-03-10 00:00"   # período analisado (UTC); a linha de comando tem prioridade
FIM = "2032-03-14 00:00"
if len(sys.argv) == 3:
    INICIO, FIM = sys.argv[1], sys.argv[2]

PASSO_S = 60             # 1 ponto por minuto
DURACAO_MIN_S = 600      # janela precisa durar >= 10 min
CORRECAO = "LT+S"        # spkpos e phaseq
CORRECAO_OCC = "LT"      # occult não aceita "+S"

JUICE_ID = -28
JANUS_ID = -28200
LUAS = {                 # nome SPICE : (código NAIF, pixels mínimos da obs. A)
    "METIS": (516, 3),
    "ADRASTEA": (515, 2),
    "AMALTHEA": (505, 10),
    "THEBE": (514, 6),
}
GALILEANAS = ["IO", "EUROPA", "GANYMEDE", "CALLISTO"]
FASE_MAX_SUL = 70        # α máximo quando a lat. sub-S/C é negativa (sul)
FASE_MAX_NORTE = 110     # α máximo quando a lat. sub-S/C é positiva (norte)
FAIXA_EQ = 10            # |b| <= 10° = "quase equatorial" (a confirmar com o TD)
LIMITE_PX = 10           # >= 10 px de qualquer limbo
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


def separacao_px(lua, R_lua, corpo, R_corpo, ets):
    """Pixels entre o limbo da lua e o limbo do corpo, vistos da JUICE."""
    p_lua, _ = spice.spkpos(lua, ets, "J2000", CORRECAO, "JUICE")
    p_corpo, _ = spice.spkpos(corpo, ets, "J2000", CORRECAO, "JUICE")
    p_lua, p_corpo = np.asarray(p_lua), np.asarray(p_corpo)
    d_lua = np.linalg.norm(p_lua, axis=1)
    d_corpo = np.linalg.norm(p_corpo, axis=1)
    cos = np.sum(p_lua * p_corpo, axis=1) / (d_lua * d_corpo)
    theta = np.arccos(np.clip(cos, -1, 1))
    rho_corpo = np.arcsin(np.clip(R_corpo / d_corpo, 0, 1))
    rho_lua = np.arcsin(np.clip(R_lua / d_lua, 0, 1))
    return (theta - rho_corpo - rho_lua) / IFOV


def ocultada(lua, corpo, ets):
    """True quando o corpo está na frente da lua (total, anular ou parcial)."""
    return np.array([spice.occult(lua, "ELLIPSOID", f"IAU_{lua}",
                                  corpo, "ELLIPSOID", f"IAU_{corpo}",
                                  CORRECAO_OCC, "JUICE", et) < 0 for et in ets])


def eclipse(lua, ets):
    """True quando Júpiter esconde o Sol (total, anular ou parcial), visto da lua."""
    return np.array([spice.occult("JUPITER", "ELLIPSOID", "IAU_JUPITER",
                                  "SUN", "ELLIPSOID", "IAU_SUN",
                                  CORRECAO_OCC, lua, et) > 0 for et in ets])


def intervalos(ets, ok):
    """Trechos seguidos com ok = True -> lista de (i_inicio, i_fim)."""
    trechos, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1] and ets[j + 1] - ets[j] <= 1.5 * PASSO_S:
                j += 1
            trechos.append((i, j))
            i = j + 1
        else:
            i += 1
    return trechos


carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
IFOV = spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]     # rad/pixel, do IK
R_JUP = spice.bodvrd("JUPITER", "RADII", 3)[1][0]       # raio equatorial (conservador)
R_GAL = {g: max(spice.bodvrd(g, "RADII", 3)[1]) for g in GALILEANAS}
print(f"Período analisado: {INICIO} -> {FIM} UTC")

cob_juice, arq_juice = cobertura_spk(JUICE_ID)
if not arq_juice:
    sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
janela = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), janela)
cob_juice = spice.wnintd(cob_juice, janela)

for nome, (id_lua, px_min) in LUAS.items():
    cob_lua, _ = cobertura_spk(id_lua)
    comum = spice.wnintd(cob_juice, cob_lua)
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue
    R_lua = np.mean(spice.bodvrd(nome, "RADII", 3)[1])
    D = 2 * R_lua

    ets = np.concatenate([np.arange(*spice.wnfetd(comum, k), PASSO_S)
                          for k in range(spice.wncard(comum))])

    # geometria (mesmas contas do dist_px_lat_zoom.py)
    pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
    v = -np.asarray(pos)                                  # lua -> JUICE
    dists = np.linalg.norm(v, axis=1)
    pixels = D / (dists * IFOV)
    lats = np.degrees(np.arcsin(v[:, 2] / dists))
    lons = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360   # leste, 0–360
    fases = np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO) for et in ets])
    fase_max = np.where(lats < 0, FASE_MAX_SUL, FASE_MAX_NORTE)
    sep_jup = separacao_px(nome, R_lua, "JUPITER", R_JUP, ets)
    sep_gal = np.min([separacao_px(nome, R_lua, g, R_GAL[g], ets)
                      for g in GALILEANAS], axis=0)
    occ = ocultada(nome, "JUPITER", ets)
    for g in GALILEANAS:
        occ |= ocultada(nome, g, ets)

    problemas = {                                         # True = problema acontece
        "ocultada": occ,
        "eclipse": eclipse(nome, ets),
        f"perto de Júpiter (< {LIMITE_PX} px)": sep_jup < LIMITE_PX,
        f"perto de galileana (< {LIMITE_PX} px)": sep_gal < LIMITE_PX,
        f"fase acima de {FASE_MAX_SUL}° S / {FASE_MAX_NORTE}° N": fases > fase_max,
        f"|lat| > {FAIXA_EQ}°": np.abs(lats) > FAIXA_EQ,
    }
    livre = ~np.any(list(problemas.values()), axis=0)
    janelas = [(i, j) for i, j in intervalos(ets, livre)
               if ets[j] - ets[i] >= DURACAO_MIN_S]

    # ---- lista e CSV das janelas ----
    print(f"\n{nome}: {len(janelas)} janela(s) >= {DURACAO_MIN_S / 60:.0f} min")
    arq_csv = f"janelas_{nome.lower()}.csv"
    with open(arq_csv, "w") as f:
        f.write("inicio_utc,fim_utc,duracao_min,dist_km,pixels,fase_graus,"
                "lat_subsc,lon_subsc_leste,sep_jupiter_px_min\n")
        for i, j in janelas:
            m = (i + j) // 2                              # valores no meio da janela
            ini = spice.et2utc(ets[i], "ISOC", 0)
            fim = spice.et2utc(ets[j], "ISOC", 0)
            dur = (ets[j] - ets[i]) / 60
            sep_min = sep_jup[i:j + 1].min()
            f.write(f"{ini},{fim},{dur:.0f},{dists[m]:.0f},{pixels[m]:.1f},"
                    f"{fases[m]:.1f},{lats[m]:.1f},{lons[m]:.1f},{sep_min:.0f}\n")
            print(f"    {ini} -> {fim} ({dur:4.0f} min) | {pixels[m]:5.1f} px | "
                  f"α = {fases[m]:5.1f}° | b = {lats[m]:5.1f}° | λ = {lons[m]:5.1f}°E")
    print(f"  tabela salva: {arq_csv}")

    # ---- figura: 5 painéis de linha + barras + janela ----
    datas = spice.et2datetime(ets)
    fig, eixos = plt.subplots(7, 1, figsize=(12, 17), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1.4, 0.35]})
    ax_d, ax_p, ax_l, ax_lon, ax_f, ax_b, ax_j = eixos

    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_title(f"{nome.capitalize()} ({id_lua}) – D ≈ {D:.0f} km ({META_KERNEL})")
    ax_d.set_ylabel("distância [km] (log)")
    ax_d.grid(True, which="both", axis="y", alpha=0.3)   # linhas menores só no y (log)

    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    ax_p.axhline(px_min, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.set_ylim(bottom=0)
    ax_p.grid(True, alpha=0.3)

    ax_l.plot(datas, lats, lw=0.6, color="tab:green")
    ax_l.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15)
    ax_l.axhline(0, color="gray", lw=0.6)
    ax_l.set_ylabel("lat. sub-S/C [°]")
    ax_l.grid(True, alpha=0.3)

    # pontos em vez de linha: a longitude dá a volta 360 -> 0 e a linha riscaria o painel
    ax_lon.plot(datas, lons, ".", ms=1.5, color="tab:brown")
    ax_lon.set_ylim(0, 360)
    ax_lon.set_yticks(range(0, 361, 90))
    ax_lon.set_ylabel("lon. sub-S/C [°E]")
    ax_lon.grid(True, alpha=0.3)

    ax_f.plot(datas, fases, lw=0.6, color="tab:purple")
    ax_f.plot(datas, fase_max, color="red", ls="--", lw=0.8)   # muda com o hemisfério
    ax_f.set_ylim(0, 180)
    ax_f.set_ylabel("fase α [°]")
    ax_f.grid(True, alpha=0.3)

    for k, (rotulo, m) in enumerate(problemas.items()):
        y = len(problemas) - 1 - k                        # primeira linha em cima
        ax_b.fill_between(datas, y - 0.35, y + 0.35, where=m,
                          color="tab:red", alpha=0.8, lw=0)
    ax_b.set_yticks(range(len(problemas)))
    ax_b.set_yticklabels(list(problemas)[::-1], fontsize=8)
    ax_b.set_ylim(-0.6, len(problemas) - 0.4)
    ax_b.set_ylabel("barra = acontece", fontsize=8)

    mask_janelas = np.zeros(len(ets), bool)
    for i, j in janelas:
        mask_janelas[i:j + 1] = True
    ax_j.fill_between(datas, 0, 1, where=mask_janelas, color="tab:green", lw=0)
    ax_j.set_yticks([0.5])
    ax_j.set_yticklabels([f"JANELA (≥ {DURACAO_MIN_S / 60:.0f} min)"], fontsize=8)
    ax_j.set_ylim(0, 1)

    for ax in eixos:
        ax.grid(True, which="major", axis="x", alpha=0.6)
    ax_j.margins(x=0)                                     # sem espaço antes/depois dos dados
    datas_eixo = mdates.AutoDateLocator()                 # serve p/ horas ou meses
    ax_j.xaxis.set_major_locator(datas_eixo)
    ax_j.xaxis.set_major_formatter(mdates.ConciseDateFormatter(datas_eixo))
    ax_j.set_xlabel("data (UTC)")
    fig.tight_layout()
    arq = f"janelas_{nome.lower()}.png"
    fig.savefig(arq, dpi=150)
    plt.close(fig)
    print(f"  gráfico salvo: {arq}")

spice.kclear()
