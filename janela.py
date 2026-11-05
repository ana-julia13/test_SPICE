"""Janelas de observação: uma imagem por lua, de cima para baixo:
distância [km], pixels na JANUS, latitude e longitude sub-S/C [°], fase α [°] com o limite em vermelho,
barras = o problema acontece (ocultada, eclipse, perto de Júpiter, ...), JANELA = nenhum problema por >= DURACAO_MIN_S.
A distância não entra na janela. As janelas também vão para um CSV.
Período: INICIO/FIM abaixo, ou na linha de comando: python janela.py "2032-03-10 00:00" "2032-03-14 00:00"
Fase: limite de 70° no sul e 110° no norte (observação B). Para a observação A, coloque FASE_MAX_NORTE = 70."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO, FIM = sys.argv[1:3] if len(sys.argv) == 3 else ("2032-05-09 00:00", "2032-05-11 12:00")
PASSO_S = 60
DURACAO_MIN_S = 600
CORRECAO = "LT+S"
CORRECAO_OCC = "LT"
LUAS = {"METIS": (516, 3), "ADRASTEA": (515, 2), "AMALTHEA": (505, 10), "THEBE": (514, 6)}
GALILEANAS = ["IO", "EUROPA", "GANYMEDE", "CALLISTO"]
FASE_MAX_SUL, FASE_MAX_NORTE = 70, 110
FAIXA_EQ = 10
LIMITE_PX = 10
SAIDA = os.getcwd()

os.chdir(os.path.dirname(MK))
spice.furnsh(MK)
IFOV = spice.gdpool("INS-28200_IFOV", 0, 1)[0]
R_JUP = spice.bodvrd("JUPITER", "RADII", 3)[1][0]
R_GAL = {g: max(spice.bodvrd(g, "RADII", 3)[1]) for g in GALILEANAS}
print(f"Período analisado: {INICIO} -> {FIM} UTC")


def cobertura(id_corpo):
    cob = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        spice.spkcov(spice.kdata(i, "SPK")[0], id_corpo, cob)
    return cob


def separacao_px(lua, R_lua, corpo, R_corpo, ets):
    """Pixels entre o limbo da lua e o limbo do corpo, vistos da JUICE."""
    p1 = np.array(spice.spkpos(lua, ets, "J2000", CORRECAO, "JUICE")[0])
    p2 = np.array(spice.spkpos(corpo, ets, "J2000", CORRECAO, "JUICE")[0])
    d1, d2 = np.linalg.norm(p1, axis=1), np.linalg.norm(p2, axis=1)
    theta = np.arccos(np.clip(np.sum(p1 * p2, axis=1) / (d1 * d2), -1, 1))
    return (theta - np.arcsin(np.clip(R_corpo / d2, 0, 1)) - np.arcsin(np.clip(R_lua / d1, 0, 1))) / IFOV


def ocultada(lua, corpo, ets):
    """True quando o corpo está na frente da lua (total, anular ou parcial)."""
    return np.array([spice.occult(lua, "ELLIPSOID", f"IAU_{lua}", corpo, "ELLIPSOID", f"IAU_{corpo}",
                                  CORRECAO_OCC, "JUICE", et) < 0 for et in ets])


def eclipse(lua, ets):
    """True quando Júpiter esconde o Sol (total, anular ou parcial), visto da lua."""
    return np.array([spice.occult("JUPITER", "ELLIPSOID", "IAU_JUPITER", "SUN", "ELLIPSOID", "IAU_SUN",
                                  CORRECAO_OCC, lua, et) > 0 for et in ets])


def trechos(ets, ok):
    """(i_inicio, i_fim) de cada trecho seguido com ok = True."""
    lista = []
    for i in np.flatnonzero(ok):
        if lista and lista[-1][1] == i - 1 and ets[i] - ets[i - 1] <= 1.5 * PASSO_S:
            lista[-1][1] = i
        else:
            lista.append([i, i])
    return lista


periodo = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), periodo)
periodo = spice.wnintd(periodo, cobertura(-28))

for nome, (id_lua, px_min) in LUAS.items():
    janela = spice.wnintd(periodo, cobertura(id_lua))
    if spice.wncard(janela) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue
    R_lua = np.mean(spice.bodvrd(nome, "RADII", 3)[1])
    D = 2 * R_lua

    ets = np.concatenate([np.arange(*spice.wnfetd(janela, k), PASSO_S) for k in range(spice.wncard(janela))])
    pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
    dists, lons, lats = np.array([spice.reclat(-p) for p in pos]).T
    lats, lons = np.degrees(lats), np.degrees(lons) % 360
    pixels = D / (dists * IFOV)
    fases = np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO) for et in ets])
    fase_max = np.where(lats < 0, FASE_MAX_SUL, FASE_MAX_NORTE)
    sep_jup = separacao_px(nome, R_lua, "JUPITER", R_JUP, ets)
    sep_gal = np.min([separacao_px(nome, R_lua, g, R_GAL[g], ets) for g in GALILEANAS], axis=0)

    problemas = {
        "ocultada": np.any([ocultada(nome, c, ets) for c in ["JUPITER", *GALILEANAS]], axis=0),
        "eclipse": eclipse(nome, ets),
        f"perto de Júpiter (< {LIMITE_PX} px)": sep_jup < LIMITE_PX,
        f"perto de galileana (< {LIMITE_PX} px)": sep_gal < LIMITE_PX,
        f"fase acima de {FASE_MAX_SUL}° S / {FASE_MAX_NORTE}° N": fases > fase_max,
        f"|lat| > {FAIXA_EQ}°": np.abs(lats) > FAIXA_EQ,
    }
    livre = ~np.any(list(problemas.values()), axis=0)
    janelas = [(i, j) for i, j in trechos(ets, livre) if ets[j] - ets[i] >= DURACAO_MIN_S]

    print(f"\n{nome}: {len(janelas)} janela(s) >= {DURACAO_MIN_S / 60:.0f} min")
    with open(f"{SAIDA}/janelas_{nome.lower()}.csv", "w") as f:
        f.write("inicio_utc,fim_utc,duracao_min,dist_km,pixels,fase_graus,lat_subsc,lon_subsc_leste,sep_jupiter_px_min\n")
        for i, j in janelas:
            m = (i + j) // 2
            ini, fim = spice.et2utc(ets[i], "ISOC", 0), spice.et2utc(ets[j], "ISOC", 0)
            dur = (ets[j] - ets[i]) / 60
            f.write(f"{ini},{fim},{dur:.0f},{dists[m]:.0f},{pixels[m]:.1f},{fases[m]:.1f},{lats[m]:.1f},"
                    f"{lons[m]:.1f},{sep_jup[i:j + 1].min():.0f}\n")
            print(f"    {ini} -> {fim} ({dur:4.0f} min) | {pixels[m]:5.1f} px | "
                  f"α = {fases[m]:5.1f}° | b = {lats[m]:5.1f}° | λ = {lons[m]:5.1f}°E")

    datas = spice.et2datetime(ets)
    fig, eixos = plt.subplots(7, 1, figsize=(12, 17), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1.4, 0.35]})
    ax_d, ax_p, ax_l, ax_lon, ax_f, ax_b, ax_j = eixos

    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_title(f"{nome.capitalize()} ({id_lua}) – D ≈ {D:.0f} km ({os.path.basename(MK)})")
    ax_d.set_ylabel("distância [km] (log)")
    ax_d.grid(True, which="both", axis="y", alpha=0.3)

    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    ax_p.axhline(px_min, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.set_ylim(bottom=0)

    ax_l.plot(datas, lats, lw=0.6, color="tab:green")
    ax_l.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15)
    ax_l.axhline(0, color="gray", lw=0.6)
    ax_l.set_ylabel("lat. sub-S/C [°]")

    ax_lon.plot(datas, lons, ".", ms=1.5, color="tab:brown")
    ax_lon.set_ylim(0, 360)
    ax_lon.set_yticks(range(0, 361, 90))
    ax_lon.set_ylabel("lon. sub-S/C [°E]")

    ax_f.plot(datas, fases, lw=0.6, color="tab:purple")
    ax_f.plot(datas, fase_max, color="red", ls="--", lw=0.8)
    ax_f.set_ylim(0, 180)
    ax_f.set_ylabel("fase α [°]")

    for k, (rotulo, m) in enumerate(problemas.items()):
        y = len(problemas) - 1 - k
        ax_b.fill_between(datas, y - 0.35, y + 0.35, where=m, color="tab:red", alpha=0.8, lw=0)
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

    for ax in (ax_p, ax_l, ax_lon, ax_f):
        ax.grid(True, alpha=0.3)
    for ax in eixos:
        ax.grid(True, which="major", axis="x", alpha=0.6)
    ax_j.margins(x=0)
    datas_eixo = mdates.AutoDateLocator()
    ax_j.xaxis.set_major_locator(datas_eixo)
    ax_j.xaxis.set_major_formatter(mdates.ConciseDateFormatter(datas_eixo))
    ax_j.set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(f"{SAIDA}/janelas_{nome.lower()}.png", dpi=150)
    plt.close(fig)
