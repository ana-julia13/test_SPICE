"""Uma imagem por lua interna (Metis, Adrastea, Amalthea, Thebe) com 5 painéis contra o 
tempo: distância JUICE–lua [km], pixels na JANUS = D / (d × IFOV), latitude e longitude 
sub-S/C [°] e fase α [°]."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_crema_5_2_v473_20260923_005.tm"

INICIO, FIM = "2031-10-01", "2032-10-01"

LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}

PASSO_S = 60
CORRECAO = "LT+S"
NIVEIS_PX = [1, 10]
FAIXA_EQ, LIMITE_B = 10, 20
LIMITE_FASE = 70
CAMPANHAS = [10, 50]
SAIDA = os.getcwd()

os.chdir(os.path.dirname(MK))
spice.furnsh(MK)
IFOV = spice.gdpool("INS-28200_IFOV", 0, 1)[0]
print(f"IFOV da JANUS (IK): {IFOV * 1e6:.2f} µrad")

def cobertura(id_corpo):
    cob = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        spice.spkcov(spice.kdata(i, "SPK")[0], id_corpo, cob)
    return cob

periodo = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), periodo)
periodo = spice.wnintd(periodo, cobertura(-28))

for nome, id_lua in LUAS.items():
    janela = spice.wnintd(periodo, cobertura(id_lua))
    if spice.wncard(janela) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    D = 2 * np.mean(spice.bodvrd(nome, "RADII", 3)[1])
    datas, dists, lats, lons, fases = [], [], [], [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.arange(a, b, PASSO_S)
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        d, lon, lat = np.array([spice.reclat(-p) for p in pos]).T
        t = spice.et2datetime(ets)
        datas += [*t, t[-1]]
        dists += [*d, np.nan]
        lats += [*np.degrees(lat), np.nan]
        lons += [*(np.degrees(lon) % 360), np.nan]
        fases += [*np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO) for et in ets]), np.nan]
    dists, lats, lons, fases = map(np.array, (dists, lats, lons, fases))
    pixels = D / (dists * IFOV)

    i = np.nanargmin(dists)
    print(f"{nome}: mín. {dists[i]:,.0f} km = {pixels[i]:.1f} px, b = {lats[i]:.1f}°, "
          f"λ = {lons[i]:.0f}°E, α = {fases[i]:.0f}° em {datas[i]:%Y-%m-%d %H:%M} UTC")

    fig, (ax_d, ax_p, ax_l, ax_lon, ax_f) = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_title(f"{nome.capitalize()} ({id_lua}) – D ≈ {D:.0f} km")
    ax_d.set_ylabel("distância [km] (log)")
    ax_d.grid(True, which="both", axis="y", alpha=0.3)

    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        ax_p.axhline(p, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.set_ylim(bottom=0)

    ax_l.plot(datas, lats, lw=0.5, color="tab:green")
    ax_l.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15)
    for y in (-LIMITE_B, LIMITE_B):
        ax_l.axhline(y, color="gray", ls=":", lw=0.8)
    ax_l.set_ylabel("lat. sub-S/C [°]")

    ax_lon.plot(datas, lons, ".", ms=1.5, color="tab:brown")
    ax_lon.set_ylim(0, 360)
    ax_lon.set_yticks(range(0, 361, 90))
    ax_lon.set_ylabel("lon. sub-S/C [°E]")

    for c in CAMPANHAS:
        ax_f.axhspan(c - 10, c + 10, color="tab:blue", alpha=0.12)
    ax_f.axhline(LIMITE_FASE, color="red", ls="--", lw=0.8)
    ax_f.plot(datas, fases, lw=0.6, color="tab:purple")
    ax_f.set_ylim(0, 180)
    ax_f.set_ylabel("fase α [°]")
    ax_f.set_xlabel("data (UTC)")
    ax_f.margins(x=0)

    for ax in (ax_p, ax_l, ax_lon, ax_f):
        ax.grid(True, alpha=0.3)
    for ax in (ax_d, ax_p, ax_l, ax_lon, ax_f):
        ax.grid(True, which="major", axis="x", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{SAIDA}/all_{nome.lower()}.png", dpi=150)
    plt.close(fig)
