"""Ponto sub-JUICE (o ponto da lua logo "embaixo" da JUICE): latitude e longitude × tempo
para Metis, Adrastea, Amalthea e Thebe, no referencial fixo de cada lua (IAU_<LUA>).
Longitude leste positiva, 0–360°. |b| <= 10° = quase equatorial; |b| >= 20° = tipo B."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO, FIM = "2031-03-01", "2034-05-01"
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}
CORRECAO = "LT+S"
FAIXA_EQ, LIMITE_B = 10, 20
PNG = os.path.abspath("lat_lon_subsc_tempo.png")

os.chdir(os.path.dirname(MK))
spice.kclear()
spice.furnsh(MK)


def cobertura(id_corpo):
    cob = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        spice.spkcov(spice.kdata(i, "SPK")[0], id_corpo, cob)
    return cob


periodo = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), periodo)
periodo = spice.wnintd(periodo, cobertura(-28))

fig, eixos = plt.subplots(len(LUAS), 2, figsize=(14, 2.8 * len(LUAS)), sharex=True, squeeze=False)
for (nome, id_lua), (eixo_lat, eixo_lon) in zip(LUAS.items(), eixos):
    janela = spice.wnintd(periodo, cobertura(id_lua))
    if spice.wncard(janela) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, lats, lons = [], [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.arange(a, b, 1800)
        pos, _ = spice.spkpos(nome, ets, f"IAU_{nome}", CORRECAO, "JUICE")
        _, lon, lat = np.array([spice.reclat(-p) for p in pos]).T
        t = spice.et2datetime(ets)
        datas += [*t, t[-1]]
        lats += [*np.degrees(lat), np.nan]
        lons += [*(np.degrees(lon) % 360), np.nan]
    lats = np.array(lats)

    perto_eq = np.mean(np.abs(lats[np.isfinite(lats)]) <= FAIXA_EQ) * 100
    print(f"{nome}: b de {np.nanmin(lats):.1f}° a {np.nanmax(lats):.1f}° | {perto_eq:.0f}% do tempo com |b| <= {FAIXA_EQ}°")

    eixo_lat.axhspan(-FAIXA_EQ, FAIXA_EQ, color="tab:green", alpha=0.15, label=f"|b| ≤ {FAIXA_EQ}° (quase equatorial)")
    for y in (-LIMITE_B, LIMITE_B):
        eixo_lat.axhline(y, color="gray", ls=":", lw=0.8)
    eixo_lat.plot(datas, lats, lw=0.5)
    eixo_lat.set_title(f"{nome.capitalize()} – latitude")
    eixo_lat.set_ylabel("latitude [°]")

    eixo_lon.plot(datas, lons, ".", ms=1)
    eixo_lon.set_ylim(0, 360)
    eixo_lon.set_yticks(range(0, 361, 90))
    eixo_lon.set_title(f"{nome.capitalize()} – longitude")
    eixo_lon.set_ylabel("longitude [°E]")

    for eixo in (eixo_lat, eixo_lon):
        eixo.grid(True, alpha=0.3)
        eixo.margins(x=0)

eixos[0, 0].legend(loc="upper right", fontsize=8)
for eixo in eixos[-1]:
    eixo.set_xlabel("data (UTC)")
fig.suptitle(f"Ponto sub-JUICE ({os.path.basename(MK)}, {CORRECAO})", fontsize=10)
fig.tight_layout()
fig.savefig(PNG, dpi=150)
