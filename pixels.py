"""Pixels que cada lua interna ocupa na JANUS ao longo do tempo: pixels = D / (d × IFOV)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO, FIM = "2031-03-01", "2034-05-01"
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}
NIVEIS_PX = [1, 10]
PNG = os.path.abspath("pixels_tempo.png")

os.chdir(os.path.dirname(MK))
spice.kclear()
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

dados = {}
for nome, id_lua in LUAS.items():
    janela = spice.wnintd(periodo, cobertura(id_lua))
    if spice.wncard(janela) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    D = 2 * np.mean(spice.bodvrd(nome, "RADII", 3)[1])
    datas, pixels = [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.arange(a, b, 600)
        pos, _ = spice.spkpos(nome, ets, "J2000", "NONE", "JUICE")
        t = spice.et2datetime(ets)
        datas += [*t, t[-1]]
        pixels += [*(D / (np.linalg.norm(pos, axis=1) * IFOV)), np.nan]
    pixels = np.array(pixels)
    dados[nome] = (datas, pixels, D)

    i = np.nanargmax(pixels)
    print(f"{nome}: D = {D:.1f} km | máx. {pixels[i]:.1f} px em {datas[i]:%Y-%m-%d %H:%M} UTC")

fig, eixos = plt.subplots(len(dados), 1, figsize=(12, 3 * len(dados)), sharex=True, squeeze=False)
for eixo, (nome, (datas, pixels, D)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        eixo.axhline(p, color="gray", ls="--", lw=0.8)
    eixo.set_title(f"{nome} (D ≈ {D:.0f} km)")
    eixo.set_ylabel("pixels através da lua")
    eixo.grid(True, which="both", alpha=0.3)
    eixo.margins(x=0)
    eixo.set_ylim(bottom=0)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(PNG, dpi=150)
