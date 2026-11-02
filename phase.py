"""Ângulo de fase (Sol – lua – JUICE, vértice na lua) × tempo para Metis, Adrastea, Amalthea e Thebe.
Observação A: α <= 70°; campanhas em α ≈ 10° (fase baixa) e α ≈ 50° (fase média)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO, FIM = "2031-07-21", "2031-07-24"
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}
CORRECAO = "LT+S"
LIMITE_FASE = 70
CAMPANHAS = [10, 50]
PNG = os.path.abspath("fase_tempo.png")

os.chdir(os.path.dirname(MK))
spice.furnsh(MK)


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

    datas, fases = [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.arange(a, b, 1800)
        t = spice.et2datetime(ets)
        datas += [*t, t[-1]]
        fases += [*np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", CORRECAO) for et in ets]), np.nan]
    fases = np.array(fases)
    dados[nome] = (datas, fases)

    ok = np.mean(fases[np.isfinite(fases)] <= LIMITE_FASE) * 100
    print(f"{nome}: α de {np.nanmin(fases):.1f}° a {np.nanmax(fases):.1f}° | {ok:.0f}% do tempo com α <= {LIMITE_FASE}°")

fig, eixos = plt.subplots(len(dados), 1, figsize=(12, 2.8 * len(dados)), sharex=True, squeeze=False)
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
fig.suptitle(f"Ângulo de fase ({os.path.basename(MK)}, {CORRECAO})", fontsize=10)
fig.tight_layout()
fig.savefig(PNG, dpi=150)
