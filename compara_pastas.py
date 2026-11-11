"""
Plota, para cada satélite, os elementos do SPICE, do DLR e do nbody juntos,
no período em comum (até DATA_FIM). Salva comp_<satelite>.png em PASTA_SAIDA.

  results_DLR/    oscelt_planetIAU_<sat>.out   data  ET  t  a  e  i  Ω  ω  M
  results_nbody/  <Sat>_osc.dat                t(dias)  a  e  i  ω  Ω  M
  results_SPICE/  elementos_<sat>.csv          saída do elementos_spice.py

Uso: python3 compara_pastas.py [pasta que contém as três]
"""

import argparse
import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= CONFIGURAÇÃO (edite aqui) =======================
PASTA_SAIDA = "results_comparacao"
NBODY_T0 = None            # t = 0 do nbody em ET ("et =" do .in); None = início do SPICE
DATA_FIM = "2040-01-01"    # UTC; None = até onde todos tiverem dados
ALPHA = 0.45               # transparência das curvas
# =========================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]
CORES = {"SPICE": "k", "DLR": "C0", "nbody": "C3"}
ET0_UTC = datetime(2000, 1, 1, 11, 58, 55, 816000)  # ET 0 em UTC (TDB − UTC ≈ 69 s)


def descobre(pasta, padrao):
    regex = re.compile("^" + re.escape(padrao).replace(r"\*", "(.+)") + "$", re.I)
    return {m.group(1).lower(): f for f in glob.glob(os.path.join(pasta, "*"))
            if (m := regex.match(os.path.basename(f)))}


def le_dlr(arquivo):
    linhas = []
    for linha in open(arquivo):
        try:
            linhas.append([float(p.replace("D", "E")) for p in linha.split()[-8:]])
        except ValueError:
            pass
    d = np.array([l for l in linhas if len(l) == 8])
    tab = d[:, 2:8]
    if np.abs(tab[:, 5]).max() <= 2 * np.pi + 1e-6:  # M até 2π: radianos
        tab[:, 2:] = np.degrees(tab[:, 2:])
    return d[:, 0], tab


def le_nbody(arquivo, et0):
    t, a, e, i, w, om, m = np.loadtxt(arquivo, ndmin=2)[:, :7].T
    return et0 + t * 86400.0, np.column_stack([a, e, i, om, w, m])


def le_spice(arquivo):
    d = np.genfromtxt(arquivo, delimiter=",", skip_header=1, usecols=range(1, 8), ndmin=2)
    return d[:, 0], d[:, 1:]


def plota(nome, series, arquivo):
    inicio = max(ets[0] for ets, _ in series.values())
    fim = min(ets[-1] for ets, _ in series.values())
    if DATA_FIM:
        fim = min(fim, (datetime.fromisoformat(DATA_FIM) - ET0_UTC).total_seconds())

    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for rotulo, (ets, tab) in series.items():
        ok = (ets >= inicio) & (ets <= fim)
        tempo = [ET0_UTC + timedelta(seconds=s) for s in ets[ok]]
        for j, eixo in enumerate(eixos):
            if j >= 3:  # ângulos: pontos, para não ligar os saltos 360 -> 0
                eixo.plot(tempo, tab[ok, j] % 360, ".", ms=2.5, mew=0,
                          color=CORES[rotulo], alpha=ALPHA, label=rotulo)
            else:
                eixo.plot(tempo, tab[ok, j], lw=0.8, color=CORES[rotulo],
                          alpha=ALPHA, label=rotulo)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        eixo.set_ylabel(rot)
        eixo.margins(x=0)
        eixo.ticklabel_format(axis="y", style="plain", useOffset=False)
        eixo.grid(alpha=0.3)
        if j >= 3:
            eixo.set_ylim(0, 360)
    leg = eixos[0].legend(loc="upper right", fontsize=8, markerscale=6)
    for linha in leg.get_lines():
        linha.set_alpha(1)
    eixos[0].set_title(nome.capitalize())
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plota SPICE, DLR e nbody juntos.")
    parser.add_argument("base", nargs="?", default=".", help="pasta que contém as três")
    base = parser.parse_args().base

    spice = descobre(os.path.join(base, "results_SPICE"), "elementos_*.csv")
    dlr = descobre(os.path.join(base, "results_DLR"), "oscelt_planetIAU_*.out")
    nbody = descobre(os.path.join(base, "results_nbody"), "*_osc.dat")
    os.makedirs(os.path.join(base, PASTA_SAIDA), exist_ok=True)

    for nome in sorted(set(spice) | set(dlr) | set(nbody)):
        series = {}
        if nome in spice:
            series["SPICE"] = le_spice(spice[nome])
        if nome in dlr:
            series["DLR"] = le_dlr(dlr[nome])
        if nome in nbody and (NBODY_T0 is not None or "SPICE" in series):
            et0 = NBODY_T0 if NBODY_T0 is not None else series["SPICE"][0][0]
            series["nbody"] = le_nbody(nbody[nome], et0)
        plota(nome, series, os.path.join(base, PASTA_SAIDA, f"comp_{nome}.png"))


if __name__ == "__main__":
    main()
