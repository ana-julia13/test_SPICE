"""
Plota, para cada satélite, os elementos orbitais do SPICE, do DLR e do
nbody juntos, lendo três pastas:

  results_DLR/    oscelt_planetIAU_metis.out, ...
                  linhas: 2030 JAN 01 00:00:00.000  ET  t  a  e  i  Ω  ω  M
  results_nbody/  Metis_osc.dat, ...
                  linhas: tempo(dias)  a  e  inc  w  om  M   ('#' = cabeçalho;
                  atenção: no nbody o ω vem ANTES do Ω)
  results_SPICE/  elementos_metis.csv, ...  (saída do elementos_spice.py)

Os satélites são achados pelos nomes dos arquivos (maiúsculas não importam).
Salva um gráfico por satélite em PASTA_SAIDA: comp_metis.png, comp_adrastea.png, ...

Uso:
  python3 compara_pastas.py                    (pastas dentro da pasta atual)
  python3 compara_pastas.py /caminho/da/pasta  (pasta que contém as três)
"""

import argparse
import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")  # só salva PNG, sem abrir janela (não precisa de Qt)
import matplotlib.pyplot as plt

# ======================= CONFIGURAÇÃO (edite aqui) =======================
PASTA_BASE = "."                 # pasta que contém as três abaixo
PASTA_DLR = "results_DLR"
PASTA_NBODY = "results_nbody"
PASTA_SPICE = "results_SPICE"
PASTA_SAIDA = "results_comparacao"

# t = 0 do nbody em ET (segundos TDB desde J2000), o "et =" do cabeçalho do
# .in. None = primeiro instante do CSV do SPICE de cada satélite.
NBODY_T0 = None

# Só plota até esta data (UTC, "AAAA-MM-DD"). None = tudo.
DATA_FIM = None

ANGULOS_DLR = "auto"             # "graus", "radianos" ou "auto" (decide pelo M)
# =========================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]
ANGULARES = [3, 4, 5]            # colunas 0–360 (Ω, ω, M)
CORES = {"SPICE": "k", "DLR": "C0", "nbody": "C1"}

# ET 0 = 2000-01-01 12:00:00 TDB = 11:58:55.816 UTC. Os 69 s de diferença
# entre TDB e UTC são tratados como constantes (não há segundo bissexto
# desde 2017): serve para o eixo dos gráficos e para o DATA_FIM.
ET0_UTC = datetime(2000, 1, 1, 11, 58, 55, 816000)


def et_para_data(ets):
    return [ET0_UTC + timedelta(seconds=float(et)) for et in ets]


def data_para_et(texto):
    return (datetime.fromisoformat(texto) - ET0_UTC).total_seconds()


def descobre(pasta, padrao):
    """{nome do satélite em minúsculas: arquivo} para um padrão com um '*'."""
    regex = re.compile("^" + re.escape(padrao).replace(r"\*", "(.+)") + "$", re.IGNORECASE)
    achados = {}
    for arquivo in glob.glob(os.path.join(pasta, "*")):
        m = regex.match(os.path.basename(arquivo))
        if m:
            achados[m.group(1).lower()] = arquivo
    return achados


def le_dlr(arquivo):
    """ET e [a, e, i, Ω, ω, M] (ângulos ainda na unidade do arquivo).
    Cada linha: data (várias palavras) seguida de 8 números: ET t a e i Ω ω M."""
    numeros = []
    with open(arquivo) as f:
        for linha in f:
            partes = linha.split()
            if len(partes) < 9 or partes[0].startswith(("#", "!")):
                continue
            try:
                numeros.append([float(p.replace("D", "E").replace("d", "e")) for p in partes[-8:]])
            except ValueError:
                continue  # cabeçalho ou linha que não é de dados
    if not numeros:
        raise ValueError(f"nenhuma linha de dados em {arquivo}")
    numeros = np.array(numeros)
    return numeros[:, 0], numeros[:, 2:8]


def le_nbody(arquivo):
    """t (dias) e [a, e, i, Ω, ω, M] (graus). No arquivo: t a e inc w om M."""
    dados = np.loadtxt(arquivo, comments="#", ndmin=2)
    t, a, e, inc, w, om, m = dados[:, :7].T
    return t, np.column_stack([a, e, inc, om, w, m])


def le_spice(arquivo):
    """ET e [a, e, i, Ω, ω, M] (graus) do CSV do elementos_spice.py."""
    dados = np.genfromtxt(arquivo, delimiter=",", skip_header=1, usecols=range(1, 8), ndmin=2)
    return dados[:, 0], dados[:, 1:7]


def para_graus(tabela, angulos, nome):
    """Passa i, Ω, ω, M para graus. 'auto' decide pelo M: em graus ele vai
    até 360, em radianos até 2π."""
    tabela = tabela.copy()
    if angulos == "auto":
        angulos = "radianos" if np.nanmax(np.abs(tabela[:, 5])) <= 2 * np.pi + 1e-6 else "graus"
        print(f"  {nome}: ângulos do DLR em {angulos} (detectado pelo M)")
    if angulos == "radianos":
        tabela[:, 2:] = np.degrees(tabela[:, 2:])
    elif angulos != "graus":
        raise ValueError(f"ANGULOS_DLR desconhecido: {angulos}")
    return tabela


def formata_eixo(eixo):
    """Eixo x começando e terminando nos dados e eixo y sem notação científica."""
    eixo.margins(x=0)
    eixo.ticklabel_format(axis="y", style="plain", useOffset=False)
    eixo.grid(alpha=0.3)


def plota_satelite(nome, arq_spice, arq_dlr, arq_nbody, arquivo_png):
    series = []
    if arq_spice:
        series.append(("SPICE",) + le_spice(arq_spice))
    if arq_dlr:
        ets, tabela = le_dlr(arq_dlr)
        series.append(("DLR", ets, para_graus(tabela, ANGULOS_DLR, nome)))
    if arq_nbody:
        t, tabela = le_nbody(arq_nbody)
        if NBODY_T0 is not None:
            et0 = NBODY_T0
        elif arq_spice:
            et0 = series[0][1][0]
        else:
            print(f"  {nome}: nbody sem NBODY_T0 nem SPICE para saber a data inicial, pulando o nbody")
            arq_nbody = None
        if arq_nbody:
            series.append(("nbody", et0 + t * 86400.0, tabela))

    if DATA_FIM is not None:
        et_fim = data_para_et(DATA_FIM)
        series = [(r, ets[ets <= et_fim], tab[ets <= et_fim]) for r, ets, tab in series]

    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        for rotulo, ets, tabela in series:
            tempo = et_para_data(ets)
            if j in ANGULARES:  # 0–360: pontos, para não ligar os saltos 360 -> 0
                eixo.plot(tempo, tabela[:, j] % 360.0, ".", ms=1.2, color=CORES[rotulo],
                          alpha=0.6, label=rotulo)
            else:
                eixo.plot(tempo, tabela[:, j], lw=0.7, color=CORES[rotulo],
                          alpha=0.8, label=rotulo)
        if j in ANGULARES:
            eixo.set_ylim(0, 360)
        eixo.set_ylabel(rot)
        formata_eixo(eixo)
    eixos[0].legend(loc="upper right", fontsize=8, markerscale=6)
    eixos[0].set_title(f"{nome.capitalize()} – " + " × ".join(r for r, _, _ in series))
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(arquivo_png, dpi=150)
    plt.close(fig)
    print(f"  salvo {arquivo_png}")


def main():
    parser = argparse.ArgumentParser(description="Plota SPICE, DLR e nbody juntos para cada satélite.")
    parser.add_argument("base", nargs="?", default=PASTA_BASE,
                        help="pasta que contém results_DLR, results_nbody e results_SPICE")
    args = parser.parse_args()

    spice_arqs = descobre(os.path.join(args.base, PASTA_SPICE), "elementos_*.csv")
    dlr_arqs = descobre(os.path.join(args.base, PASTA_DLR), "oscelt_planetIAU_*.out")
    nbody_arqs = descobre(os.path.join(args.base, PASTA_NBODY), "*_osc.dat")

    saida = os.path.join(args.base, PASTA_SAIDA)
    os.makedirs(saida, exist_ok=True)
    for nome in sorted(set(spice_arqs) | set(dlr_arqs) | set(nbody_arqs)):
        print(nome)
        plota_satelite(nome, spice_arqs.get(nome), dlr_arqs.get(nome), nbody_arqs.get(nome),
                       os.path.join(saida, f"comp_{nome}.png"))


if __name__ == "__main__":
    main()
