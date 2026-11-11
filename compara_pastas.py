"""
Compara, satélite por satélite, os resultados de três pastas:

  results_DLR/    oscelt_planetIAU_metis.out, oscelt_planetIAU_io.out, ...
                  linhas: 2030 JAN 01 00:00:00.000  ET  t  a  e  i  Ω  ω  M
  results_nbody/  Metis_osc.dat, Adrastea_osc.dat, ...
                  linhas: tempo(dias)  a  e  inc  w  om  M   ('#' = cabeçalho;
                  atenção: no nbody o ω vem ANTES do Ω)
  results_SPICE/  elementos_metis.csv, elementos_adrastea.csv, ...
                  (saída do elementos_spice.py: data_utc, et_tdb_s, a, e, i, Ω, ω, M)

Os satélites são descobertos sozinhos pelos nomes dos arquivos (maiúsculas
não importam). Cada código é comparado com o SPICE nos seus próprios
instantes; o SPICE é interpolado nesses instantes (ângulos sem o salto de
360°). Use no elementos_spice.py um PASSO_S pequeno (600 s serve até para o
Metis).

Não precisa de kernels: o SPICE já está nos CSVs. Os três arquivos têm que
estar no mesmo frame (o do elementos_spice.py, equador de Júpiter congelado
no INICIO) e o tempo do nbody é contado a partir de NBODY_T0.

Gera em PASTA_SAIDA/<satelite>/:
  tres_elementos.png    os 6 elementos das três fontes sobrepostos
  tres_diferencas.png   DLR − SPICE e nbody − SPICE
  tres_distancia.png    distância de cada código à posição do SPICE (km)
  dlr_vs_spice.csv e nbody_vs_spice.csv
e PASTA_SAIDA/resumo.csv com o RMS das diferenças de todos os satélites.

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

# Só usa dados até esta data (UTC, "AAAA-MM-DD"). None = tudo.
DATA_FIM = None

# μ (km³/s²) para refazer as posições dos elementos na comparação em km.
# Use o mesmo GM de Júpiter do SPICE e dos códigos.
MU_KM3S2 = 126686595.050296

ANGULOS_DLR = "auto"             # "graus", "radianos" ou "auto" (decide pelo M)
# =========================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]
NOMES_CSV = ["a_km", "e", "i_deg", "Omega_deg", "omega_deg", "M_deg"]
ANGULARES = [3, 4, 5]            # colunas 0–360 (Ω, ω, M)
CORES = {"SPICE": "k", "DLR": "C0", "nbody": "C1"}

# ET 0 = 2000-01-01 12:00:00 TDB = 11:58:55.816 UTC. Os 69 s de diferença
# entre TDB e UTC são tratados como constantes (não há segundo bissexto
# desde 2017): serve para os eixos dos gráficos e para o DATA_FIM.
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


def para_graus(tabela, angulos, rotulo):
    """Passa i, Ω, ω, M para graus. 'auto' decide pelo M: em graus ele vai
    até 360, em radianos até 2π."""
    tabela = tabela.copy()
    if angulos == "auto":
        angulos = "radianos" if np.nanmax(np.abs(tabela[:, 5])) <= 2 * np.pi + 1e-6 else "graus"
        print(f"  ângulos do {rotulo}: {angulos} (detectado pelo M)")
    if angulos == "radianos":
        tabela[:, 2:] = np.degrees(tabela[:, 2:])
    elif angulos != "graus":
        raise ValueError(f"ANGULOS_DLR desconhecido: {angulos}")
    return tabela


def interpola_spice(ets_spice, spc, ets):
    """SPICE interpolado nos instantes 'ets'; ângulos interpolados sem o
    salto de 360° (unwrap) e devolvidos em 0–360."""
    saida = np.empty((len(ets), 6))
    for j in range(6):
        coluna = spc[:, j]
        if j in ANGULARES:
            coluna = np.degrees(np.unwrap(np.radians(coluna)))
        saida[:, j] = np.interp(ets, ets_spice, coluna)
    saida[:, ANGULARES] %= 360.0
    return saida


def posicoes(tabela, mu):
    """Posição (km) de cada linha de [a, e, i, Ω, ω, M] (graus)."""
    a, e = tabela[:, 0], tabela[:, 1]
    i, om, w, m = np.radians(tabela[:, 2:6]).T
    u = m + e * np.sin(m)                       # equação de Kepler (Newton)
    for _ in range(50):
        du = (u - e * np.sin(u) - m) / (1.0 - e * np.cos(u))
        u -= du
        if np.max(np.abs(du)) < 1e-14:
            break
    x = a * (np.cos(u) - e)
    y = a * np.sqrt(1.0 - e * e) * np.sin(u)
    p = np.column_stack([np.cos(om) * np.cos(w) - np.sin(om) * np.sin(w) * np.cos(i),
                         np.sin(om) * np.cos(w) + np.cos(om) * np.sin(w) * np.cos(i),
                         np.sin(w) * np.sin(i)])
    q = np.column_stack([-np.cos(om) * np.sin(w) - np.sin(om) * np.cos(w) * np.cos(i),
                         -np.sin(om) * np.sin(w) + np.cos(om) * np.cos(w) * np.cos(i),
                         np.cos(w) * np.sin(i)])
    return x[:, None] * p + y[:, None] * q


def diferencas(tabela, referencia):
    """tabela − referência; nos ângulos Ω, ω, M, entre −180° e 180°."""
    dif = tabela - referencia
    dif[:, ANGULARES] = (dif[:, ANGULARES] + 180.0) % 360.0 - 180.0
    return dif


def formata_eixo(eixo):
    """Eixo x começando e terminando nos dados e eixo y sem notação científica."""
    eixo.margins(x=0)
    eixo.ticklabel_format(axis="y", style="plain", useOffset=False)
    eixo.grid(alpha=0.3)


def salva_csv(arquivo, ets, codigo, spc, dif, distancia):
    with open(arquivo, "w") as f:
        f.write("et_tdb_s,"
                + ",".join(f"{n}_codigo" for n in NOMES_CSV) + ","
                + ",".join(f"{n}_spice" for n in NOMES_CSV) + ","
                + ",".join(f"d{n}" for n in NOMES_CSV) + ",distancia_km\n")
        for et, lc, ls, ld, dist in zip(ets, codigo, spc, dif, distancia):
            f.write(f"{et:.3f}," + ",".join(f"{v:.10g}" for v in np.concatenate([lc, ls, ld]))
                    + f",{dist:.6g}\n")


def compara_satelite(nome, arq_spice, arq_dlr, arq_nbody, pasta):
    """Faz os gráficos e CSVs de um satélite; devolve {código: (dif, distância)}."""
    print(f"\n=== {nome.upper()} ===")
    ets_spice, spc = le_spice(arq_spice)
    et_fim = data_para_et(DATA_FIM) if DATA_FIM else np.inf

    fontes = {}
    if arq_dlr:
        ets, tabela = le_dlr(arq_dlr)
        fontes["DLR"] = (ets, para_graus(tabela, ANGULOS_DLR, "DLR"))
    if arq_nbody:
        t, tabela = le_nbody(arq_nbody)
        et0 = ets_spice[0] if NBODY_T0 is None else NBODY_T0
        fontes["nbody"] = (et0 + t * 86400.0, tabela)
        print(f"  nbody: t = 0 em ET {et0:.3f}"
              + ("  [= primeiro instante do SPICE]" if NBODY_T0 is None else ""))

    resultados = {}
    for rotulo, (ets, tabela) in fontes.items():
        # só onde há SPICE para interpolar (e até DATA_FIM)
        ok = (ets >= ets_spice[0]) & (ets <= ets_spice[-1]) & (ets <= et_fim)
        if not ok.any():
            print(f"  {rotulo}: nenhum instante dentro do período do SPICE, pulando")
            continue
        if not ok.all():
            print(f"  {rotulo}: {np.count_nonzero(~ok)} instantes fora do período do SPICE"
                  + (" / depois de DATA_FIM" if DATA_FIM else "") + ", ignorados")
        ets, tabela = ets[ok], tabela[ok].copy()
        tabela[:, ANGULARES] %= 360.0
        ref = interpola_spice(ets_spice, spc, ets)
        dif = diferencas(tabela, ref)
        distancia = np.linalg.norm(posicoes(tabela, MU_KM3S2) - posicoes(ref, MU_KM3S2), axis=1)
        resultados[rotulo] = (ets, tabela, ref, dif, distancia)
        salva_csv(os.path.join(pasta, f"{rotulo.lower()}_vs_spice.csv"),
                  ets, tabela, ref, dif, distancia)
        print(f"  {rotulo} − SPICE ({len(ets)} instantes):  RMS  " + "  ".join(
            f"{r.split()[0]}={np.sqrt(np.mean(c**2)):.4g}" for r, c in zip(ROTULOS, dif.T))
            + f"  dist={np.sqrt(np.mean(distancia**2)):.4g} km")

    if not resultados:
        return {}

    ok = ets_spice <= et_fim
    series = [("SPICE", ets_spice[ok], spc[ok])] + \
             [(rotulo, r[0], r[1]) for rotulo, r in resultados.items()]

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
    eixos[0].set_title(f"{nome.capitalize()} – SPICE × DLR × nbody (osculadores)")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "tres_elementos.png"), dpi=150)
    plt.close(fig)

    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        for rotulo, (ets, _, _, dif, _) in resultados.items():
            eixo.plot(et_para_data(ets), dif[:, j], lw=0.7, color=CORES[rotulo],
                      alpha=0.8, label=f"{rotulo} − SPICE")
        eixo.axhline(0.0, color="k", lw=0.5)
        eixo.set_ylabel("Δ" + rot)
        formata_eixo(eixo)
    eixos[0].legend(loc="upper right", fontsize=8)
    eixos[0].set_title(f"{nome.capitalize()} – diferença em relação ao SPICE")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "tres_diferencas.png"), dpi=150)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(11, 4))
    for rotulo, (ets, _, _, _, distancia) in resultados.items():
        eixo.plot(et_para_data(ets), distancia, lw=0.7, color=CORES[rotulo],
                  label=f"{rotulo} − SPICE")
    eixo.set_ylabel("distância ao SPICE [km]")
    eixo.set_xlabel("data (UTC)")
    eixo.set_title(f"{nome.capitalize()} – distância de cada código à posição do SPICE")
    eixo.legend(fontsize=8)
    formata_eixo(eixo)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "tres_distancia.png"), dpi=150)
    plt.close(fig)

    return {rotulo: (r[3], r[4]) for rotulo, r in resultados.items()}


def main():
    parser = argparse.ArgumentParser(description="Compara DLR, nbody e SPICE para todos os satélites.")
    parser.add_argument("base", nargs="?", default=PASTA_BASE,
                        help="pasta que contém results_DLR, results_nbody e results_SPICE")
    args = parser.parse_args()

    spice_arqs = descobre(os.path.join(args.base, PASTA_SPICE), "elementos_*.csv")
    dlr_arqs = descobre(os.path.join(args.base, PASTA_DLR), "oscelt_planetIAU_*.out")
    nbody_arqs = descobre(os.path.join(args.base, PASTA_NBODY), "*_osc.dat")
    print(f"SPICE: {sorted(spice_arqs)}\nDLR:   {sorted(dlr_arqs)}\nnbody: {sorted(nbody_arqs)}")
    if not spice_arqs:
        raise SystemExit(f"nenhum elementos_*.csv em {os.path.join(args.base, PASTA_SPICE)}")

    for nome in sorted((set(dlr_arqs) | set(nbody_arqs)) - set(spice_arqs)):
        print(f"{nome}: tem resultado de código mas não tem CSV do SPICE, pulando")

    resumo = []
    for nome in sorted(spice_arqs):
        if nome not in dlr_arqs and nome not in nbody_arqs:
            print(f"\n{nome}: só tem SPICE, nada para comparar")
            continue
        pasta = os.path.join(args.base, PASTA_SAIDA, nome)
        os.makedirs(pasta, exist_ok=True)
        for rotulo, (dif, distancia) in compara_satelite(
                nome, spice_arqs[nome], dlr_arqs.get(nome), nbody_arqs.get(nome), pasta).items():
            resumo.append([nome, rotulo] + list(np.sqrt(np.mean(dif**2, axis=0)))
                          + [np.sqrt(np.mean(distancia**2)), distancia.max()])

    if resumo:
        arquivo = os.path.join(args.base, PASTA_SAIDA, "resumo.csv")
        with open(arquivo, "w") as f:
            f.write("satelite,codigo," + ",".join(f"rms_d{n}" for n in NOMES_CSV)
                    + ",rms_distancia_km,max_distancia_km\n")
            for linha in resumo:
                f.write(",".join(linha[:2]) + "," + ",".join(f"{v:.6g}" for v in linha[2:]) + "\n")
        print(f"\nResumo em {os.path.abspath(arquivo)}")
    print(f"Gráficos em {os.path.abspath(os.path.join(args.base, PASTA_SAIDA))}/<satelite>/")


if __name__ == "__main__":
    main()
