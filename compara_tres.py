"""
Compara três fontes para o mesmo satélite: SPICE, o outro integrador e o
nbody. Cada código é comparado com o SPICE nos seus próprios instantes.

Arquivos de entrada:
  integrador (ex.: oscelt_planetIAU.out), uma linha por instante:
    2030 JAN 01 00:00:00.000  ET  t  a  e  i  Ω  ω  M
  nbody (ex.: Metis_osc.dat), linhas '#' são cabeçalho:
    tempo(dias)  a(km)  e  inc(deg)  w(deg)  om(deg)  M(deg)
    (atenção: no nbody o ω vem ANTES do Ω)
O tempo do nbody é contado em dias a partir de NBODY_T0.

Gera em PASTA_SAIDA:
  LUA_tres_elementos.png    os 6 elementos das três fontes sobrepostos
  LUA_tres_diferencas.png   integrador − SPICE e nbody − SPICE
  LUA_tres_distancia.png    distância de cada código à posição do SPICE (km)
  LUA_integrador_vs_spice.csv e LUA_nbody_vs_spice.csv

Uso:
  python3 compara_tres.py                                    (usa a CONFIGURAÇÃO)
  python3 compara_tres.py --integrador oscelt_planetIAU.out --nbody Metis_osc.dat --lua METIS
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

from compara_spice import (ANGULARES, NOMES_CSV, ROTULOS, calcula_mu, carrega_kernels,
                           converte_angulos, diferencas, elementos_spice, formata_eixo,
                           identifica_lua, imprime_resumo, le_integrador, matriz_frame,
                           na_cobertura, posicoes)

# ======================= CONFIGURAÇÃO (edite aqui) =======================
# Pode ser um meta-kernel (.tm) ou uma lista de arquivos de kernel.
KERNELS = [r"/caminho/para/JUICE/kernels/mk/juice_crema_5_1_150la.tm"]
ARQUIVO_INTEGRADOR = r"oscelt_planetIAU.out"
ARQUIVO_NBODY = r"Metis_osc.dat"
LUA = "METIS"            # nome SPICE (METIS, ADRASTEA, ..., CALLISTO) ou código NAIF

# Instante do t = 0 do nbody: data UTC (texto, ex. "2030-01-01T00:00:00")
# ou ET em segundos TDB (número, ex. 946728069.184 — o "et =" do cabeçalho
# do .in). None = primeiro ET do arquivo do integrador.
NBODY_T0 = None

# "EQUADOR_JUPITER": frame inercial alinhado ao equador de Júpiter na época
#                    EPOCA_FRAME (o mesmo do elementos_spice.py)
# "J2000":           equador terrestre J2000 (ICRF)
FRAME = "EQUADOR_JUPITER"
# Época (UTC) em que o equador de Júpiter é congelado. None = primeiro ET do
# integrador. Use a mesma época em que as simulações começaram.
EPOCA_FRAME = None

# μ da conversão estado <-> elementos. None = GM de Júpiter (+ GM da lua, se
# SOMAR_GM_LUA) dos kernels. Deixe igual ao μ dos códigos.
MU_KM3S2 = None
SOMAR_GM_LUA = True

ANGULOS_INTEGRADOR = "auto"   # "graus", "radianos" ou "auto" (decide pelo M)
PASTA_SAIDA = "saida_comparacao"
# =========================================================================

CORES = {"SPICE": "k", "integrador": "C0", "nbody": "C1"}


def le_nbody(arquivo):
    """Lê o NOME_osc.dat do nbody. Devolve t (dias) e [a, e, i, Ω, ω, M] (graus)."""
    dados = np.loadtxt(arquivo, comments="#", ndmin=2)
    if dados.shape[1] < 7:
        raise ValueError(f"{arquivo}: esperava 7 colunas (t a e inc w om M)")
    t, a, e, inc, w, om, m = dados[:, :7].T
    return t, np.column_stack([a, e, inc, om % 360.0, w % 360.0, m % 360.0])


def et_de(valor):
    """Data UTC (texto) ou ET (número) -> ET."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return spice.str2et(valor)


def salva_csv(arquivo, ets, codigo, spc, dif, distancia):
    with open(arquivo, "w") as f:
        f.write("data_utc,et_tdb_s,"
                + ",".join(f"{n}_codigo" for n in NOMES_CSV) + ","
                + ",".join(f"{n}_spice" for n in NOMES_CSV) + ","
                + ",".join(f"d{n}" for n in NOMES_CSV) + ",distancia_km\n")
        for et, lc, ls, ld, dist in zip(ets, codigo, spc, dif, distancia):
            f.write(f"{spice.et2utc(et, 'ISOC', 3)},{et:.3f},"
                    + ",".join(f"{v:.10g}" for v in np.concatenate([lc, ls, ld]))
                    + f",{dist:.6g}\n")


def main():
    parser = argparse.ArgumentParser(description="Compara integrador, nbody e SPICE.")
    parser.add_argument("--integrador", default=ARQUIVO_INTEGRADOR)
    parser.add_argument("--nbody", default=ARQUIVO_NBODY)
    parser.add_argument("--lua", default=LUA, help="nome SPICE ou código NAIF")
    parser.add_argument("--t0", default=NBODY_T0,
                        help="t = 0 do nbody: data UTC ou ET (padrão: NBODY_T0)")
    args = parser.parse_args()

    carrega_kernels(KERNELS)
    id_lua, nome_lua = identifica_lua(args.lua)

    # --- integrador
    datas, ets_i, integ = le_integrador(args.integrador)
    print(f"integrador: {args.integrador}: {len(ets_i)} linhas, de {datas[0]} a {datas[-1]}")
    integ = converte_angulos(integ, ANGULOS_INTEGRADOR, "integrador")

    # --- nbody
    t_n, nbody = le_nbody(args.nbody)
    et0 = ets_i[0] if args.t0 is None else et_de(args.t0)
    ets_n = et0 + t_n * 86400.0
    print(f"nbody: {args.nbody}: {len(t_n)} linhas, t = 0 em "
          f"{spice.et2utc(et0, 'C', 3)} UTC (ET {et0:.3f})"
          + ("  [= primeiro ET do integrador]" if args.t0 is None else ""))

    ok = na_cobertura(ets_i, id_lua, f"{nome_lua} (integrador)")
    ets_i, integ = ets_i[ok], integ[ok]
    ok = na_cobertura(ets_n, id_lua, f"{nome_lua} (nbody)")
    ets_n, nbody = ets_n[ok], nbody[ok]

    et_frame = ets_i[0] if EPOCA_FRAME is None else spice.str2et(EPOCA_FRAME)
    R = matriz_frame(FRAME, et_frame)
    mu = calcula_mu(id_lua, MU_KM3S2, SOMAR_GM_LUA)
    print(f"μ = {mu:.6f} km³/s²")

    # --- cada código contra o SPICE nos seus próprios instantes
    resultados = {}
    for rotulo, ets, tabela in [("integrador", ets_i, integ), ("nbody", ets_n, nbody)]:
        spc, pos_spice = elementos_spice(ets, id_lua, R, mu)
        distancia = np.linalg.norm(posicoes(tabela, ets, mu) - pos_spice, axis=1)
        dif = diferencas(tabela, spc)
        resultados[rotulo] = (ets, tabela, spc, dif, distancia)
        imprime_resumo(f"{nome_lua}: {rotulo} − SPICE ({len(ets)} instantes)", dif, distancia)

    # SPICE em todos os instantes, para a curva de referência
    ets_todos = np.union1d(ets_i, ets_n)
    spc_todos, _ = elementos_spice(ets_todos, id_lua, R, mu)

    # ------------------------------ saída ---------------------------------
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    base = os.path.join(PASTA_SAIDA, nome_lua.lower())
    for rotulo, (ets, tabela, spc, dif, distancia) in resultados.items():
        salva_csv(f"{base}_{rotulo}_vs_spice.csv", ets, tabela, spc, dif, distancia)

    series = [("SPICE", ets_todos, spc_todos)] + \
             [(rotulo, r[0], r[1]) for rotulo, r in resultados.items()]

    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        for rotulo, ets, tabela in series:
            tempo = spice.et2datetime(ets)
            if j in ANGULARES:  # 0–360: pontos, para não ligar os saltos 360 -> 0
                eixo.plot(tempo, tabela[:, j], ".", ms=1.2, color=CORES[rotulo],
                          alpha=0.6, label=rotulo)
            else:
                eixo.plot(tempo, tabela[:, j], lw=0.7, color=CORES[rotulo],
                          alpha=0.8, label=rotulo)
        if j in ANGULARES:
            eixo.set_ylim(0, 360)
        eixo.set_ylabel(rot)
        formata_eixo(eixo)
    eixos[0].legend(loc="upper right", fontsize=8, markerscale=6)
    eixos[0].set_title(f"{nome_lua} – SPICE × integrador × nbody ({FRAME}, osculadores)")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(f"{base}_tres_elementos.png", dpi=150)
    plt.close(fig)

    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        for rotulo, (ets, _, _, dif, _) in resultados.items():
            eixo.plot(spice.et2datetime(ets), dif[:, j], lw=0.7, color=CORES[rotulo],
                      alpha=0.8, label=f"{rotulo} − SPICE")
        eixo.axhline(0.0, color="k", lw=0.5)
        eixo.set_ylabel("Δ" + rot)
        formata_eixo(eixo)
    eixos[0].legend(loc="upper right", fontsize=8)
    eixos[0].set_title(f"{nome_lua} – diferença em relação ao SPICE")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(f"{base}_tres_diferencas.png", dpi=150)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(11, 4))
    for rotulo, (ets, _, _, _, distancia) in resultados.items():
        eixo.plot(spice.et2datetime(ets), distancia, lw=0.7, color=CORES[rotulo],
                  label=f"{rotulo} − SPICE")
    eixo.set_ylabel("distância ao SPICE [km]")
    eixo.set_xlabel("data (UTC)")
    eixo.set_title(f"{nome_lua} – distância de cada código à posição do SPICE")
    eixo.legend(fontsize=8)
    formata_eixo(eixo)
    fig.tight_layout()
    fig.savefig(f"{base}_tres_distancia.png", dpi=150)
    plt.close(fig)

    print(f"\nArquivos salvos em {os.path.abspath(PASTA_SAIDA)}")
    spice.kclear()


if __name__ == "__main__":
    main()
