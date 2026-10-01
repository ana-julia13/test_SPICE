"""
Distância entre a JUICE e um corpo (Júpiter, Thebe, Metis, ...).

1. Carrega o meta-kernel da JUICE
2. Recorta o período pedido à cobertura da trajetória da JUICE
3. Calcula a distância JUICE–ALVO ao longo do tempo
4. Acha as aproximações (mínimos locais) com o Geometry Finder (gfdist)
5. Salva um gráfico (PNG) e uma tabela (CSV) na pasta de resultados,
   com um zoom em volta da aproximação mais próxima
"""

import os
import re

import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

# ======================= CONFIGURAÇÃO (edite aqui) =======================
META_KERNEL = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO = "2031-07-01"    # início do período de interesse (UTC)
FIM = "2035-10-13"       # fim do período de interesse (UTC)
PASSO_S = 600            # passo do gráfico em segundos (600 s = 10 min)
PASSO_BUSCA_S = 2 * 3600 # passo da busca de mínimos (menor que meio período
                         # orbital do alvo: Thebe ~16 h, Metis ~7 h)
ZOOM_DIAS = 2            # meia-largura do zoom em volta da menor distância
N_LISTA = 10             # quantas aproximações mais próximas mostrar
PASTA_SAIDA = "resultados"

JUICE = "JUICE"
JUICE_ID = -28
ALVO = "THEBE"           # "JUPITER", "THEBE", "METIS", "ADRASTEA", "AMALTHEA"...
# =========================================================================


def cobertura_spk(id_corpo):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(20000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        ids = spice.spkobj(arquivo)
        if id_corpo in [ids[k] for k in range(spice.card(ids))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


def raio_jupiter():
    try:
        _, raios = spice.bodvrd("JUPITER", "RADII", 3)
        return raios[0]
    except SpiceyError:
        return 71492.0   # raio equatorial de Júpiter [km]


def minimos_locais(janela):
    """Instantes de mínimo local da distância JUICE–ALVO dentro da janela."""
    max_intervalos = 200000
    resultado = spice.cell_double(2 * max_intervalos)
    spice.gfdist(ALVO, "NONE", JUICE, "LOCMIN", 0.0, 0.0,
                 PASSO_BUSCA_S, max_intervalos, janela, resultado)
    return [spice.wnfetd(resultado, k)[0] for k in range(spice.wncard(resultado))]


def distancia(ets, alvo=ALVO):
    pos, _ = spice.spkpos(alvo, ets, "J2000", "NONE", JUICE)
    return np.linalg.norm(np.atleast_2d(pos), axis=1)


def carrega_meta_kernel(caminho):
    """Lê o KERNELS_TO_LOAD do meta-kernel e carrega um por um, pulando os
    arquivos que não existem no disco (em vez de parar no primeiro que falta).
    Caminhos relativos (../ck/...) são resolvidos a partir da pasta do .tm."""
    pasta_mk = os.path.dirname(os.path.abspath(caminho))
    with open(caminho) as f:
        texto = f.read()

    # Só o que está entre \begindata e \begintext é dado
    dados = " ".join(re.findall(r"\\begindata(.*?)(?=\\begintext|$)", texto, re.S))
    variaveis = {}
    for nome, valor in re.findall(r"(\w+)\s*\+?=\s*(\([^)]*\)|'[^']*')", dados):
        variaveis.setdefault(nome, []).extend(re.findall(r"'([^']*)'", valor))

    simbolos = dict(zip(variaveis.get("PATH_SYMBOLS", []),
                        variaveis.get("PATH_VALUES", [])))

    faltando = []
    for kernel in variaveis.get("KERNELS_TO_LOAD", []):
        for simbolo, valor in simbolos.items():
            kernel = kernel.replace("$" + simbolo, valor)
        if not os.path.isabs(kernel):
            kernel = os.path.normpath(os.path.join(pasta_mk, kernel))
        if os.path.isfile(kernel):
            spice.furnsh(kernel)
        else:
            faltando.append(kernel)

    if faltando:
        print(f"AVISO: {len(faltando)} kernel(s) do meta-kernel não existem e foram pulados:")
        for kernel in faltando:
            print(f"  {kernel}")


def main():
    carrega_meta_kernel(META_KERNEL)
    print(f"Meta-kernel carregado: {META_KERNEL}")
    print(f"  ({spice.ktotal('ALL')} kernels no total)")

    pedido = spice.cell_double(2)
    spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), pedido)
    janela = spice.wnintd(cobertura_spk(JUICE_ID), pedido)
    if spice.wncard(janela) == 0:
        print("A trajetória da JUICE não cobre o período pedido.")
        return

    print("\nPeríodo calculado:")
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        print(f"  {spice.et2utc(a, 'C', 0)}  ->  {spice.et2utc(b, 'C', 0)}")

    rj = raio_jupiter()

    # Distância ao longo do tempo
    datas, dists = [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.arange(a, b, PASSO_S)
        datas.extend(spice.et2datetime(ets))
        dists.extend(distancia(ets))
    dists = np.array(dists)

    # Mínimos locais (para Júpiter são os perijovos); mostra os mais próximos
    ets_min = minimos_locais(janela)
    d_min = distancia(ets_min) if ets_min else np.array([])
    ordem = np.argsort(d_min)[:N_LISTA]
    ets_top = [ets_min[i] for i in ordem]
    datas_top = [spice.et2datetime(et) for et in ets_top]

    print(f"\n{len(ets_min)} mínimo(s) local(is) de distância encontrado(s).")
    print(f"As {len(ordem)} aproximações mais próximas de {ALVO}:")
    for data, d in zip(datas_top, d_min[ordem]):
        print(f"  {data:%Y-%m-%d %H:%M} UTC   {d:14,.0f} km   {d / rj:6.2f} RJ")

    i_min = int(np.argmin(dists))
    et_mais_perto = ets_top[0] if ets_top else spice.datetime2et(datas[i_min])
    d_mais_perto = d_min[ordem[0]] if ets_top else dists[i_min]
    print(f"\nDistância mínima no período: {d_mais_perto:,.0f} km "
          f"({d_mais_perto / rj:.2f} RJ) em {spice.et2utc(et_mais_perto, 'C', 0)} UTC")

    # Saídas
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    nome = ALVO.lower()

    caminho_csv = os.path.join(PASTA_SAIDA, f"distancia_juice_{nome}.csv")
    with open(caminho_csv, "w") as f:
        f.write("data_utc,distancia_km,distancia_rj\n")
        for data, d in zip(datas, dists):
            f.write(f"{data:%Y-%m-%dT%H:%M:%S},{d:.3f},{d / rj:.5f}\n")

    fig, (eixo, zoom) = plt.subplots(2, 1, figsize=(12, 9))
    eixo.semilogy(datas, dists / rj, lw=0.6, label=f"JUICE–{ALVO}")
    if ets_top:
        eixo.plot(datas_top, d_min[ordem] / rj, "ro", ms=4,
                  label=f"{len(ordem)} aproximações mais próximas")
    eixo.set_title(f"Distância JUICE–{ALVO}  (mín: {d_mais_perto / rj:.2f} RJ = "
                   f"{d_mais_perto:,.0f} km)")
    eixo.set_xlabel("data (UTC)")
    eixo.set_ylabel("distância [raios de Júpiter]")
    eixo.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    eixo.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    eixo.grid(True, which="both", alpha=0.3)
    eixo.legend()

    # Zoom de ±ZOOM_DIAS em volta da menor distância (amostragem de 2 min)
    ets_zoom = np.arange(et_mais_perto - ZOOM_DIAS * 86400,
                         et_mais_perto + ZOOM_DIAS * 86400, 120)
    ets_zoom = np.array([et for et in ets_zoom if spice.wnelmd(et, janela)])
    horas = (ets_zoom - et_mais_perto) / 3600
    zoom.plot(horas, distancia(ets_zoom) / 1000, lw=1, label=f"JUICE–{ALVO}")
    if ALVO != "JUPITER":
        zoom.plot(horas, distancia(ets_zoom, "JUPITER") / 1000, lw=1, ls="--",
                  color="gray", label="JUICE–JUPITER (referência)")
    zoom.plot(0, d_mais_perto / 1000, "ro", ms=5)
    zoom.set_title(f"Zoom de ±{ZOOM_DIAS} dias em volta da menor distância "
                   f"({spice.et2utc(et_mais_perto, 'C', 0)} UTC)")
    zoom.set_xlabel("horas em relação à menor distância")
    zoom.set_ylabel("distância [mil km]")
    zoom.grid(True, alpha=0.3)
    zoom.legend()
    plt.tight_layout()

    caminho_png = os.path.join(PASTA_SAIDA, f"distancia_juice_{nome}.png")
    plt.savefig(caminho_png, dpi=150)
    print(f"\nGráfico salvo em: {caminho_png}")
    print(f"Tabela salva em:  {caminho_csv}")
    plt.show()

    spice.kclear()


if __name__ == "__main__":
    main()
