"""
Distância entre a JUICE e Júpiter.

1. Carrega o meta-kernel da JUICE
2. Recorta o período pedido à cobertura da trajetória da JUICE
3. Calcula a distância JUICE–Júpiter ao longo do tempo
4. Acha os perijovos (mínimos locais) com o Geometry Finder (gfdist)
5. Salva um gráfico (PNG) e uma tabela (CSV) na pasta de resultados
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

# ======================= CONFIGURAÇÃO (edite aqui) =======================
META_KERNEL = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO = "2031-07-01"    # início do período de interesse (UTC)
FIM = "2035-10-13"       # fim do período de interesse (UTC)
PASSO_S = 600            # passo do gráfico em segundos (600 s = 10 min)
PASSO_BUSCA_S = 6 * 3600 # passo da busca de perijovos (menor que meia órbita)
PASTA_SAIDA = "resultados"

JUICE = "JUICE"
JUICE_ID = -28
ALVO = "JUPITER"
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
        _, raios = spice.bodvrd(ALVO, "RADII", 3)
        return raios[0]
    except SpiceyError:
        return 71492.0   # raio equatorial de Júpiter [km]


def perijovos(janela):
    """Instantes de mínimo local da distância JUICE–Júpiter dentro da janela."""
    max_intervalos = 200000
    resultado = spice.cell_double(2 * max_intervalos)
    spice.gfdist(ALVO, "NONE", JUICE, "LOCMIN", 0.0, 0.0,
                 PASSO_BUSCA_S, max_intervalos, janela, resultado)
    return [spice.wnfetd(resultado, k)[0] for k in range(spice.wncard(resultado))]


def distancia(ets):
    pos, _ = spice.spkpos(ALVO, ets, "J2000", "NONE", JUICE)
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

    # Perijovos
    ets_peri = perijovos(janela)
    d_peri = distancia(ets_peri) if ets_peri else np.array([])
    datas_peri = [spice.et2datetime(et) for et in ets_peri]

    print(f"\n{len(ets_peri)} perijovo(s) encontrado(s):")
    for data, d in zip(datas_peri, d_peri):
        print(f"  {data:%Y-%m-%d %H:%M} UTC   {d:14,.0f} km   {d / rj:6.2f} RJ")

    i_min = int(np.argmin(dists))
    print(f"\nDistância mínima no período: {dists[i_min]:,.0f} km "
          f"({dists[i_min] / rj:.2f} RJ) em {datas[i_min]:%Y-%m-%d %H:%M} UTC")

    # Saídas
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    caminho_csv = os.path.join(PASTA_SAIDA, "distancia_juice_jupiter.csv")
    with open(caminho_csv, "w") as f:
        f.write("data_utc,distancia_km,distancia_rj\n")
        for data, d in zip(datas, dists):
            f.write(f"{data:%Y-%m-%dT%H:%M:%S},{d:.3f},{d / rj:.5f}\n")

    fig, eixo = plt.subplots(figsize=(12, 5))
    eixo.semilogy(datas, dists / rj, lw=0.6, label="JUICE–Júpiter")
    if ets_peri:
        eixo.plot(datas_peri, d_peri / rj, "ro", ms=3, label="perijovos")
    eixo.set_title(f"Distância JUICE–Júpiter  (mín: {dists[i_min] / rj:.2f} RJ)")
    eixo.set_xlabel("data (UTC)")
    eixo.set_ylabel("distância [raios de Júpiter]")
    eixo.grid(True, which="both", alpha=0.3)
    eixo.legend()
    plt.tight_layout()

    caminho_png = os.path.join(PASTA_SAIDA, "distancia_juice_jupiter.png")
    plt.savefig(caminho_png, dpi=150)
    print(f"\nGráfico salvo em: {caminho_png}")
    print(f"Tabela salva em:  {caminho_csv}")
    plt.show()

    spice.kclear()


if __name__ == "__main__":
    main()
