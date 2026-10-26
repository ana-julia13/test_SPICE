"""
Distância entre a JUICE e as luas internas de Júpiter (Metis, Adrastea, Amalthea, Thebe).

1. Carrega um meta-kernel da JUICE
2. Verifica QUAIS arquivos SPK têm dados de cada lua e da JUICE, e em que período
3. Calcula a distância JUICE–lua ao longo do tempo
4. Faz um gráfico por lua e mostra a aproximação mínima de cada uma
"""

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

# ======================= CONFIGURAÇÃO (edite aqui) =======================
META_KERNEL = r"/caminho/para/JUICE/kernels/mk/juice_crema_5_1_150la.tm"
INICIO = "2031-07-01"    # início do período de interesse (UTC)
FIM = "2035-12-31"       # fim do período de interesse (UTC)
PASSO_S = 600            # passo de amostragem em segundos (600 s = 10 min)

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
# =========================================================================


def ids_na_celula(celula):
    return [celula[i] for i in range(spice.card(celula))]


def cobertura_spk(id_corpo):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(20000)
    arquivos = []
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        if id_corpo in ids_na_celula(spice.spkobj(arquivo)):
            spice.spkcov(arquivo, id_corpo, cobertura)
            arquivos.append(arquivo)
    return cobertura, arquivos


def imprime_cobertura(nome, cobertura, arquivos):
    print(f"\n=== {nome} ===")
    if not arquivos:
        print("  NENHUM SPK carregado contém este corpo!")
        return
    for a in arquivos:
        print(f"  arquivo: {a}")
    for k in range(spice.wncard(cobertura)):
        a, b = spice.wnfetd(cobertura, k)
        print(f"  cobertura: {spice.et2utc(a, 'C', 0)}  ->  {spice.et2utc(b, 'C', 0)}")


spice.furnsh(META_KERNEL)

# Janela de interesse definida pelo usuário
janela = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), janela)

# Cobertura da JUICE
cob_juice, arq_juice = cobertura_spk(JUICE_ID)
imprime_cobertura("JUICE", cob_juice, arq_juice)

fig, eixos = plt.subplots(len(LUAS), 1, figsize=(12, 3 * len(LUAS)), sharex=True)

for eixo, (nome, id_lua) in zip(eixos, LUAS.items()):
    cob_lua, arq_lua = cobertura_spk(id_lua)
    imprime_cobertura(nome, cob_lua, arq_lua)

    # Período em que JUICE e a lua têm dados E que está dentro da janela pedida
    comum = spice.wnintd(spice.wnintd(cob_juice, cob_lua), janela)
    if spice.wncard(comum) == 0:
        eixo.set_title(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, dists = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        pos, _ = spice.spkpos(nome, ets, "J2000", "NONE", "JUICE")
        d = np.linalg.norm(pos, axis=1)          # km
        datas.extend(spice.et2datetime(ets))
        dists.extend(d)
        datas.append(datas[-1])                   # NaN separa lacunas no gráfico
        dists.append(np.nan)

    dists = np.array(dists)
    i_min = np.nanargmin(dists)
    print(f"  -> distância mínima: {dists[i_min]:,.0f} km em {datas[i_min]:%Y-%m-%d %H:%M} UTC")

    eixo.semilogy(datas, dists, lw=0.6)
    eixo.plot(datas[i_min], dists[i_min], "ro", ms=4)
    eixo.set_title(f"JUICE – {nome}  (mín: {dists[i_min]:,.0f} km)")
    eixo.set_ylabel("distância [km]")
    eixo.grid(True, which="both", alpha=0.3)

eixos[-1].set_xlabel("data (UTC)")
plt.tight_layout()
plt.savefig("distancia_juice_luas.png", dpi=150)
plt.show()

spice.kclear()
