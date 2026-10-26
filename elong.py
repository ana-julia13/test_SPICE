"""
Elongação solar das luas internas de Júpiter vistas da JUICE, e a distância
JUICE–lua no mesmo período.

  elongação = ângulo Sol–JUICE–lua (vértice na JUICE): quão longe do Sol a
              câmera teria que apontar para ver a lua. Pequena = perto do Sol.

Carrega os kernels igual ao dist_juice_luas_V2.py (juice_plan.tm).

Gera (na pasta PASTA_SAIDA):
  elongacao_tempo.png            elongação × tempo (um quadro por lua)
  distancia_elongacao_tempo.png  distância e elongação × tempo, por lua

Uso:
    python3 dist_elong.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # só salva em arquivo, não abre janela
import matplotlib.pyplot as plt
import spiceypy as spice

# ======================= CONFIGURAÇÃO (edite aqui) =======================
# Pasta "kernels" da JUICE (a que tem ck/, fk/, lsk/, mk/, spk/ ...)
PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL = "juice_plan.tm"   # fica dentro de PASTA_KERNELS/mk

# Período: None = usa todo o período em que há dados.
# Ou datas, ex.: INICIO = "2031-07-01", FIM = "2035-12-31"
INICIO = "2032-06-01"
FIM = "2032-09-01"
PASSO_S = 1800           # 1 ponto a cada 30 min

JUICE = "JUICE"
JUICE_ID = -28
LUAS = [                 # (nome SPICE, código NAIF), na ordem dos gráficos
    ("METIS", 516),
    ("ADRASTEA", 515),
    ("AMALTHEA", 505),
    ("THEBE", 514),
]
LIMITE_SOL = 30          # graus: elongação mínima (EXEMPLO, confirme com a equipe)

PASTA_SAIDA = "elong"
# =========================================================================


def carrega_meta_kernel(pasta_kernels, nome_mk):
    """Os meta-kernels da JUICE usam PATH_VALUES = ( '..' ): os caminhos são
    relativos à pasta mk/. Por isso entramos na pasta mk/ antes do furnsh."""
    pasta_mk = os.path.join(pasta_kernels, "mk")
    caminho = os.path.join(pasta_mk, nome_mk)
    if not os.path.isfile(caminho):
        print(f"ERRO: meta-kernel não encontrado: {caminho}")
        sys.exit(1)

    pasta_original = os.getcwd()
    os.chdir(pasta_mk)
    try:
        spice.furnsh(nome_mk)
    finally:
        os.chdir(pasta_original)
    print(f"Meta-kernel carregado: {caminho}")
    return pasta_mk


def cobertura_spk(id_corpo, pasta_mk):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        # caminhos do meta-kernel são relativos à pasta mk/ (ex.: ../spk/x.bsp)
        arquivo = os.path.normpath(os.path.join(pasta_mk, arquivo))
        objetos = spice.spkobj(arquivo)
        if id_corpo in [objetos[k] for k in range(spice.card(objetos))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


def angulo_entre(v1, v2):
    """Ângulo (graus) entre vetores linha a linha, como o vsep do SPICE."""
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def distancia_e_elongacao(nome_lua, janela):
    """Calcula, a cada PASSO_S segundos dentro da janela, a distância
    JUICE–lua (km) e a elongação solar da lua vista da JUICE (graus).
    Entre intervalos sem dados coloca NaN para o gráfico não ligar os pontos."""
    ets_todos, dists_todos, elongs_todos = [], [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.append(np.arange(a, b, PASSO_S), b)

        # distância: posição geométrica da lua vista da JUICE (igual ao V2)
        pos, _ = spice.spkpos(nome_lua, ets, "J2000", "NONE", JUICE)
        dists = np.linalg.norm(np.asarray(pos), axis=1)

        # elongação: ângulo entre as direções da lua e do Sol, vistas da JUICE
        # ("LT+S" = direção aparente, como a câmera veria)
        p_lua, _ = spice.spkpos(nome_lua, ets, "J2000", "LT+S", JUICE)
        p_sol, _ = spice.spkpos("SUN", ets, "J2000", "LT+S", JUICE)
        elongs = angulo_entre(np.asarray(p_lua), np.asarray(p_sol))

        ets_todos.extend(ets)
        dists_todos.extend(dists)
        elongs_todos.extend(elongs)
        ets_todos.append(b)            # ponto NaN = "buraco" no gráfico
        dists_todos.append(np.nan)
        elongs_todos.append(np.nan)
    return np.array(ets_todos), np.array(dists_todos), np.array(elongs_todos)


def grafico_elongacao(eixo, datas, elongs):
    eixo.plot(datas, elongs, lw=0.6, color="tab:green")
    eixo.axhline(LIMITE_SOL, color="gray", ls="--", lw=0.8,
                 label=f"limite {LIMITE_SOL}°")
    eixo.set_ylim(0, 180)
    eixo.set_ylabel("elongação [°]")
    eixo.legend(loc="upper right", fontsize=8)
    eixo.grid(True, alpha=0.3)


def grafico_distancia(eixo, datas, dists):
    eixo.semilogy(datas, dists, lw=0.6, color="tab:blue")
    eixo.set_ylabel("distância [km]")
    eixo.grid(True, which="both", alpha=0.3)


def main():
    pasta_mk = carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # Período: o pedido (INICIO/FIM) ou a cobertura toda da JUICE
    cob_juice = cobertura_spk(JUICE_ID, pasta_mk)
    if spice.wncard(cob_juice) == 0:
        sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
    if INICIO or FIM:
        n = spice.wncard(cob_juice)
        et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
        et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
        janela_usuario = spice.cell_double(2)
        spice.wninsd(et0, et1, janela_usuario)
        cob_juice = spice.wnintd(cob_juice, janela_usuario)

    resultados = []   # (nome, datas, distâncias, elongações)
    for nome, id_lua in LUAS:
        janela = spice.wnintd(cob_juice, cobertura_spk(id_lua, pasta_mk))
        if spice.wncard(janela) == 0:
            print(f"{nome}: sem cobertura em comum com a JUICE no período -> pulando")
            continue

        ets, dists, elongs = distancia_e_elongacao(nome, janela)
        datas = spice.et2datetime(ets)
        i_min = np.nanargmin(elongs)
        print(f"{nome}: elongação {np.nanmin(elongs):.0f}°–{np.nanmax(elongs):.0f}° "
              f"(mínima em {spice.et2utc(ets[i_min], 'C', 0)} UTC) | "
              f"distância {np.nanmin(dists):,.0f}–{np.nanmax(dists):,.0f} km")
        resultados.append((nome, datas, dists, elongs))

    spice.kclear()

    if not resultados:
        sys.exit("Nenhuma lua com dados no período: nenhum gráfico gerado.")
    n = len(resultados)
    # eixo x do primeiro ao último ponto de todas as luas (sem espaço em branco)
    inicio = min(r[1][0] for r in resultados)
    fim = max(r[1][-1] for r in resultados)

    # ---- elongação × tempo ----
    fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
    for eixo, (nome, datas, dists, elongs) in zip(eixos[:, 0], resultados):
        grafico_elongacao(eixo, datas, elongs)
        eixo.set_title(f"Elongação solar de {nome.capitalize()} vista da JUICE")
    eixos[-1, 0].set_xlim(inicio, fim)
    eixos[-1, 0].set_xlabel("data (UTC)")
    fig.tight_layout()
    arq = os.path.join(PASTA_SAIDA, "elongacao_tempo.png")
    fig.savefig(arq, dpi=150)
    print(f"\ngráfico salvo: {arq}")

    # ---- distância e elongação × tempo (por lua: distância em cima, elongação embaixo) ----
    fig, eixos = plt.subplots(2 * n, 1, figsize=(12, 2.5 * 2 * n), sharex=True, squeeze=False)
    for k, (nome, datas, dists, elongs) in enumerate(resultados):
        ax_d, ax_e = eixos[2 * k, 0], eixos[2 * k + 1, 0]
        grafico_distancia(ax_d, datas, dists)
        ax_d.set_title(f"{nome.capitalize()}: distância e elongação")
        grafico_elongacao(ax_e, datas, elongs)
    eixos[-1, 0].set_xlim(inicio, fim)
    eixos[-1, 0].set_xlabel("data (UTC)")
    fig.tight_layout()
    arq = os.path.join(PASTA_SAIDA, "distancia_elongacao_tempo.png")
    fig.savefig(arq, dpi=150)
    print(f"gráfico salvo: {arq}")


if __name__ == "__main__":
    main()
