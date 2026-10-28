"""
Distância entre a JUICE e as luas internas de Júpiter, ao longo da missão.

"""

import os
import sys

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import spiceypy as spice

# CONFIG
# Pasta "kernels" da JUICE (a que tem ck/, fk/, lsk/, mk/, spk/ ...)
PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"

# juice_plan.tm = trajetória planejada para a missão inteira.
# Se quiser um cenário "crema" específico, troque o nome aqui
META_KERNEL = "juice_plan.tm"

# Período: None = usa TODO o período em que há dados (a missão inteira).
# Ou coloque datas, ex.: INICIO = "2031-07-01", FIM = "2035-12-31"
INICIO = "2031-03-01"
FIM = "2034-05-01"

PASSO_GRAFICO_S = 1800   # 1 ponto a cada 30 min no gráfico
PASSO_BUSCA_S = 600      # passo da busca da distância mínima (10 min)
BLOCO_BUSCA_S = 30 * 86400  # a busca do mínimo é feita em blocos de 30 dias

JUICE = "JUICE"          # código NAIF da JUICE: -28
LUAS = [                 # (nome SPICE, código NAIF), na ordem dos gráficos
    ("METIS", 516),
    ("ADRASTEA", 515),
    ("AMALTHEA", 505),
    ("THEBE", 514),
]
JUICE_ID = -28

PASTA_SAIDA = "resultados"
# =========================================================================


def carrega_meta_kernel(pasta_kernels, nome_mk):
    pasta_mk = os.path.join(pasta_kernels, "mk")
    caminho = os.path.join(pasta_mk, nome_mk)
    if not os.path.isfile(caminho):
        print(f"ERRO: meta-kernel não encontrado: {caminho}")
        if os.path.isdir(pasta_mk):
            print("Meta-kernels disponíveis em", pasta_mk)
            for f in sorted(os.listdir(pasta_mk)):
                if f.endswith(".tm"):
                    print("   ", f)
        sys.exit(1)

    pasta_original = os.getcwd()
    os.chdir(pasta_mk)
    try:
        spice.furnsh(nome_mk)
        arquivos = []
        for i in range(spice.ktotal("ALL")):
            arquivo, tipo, _, _ = spice.kdata(i, "ALL")
            if tipo != "META":
                arquivos.append(os.path.abspath(arquivo))
    finally:
        os.chdir(pasta_original)

    spice.kclear()
    for arquivo in arquivos:
        spice.furnsh(arquivo)
    print(f"Meta-kernel carregado: {caminho}")
    print(f"  ({len(arquivos)} kernels no total)")


def cobertura_spk(id_corpo):
    cobertura = spice.cell_double(200000)
    arquivos = []
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        objetos = spice.spkobj(arquivo)
        if id_corpo in [objetos[k] for k in range(spice.card(objetos))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
            arquivos.append(arquivo)
    return cobertura, arquivos


def imprime_janela(titulo, janela, max_linhas=10):
    n = spice.wncard(janela)
    print(f"  {titulo}: {n} intervalo(s)")
    for k in range(min(n, max_linhas)):
        a, b = spice.wnfetd(janela, k)
        print(f"    {spice.et2utc(a, 'C', 0)}  ->  {spice.et2utc(b, 'C', 0)}")
    if n > max_linhas:
        print(f"    ... (+{n - max_linhas} intervalos)")


def distancia_minima(nome_lua, janela):
    # espaço suficiente mesmo que houvesse um mínimo a cada passo de busca
    n_max = int(BLOCO_BUSCA_S / PASSO_BUSCA_S) + 10
    et_min, d_min = None, np.inf
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        inicio = a
        while inicio < b:
            fim = min(inicio + BLOCO_BUSCA_S, b)
            bloco = spice.cell_double(2)
            spice.wninsd(inicio, fim, bloco)
            resultado = spice.cell_double(2 * n_max)
            spice.gfdist(nome_lua, "NONE", JUICE, "ABSMIN", 0.0, 0.0,
                         PASSO_BUSCA_S, n_max, bloco, resultado)
            if spice.wncard(resultado) > 0:
                et, _ = spice.wnfetd(resultado, 0)
                pos, _ = spice.spkpos(nome_lua, et, "J2000", "NONE", JUICE)
                d = spice.vnorm(pos)
                if d < d_min:
                    et_min, d_min = et, d
            inicio = fim
    if et_min is None:
        return None, None
    return et_min, d_min


def serie_distancias(nome_lua, janela):
    """Calcula a distância em vários instantes dentro da janela.
    Entre intervalos sem dados coloca NaN para o gráfico não ligar os pontos."""
    ets_todos, dists_todos = [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        ets = np.append(np.arange(a, b, PASSO_GRAFICO_S), b)
        # spkpos: posição da LUA vista da JUICE, em km
        pos, _ = spice.spkpos(nome_lua, ets, "J2000", "NONE", JUICE)
        dists = np.linalg.norm(np.asarray(pos), axis=1)
        ets_todos.extend(ets)
        dists_todos.extend(dists)
        ets_todos.append(b)            # ponto NaN = "buraco" no gráfico
        dists_todos.append(np.nan)
    return np.array(ets_todos), np.array(dists_todos)


def grafico(eixo, nome, datas, dists, et_min, d_min):
    eixo.semilogy(datas, dists, lw=0.6)
    if et_min is not None:
        data_min = spice.et2datetime(et_min)
        eixo.plot(data_min, d_min, "ro", ms=5,
                  label=f"min: {d_min:,.0f} km em {data_min:%Y-%m-%d %H:%M} UTC")
        eixo.legend(loc="upper right", fontsize=8)
    eixo.set_title(f"Distance JUICE – {nome.capitalize()}")
    eixo.set_ylabel("distance [km] (log)")
    eixo.grid(True, which="both", alpha=0.3)
    eixo.margins(x=0)    # eixo x começa no 1º ponto e termina no último (sem folga)
    datas_eixo = mdates.AutoDateLocator()   # datas compactas, sem rótulos sobrepostos
    eixo.xaxis.set_major_locator(datas_eixo)
    eixo.xaxis.set_major_formatter(mdates.ConciseDateFormatter(datas_eixo))



def main():
    carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    print("\n=== JUICE ===")
    cob_juice, arq_juice = cobertura_spk(JUICE_ID)
    if not arq_juice:
        sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
    imprime_janela("cobertura da trajetória", cob_juice)

    # Janela pedida pelo usuário (ou a cobertura toda da JUICE)
    if INICIO or FIM:
        n = spice.wncard(cob_juice)
        et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
        et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
        janela_usuario = spice.cell_double(2)
        spice.wninsd(et0, et1, janela_usuario)
        cob_juice = spice.wnintd(cob_juice, janela_usuario)

    resultados = []
    for nome, id_lua in LUAS:
        print(f"\n=== {nome} (NAIF {id_lua}) ===")
        cob_lua, arq_lua = cobertura_spk(id_lua)
        if not arq_lua:
            print("  NENHUM SPK carregado tem a efeméride desta lua -> pulando.")
            print("  (Carregue um SPK de satélites de Júpiter, ex.: jup365.bsp)")
            continue

        # Só dá para calcular onde há dados da JUICE E da lua ao mesmo tempo
        janela = spice.wnintd(cob_juice, cob_lua)
        if spice.wncard(janela) == 0:
            print("  Sem período em comum com a JUICE -> pulando.")
            continue
        imprime_janela("período calculado", janela, max_linhas=3)

        ets, dists = serie_distancias(nome, janela)
        et_min, d_min = distancia_minima(nome, janela)
        if et_min is not None:
            print(f"  -> menor distância: {d_min:,.0f} km em "
                  f"{spice.et2utc(et_min, 'C', 0)} UTC")

        datas = spice.et2datetime(ets)

        # Gráfico individual
        fig, eixo = plt.subplots(figsize=(12, 4))
        grafico(eixo, nome, datas, dists, et_min, d_min)
        eixo.set_xlabel("date (UTC)")
        fig.tight_layout()
        arq_png = os.path.join(PASTA_SAIDA, f"distancia_juice_{nome.lower()}.png")
        fig.savefig(arq_png, dpi=150)


        # Tabela CSV (data UTC, distância em km)
        arq_csv = os.path.join(PASTA_SAIDA, f"distancia_juice_{nome.lower()}.csv")
        with open(arq_csv, "w") as f:
            f.write("data_utc,distancia_km\n")
            for et, d in zip(ets, dists):
                if np.isfinite(d):
                    f.write(f"{spice.et2utc(et, 'ISOC', 0)},{d:.3f}\n")

        resultados.append((nome, datas, dists, et_min, d_min))

    # Gráfico com as 4 luas, uma embaixo da outra
    if resultados:
        fig, eixos = plt.subplots(len(resultados), 1, sharex=True,
                                  figsize=(12, 3 * len(resultados)), squeeze=False)
        for eixo, (nome, datas, dists, et_min, d_min) in zip(eixos[:, 0], resultados):
            grafico(eixo, nome, datas, dists, et_min, d_min)
        eixos[-1, 0].set_xlabel("data (UTC)")
        fig.tight_layout()
        arq_png = os.path.join(PASTA_SAIDA, "distancia_juice_todas.png")
        fig.savefig(arq_png, dpi=150)

    spice.kclear()


if __name__ == "__main__":
    main()
