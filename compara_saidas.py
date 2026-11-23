"""
Compara duas saídas do SPICE olhando: para cada CSV que existe nas pastas
1 e 2, plota as colunas das duas saídas no MESMO gráfico (1 = linha azul,
2 = linha laranja tracejada), uma coluna por painel.

O eixo x é a 1ª coluna do CSV (data_utc, nome da lua, ...). Linhas com
comentário (# ...) são ignoradas.

Uso:
  python compara_saidas.py                # pastas 1 e 2
  python compara_saidas.py runA runB      # outras pastas
  python compara_saidas.py --mostrar      # também abre as janelas dos gráficos

Os PNG vão para a pasta comparacao_1_vs_2/ (mude com --saida).
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime

import matplotlib


def le_csv(caminho):
    """Retorna (cabecalho, linhas) pulando as linhas de comentário '#'."""
    with open(caminho, newline="", encoding="utf-8", errors="replace") as f:
        linhas = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    tabela = [[c.strip() for c in l] for l in csv.reader(linhas)]
    if not tabela:
        return [], []
    return tabela[0], tabela[1:]


def para_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def para_data(s):
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


def eixo_x(valores):
    """Datas se a 1ª coluna for data; números se for número; senão, texto."""
    datas = [para_data(v) for v in valores]
    if valores and all(datas):
        return datas, "data (UTC)"
    nums = [para_float(v) for v in valores]
    if valores and not any(math.isnan(n) for n in nums):
        return nums, None
    return valores, None


def plota(nome, p1, p2, rotulo1, rotulo2, pasta_saida):
    import matplotlib.pyplot as plt

    cab1, l1 = le_csv(p1)
    cab2, l2 = le_csv(p2)
    if not cab1 or not cab2:
        print(f"[{nome}] vazio, pulando")
        return
    x1, rot_x = eixo_x([l[0] for l in l1])
    x2, _ = eixo_x([l[0] for l in l2])

    colunas = []
    for c in cab1[1:]:
        if c not in cab2:
            continue
        if rot_x and c.lower().startswith("et_"):   # et é a mesma coisa que a data
            continue
        j1, j2 = cab1.index(c), cab2.index(c)
        y1 = [para_float(l[j1]) if j1 < len(l) else math.nan for l in l1]
        y2 = [para_float(l[j2]) if j2 < len(l) else math.nan for l in l2]
        if sum(not math.isnan(v) for v in y1) == 0:
            continue                                 # coluna de texto
        colunas.append((c, y1, y2))
    if not colunas:
        print(f"[{nome}] sem colunas numéricas em comum, pulando")
        return

    poucos = max(len(l1), len(l2)) <= 50
    fig, eixos = plt.subplots(len(colunas), 1, sharex=True, squeeze=False,
                              figsize=(12, 2.3 * len(colunas) + 1))
    for eixo, (c, y1, y2) in zip(eixos[:, 0], colunas):
        if poucos:
            eixo.plot(x1, y1, "o-", ms=5, lw=1.0, label=rotulo1)
            eixo.plot(x2, y2, "x--", ms=7, lw=1.0, label=rotulo2)
        else:
            eixo.plot(x1, y1, "-", lw=1.0, label=rotulo1)
            eixo.plot(x2, y2, "--", lw=1.0, label=rotulo2)
        eixo.set_ylabel(c)
        eixo.grid(alpha=0.3)
    eixos[0, 0].legend(loc="upper right")
    eixos[0, 0].set_title(f"{nome}: {rotulo1} x {rotulo2}")
    eixos[-1, 0].set_xlabel(rot_x or cab1[0])
    if not rot_x and isinstance(x1[0], str):
        plt.setp(eixos[-1, 0].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    arq = os.path.join(pasta_saida,
                       os.path.splitext(nome)[0].replace(os.sep, "_") + ".png")
    fig.savefig(arq, dpi=130)
    print(f"[{nome}] gráfico salvo: {arq}")


def lista_csv(pasta):
    arqs = set()
    for raiz, _, nomes in os.walk(pasta):
        for n in nomes:
            if n.lower().endswith(".csv"):
                arqs.add(os.path.relpath(os.path.join(raiz, n), pasta))
    return arqs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta1", nargs="?", default="1")
    ap.add_argument("pasta2", nargs="?", default="2")
    ap.add_argument("--saida", default="comparacao_1_vs_2")
    ap.add_argument("--mostrar", action="store_true",
                    help="abre as janelas dos gráficos no final")
    a = ap.parse_args()

    if not a.mostrar:
        matplotlib.use("Agg")   # só salva PNG, sem abrir janela
    import matplotlib.pyplot as plt

    for p in (a.pasta1, a.pasta2):
        if not os.path.isdir(p):
            sys.exit(f"ERRO: pasta '{p}' não encontrada")
    os.makedirs(a.saida, exist_ok=True)

    f1, f2 = lista_csv(a.pasta1), lista_csv(a.pasta2)
    for n in sorted(f1 - f2):
        print(f"só em {a.pasta1}: {n}")
    for n in sorted(f2 - f1):
        print(f"só em {a.pasta2}: {n}")

    for nome in sorted(f1 & f2):
        plota(nome, os.path.join(a.pasta1, nome), os.path.join(a.pasta2, nome),
              a.pasta1, a.pasta2, a.saida)

    if a.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
