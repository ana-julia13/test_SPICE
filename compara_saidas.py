"""
Compara duas saídas do SPICE entre si (pastas 1 e 2 com os mesmos arquivos).

Para cada arquivo que existe nas duas pastas:
  * .csv  -> alinha as linhas pela 1ª coluna (data_utc, nome, inicio_utc...)
             e, para cada coluna numérica, calcula a diferença 2 - 1:
             máx |Δ|, RMS, máx |Δ| relativo e onde ocorreu o máximo.
             Colunas de ângulo (_deg, graus, fase) têm Δ reduzido a [-180, 180)
             para não acusar 359.9° vs 0.1° como diferença de 359.8°.
             As linhas de comentário (# ...) também são comparadas.
  * outros arquivos de texto -> diz quantas linhas diferem
  * binários (.png etc.) -> só diz se são idênticos ou não

Gera:
  <saida>/resumo.csv          uma linha por (arquivo, coluna) com as estatísticas
  <saida>/delta_<arquivo>.png Δ de cada coluna no tempo (se houver matplotlib)

Uso:
  python compara_saidas.py                   # compara as pastas 1 e 2
  python compara_saidas.py runA runB         # outras pastas
  python compara_saidas.py --tol 1e-8        # tolerância relativa p/ "igual"
  python compara_saidas.py --sem-graficos
"""

import argparse
import csv
import difflib
import filecmp
import math
import os
import sys
from datetime import datetime

import numpy as np

EXT_TEXTO = {".txt", ".dat", ".in", ".out", ".log", ".tm", ".tab"}


# ----------------------------------------------------------------- leitura
def le_csv(caminho):
    """Retorna (comentarios, cabecalho, linhas). Linhas '#' viram comentários."""
    comentarios, linhas = [], []
    with open(caminho, newline="", encoding="utf-8", errors="replace") as f:
        for linha in f:
            if linha.lstrip().startswith("#"):
                comentarios.append(linha.rstrip("\n"))
            elif linha.strip():
                linhas.append(linha)
    tabela = list(csv.reader(linhas))
    if not tabela:
        return comentarios, [], []
    cab = [c.strip() for c in tabela[0]]
    return comentarios, cab, [[c.strip() for c in l] for l in tabela[1:]]


def para_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def eh_angulo(nome):
    n = nome.lower()
    return n.endswith("_deg") or "graus" in n or "fase" in n or n.endswith("deg")


def tenta_data(s):
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


# --------------------------------------------------------------- comparação
def alinha(cab, l1, l2):
    """Alinha as linhas pela 1ª coluna se ela for uma chave única não numérica;
    caso contrário, alinha pela posição."""
    chaves1 = [l[0] for l in l1 if l]
    chaves2 = [l[0] for l in l2 if l]
    chave_ok = (cab and chaves1 and math.isnan(para_float(chaves1[0]))
                and len(set(chaves1)) == len(chaves1)
                and len(set(chaves2)) == len(chaves2))
    if not chave_ok:
        n = min(len(l1), len(l2))
        so1 = [f"linha {i + 2}" for i in range(n, len(l1))]
        so2 = [f"linha {i + 2}" for i in range(n, len(l2))]
        return [str(i + 2) for i in range(n)], l1[:n], l2[:n], so1, so2, False
    d1 = {l[0]: l for l in l1 if l}
    d2 = {l[0]: l for l in l2 if l}
    comuns = [k for k in chaves1 if k in d2]
    so1 = [k for k in chaves1 if k not in d2]
    so2 = [k for k in chaves2 if k not in d1]
    return comuns, [d1[k] for k in comuns], [d2[k] for k in comuns], so1, so2, True


def compara_csv(nome, p1, p2, tol, pasta_graf):
    com1, cab1, l1 = le_csv(p1)
    com2, cab2, l2 = le_csv(p2)
    resultados = []

    if com1 != com2:
        print("  comentários (#) diferem:")
        for c in difflib.unified_diff(com1, com2, "1", "2", lineterm="", n=0):
            if not c.startswith(("---", "+++", "@@")):
                print("    " + c)
    if cab1 != cab2:
        print(f"  ATENÇÃO: cabeçalhos diferentes\n    1: {cab1}\n    2: {cab2}")
    colunas = [c for c in cab1 if c in cab2]
    if not colunas:
        print("  nenhuma coluna em comum, pulando")
        return resultados

    chaves, a1, a2, so1, so2, por_chave = alinha(cab1, l1, l2)
    print(f"  linhas: 1={len(l1)}  2={len(l2)}  comparadas={len(chaves)}"
          + ("  (alinhadas pela coluna '" + cab1[0] + "')" if por_chave else ""))
    if so1:
        print(f"  {len(so1)} só em 1 (ex.: {', '.join(so1[:3])})")
    if so2:
        print(f"  {len(so2)} só em 2 (ex.: {', '.join(so2[:3])})")
    if not chaves:
        return resultados

    i1 = {c: cab1.index(c) for c in colunas}
    i2 = {c: cab2.index(c) for c in colunas}
    pega = lambda linhas, j: [l[j] if j < len(l) else "" for l in linhas]

    print(f"  {'coluna':14s} {'máx |Δ|':>12s} {'RMS Δ':>12s} {'máx |Δ|/|v|':>12s}"
          f"  {'onde (máx |Δ|)':s}")
    deltas_graf = {}
    for c in colunas:
        s1, s2 = pega(a1, i1[c]), pega(a2, i2[c])
        v1 = np.array([para_float(x) for x in s1])
        v2 = np.array([para_float(x) for x in s2])
        numerica = np.isfinite(v1).sum() > 0.5 * len(v1)
        if not numerica:
            ndif = sum(x != y for x, y in zip(s1, s2))
            if ndif and c != (cab1[0] if por_chave else None):
                print(f"  {c:14s} {ndif} valor(es) de texto diferentes")
                resultados.append(dict(arquivo=nome, coluna=c, n=len(s1),
                                       texto_diferente=ndif))
            continue
        d = v2 - v1
        if eh_angulo(c):
            d = (d + 180.0) % 360.0 - 180.0
        ok = np.isfinite(d)
        if not ok.any():
            continue
        ad = np.abs(d[ok])
        rel = ad / np.where(np.abs(v1[ok]) > 0, np.abs(v1[ok]), np.nan)
        k = int(np.argmax(ad))
        onde = np.array(chaves)[ok][k]
        maxrel = np.nanmax(rel) if np.isfinite(rel).any() else math.nan
        rms = math.sqrt(np.mean(d[ok] ** 2))
        igual = ad.max() == 0 or (np.isfinite(maxrel) and maxrel <= tol)
        marca = "" if igual else "  <--"
        print(f"  {c:14s} {ad.max():12.4e} {rms:12.4e} {maxrel:12.4e}  "
              f"{onde}{marca}")
        resultados.append(dict(arquivo=nome, coluna=c, n=int(ok.sum()),
                               max_abs=ad.max(), rms=rms, max_rel=maxrel,
                               media=float(np.mean(d[ok])), onde_max=onde,
                               v1_no_max=v1[ok][k], v2_no_max=v2[ok][k],
                               dentro_tol=igual))
        if not igual:
            deltas_graf[c] = d
    if pasta_graf and deltas_graf:
        grafico(nome, chaves, deltas_graf, pasta_graf)
    return resultados


def grafico(nome, chaves, deltas, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    datas = [tenta_data(k) for k in chaves]
    x = datas if all(datas) else np.arange(len(chaves))
    fig, eixos = plt.subplots(len(deltas), 1, figsize=(11, 2.2 * len(deltas) + 1),
                              sharex=True, squeeze=False)
    for eixo, (c, d) in zip(eixos[:, 0], deltas.items()):
        if len(chaves) > 50:
            eixo.plot(x, d, lw=0.6)
        else:
            eixo.plot(x, d, "o-", ms=3, lw=0.6)
        eixo.set_ylabel(f"Δ {c}", fontsize=8)
        eixo.grid(alpha=0.3)
        eixo.axhline(0, color="k", lw=0.5)
    eixos[0, 0].set_title(f"{nome}: diferença (2 − 1)")
    eixos[-1, 0].set_xlabel("data (UTC)" if all(datas) else "linha")
    fig.tight_layout()
    arq = os.path.join(pasta, "delta_" + nome.replace(os.sep, "_")
                       .rsplit(".", 1)[0] + ".png")
    fig.savefig(arq, dpi=130)
    plt.close(fig)
    print(f"  gráfico: {arq}")


def compara_texto(p1, p2):
    with open(p1, encoding="utf-8", errors="replace") as f:
        t1 = f.read().splitlines()
    with open(p2, encoding="utf-8", errors="replace") as f:
        t2 = f.read().splitlines()
    dif = [l for l in difflib.unified_diff(t1, t2, "1", "2", lineterm="", n=0)
           if l[:1] in "+-" and not l.startswith(("---", "+++"))]
    print(f"  {sum(l[0] == '-' for l in dif)} linha(s) removidas / "
          f"{sum(l[0] == '+' for l in dif)} adicionadas")
    for l in dif[:10]:
        print("    " + l)
    if len(dif) > 10:
        print(f"    ... (+{len(dif) - 10})")


# ---------------------------------------------------------------------- main
def lista(pasta):
    arqs = set()
    for raiz, _, nomes in os.walk(pasta):
        for n in nomes:
            arqs.add(os.path.relpath(os.path.join(raiz, n), pasta))
    return arqs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta1", nargs="?", default="1")
    ap.add_argument("pasta2", nargs="?", default="2")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="diferença relativa abaixo da qual a coluna conta "
                         "como igual (padrão 1e-10)")
    ap.add_argument("--saida", default="comparacao_1_vs_2",
                    help="pasta para resumo.csv e gráficos")
    ap.add_argument("--sem-graficos", action="store_true")
    a = ap.parse_args()

    for p in (a.pasta1, a.pasta2):
        if not os.path.isdir(p):
            sys.exit(f"ERRO: pasta '{p}' não encontrada")
    os.makedirs(a.saida, exist_ok=True)

    f1, f2 = lista(a.pasta1), lista(a.pasta2)
    comuns = sorted(f1 & f2)
    print(f"Comparando '{a.pasta1}' x '{a.pasta2}': {len(comuns)} arquivo(s) em comum")
    for n in sorted(f1 - f2):
        print(f"  só em {a.pasta1}: {n}")
    for n in sorted(f2 - f1):
        print(f"  só em {a.pasta2}: {n}")

    resultados, identicos = [], 0
    for nome in comuns:
        p1, p2 = os.path.join(a.pasta1, nome), os.path.join(a.pasta2, nome)
        if filecmp.cmp(p1, p2, shallow=False):
            identicos += 1
            print(f"\n[{nome}] idênticos")
            continue
        print(f"\n[{nome}]")
        ext = os.path.splitext(nome)[1].lower()
        if ext == ".csv":
            resultados += compara_csv(nome, p1, p2, a.tol,
                                      None if a.sem_graficos else a.saida)
        elif ext in EXT_TEXTO:
            compara_texto(p1, p2)
        else:
            print("  binário diferente (ex.: PNG — o conteúdo não é comparado)")

    campos = ["arquivo", "coluna", "n", "max_abs", "rms", "max_rel", "media",
              "onde_max", "v1_no_max", "v2_no_max", "dentro_tol", "texto_diferente"]
    arq_resumo = os.path.join(a.saida, "resumo.csv")
    with open(arq_resumo, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resultados)

    fora = [r for r in resultados if not r.get("dentro_tol", False)]
    print(f"\n{identicos} arquivo(s) idênticos, {len(comuns) - identicos} diferentes; "
          f"{len(fora)} coluna(s) acima da tolerância relativa {a.tol:g}")
    print(f"resumo salvo em {arq_resumo}")


if __name__ == "__main__":
    main()
