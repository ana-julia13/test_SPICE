"""
Compara os elementos orbitais de outro integrador com os do SPICE, para um
satélite escolhido (por enquanto Metis, mas vale para qualquer lua do SPK).

Formato do arquivo do integrador (uma linha por instante):
  2030 JAN 01 00:00:00.000  ET  t  a  e  i  Ω  ω  M
  - a data UTC (só para referência; o tempo usado é o ET)
  - ET: segundos TDB desde J2000, o mesmo tempo do SPICE
  - t: tempo da simulação (dias)
  - a (km), e, i, Ω (long. do nodo), ω (arg. do pericentro), M (anom. média)
Os ângulos podem estar em graus ou radianos (detectado sozinho, ver ANGULOS).

Para cada linha, o script pega o estado da lua no SPICE no mesmo ET, gira
para o mesmo frame (equador de Júpiter congelado na época EPOCA_FRAME) e
converte em elementos osculadores com oscltx. Gera em PASTA_SAIDA:
  comparacao_LUA.png   elementos do SPICE e do integrador sobrepostos
  diferencas_LUA.png   integrador − SPICE (ângulos entre −180° e 180°)
  distancia_LUA.png    distância entre as duas posições (km)
  comparacao_LUA.csv   tudo junto, linha a linha

Uso:
  python3 compara_spice.py                              (usa a CONFIGURAÇÃO)
  python3 compara_spice.py oscelt_planetIAU.out --lua METIS
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # só salva PNG, sem abrir janela (não precisa de Qt)
import matplotlib.pyplot as plt
import spiceypy as spice

# ======================= CONFIGURAÇÃO (edite aqui) =======================
# Pode ser um meta-kernel (.tm) ou uma lista de arquivos de kernel.
KERNELS = [r"/caminho/para/JUICE/kernels/mk/juice_crema_5_1_150la.tm"]
ARQUIVO_INTEGRADOR = r"oscelt_planetIAU.out"
LUA = "METIS"            # nome SPICE (METIS, ADRASTEA, ..., CALLISTO) ou código NAIF

# "EQUADOR_JUPITER": frame inercial alinhado ao equador de Júpiter na época
#                    EPOCA_FRAME (o mesmo do elementos_spice.py)
# "J2000":           equador terrestre J2000 (ICRF)
FRAME = "EQUADOR_JUPITER"
# Época (UTC) em que o equador de Júpiter é congelado. None = primeira linha
# do arquivo do integrador. Use a mesma época em que o integrador começou.
EPOCA_FRAME = None

# μ usado para converter o estado do SPICE em elementos (e para refazer as
# posições na comparação em km). None = GM de Júpiter (+ GM da lua, se
# SOMAR_GM_LUA) dos kernels. Deixe igual ao μ do integrador.
MU_KM3S2 = None
SOMAR_GM_LUA = True

ANGULOS = "auto"         # "graus", "radianos" ou "auto" (decide pelo M)
PASTA_SAIDA = "saida_comparacao"
# =========================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]
ANGULARES = [2, 3, 4, 5]  # colunas de ângulo em ROTULOS

PASTA_DE_CARGA = {}  # nome usado no furnsh -> pasta de onde foi carregado


def carrega_kernels(kernels):
    """furnsh de cada kernel; meta-kernels são carregados da própria pasta,
    pois os caminhos dentro deles (PATH_VALUES = '..') são relativos."""
    pasta_atual = os.getcwd()
    for k in kernels:
        k = os.path.abspath(k)
        os.chdir(os.path.dirname(k))
        try:
            spice.furnsh(os.path.basename(k))
        finally:
            os.chdir(pasta_atual)
        PASTA_DE_CARGA[os.path.basename(k)] = os.path.dirname(k)


def caminho_absoluto(arquivo, origem):
    """Caminho real de um kernel listado por kdata (que pode ser relativo à
    pasta do meta-kernel que o carregou)."""
    pasta = PASTA_DE_CARGA.get(origem or arquivo, os.getcwd())
    return os.path.join(pasta, arquivo)


def cobertura_spk(id_corpo):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(20000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, origem, _ = spice.kdata(i, "SPK")
        arquivo = caminho_absoluto(arquivo, origem)
        ids = spice.spkobj(arquivo)
        if id_corpo in [ids[j] for j in range(spice.card(ids))]:
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


def matriz_equador_jupiter(et):
    """Matriz J2000 -> frame inercial com z no polo de Júpiter e x no nó
    ascendente do equador de Júpiter sobre o equador J2000, na época 'et'."""
    polo = spice.pxform("IAU_JUPITER", "J2000", et)[:, 2].copy()  # contíguo p/ spiceypy
    nodo = spice.ucrss([0.0, 0.0, 1.0], polo)
    y = spice.ucrss(polo, nodo)
    return np.array([nodo, y, polo])


def gm_corpo(codigo):
    """GM (km³/s²) do kernel de constantes, ou 0 se não houver."""
    if not spice.bodfnd(codigo, "GM"):
        return 0.0
    return spice.bodvcd(codigo, "GM", 1)[1][0]


def formata_eixo(eixo):
    """Eixo x começando e terminando nos dados e eixo y sem notação
    científica (nem o '1e-7+4.217e5' em cima do eixo)."""
    eixo.margins(x=0)
    eixo.ticklabel_format(axis="y", style="plain", useOffset=False)
    eixo.grid(alpha=0.3)


def le_integrador(arquivo):
    """Lê o arquivo do integrador. Devolve as datas (texto), o ET e a tabela
    [a, e, i, Ω, ω, M], com os ângulos ainda na unidade do arquivo.
    Cada linha: data (várias palavras) seguida de 8 números."""
    datas, numeros = [], []
    with open(arquivo) as f:
        for n_linha, linha in enumerate(f, 1):
            partes = linha.split()
            if not partes or partes[0].startswith(("#", "!")):
                continue
            try:
                valores = [float(p.replace("D", "E").replace("d", "e")) for p in partes[-8:]]
            except ValueError:
                continue  # cabeçalho ou linha que não é de dados
            if len(partes) < 9:
                raise ValueError(f"{arquivo}:{n_linha}: esperava data + 8 números")
            datas.append(" ".join(partes[:-8]))
            numeros.append(valores)
    if not numeros:
        raise ValueError(f"nenhuma linha de dados em {arquivo}")
    numeros = np.array(numeros)
    # colunas: ET, t (dias), a, e, i, Ω, ω, M
    return datas, numeros[:, 0], numeros[:, 2:8]


def diferenca_angular(a, b):
    """a − b em graus, entre −180 e 180."""
    return (a - b + 180.0) % 360.0 - 180.0


# --------------------------- argumentos -----------------------------------
parser = argparse.ArgumentParser(description="Compara elementos de um integrador com o SPICE.")
parser.add_argument("arquivo", nargs="?", default=ARQUIVO_INTEGRADOR,
                    help="saída do integrador (padrão: ARQUIVO_INTEGRADOR)")
parser.add_argument("--lua", default=LUA, help="nome SPICE ou código NAIF (padrão: LUA)")
args = parser.parse_args()

carrega_kernels(KERNELS)

try:
    id_lua = int(args.lua)
except ValueError:
    id_lua = spice.bods2c(args.lua.upper())
nome_lua = spice.bodc2n(id_lua)

datas, ets, integ = le_integrador(args.arquivo)
print(f"{args.arquivo}: {len(ets)} linhas, de {datas[0]} a {datas[-1]}")

# Unidade dos ângulos: em graus, M percorre até 360; em radianos, até 2π
angulos = ANGULOS
if angulos == "auto":
    angulos = "radianos" if np.nanmax(np.abs(integ[:, 5])) <= 2 * np.pi + 1e-6 else "graus"
    print(f"ângulos do integrador: {angulos} (detectado pelo M)")
if angulos == "radianos":
    integ[:, 2:] = np.degrees(integ[:, 2:])
elif angulos != "graus":
    raise ValueError(f"ANGULOS desconhecido: {angulos}")
integ[:, 3:] %= 360.0

# Período coberto pelo SPK desta lua
cobertura = cobertura_spk(id_lua)
dentro = np.array([spice.wnelmd(et, cobertura) for et in ets])
if not dentro.any():
    raise RuntimeError(f"{nome_lua}: nenhum ET do arquivo está na cobertura do SPK")
if not dentro.all():
    print(f"{nome_lua}: {np.count_nonzero(~dentro)} linhas fora da cobertura do SPK, ignoradas")
datas = [d for d, ok in zip(datas, dentro) if ok]
ets, integ = ets[dentro], integ[dentro]

# Frame
if FRAME == "EQUADOR_JUPITER":
    et_frame = ets[0] if EPOCA_FRAME is None else spice.str2et(EPOCA_FRAME)
    R = matriz_equador_jupiter(et_frame)
    print(f"frame: equador de Júpiter congelado em {spice.et2utc(et_frame, 'C', 3)} UTC")
elif FRAME == "J2000":
    R = np.eye(3)
else:
    raise ValueError(f"FRAME desconhecido: {FRAME}")

# μ
if MU_KM3S2 is not None:
    mu = MU_KM3S2
else:
    gm_jup = gm_corpo(599)
    if gm_jup == 0.0:
        raise RuntimeError("GM de Júpiter não encontrado: carregue gm_de440.tpc")
    mu = gm_jup + (gm_corpo(id_lua) if SOMAR_GM_LUA else 0.0)
print(f"μ = {mu:.6f} km³/s²")

# Elementos do SPICE nos mesmos ETs
estados, _ = spice.spkezr(str(id_lua), ets, "J2000", "NONE", "JUPITER")
estados = np.asarray(estados)
spc = np.empty_like(integ)
pos_spice = np.empty((len(ets), 3))
for k, (et, s) in enumerate(zip(ets, estados)):
    s_rot = np.concatenate([R @ s[:3], R @ s[3:]])
    el = spice.oscltx(s_rot, et, mu)
    spc[k] = [el[9], el[1], np.degrees(el[2]),
              np.degrees(el[3]) % 360.0, np.degrees(el[4]) % 360.0, np.degrees(el[5]) % 360.0]
    pos_spice[k] = s_rot[:3]

# Posição do integrador refeita a partir dos elementos dele (mesmo μ)
pos_integ = np.empty_like(pos_spice)
for k, (et, (a, e, i, om, w, m)) in enumerate(zip(ets, integ)):
    elts = [a * (1.0 - e), e, np.radians(i), np.radians(om), np.radians(w),
            np.radians(m), et, mu]
    pos_integ[k] = spice.conics(elts, et)[:3]
distancia = np.linalg.norm(pos_integ - pos_spice, axis=1)

dif = integ - spc
for j in ANGULARES[1:]:
    dif[:, j] = diferenca_angular(integ[:, j], spc[:, j])

# ------------------------------ saída -------------------------------------
os.makedirs(PASTA_SAIDA, exist_ok=True)
base = os.path.join(PASTA_SAIDA, nome_lua.lower())
nomes = ["a_km", "e", "i_deg", "Omega_deg", "omega_deg", "M_deg"]

with open(f"{base}_comparacao.csv", "w") as f:
    f.write("data_utc,et_tdb_s,"
            + ",".join(f"{n}_integ" for n in nomes) + ","
            + ",".join(f"{n}_spice" for n in nomes) + ","
            + ",".join(f"d{n}" for n in nomes) + ",distancia_km\n")
    for d, et, li, ls, ld, dist in zip(datas, ets, integ, spc, dif, distancia):
        f.write(f"{d},{et:.3f},"
                + ",".join(f"{v:.10g}" for v in np.concatenate([li, ls, ld]))
                + f",{dist:.6g}\n")

print(f"\n{nome_lua}: integrador − SPICE ({len(ets)} instantes)")
print(f"  {'':10s} {'média':>14s} {'RMS':>14s} {'máx |dif|':>14s}")
for rot, col in zip(ROTULOS, dif.T):
    print(f"  {rot:10s} {col.mean():14.6g} {np.sqrt(np.mean(col**2)):14.6g} {np.abs(col).max():14.6g}")
print(f"  {'dist [km]':10s} {distancia.mean():14.6g} {np.sqrt(np.mean(distancia**2)):14.6g} "
      f"{distancia.max():14.6g}")

tempo = spice.et2datetime(ets)
titulo = f"{nome_lua} – integrador × SPICE ({FRAME}, osculadores)"

fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
    if j in ANGULARES[1:]:  # 0–360: pontos, para não ligar os saltos 360 -> 0
        eixo.plot(tempo, spc[:, j], ".", ms=1.2, label="SPICE")
        eixo.plot(tempo, integ[:, j], ".", ms=1.2, alpha=0.6, label="integrador")
        eixo.set_ylim(0, 360)
    else:
        eixo.plot(tempo, spc[:, j], lw=0.7, label="SPICE")
        eixo.plot(tempo, integ[:, j], lw=0.7, alpha=0.8, label="integrador")
    eixo.set_ylabel(rot)
    formata_eixo(eixo)
eixos[0].legend(loc="upper right", fontsize=8, markerscale=6)
eixos[0].set_title(titulo)
eixos[-1].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(f"{base}_comparacao.png", dpi=150)
plt.close(fig)

fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
    eixo.plot(tempo, dif[:, j], lw=0.7, color="C3")
    eixo.axhline(0.0, color="k", lw=0.5)
    eixo.set_ylabel("Δ" + rot)
    formata_eixo(eixo)
eixos[0].set_title(f"{nome_lua} – diferença integrador − SPICE")
eixos[-1].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(f"{base}_diferencas.png", dpi=150)
plt.close(fig)

fig, eixo = plt.subplots(figsize=(11, 4))
eixo.plot(tempo, distancia, lw=0.7, color="C3")
eixo.set_ylabel("|r_integ − r_SPICE| [km]")
eixo.set_xlabel("data (UTC)")
eixo.set_title(f"{nome_lua} – distância entre as posições do integrador e do SPICE")
formata_eixo(eixo)
fig.tight_layout()
fig.savefig(f"{base}_distancia.png", dpi=150)
plt.close(fig)

print(f"\nArquivos salvos em {os.path.abspath(PASTA_SAIDA)}")
spice.kclear()
