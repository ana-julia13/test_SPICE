"""
Barras no tempo (zoom de poucos dias) com os critérios de fundo/iluminação
para as 4 luas internas. Barra preenchida = o problema ACONTECE:

  perto de Júpiter   lua a menos de 10 px do limbo de Júpiter
                     (inclui trânsito e ocultação: aí a separação é negativa)
  perto de galileana lua a menos de 10 px do limbo de Io/Europa/Ganimedes/Calisto
  ocultada           Júpiter ou uma galileana NA FRENTE da lua, vista da JUICE
  eclipse            lua na sombra de Júpiter (Sol escondido por Júpiter, visto da lua)
  livre              nenhum dos acima

separação até o limbo = θ − ρ_corpo − ρ_lua   (÷ IFOV = pixels)
  θ = ângulo corpo–lua visto da JUICE, ρ = raio angular (asin(R / d))
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import spiceypy as spice

PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL = "juice_plan.tm"
INICIO = "2032-03-10 00:00"   # janela do zoom (UTC), ex.: ~4 dias
FIM = "2032-03-14 00:00"

PASSO_S = 60             # 1 ponto por minuto
CORRECAO = "LT+S"        # para a separação (spkpos)
CORRECAO_OCC = "LT"      # occult não aceita "+S"

JUICE_ID = -28
JANUS_ID = -28200
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
GALILEANAS = ["IO", "EUROPA", "GANYMEDE", "CALLISTO"]
LIMITE_PX = 10           # >= 10 px de qualquer limbo
# =========================================================================


def carrega_meta_kernel(pasta_kernels, nome_mk):
    pasta_mk = os.path.join(pasta_kernels, "mk")
    caminho = os.path.join(pasta_mk, nome_mk)
    if not os.path.isfile(caminho):
        sys.exit(f"ERRO: meta-kernel não encontrado: {caminho}")
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
    # recarrega com caminho absoluto (assim spkcov acha os arquivos)
    spice.kclear()
    for arquivo in arquivos:
        spice.furnsh(arquivo)
    print(f"Meta-kernel carregado: {caminho} ({len(arquivos)} kernels)")


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


def separacao_px(lua, R_lua, corpo, R_corpo, ets):
    """Pixels entre o limbo da lua e o limbo do corpo, vistos da JUICE."""
    p_lua, _ = spice.spkpos(lua, ets, "J2000", CORRECAO, "JUICE")
    p_corpo, _ = spice.spkpos(corpo, ets, "J2000", CORRECAO, "JUICE")
    p_lua, p_corpo = np.asarray(p_lua), np.asarray(p_corpo)
    d_lua = np.linalg.norm(p_lua, axis=1)
    d_corpo = np.linalg.norm(p_corpo, axis=1)
    cos = np.sum(p_lua * p_corpo, axis=1) / (d_lua * d_corpo)
    theta = np.arccos(np.clip(cos, -1, 1))
    rho_corpo = np.arcsin(np.clip(R_corpo / d_corpo, 0, 1))
    rho_lua = np.arcsin(np.clip(R_lua / d_lua, 0, 1))
    return (theta - rho_corpo - rho_lua) / IFOV


def ocultada(lua, corpo, ets):
    """True quando o corpo está na frente da lua (total, anular ou parcial)."""
    return np.array([spice.occult(lua, "ELLIPSOID", f"IAU_{lua}",
                                  corpo, "ELLIPSOID", f"IAU_{corpo}",
                                  CORRECAO_OCC, "JUICE", et) < 0 for et in ets])


def eclipse(lua, ets):
    """True quando Júpiter esconde o Sol (total, anular ou parcial), visto da lua."""
    return np.array([spice.occult("JUPITER", "ELLIPSOID", "IAU_JUPITER",
                                  "SUN", "ELLIPSOID", "IAU_SUN",
                                  CORRECAO_OCC, lua, et) > 0 for et in ets])


carrega_meta_kernel(PASTA_KERNELS, META_KERNEL)
IFOV = spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]     # rad/pixel, do IK
R_JUP = spice.bodvrd("JUPITER", "RADII", 3)[1][0]       # raio equatorial (conservador)
R_GAL = {g: max(spice.bodvrd(g, "RADII", 3)[1]) for g in GALILEANAS}

cob_juice, arq_juice = cobertura_spk(JUICE_ID)
if not arq_juice:
    sys.exit("ERRO: nenhum SPK carregado tem a trajetória da JUICE.")
janela = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), janela)
cob_juice = spice.wnintd(cob_juice, janela)

dados = {}   # nome -> (datas, {linha: máscara})
for nome, id_lua in LUAS.items():
    cob_lua, _ = cobertura_spk(id_lua)
    comum = spice.wnintd(cob_juice, cob_lua)
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue
    R_lua = np.mean(spice.bodvrd(nome, "RADII", 3)[1])

    ets = np.concatenate([np.arange(*spice.wnfetd(comum, k), PASSO_S)
                          for k in range(spice.wncard(comum))])

    sep_jup = separacao_px(nome, R_lua, "JUPITER", R_JUP, ets)
    sep_gal = np.min([separacao_px(nome, R_lua, g, R_GAL[g], ets)
                      for g in GALILEANAS], axis=0)
    occ = ocultada(nome, "JUPITER", ets)
    for g in GALILEANAS:
        occ |= ocultada(nome, g, ets)

    linhas = {
        f"perto de Júpiter (< {LIMITE_PX} px)": sep_jup < LIMITE_PX,
        f"perto de galileana (< {LIMITE_PX} px)": sep_gal < LIMITE_PX,
        "ocultada": occ,
        "eclipse": eclipse(nome, ets),
    }
    linhas["livre"] = ~np.any(list(linhas.values()), axis=0)
    dados[nome] = (spice.et2datetime(ets), linhas)

    print(f"{nome}:")
    for rotulo, m in linhas.items():
        print(f"    {rotulo:28s} {100 * m.mean():5.1f}% do tempo")

spice.kclear()

# ---- barras × tempo (uma faixa por lua) ----
n = len(dados)
fig, eixos = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, linhas)) in zip(eixos[:, 0], dados.items()):
    rotulos = list(linhas)
    for k, rotulo in enumerate(rotulos):
        y = len(rotulos) - 1 - k                 # primeira linha em cima
        cor = "tab:green" if rotulo == "livre" else "tab:red"
        eixo.fill_between(datas, y - 0.35, y + 0.35, where=linhas[rotulo],
                          color=cor, alpha=0.8, lw=0)
    eixo.set_yticks(range(len(rotulos)))
    eixo.set_yticklabels(rotulos[::-1], fontsize=8)
    eixo.set_ylim(-0.6, len(rotulos) - 0.4)
    eixo.set_title(f"{nome.capitalize()} ({LUAS[nome]})")
    eixo.grid(True, axis="x", alpha=0.6)
    eixo.margins(x=0)
eixo = eixos[-1, 0]
eixo.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
eixo.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%Hh"))
eixo.xaxis.set_minor_locator(mdates.HourLocator())      # tracinho a cada hora
eixo.set_xlabel("data (UTC)")
fig.suptitle(f"Barra = acontece ({META_KERNEL})", fontsize=10)
fig.tight_layout()
fig.savefig("barras_tempo.png", dpi=150)
print("gráfico salvo: barras_tempo.png")
