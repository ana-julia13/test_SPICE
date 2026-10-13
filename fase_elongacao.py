"""
Ângulo de fase e elongação solar das luas internas de Júpiter, vistas da JUICE.

  fase      = ângulo Sol–lua–JUICE (vértice na lua): quanto da lua aparece
              iluminado. 0° = cheia, 180° = contraluz.
  elongação = ângulo Sol–JUICE–lua (vértice na JUICE): quão longe do Sol a
              câmera teria que apontar. Pequena = proibido / luz espúria.

Carrega os kernels igual ao distancia_juice_luas.py (juice_plan.tm).

Gera:
  fase_tempo.png             ângulo de fase × tempo
  elongacao_tempo.png        elongação solar × tempo
  fase_elongacao_tempo.png   os dois juntos, por lua (para correlacionar)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

# ======================= CONFIGURAÇÃO (edite aqui) =======================
# Pasta "kernels" da JUICE (a que tem ck/, fk/, lsk/, mk/, spk/ ...)
PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL = "juice_plan.tm"   # fica dentro de PASTA_KERNELS/mk

# Período: None = usa TODO o período em que há dados (a missão inteira).
# Ou coloque datas, ex.: INICIO = "2031-07-01", FIM = "2035-12-31"
INICIO = None
FIM = None
PASSO_S = 1800           # passo de amostragem em segundos (1800 s = 30 min)

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
LIMITE_FASE = 90         # graus: abaixo disso a lua aparece mais de meio iluminada
LIMITE_SOL = 30          # graus: elongação mínima (EXEMPLO, confirme com a equipe)
# =========================================================================


def carrega_meta_kernel():
    """Os meta-kernels da JUICE usam PATH_VALUES = ( '..' ): os caminhos são
    relativos à pasta mk/. Por isso entramos na pasta mk/ antes do furnsh."""
    pasta_mk = os.path.join(PASTA_KERNELS, "mk")
    pasta_original = os.getcwd()
    os.chdir(pasta_mk)
    try:
        spice.furnsh(META_KERNEL)
    finally:
        os.chdir(pasta_original)
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


pasta_mk = carrega_meta_kernel()

# Período: o pedido (INICIO/FIM) ou a cobertura toda da JUICE
cob_juice = cobertura_spk(JUICE_ID, pasta_mk)
if INICIO or FIM:
    n = spice.wncard(cob_juice)
    et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
    et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
    janela = spice.cell_double(2)
    spice.wninsd(et0, et1, janela)
    cob_juice = spice.wnintd(cob_juice, janela)

dados = {}   # nome -> (datas, fase, elongação)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua, pasta_mk))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    datas, fase, elong = [], [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        # fase: ângulo Sol–lua–JUICE, vértice na lua
        fase.extend(np.degrees([spice.phaseq(et, nome, "SUN", "JUICE", "LT+S") for et in ets]))
        # elongação: ângulo entre as direções da lua e do Sol, vistas da JUICE
        p_lua, _ = spice.spkpos(nome, ets, "J2000", "LT+S", "JUICE")
        p_sol, _ = spice.spkpos("SUN", ets, "J2000", "LT+S", "JUICE")
        elong.extend(angulo_entre(np.asarray(p_lua), np.asarray(p_sol)))
        datas.extend(spice.et2datetime(ets))
        datas.append(datas[-1])                   # NaN separa lacunas no gráfico
        fase.append(np.nan)
        elong.append(np.nan)

    dados[nome] = (datas, np.array(fase), np.array(elong))
    print(f"{nome}: fase {np.nanmin(fase):.0f}°–{np.nanmax(fase):.0f}° | "
          f"elongação {np.nanmin(elong):.0f}°–{np.nanmax(elong):.0f}°")

spice.kclear()

n = len(dados)

# ---- fase × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, fase, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, fase, lw=0.6, color="tab:blue")
    eixo.axhline(LIMITE_FASE, color="gray", ls="--", lw=0.8)
    eixo.set_ylim(0, 180)
    eixo.set_title(nome)
    eixo.set_ylabel("ângulo de fase [°]")
    eixo.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("fase_tempo.png", dpi=150)

# ---- elongação × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, fase, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, elong, lw=0.6, color="tab:green")
    eixo.axhline(LIMITE_SOL, color="gray", ls="--", lw=0.8)
    eixo.set_ylim(0, 180)
    eixo.set_title(nome)
    eixo.set_ylabel("elongação solar [°]")
    eixo.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("elongacao_tempo.png", dpi=150)

# ---- fase e elongação × tempo, juntos ----
fig, eixos = plt.subplots(2 * n, 1, figsize=(12, 2.5 * 2 * n), sharex=True, squeeze=False)
for k, (nome, (datas, fase, elong)) in enumerate(dados.items()):
    ax_f, ax_e = eixos[2 * k, 0], eixos[2 * k + 1, 0]
    ax_f.plot(datas, fase, lw=0.6, color="tab:blue")
    ax_f.axhline(LIMITE_FASE, color="gray", ls="--", lw=0.8)
    ax_f.set_ylabel("fase [°]")
    ax_f.set_title(nome)
    ax_e.plot(datas, elong, lw=0.6, color="tab:green")
    ax_e.axhline(LIMITE_SOL, color="gray", ls="--", lw=0.8)
    ax_e.set_ylabel("elongação [°]")
    for ax in (ax_f, ax_e):
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("fase_elongacao_tempo.png", dpi=150)

plt.show()
