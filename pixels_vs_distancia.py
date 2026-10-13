"""
Pixels que cada lua interna ocupa na JANUS, a partir da distância JUICE–lua.

Carrega os kernels igual ao distancia_juice_luas.py (juice_plan.tm):
distâncias dos SPKs, raios das luas do PCK e IFOV da JANUS do IK.
pixels = D / (d × IFOV)
elongação solar = ângulo, visto da JUICE, entre o Sol e a lua (pequeno = câmera
                  teria que apontar perto do Sol)

Gera:
  distancia_tempo.png         distância × tempo
  escala_tempo.png            escala (km/pixel) × tempo
  pixels_tempo.png            pixels × tempo
  pixels_distancia.png        pixels × distância
  elongacao_tempo.png         elongação solar × tempo
  distancia_pixels_elongacao_tempo.png
                              distância, pixels e elongação × tempo, juntos
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
PASSO_S = 600            # passo de amostragem em segundos (600 s = 10 min)

JUICE_ID = -28
LUAS = {                 # nome SPICE : código NAIF
    "METIS": 516,
    "ADRASTEA": 515,
    "AMALTHEA": 505,
    "THEBE": 514,
}
JANUS_ID = -28200        # código NAIF da JANUS (do IK)
NIVEIS_PX = [1, 10]      # linhas de referência nos gráficos de pixels
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


def ifov_janus():
    """IFOV (rad/pixel) lido direto do IK da JANUS."""
    return spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]


pasta_mk = carrega_meta_kernel()
IFOV = ifov_janus()
print(f"{spice.ktotal('ALL')} kernels carregados | IFOV da JANUS (IK): {IFOV * 1e6:.2f} µrad")

# Período: o pedido (INICIO/FIM) ou a cobertura toda da JUICE
cob_juice = cobertura_spk(JUICE_ID, pasta_mk)
if INICIO or FIM:
    n = spice.wncard(cob_juice)
    et0 = spice.str2et(INICIO) if INICIO else spice.wnfetd(cob_juice, 0)[0]
    et1 = spice.str2et(FIM) if FIM else spice.wnfetd(cob_juice, n - 1)[1]
    janela = spice.cell_double(2)
    spice.wninsd(et0, et1, janela)
    cob_juice = spice.wnintd(cob_juice, janela)

dados = {}   # nome -> (datas, distâncias, pixels, diâmetro, elongação)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(cob_juice, cobertura_spk(id_lua, pasta_mk))
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    _, raios = spice.bodvrd(nome, "RADII", 3)
    D = 2 * np.mean(raios)                       # diâmetro médio (km)

    datas, dists, elong = [], [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        pos, _ = spice.spkpos(nome, ets, "J2000", "NONE", "JUICE")
        # elongação: direções da lua e do Sol vistas da JUICE ("onde eu vejo" -> LT+S)
        p_lua, _ = spice.spkpos(nome, ets, "J2000", "LT+S", "JUICE")
        p_sol, _ = spice.spkpos("SUN", ets, "J2000", "LT+S", "JUICE")
        datas.extend(spice.et2datetime(ets))
        dists.extend(np.linalg.norm(pos, axis=1))
        elong.extend(angulo_entre(np.asarray(p_lua), np.asarray(p_sol)))
        datas.append(datas[-1])                   # NaN separa lacunas no gráfico
        dists.append(np.nan)
        elong.append(np.nan)

    dists = np.array(dists)
    elong = np.array(elong)
    pixels = D / (dists * IFOV)
    dados[nome] = (datas, dists, pixels, D, elong)

    i = np.nanargmin(dists)
    print(f"{nome}: D = {D:.1f} km | mín. {dists[i]:,.0f} km = {pixels[i]:.1f} px, "
          f"elongação {elong[i]:.0f}° em {datas[i]:%Y-%m-%d %H:%M} UTC")

spice.kclear()

n = len(dados)

# ---- distância × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.semilogy(datas, dists, lw=0.6)
    eixo.set_title(f"JUICE – {nome}")
    eixo.set_ylabel("distância [km]")
    eixo.grid(True, which="both", alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("distancia_tempo.png", dpi=150)

# ---- escala (km/pixel) × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.semilogy(datas, dists * IFOV, lw=0.6, color="tab:purple")
    eixo.set_title(nome)
    eixo.set_ylabel("escala [km/pixel]")
    eixo.grid(True, which="both", alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("escala_tempo.png", dpi=150)

# ---- pixels × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        eixo.axhline(p, color="gray", ls="--", lw=0.8)
    eixo.set_title(f"{nome} (D ≈ {D:.0f} km)")
    eixo.set_ylabel("pixels através da lua")
    eixo.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("pixels_tempo.png", dpi=150)

# ---- elongação solar × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D, elong)) in zip(eixos[:, 0], dados.items()):
    eixo.plot(datas, elong, lw=0.6, color="tab:green")
    eixo.axhline(LIMITE_SOL, color="gray", ls="--", lw=0.8)
    eixo.set_ylim(0, 180)
    eixo.set_title(nome)
    eixo.set_ylabel("elongação solar [°]")
    eixo.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("elongacao_tempo.png", dpi=150)

# ---- pixels × distância ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for nome, (datas, dists, pixels, D, elong) in dados.items():
    ordem = np.argsort(dists)                     # NaN vai para o fim e não é desenhado
    ax.loglog(dists[ordem], pixels[ordem], lw=2, label=f"{nome} (D ≈ {D:.0f} km)")
for p in NIVEIS_PX:
    ax.axhline(p, color="gray", ls="--", lw=0.8)
ax.set_xlabel("distância JUICE → lua [km]")
ax.set_ylabel("pixels através da lua")
ax.set_title(f"Pixels na JANUS (IFOV = {IFOV * 1e6:.0f} µrad)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("pixels_distancia.png", dpi=150)

# ---- distância, pixels e elongação × tempo, juntos ----
fig, eixos = plt.subplots(3 * n, 1, figsize=(12, 2.3 * 3 * n), sharex=True, squeeze=False)
for k, (nome, (datas, dists, pixels, D, elong)) in enumerate(dados.items()):
    ax_d, ax_p, ax_e = eixos[3 * k, 0], eixos[3 * k + 1, 0], eixos[3 * k + 2, 0]
    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_ylabel("distância [km]")
    ax_d.set_title(nome)
    ax_d.grid(True, which="both", alpha=0.3)
    ax_p.plot(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        ax_p.axhline(p, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    ax_p.grid(True, alpha=0.3)
    ax_e.plot(datas, elong, lw=0.6, color="tab:green")
    ax_e.axhline(LIMITE_SOL, color="gray", ls="--", lw=0.8)
    ax_e.set_ylim(0, 180)
    ax_e.set_ylabel("elongação [°]")
    ax_e.grid(True, alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("distancia_pixels_elongacao_tempo.png", dpi=150)

plt.show()
