"""
Quantos pixels cada lua interna ocupa numa imagem da JANUS, em função da distância.

Não depende de data: é só a fórmula  N = D / (d × IFOV), onde
  D    = diâmetro da lua (km)
  d    = distância JUICE -> lua (km)
  IFOV = ângulo de um pixel da JANUS (rad)

Serve de "régua" para traduzir o gráfico de distância em "o que dá para ver",
e para escolher as linhas de corte (N_min) de cada lua.
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================= CONFIGURAÇÃO (edite aqui) =======================
IFOV = 15e-6             # rad/pixel da JANUS (confirme no IK)

# True: lê os raios do PCK pelo SPICE. False: usa os valores fixos abaixo.
USAR_SPICE = False
META_KERNEL = r"/caminho/para/JUICE/kernels/mk/juice_crema_5_1_150la.tm"

# Raios (a, b, c) em km, do pck00010/pck00011. Usados se USAR_SPICE = False.
RAIOS_FIXOS = {
    "AMALTHEA": (125.0, 73.0, 64.0),
    "THEBE": (58.0, 49.0, 42.0),
    "METIS": (30.0, 20.0, 17.0),
    "ADRASTEA": (10.0, 8.0, 7.0),
}

# Qual diâmetro usar: "medio" (média dos 3 raios) ou "maior" (eixo maior, caso otimista)
DIAMETRO = "medio"

# Faixa de distâncias do gráfico (km)
D_MIN_KM, D_MAX_KM = 1e4, 2e7

# Linhas de referência: pixels -> o que dá para ver
NIVEIS = {1: "1 px: só um ponto", 10: "10 px: forma visível", 100: "100 px: detalhes de superfície"}

# Opcional: distâncias para marcar com linha vertical, ex.: a mínima do seu
# script de distância. Deixe {} para não marcar nada.
DISTANCIAS_MARCAR = {}   # ex.: {"mín. Amalthea": 5.0e5}

# Cores fixas por lua (a cor segue a lua, não a ordem)
CORES = {"AMALTHEA": "#2a78d6", "THEBE": "#eb6834", "METIS": "#1baf7a", "ADRASTEA": "#e87ba4"}
# =========================================================================


def le_raios():
    if not USAR_SPICE:
        return RAIOS_FIXOS
    import spiceypy as spice
    spice.furnsh(META_KERNEL)
    raios = {lua: tuple(spice.bodvrd(lua, "RADII", 3)[1]) for lua in RAIOS_FIXOS}
    spice.kclear()
    return raios


def diametro(raios):
    return 2 * (max(raios) if DIAMETRO == "maior" else np.mean(raios))


diametros = {lua: diametro(r) for lua, r in le_raios().items()}
dist = np.logspace(np.log10(D_MIN_KM), np.log10(D_MAX_KM), 400)

# ---- tabela no terminal: distância máxima para atingir cada nível ----
print(f"IFOV = {IFOV * 1e6:.1f} µrad   |   diâmetro: {DIAMETRO}")
print(f"{'lua':<10}{'D (km)':>8}" + "".join(f"{f'd p/ {n} px':>16}" for n in NIVEIS))
for lua, D in diametros.items():
    linhas = "".join(f"{D / (n * IFOV):>13,.0f} km" for n in NIVEIS)
    print(f"{lua:<10}{D:>8.1f}{linhas}")
for rotulo, d in DISTANCIAS_MARCAR.items():
    print(f"\nA {d:,.0f} km ({rotulo}):")
    for lua, D in diametros.items():
        print(f"  {lua:<10} {D / (d * IFOV):6.1f} px")

# ---- gráfico ----
fig, ax = plt.subplots(figsize=(9, 5.5))

for n, texto in NIVEIS.items():
    ax.axhline(n, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.text(D_MIN_KM * 1.15, n * 1.12, texto, color="0.35", fontsize=8.5, va="bottom")

for rotulo, d in DISTANCIAS_MARCAR.items():
    ax.axvline(d, color="0.3", lw=0.8, ls=":", zorder=1)
    ax.text(d * 1.05, 0.06, rotulo, color="0.3", fontsize=8, rotation=90, va="bottom")

for lua, D in diametros.items():
    px = D / (dist * IFOV)
    ax.plot(dist, px, color=CORES[lua], lw=2, label=f"{lua.title()} (D ≈ {D:.0f} km)", zorder=3)
    # rótulo direto no fim da linha
    ax.text(dist[-1] * 1.08, px[-1], lua.title(), color="0.2", fontsize=8.5, va="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_MIN_KM, D_MAX_KM * 2.2)   # espaço para os rótulos à direita
ax.set_ylim(0.04, 2000)
ax.set_xlabel("Distância JUICE → lua (km)")
ax.set_ylabel("Pixels através da lua")
ax.set_title(f"Quantos pixels a lua ocupa na JANUS (IFOV = {IFOV * 1e6:.0f} µrad)",
             loc="left", fontsize=12)
ax.grid(True, which="major", color="0.9", lw=0.6)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("pixels_vs_distancia.png", dpi=150)
plt.show()
