"""
Pixels que cada lua interna ocupa na JANUS, a partir da distância JUICE–lua.

Usa os mesmos kernels do distancia_juice_luas.py (meta-kernel):
distâncias dos SPKs, raios das luas do PCK e IFOV da JANUS do IK.
pixels = D / (d × IFOV)

Gera:
  distancia_tempo.png         distância × tempo
  pixels_tempo.png            pixels × tempo
  pixels_distancia.png        pixels × distância
  distancia_pixels_tempo.png  distância e pixels × tempo, juntos (para correlacionar)
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
JANUS_ID = -28200        # código NAIF da JANUS (do IK)
NIVEIS_PX = [1, 10, 100] # linhas de referência nos gráficos de pixels
# =========================================================================


def ids_na_celula(celula):
    return [celula[i] for i in range(spice.card(celula))]


def cobertura_spk(id_corpo):
    """Junta a cobertura de todos os SPKs carregados que têm dados deste corpo."""
    cobertura = spice.cell_double(20000)
    for i in range(spice.ktotal("SPK")):
        arquivo, _, _, _ = spice.kdata(i, "SPK")
        if id_corpo in ids_na_celula(spice.spkobj(arquivo)):
            spice.spkcov(arquivo, id_corpo, cobertura)
    return cobertura


def ifov_janus():
    """IFOV (rad/pixel) lido direto do IK da JANUS."""
    return spice.gdpool(f"INS{JANUS_ID}_IFOV", 0, 1)[0]


spice.furnsh(META_KERNEL)
IFOV = ifov_janus()
print(f"IFOV da JANUS (IK): {IFOV * 1e6:.2f} µrad")

janela = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), janela)
cob_juice = cobertura_spk(JUICE_ID)

dados = {}   # nome -> (datas, distâncias, pixels, diâmetro)
for nome, id_lua in LUAS.items():
    comum = spice.wnintd(spice.wnintd(cob_juice, cobertura_spk(id_lua)), janela)
    if spice.wncard(comum) == 0:
        print(f"{nome}: sem cobertura em comum com a JUICE no período")
        continue

    _, raios = spice.bodvrd(nome, "RADII", 3)
    D = 2 * np.mean(raios)                       # diâmetro médio (km)

    datas, dists = [], []
    for k in range(spice.wncard(comum)):
        a, b = spice.wnfetd(comum, k)
        ets = np.arange(a, b, PASSO_S)
        pos, _ = spice.spkpos(nome, ets, "J2000", "NONE", "JUICE")
        datas.extend(spice.et2datetime(ets))
        dists.extend(np.linalg.norm(pos, axis=1))
        datas.append(datas[-1])                   # NaN separa lacunas no gráfico
        dists.append(np.nan)

    dists = np.array(dists)
    pixels = D / (dists * IFOV)
    dados[nome] = (datas, dists, pixels, D)

    i = np.nanargmin(dists)
    print(f"{nome}: D = {D:.1f} km | mín. {dists[i]:,.0f} km = {pixels[i]:.1f} px "
          f"em {datas[i]:%Y-%m-%d %H:%M} UTC")

spice.kclear()

n = len(dados)

# ---- distância × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D)) in zip(eixos[:, 0], dados.items()):
    eixo.semilogy(datas, dists, lw=0.6)
    eixo.set_title(f"JUICE – {nome}")
    eixo.set_ylabel("distância [km]")
    eixo.grid(True, which="both", alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("distancia_tempo.png", dpi=150)

# ---- pixels × tempo ----
fig, eixos = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, squeeze=False)
for eixo, (nome, (datas, dists, pixels, D)) in zip(eixos[:, 0], dados.items()):
    eixo.semilogy(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        eixo.axhline(p, color="gray", ls="--", lw=0.8)
    eixo.set_title(f"{nome} (D ≈ {D:.0f} km)")
    eixo.set_ylabel("pixels através da lua")
    eixo.grid(True, which="both", alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("pixels_tempo.png", dpi=150)

# ---- pixels × distância ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for nome, (datas, dists, pixels, D) in dados.items():
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

# ---- distância e pixels × tempo, juntos ----
fig, eixos = plt.subplots(2 * n, 1, figsize=(12, 2.5 * 2 * n), sharex=True, squeeze=False)
for k, (nome, (datas, dists, pixels, D)) in enumerate(dados.items()):
    ax_d, ax_p = eixos[2 * k, 0], eixos[2 * k + 1, 0]
    ax_d.semilogy(datas, dists, lw=0.6)
    ax_d.set_ylabel("distância [km]")
    ax_d.set_title(nome)
    ax_p.semilogy(datas, pixels, lw=0.6, color="tab:orange")
    for p in NIVEIS_PX:
        ax_p.axhline(p, color="gray", ls="--", lw=0.8)
    ax_p.set_ylabel("pixels")
    for ax in (ax_d, ax_p):
        ax.grid(True, which="both", alpha=0.3)
eixos[-1, 0].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig("distancia_pixels_tempo.png", dpi=150)

plt.show()
