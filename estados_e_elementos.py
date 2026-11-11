"""
Estados iniciais e elementos orbitais das luas de Júpiter numa data.

Gera duas tabelas:
  estados_iniciais.csv    posição e velocidade de cada lua (km e km/s)
  elementos_iniciais.csv  elementos orbitais calculados a partir desses estados
"""
import os

import numpy as np
import spiceypy as spice

# ----------------------------- CONFIGURAÇÃO -----------------------------
META_KERNEL = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
DATA = "2030-07-01"          # data (UTC) dos estados iniciais
FRAME = "EQUADOR_JUPITER"    # "EQUADOR_JUPITER" ou "J2000"
SOMAR_GM_LUA = True          # mu = GM de Júpiter + GM da lua
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514,
        "IO": 501, "EUROPA": 502, "GANYMEDE": 503, "CALLISTO": 504}
# ------------------------------------------------------------------------

# 1) Carregar os kernels.
#    Entramos na pasta do meta-kernel porque ele usa caminhos relativos.
pasta_original = os.getcwd()
os.chdir(os.path.dirname(META_KERNEL))
spice.furnsh(META_KERNEL)
os.chdir(pasta_original)

# 2) Converter a data para o tempo do SPICE (ET, segundos desde J2000).
et = spice.str2et(DATA)

# 3) Matriz de rotação: J2000 -> equador de Júpiter (fixa nesta data).
if FRAME == "J2000":
    R = np.eye(3)                                         # não gira nada
else:
    z = spice.pxform("IAU_JUPITER", "J2000", et)[:, 2].copy()  # polo de Júpiter
    x = spice.ucrss([0.0, 0.0, 1.0], z)                   # nó dos dois equadores
    y = spice.ucrss(z, x)                                 # completa os eixos
    R = np.array([x, y, z])

# 4) GM de Júpiter (km³/s²), lido do kernel PCK.
gm_jupiter = spice.bodvrd("JUPITER", "GM", 1)[1][0]

# 5) Para cada lua: ler o estado, girar, calcular os elementos.
linhas_estados = []
linhas_elementos = []

for nome, codigo in LUAS.items():
    # GM da lua (0 se o kernel não tiver)
    if spice.bodfnd(codigo, "GM"):
        gm_lua = spice.bodvrd(nome, "GM", 1)[1][0]
    else:
        gm_lua = 0.0

    # estado = [x, y, z, vx, vy, vz] da lua em relação a Júpiter, em J2000
    estado, _ = spice.spkezr(nome, et, "J2000", "NONE", "JUPITER")

    # gira posição e velocidade para o FRAME escolhido
    posicao = R @ estado[:3]
    velocidade = R @ estado[3:]
    estado = np.concatenate([posicao, velocidade])

    # estado -> elementos orbitais
    mu = gm_jupiter + gm_lua if SOMAR_GM_LUA else gm_jupiter
    el = spice.oscltx(estado, et, mu)
    a = el[9]                            # semieixo maior [km]
    e = el[1]                            # excentricidade
    i = np.degrees(el[2])                # inclinação [graus]
    Omega = np.degrees(el[3]) % 360      # nó ascendente [graus]
    omega = np.degrees(el[4]) % 360      # argumento do periapside [graus]
    M = np.degrees(el[5]) % 360          # anomalia média [graus]
    periodo_h = el[10] / 3600            # período [horas]

    x, y, z, vx, vy, vz = estado
    linhas_estados.append(
        f"{nome},{codigo},{gm_lua:.15g},{x:.15g},{y:.15g},{z:.15g},"
        f"{vx:.15g},{vy:.15g},{vz:.15g}")
    linhas_elementos.append(
        f"{nome},{codigo},{a:.10g},{e:.10g},{i:.10g},{Omega:.10g},"
        f"{omega:.10g},{M:.10g},{periodo_h:.10g}")

    print(f"{nome:9s} a = {a:10.1f} km   e = {e:.5f}   i = {i:.4f}°   P = {periodo_h:.3f} h")

# 6) Salvar as duas tabelas.
data_utc = spice.et2utc(et, "ISOC", 3)
info = (f"# epoca_utc={data_utc} et_tdb_s={et:.3f} frame={FRAME} "
        f"centro=JUPITER GM_JUPITER={gm_jupiter:.10g}")

with open("estados_iniciais.csv", "w") as f:
    f.write(info + "\n")
    f.write("nome,naif_id,GM_km3_s2,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n")
    for linha in linhas_estados:
        f.write(linha + "\n")

with open("elementos_iniciais.csv", "w") as f:
    f.write(info + "\n")
    f.write("nome,naif_id,a_km,e,i_deg,Omega_deg,omega_deg,M_deg,periodo_h\n")
    for linha in linhas_elementos:
        f.write(linha + "\n")

print("\nSalvo: estados_iniciais.csv e elementos_iniciais.csv")
