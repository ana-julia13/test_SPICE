import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

META_KERNEL = r"/caminho/para/JUICE/kernels/mk/juice_crema_5_1_150la.tm"
INICIO, FIM = "2031-07-01", "2032-07-01"
PASSO_S = 3600
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}

spice.furnsh(META_KERNEL)

# GM de Júpiter (para as galileanas, some o GM da lua também)
_, gm = spice.bodvrd("JUPITER", "GM", 1)
mu = gm[0]


def matriz_equador_jupiter(et):
    """Matriz J2000 -> frame INERCIAL alinhado ao equador de Júpiter em 'et'."""
    polo = spice.pxform("IAU_JUPITER", "J2000", et)[:, 2]  # eixo z de Júpiter em J2000
    nodo = spice.ucrss([0.0, 0.0, 1.0], polo)             # nó do equador de Júpiter
    y = spice.ucrss(polo, nodo)
    return np.array([nodo, y, polo])


et0, et1 = spice.str2et(INICIO), spice.str2et(FIM)
ets = np.arange(et0, et1, PASSO_S)
datas = spice.et2datetime(ets)
R = matriz_equador_jupiter(et0)  # fixa na época inicial

fig, eixos = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for nome, id_lua in LUAS.items():
    estados, _ = spice.spkezr(str(id_lua), ets, "J2000", "NONE", "JUPITER")
    elems = []
    for et, s in zip(ets, np.asarray(estados)):
        s_eq = np.concatenate([R @ s[:3], R @ s[3:]])  # gira pos. e vel.
        elems.append(spice.oscltx(s_eq, et, mu))
    elems = np.array(elems)

    a = elems[:, 9]
    e = elems[:, 1]
    inc = np.degrees(elems[:, 2])
    varpi = np.degrees(np.unwrap(elems[:, 3] + elems[:, 4]))  # longitude do periapse

    eixos[0].plot(datas, a, lw=0.6, label=nome)
    eixos[1].plot(datas, e, lw=0.6)
    eixos[2].plot(datas, inc, lw=0.6)
    eixos[3].plot(datas, varpi, lw=0.6)

for eixo, rot in zip(eixos, ["a [km]", "e", "i [°]", "ϖ = Ω+ω [°]"]):
    eixo.set_ylabel(rot)
    eixo.grid(alpha=0.3)
eixos[0].legend()
eixos[-1].set_xlabel("data (UTC)")
plt.tight_layout()
plt.savefig("elementos_luas.png", dpi=150)
spice.kclear()
