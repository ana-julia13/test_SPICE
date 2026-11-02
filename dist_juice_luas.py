import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import spiceypy as spice

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INICIO, FIM = "2032-02-01", "2032-10-01"
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514}
SAIDA = os.path.abspath("resultados")

os.makedirs(SAIDA, exist_ok=True)
os.chdir(os.path.dirname(MK))  # o meta-kernel usa caminhos relativos à pasta mk/
spice.kclear()
spice.furnsh(MK)


def cobertura(id_corpo):
    """Período em que existe trajetória deste corpo nos kernels carregados."""
    cob = spice.cell_double(200000)
    for i in range(spice.ktotal("SPK")):
        spice.spkcov(spice.kdata(i, "SPK")[0], id_corpo, cob)
    return cob


def grafico(eixo, nome, datas, dists, et_min, d_min):
    data_min = spice.et2datetime(et_min)
    eixo.semilogy(datas, dists, lw=0.6)
    eixo.plot(data_min, d_min, "ro", ms=5,
              label=f"mínimo: {d_min:,.0f} km em {data_min:%Y-%m-%d %H:%M} UTC")
    eixo.legend(loc="upper right", fontsize=8)
    eixo.set_xlim(datas[0], datas[-1])
    km = FuncFormatter(lambda v, _: f"{v:,.0f}")
    eixo.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    eixo.yaxis.set_major_formatter(km)
    eixo.yaxis.set_minor_formatter(km if np.nanmax(dists) / np.nanmin(dists) < 10 else NullFormatter())
    eixo.set_title(f"Distância JUICE – {nome.capitalize()}")
    eixo.set_ylabel("distância [km] (escala log)")
    eixo.grid(True, which="both", alpha=0.3)


# período pedido, cortado para onde há trajetória da JUICE
periodo = spice.cell_double(2)
spice.wninsd(spice.str2et(INICIO), spice.str2et(FIM), periodo)
periodo = spice.wnintd(periodo, cobertura(-28))

resultados = []
for nome, id_lua in LUAS.items():
    janela = spice.wnintd(periodo, cobertura(id_lua))

    # distância a cada 30 min (NaN entre intervalos para o gráfico não ligar os pontos)
    ets, dists = [], []
    for k in range(spice.wncard(janela)):
        a, b = spice.wnfetd(janela, k)
        t = np.append(np.arange(a, b, 1800), b)
        pos, _ = spice.spkpos(nome, t, "J2000", "NONE", "JUICE")
        ets += [*t, b]
        dists += [*np.linalg.norm(pos, axis=1), np.nan]
    ets, dists = np.array(ets), np.array(dists)

    # menor distância (busca a cada 10 min)
    minimos = spice.cell_double(400000)
    spice.gfdist(nome, "NONE", "JUICE", "ABSMIN", 0.0, 0.0, 600, 200000, janela, minimos)
    et_min = spice.wnfetd(minimos, 0)[0]
    d_min = spice.vnorm(spice.spkpos(nome, et_min, "J2000", "NONE", "JUICE")[0])
    print(f"{nome}: menor distância {d_min:,.0f} km em {spice.et2utc(et_min, 'C', 0)} UTC")

    datas = spice.et2datetime(ets)
    arquivo = f"{SAIDA}/distancia_juice_{nome.lower()}"

    fig, eixo = plt.subplots(figsize=(12, 4))
    grafico(eixo, nome, datas, dists, et_min, d_min)
    eixo.set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(arquivo + ".png", dpi=150)

    with open(arquivo + ".csv", "w") as f:
        f.write("data_utc,distancia_km\n")
        for et, d in zip(ets, dists):
            if np.isfinite(d):
                f.write(f"{spice.et2utc(et, 'ISOC', 0)},{d:.3f}\n")

    resultados.append((nome, datas, dists, et_min, d_min))

# as 4 luas juntas, uma embaixo da outra
fig, eixos = plt.subplots(len(resultados), 1, sharex=True, figsize=(12, 3 * len(resultados)))
for eixo, r in zip(eixos, resultados):
    grafico(eixo, *r)
eixos[-1].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(f"{SAIDA}/distancia_juice_todas.png", dpi=150)
