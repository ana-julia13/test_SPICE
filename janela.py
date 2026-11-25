import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as sp

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INI = "2032-05-09 00:00"
FIM = "2032-05-11 12:00"
PASSO = 60        # s
DUR_MIN = 600     # s, duração mínima da janela
PX_LIM = 10       # px, distância mínima de qualquer limbo
LAT_MAX = 10      # graus
F_SUL = 70        # fase máxima com lat < 0
F_NORTE = 110     # fase máxima com lat > 0
LUAS = {"METIS": 3, "ADRASTEA": 2, "AMALTHEA": 10, "THEBE": 6}  # lua: pixels mínimos
GAL = ["IO", "EUROPA", "GANYMEDE", "CALLISTO"]

pasta = os.getcwd()
os.chdir(os.path.dirname(MK))   # o meta-kernel usa caminhos relativos à pasta dele
sp.furnsh(MK)
os.chdir(pasta)

IFOV = sp.gdpool("INS-28200_IFOV", 0, 1)[0]
ets = np.arange(sp.str2et(INI), sp.str2et(FIM), PASSO)
t = sp.et2datetime(ets)


def pos(corpo, frame="J2000"):
    return np.array(sp.spkpos(corpo, ets, frame, "LT+S", "JUICE")[0])


def raio(corpo):
    return sp.bodvrd(corpo, "RADII", 3)[1].max()


def ocultado(alvo, frente, obs):
    return np.array([sp.occult(alvo, "ELLIPSOID", f"IAU_{alvo}", frente, "ELLIPSOID", f"IAU_{frente}",
                               "LT", obs, et) for et in ets])


def separacao_px(lua, corpo):
    """Pixels entre os limbos da lua e do corpo, vistos da JUICE."""
    a, b = pos(lua), pos(corpo)
    da, db = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ang = np.arccos(np.clip(np.sum(a * b, axis=1) / (da * db), -1, 1))
    return (ang - np.arcsin(raio(lua) / da) - np.arcsin(raio(corpo) / db)) / IFOV


for lua, px_min in LUAS.items():
    v = -pos(lua, f"IAU_{lua}")                       # lua -> JUICE
    dist = np.linalg.norm(v, axis=1)
    px = 2 * sp.bodvrd(lua, "RADII", 3)[1].mean() / (dist * IFOV)
    lat = np.degrees(np.arcsin(v[:, 2] / dist))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    fase = np.degrees([sp.phaseq(et, lua, "SUN", "JUICE", "LT+S") for et in ets])
    fase_max = np.where(lat < 0, F_SUL, F_NORTE)

    problemas = {
        "ocultada": np.any([ocultado(lua, c, "JUICE") < 0 for c in ["JUPITER"] + GAL], axis=0),
        "eclipse": ocultado("JUPITER", "SUN", lua) > 0,
        "perto de Júpiter": separacao_px(lua, "JUPITER") < PX_LIM,
        "perto de galileana": np.min([separacao_px(lua, g) for g in GAL], axis=0) < PX_LIM,
        "fase alta": fase > fase_max,
        f"|lat| > {LAT_MAX}°": np.abs(lat) > LAT_MAX,
    }

    # janelas = trechos sem nenhum problema que duram >= DUR_MIN
    livre = ~np.any(list(problemas.values()), axis=0)
    bordas = np.diff(np.concatenate([[0], livre.astype(int), [0]]))
    janelas = [(i, j) for i, j in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1) - 1)
               if ets[j] - ets[i] >= DUR_MIN]
    na_janela = np.zeros(len(ets), bool)

    with open(f"janelas_{lua.lower()}.csv", "w") as f:
        f.write("inicio_utc,fim_utc,duracao_min,dist_km,pixels,fase,lat,lon\n")
        for i, j in janelas:
            na_janela[i:j + 1] = True
            m = (i + j) // 2
            f.write(f"{sp.et2utc(ets[i], 'ISOC', 0)},{sp.et2utc(ets[j], 'ISOC', 0)},{(ets[j] - ets[i]) / 60:.0f},{dist[m]:.0f},{px[m]:.1f},"
                    f"{fase[m]:.1f},{lat[m]:.1f},{lon[m]:.1f}\n")
    print(f"{lua}: {len(janelas)} janela(s)")

    fig, ax = plt.subplots(7, 1, figsize=(12, 17), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1.4, 0.35]})
    paineis = [
        (dist, "distância [km]", "tab:blue", "-"),
        (px, "pixels", "tab:orange", "-"),
        (lat, "lat. sub-S/C [°]", "tab:green", "-"),
        (lon, "lon. sub-S/C [°E]", "tab:brown", "."),   # pontos: a lon pula de 360 para 0
        (fase, "fase α [°]", "tab:purple", "-"),
    ]
    for a, (y, rotulo, cor, estilo) in zip(ax, paineis):
        a.plot(t, y, estilo, color=cor, lw=0.6, ms=1.5)
        a.set_ylabel(rotulo)
        a.grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[1].axhline(px_min, color="gray", ls="--")
    ax[2].axhspan(-LAT_MAX, LAT_MAX, color="tab:green", alpha=0.15)
    ax[4].plot(t, fase_max, "r--", lw=0.8)

    for k, acontece in enumerate(problemas.values()):
        ax[5].fill_between(t, k - 0.35, k + 0.35, where=acontece, color="tab:red", lw=0)
    ax[5].set_yticks(range(len(problemas)), list(problemas), fontsize=8)
    ax[5].invert_yaxis()

    ax[6].fill_between(t, 0, 1, where=na_janela, color="tab:green", lw=0)
    ax[6].set_yticks([])
    ax[6].set_ylabel("JANELA", fontsize=8)
    ax[6].set_xlim(t[0], t[-1])

    ax[0].set_title(lua)
    fig.tight_layout()
    fig.savefig(f"janelas_{lua.lower()}.png", dpi=150)
    plt.close(fig)
