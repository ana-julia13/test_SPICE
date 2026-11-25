import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spiceypy as sp

MK = "/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"
INI, FIM = sys.argv[1:3] if len(sys.argv) == 3 else ("2032-05-09 00:00", "2032-05-11 12:00")
PASSO, DUR_MIN, PX_LIM, LAT_MAX, F_SUL, F_NORTE = 60, 600, 10, 10, 70, 110
LUAS = {"METIS": 3, "ADRASTEA": 2, "AMALTHEA": 10, "THEBE": 6}  # lua: pixels mínimos
GAL = ["IO", "EUROPA", "GANYMEDE", "CALLISTO"]

cwd = os.getcwd(); os.chdir(os.path.dirname(MK)); sp.furnsh(MK); os.chdir(cwd)
IFOV = sp.gdpool("INS-28200_IFOV", 0, 1)[0]
ets = np.arange(sp.str2et(INI), sp.str2et(FIM), PASSO)
R = lambda c: sp.bodvrd(c, "RADII", 3)[1]
pos = lambda c, f="J2000": np.array(sp.spkpos(c, ets, f, "LT+S", "JUICE")[0])
occ = lambda a, b, obs: np.array([sp.occult(a, "ELLIPSOID", f"IAU_{a}", b, "ELLIPSOID", f"IAU_{b}", "LT", obs, t) for t in ets])

def sep(a, Ra, b, Rb):  # pixels entre os limbos de a e b, vistos da JUICE
    pa, pb = pos(a), pos(b)
    da, db = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
    th = np.arccos(np.clip((pa * pb).sum(1) / (da * db), -1, 1))
    return (th - np.arcsin(Rb / db) - np.arcsin(Ra / da)) / IFOV

for lua, px_min in LUAS.items():
    Rl = R(lua).mean()
    v = -pos(lua, f"IAU_{lua}"); d = np.linalg.norm(v, axis=1)
    px = 2 * Rl / (d * IFOV)
    lat = np.degrees(np.arcsin(v[:, 2] / d))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    fase = np.degrees([sp.phaseq(t, lua, "SUN", "JUICE", "LT+S") for t in ets])
    fmax = np.where(lat < 0, F_SUL, F_NORTE)
    prob = {"ocultada": np.any([occ(lua, c, "JUICE") < 0 for c in ["JUPITER"] + GAL], axis=0),
            "eclipse": occ("JUPITER", "SUN", lua) > 0,
            "perto de Júpiter": sep(lua, Rl, "JUPITER", R("JUPITER")[0]) < PX_LIM,
            "perto de galileana": np.min([sep(lua, Rl, g, max(R(g))) for g in GAL], axis=0) < PX_LIM,
            "fase alta": fase > fmax,
            f"|lat| > {LAT_MAX}°": abs(lat) > LAT_MAX}
    livre = ~np.any(list(prob.values()), axis=0)
    dl = np.diff(np.r_[0, livre.astype(int), 0])
    jan = [(i, j) for i, j in zip(np.where(dl == 1)[0], np.where(dl == -1)[0] - 1) if ets[j] - ets[i] >= DUR_MIN]

    mj = np.zeros(len(ets), bool)
    with open(f"janelas_{lua.lower()}.csv", "w") as f:
        f.write("inicio_utc,fim_utc,duracao_min,dist_km,pixels,fase,lat,lon\n")
        for i, j in jan:
            m = (i + j) // 2; mj[i:j + 1] = True
            f.write(f"{sp.et2utc(ets[i], 'ISOC', 0)},{sp.et2utc(ets[j], 'ISOC', 0)},{(ets[j] - ets[i]) / 60:.0f},"
                    f"{d[m]:.0f},{px[m]:.1f},{fase[m]:.1f},{lat[m]:.1f},{lon[m]:.1f}\n")
    print(f"{lua}: {len(jan)} janela(s)")

    t = sp.et2datetime(ets)
    fig, ax = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    for a, y, nome in zip(ax, [d, px, lat, lon, fase], ["dist [km]", "pixels", "lat [°]", "lon [°E]", "fase [°]"]):
        a.plot(t, y, ".", ms=1); a.set_ylabel(nome); a.grid(alpha=.3)
    ax[0].set_yscale("log"); ax[1].axhline(px_min, c="gray", ls="--")
    ax[2].axhspan(-LAT_MAX, LAT_MAX, alpha=.15); ax[4].plot(t, fmax, "r--")
    for k, m in enumerate(prob.values()):
        ax[5].fill_between(t, k - .35, k + .35, where=m, color="r", lw=0)
    ax[5].set_yticks(range(len(prob)), list(prob), fontsize=8)
    ax[6].fill_between(t, 0, 1, where=mj, color="g", lw=0)
    ax[6].set_ylabel("JANELA"); ax[6].set_yticks([])
    ax[0].set_title(lua); fig.tight_layout(); fig.savefig(f"janelas_{lua.lower()}.png", dpi=150); plt.close(fig)
