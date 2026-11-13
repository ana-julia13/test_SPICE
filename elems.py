"""
Elementos orbitais osculadores (a, e, i, Ω, ω, M) das luas de Júpiter, de
Metis a Callisto, ao longo do tempo, a partir dos kernels SPICE. Serve para
comparar com a integração N-corpos.

Saídas (em PASTA_SAIDA):
  elementos_<lua>.csv / .png   série temporal dos 6 elementos de cada lua
  resumo_a_e_i.png             a/<a>, e, i de todas as luas juntas
  estados_iniciais.csv         estado jovicêntrico em INICIO (condição inicial do N-corpos)

Kernels (https://naif.jpl.nasa.gov/pub/naif/generic_kernels/), ou o meta-kernel da JUICE:
  lsk/naif0012.tls, pck/pck00011.tpc, pck/gm_de440.tpc, spk/satellites/jup365.bsp
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import spiceypy as spice

# ============================ CONFIGURAÇÃO ============================
KERNELS = ["/pho/figaro/estec/juice/kernels/mk/juice_plan.tm"]  # .tm ou lista de kernels
INICIO, FIM = "2030-07-01", "2040-07-01"   # UTC
PASSO_S = 600                              # s (Metis: período ~7 h)
FRAME = "EQUADOR_JUPITER"                  # ou "J2000"
SOMAR_GM_LUA = True                        # mu = GM_Júpiter + GM_lua (igual ao N-corpos)
PASTA_SAIDA = "saida_elementos"
LUAS = {"METIS": 516, "ADRASTEA": 515, "AMALTHEA": 505, "THEBE": 514,
        "IO": 501, "EUROPA": 502, "GANYMEDE": 503, "CALLISTO": 504}
# ======================================================================

ROTULOS = ["a [km]", "e", "i [°]", "Ω [°]", "ω [°]", "M [°]"]
PASTAS = {}  # kernel carregado -> sua pasta (caminhos no .tm são relativos a ela)


def carrega(kernels):
    """furnsh a partir da pasta de cada kernel, por causa dos caminhos relativos."""
    cwd = os.getcwd()
    for k in map(os.path.abspath, kernels):
        os.chdir(os.path.dirname(k))
        try:
            spice.furnsh(os.path.basename(k))
        finally:
            os.chdir(cwd)
        PASTAS[os.path.basename(k)] = os.path.dirname(k)


def cobertura(id_corpo):
    """União da cobertura de todos os SPKs carregados que têm este corpo."""
    cov = spice.cell_double(20000)
    for i in range(spice.ktotal("SPK")):
        arq, _, origem, _ = spice.kdata(i, "SPK")
        arq = os.path.join(PASTAS.get(origem or arq, os.getcwd()), arq)
        ids = spice.spkobj(arq)
        if id_corpo in [ids[j] for j in range(spice.card(ids))]:
            spice.spkcov(arq, id_corpo, cov)
    return cov


def gm(nome):
    """GM (km³/s²) do PCK, ou 0 se não houver."""
    return spice.bodvrd(nome, "GM", 1)[1][0] if spice.bodfnd(spice.bods2c(nome), "GM") else 0.0


def rotacao(et):
    """J2000 -> frame inercial: z no polo de Júpiter, x no nó do equador de Júpiter."""
    if FRAME == "J2000":
        return np.eye(3)
    if FRAME != "EQUADOR_JUPITER":
        raise ValueError(f"FRAME desconhecido: {FRAME}")
    z = spice.pxform("IAU_JUPITER", "J2000", et)[:, 2].copy()
    x = spice.ucrss([0.0, 0.0, 1.0], z)
    return np.array([x, spice.ucrss(z, x), z])


def formata(eixo):
    """x justo nos dados; y sem notação científica nem offset."""
    eixo.margins(x=0)
    eixo.grid(alpha=0.3)
    if eixo.get_yscale() == "log":
        eixo.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        eixo.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: np.format_float_positional(float(f"{v:.6g}"), trim="-")))
        eixo.yaxis.set_minor_formatter(NullFormatter())
    else:
        eixo.ticklabel_format(axis="y", style="plain", useOffset=False)


def salva_csv(caminho, cabecalho, linhas):
    with open(caminho, "w") as f:
        f.write(cabecalho + "\n" + "\n".join(linhas) + "\n")


carrega(KERNELS)
os.makedirs(PASTA_SAIDA, exist_ok=True)
saida = lambda nome: os.path.join(PASTA_SAIDA, nome)

et0, et1 = spice.str2et(INICIO), spice.str2et(FIM)
janela = spice.cell_double(2)
spice.wninsd(et0, et1, janela)
R = rotacao(et0)  # fixa em INICIO -> frame inercial
GM_JUP = gm("JUPITER")
if GM_JUP == 0.0:
    raise RuntimeError("GM de Júpiter não encontrado: carregue gm_de440.tpc")
print(f"GM Júpiter = {GM_JUP:.6f} km³/s²   frame = {FRAME}")

iniciais, dados = [], {}
for nome, naif in LUAS.items():
    periodo = spice.wnintd(cobertura(naif), janela)
    n = spice.wncard(periodo)
    if n == 0:
        print(f"{nome}: SEM cobertura SPK no período, pulando")
        continue
    if n > 1 or spice.wnfetd(periodo, 0) != (et0, et1):
        print(f"{nome}: cobertura parcial no período, usando só o que existe")

    gm_lua = gm(nome)
    mu = GM_JUP + (gm_lua if SOMAR_GM_LUA else 0.0)

    # estados jovicêntricos em J2000, girados para FRAME
    ets = np.concatenate([np.arange(*spice.wnfetd(periodo, k), PASSO_S) for k in range(n)])
    S = np.asarray(spice.spkezr(str(naif), ets, "J2000", "NONE", "JUPITER")[0])
    S = np.hstack([S[:, :3] @ R.T, S[:, 3:] @ R.T])
    iniciais.append((nome, naif, gm_lua, *S[0]))

    # oscltx: [rp, ecc, inc, lnode, argp, m0, t0, mu, nu, a, tau, ...]
    el = np.array([spice.oscltx(s, et, mu) for s, et in zip(S, ets)])
    tab = np.column_stack([el[:, 9], el[:, 1], np.degrees(el[:, 2]),
                           np.degrees(el[:, 3:6]) % 360.0])  # a, e, i, Ω, ω, M
    dados[nome] = (ets, tab)

    salva_csv(saida(f"elementos_{nome.lower()}.csv"),
              "data_utc,et_tdb_s,a_km,e,i_deg,Omega_deg,omega_deg,M_deg",
              [f"{spice.et2utc(et, 'ISOC', 3)},{et:.3f}," + ",".join(f"{v:.10g}" for v in linha)
               for et, linha in zip(ets, tab)])

    a_m, e_m, i_m = tab[:, :3].mean(axis=0)
    print(f"{nome:9s} GM={gm_lua:10.4f}  <a>={a_m:12.1f} km  <e>={e_m:.5f}  <i>={i_m:.4f}°")

    datas = spice.et2datetime(ets)
    fig, eixos = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    for j, (eixo, rot) in enumerate(zip(eixos, ROTULOS)):
        if j >= 3:  # ângulos: pontos, para não ligar o salto 360 -> 0
            eixo.plot(datas, tab[:, j], ".", ms=0.8)
            eixo.set_ylim(0, 360)
        else:
            eixo.plot(datas, tab[:, j], lw=0.6)
        eixo.set_ylabel(rot)
        formata(eixo)
    eixos[0].set_title(f"{nome} – elementos osculadores jovicêntricos (SPICE, {FRAME})")
    eixos[-1].set_xlabel("data (UTC)")
    fig.tight_layout()
    fig.savefig(saida(f"elementos_{nome.lower()}.png"), dpi=150)
    plt.close(fig)

# estados iniciais (km, km/s) no mesmo frame, para o N-corpos
salva_csv(saida("estados_iniciais.csv"),
          f"# epoca_utc={spice.et2utc(et0, 'ISOC', 3)} et_tdb_s={et0:.3f} "
          f"frame={FRAME} centro=JUPITER GM_JUPITER={GM_JUP:.10g}\n"
          "nome,naif_id,GM_km3_s2,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s",
          [f"{nome},{naif}," + ",".join(f"{v:.15g}" for v in resto)
           for nome, naif, *resto in iniciais])

# resumo: a/<a>, e, i de todas as luas
fig, eixos = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for nome, (ets, tab) in dados.items():
    datas = spice.et2datetime(ets)
    eixos[0].plot(datas, tab[:, 0] / tab[:, 0].mean(), lw=0.5, label=nome)
    eixos[1].plot(datas, tab[:, 1], lw=0.5)
    eixos[2].plot(datas, tab[:, 2], lw=0.5)
eixos[1].set_yscale("log")
for eixo, rot in zip(eixos, ["a / <a>", "e", "i [°]"]):
    eixo.set_ylabel(rot)
    formata(eixo)
eixos[0].legend(ncol=4, fontsize=8)
eixos[-1].set_xlabel("data (UTC)")
fig.tight_layout()
fig.savefig(saida("resumo_a_e_i.png"), dpi=150)
plt.close(fig)

print(f"\nArquivos salvos em {os.path.abspath(PASTA_SAIDA)}")
