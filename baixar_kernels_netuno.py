"""
Baixa os kernels de Netuno e suas luas e cria um meta-kernel pronto pra usar.

Como usar:
    1. Salve este arquivo na pasta onde voce quer guardar os kernels.
    2. No terminal, dentro dessa pasta:   python baixar_kernels_netuno.py
    3. No seu codigo:                      spice.furnsh("kernels/netuno.tm")
       (use o caminho completo que o script imprimir no final)

Arquivos que ele ja tiver baixado antes sao pulados.
"""
from pathlib import Path
import urllib.request

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# pasta de destino -> lista de URLs
KERNELS = {
    "lsk": [f"{BASE}/lsk/naif0012.tls"],               # segundos bissextos (tempo)
    "pck": [f"{BASE}/pck/pck00011.tpc"],               # tamanho/rotacao dos corpos
    # ORDEM IMPORTA: se dois arquivos tem o mesmo corpo no mesmo instante,
    # o SPICE usa o carregado por ULTIMO. Por isso os "xl" (cobertura longa,
    # usados so como reserva fora do periodo dos outros) e o nep098 (mais
    # antigo) vem primeiro, e os mais novos (nep097/104/105) por ultimo.
    "spk": [
        f"{BASE}/spk/planets/de440s.bsp",              # Sol, Terra, baricentro de Netuno (8)
        f"{BASE}/spk/satellites/nep097xl-899.bsp",     # Netuno (899), cobertura longa
        f"{BASE}/spk/satellites/nep097xl-801.bsp",     # Tritao (801), cobertura longa
        f"{BASE}/spk/satellites/nep101xl-802.bsp",     # Nereida (802), cobertura longa
        f"{BASE}/spk/satellites/nep101xl.bsp",         # cobertura longa (confira com brief)
        f"{BASE}/spk/satellites/nep098_part-1.bsp",    # irregulares (precisa do nameid)
        f"{BASE}/spk/satellites/nep098_part-2.bsp",
        f"{BASE}/spk/satellites/nep098_part-3.bsp",
        f"{BASE}/spk/satellites/nep097.bsp",           # Netuno (899), Tritao e luas internas
        f"{BASE}/spk/satellites/nep104.bsp",           # irregulares externas
        f"{BASE}/spk/satellites/nep105.bsp",           # Nereida
    ],
    "fk": [f"{BASE}/fk/satellites/nep098_nameid.tf"],  # nomes das luas mais novas
}

raiz = Path(__file__).resolve().parent / "kernels"


def baixar(url: str, destino: Path) -> bool:
    """Baixa o arquivo. Retorna True se ele estiver disponivel no fim."""
    if destino.exists() and destino.stat().st_size > 0:
        print(f"  ja existe: {destino.name}")
        return True
    print(f"  baixando:  {destino.name} ...", end="", flush=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:  # 404, queda de conexao etc.
        print(f" FALHOU ({e}) -- pulando, nao vai entrar no meta-kernel")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(destino)
    tam = destino.stat().st_size
    print(f" ok ({tam / 1e6:.1f} MB)" if tam >= 1e6 else f" ok ({tam / 1e3:.1f} KB)")
    return True


carregar = []
for pasta, urls in KERNELS.items():
    (raiz / pasta).mkdir(parents=True, exist_ok=True)
    for url in urls:
        nome = url.rsplit("/", 1)[1]
        if baixar(url, raiz / pasta / nome):
            carregar.append(f"$KERNELS/{pasta}/{nome}")


def quebrar(texto: str, n: int = 60) -> list[str]:
    """Strings de kernel tem limite de 80 caracteres; quebra com '+' no fim."""
    partes = [texto[i:i + n] for i in range(0, len(texto), n)]
    return [p + "+" for p in partes[:-1]] + [partes[-1]]


caminho = raiz.as_posix()
linhas_path = "\n".join(f"                     '{p}'" for p in quebrar(caminho))
linhas_kern = "\n".join(f"                        '{k}'" for k in carregar)

meta = f"""KPL/MK

Meta-kernel para Netuno e suas luas (gerado por baixar_kernels_netuno.py).

\\begindata

   PATH_VALUES     = (
{linhas_path}
                     )
   PATH_SYMBOLS    = ( 'KERNELS' )

   KERNELS_TO_LOAD = (
{linhas_kern}
                     )

\\begintext
"""

mk = raiz / "netuno.tm"
mk.write_text(meta)
print(f"\nMeta-kernel criado: {mk}")

# Teste rapido, se o SpiceyPy estiver instalado
try:
    import spiceypy as spice

    spice.furnsh(str(mk))
    et = spice.str2et("2026-11-09T12:00:00")
    pos, _ = spice.spkpos("TRITON", et, "J2000", "NONE", "NEPTUNE")
    dist = (pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5
    print(f"Teste OK: Tritao esta a {dist:,.0f} km do centro de Netuno (500@899).")
    spice.kclear()
except ImportError:
    print("(SpiceyPy nao instalado: pulei o teste. Instale com: pip install spiceypy)")
