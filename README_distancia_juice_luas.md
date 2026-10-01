# Distância JUICE ↔ luas internas de Júpiter

O script `distancia_juice_luas.py` faz o gráfico da **distância entre a sonda JUICE e
Metis, Adrastea, Amalthea e Thebe** (nessa ordem) ao longo da missão, usando SPICE
(pela biblioteca Python **SpiceyPy**).

Este README começa do zero: primeiro explica o que é SPICE, depois como rodar o
script e por último o que cada parte do código faz.

---

## 1. Como rodar (versão rápida)

```bash
pip install spiceypy numpy matplotlib
python distancia_juice_luas.py
```

O script já está configurado para os seus kernels em `/pho/figaro/estec/juice/kernels`.
Os resultados vão para a pasta `resultados/`:

| Arquivo | O que é |
|---|---|
| `distancia_juice_metis.png` | gráfico distância × tempo para Metis |
| `distancia_juice_adrastea.png` | ... para Adrastea |
| `distancia_juice_amalthea.png` | ... para Amalthea |
| `distancia_juice_thebe.png` | ... para Thebe |
| `distancia_juice_todas.png` | as 4, uma embaixo da outra, mesmo eixo de tempo |
| `distancia_juice_<lua>.csv` | tabela `data_utc, distancia_km` (dá para abrir no Excel) |

No terminal ele também mostra o período com dados e a **menor distância** (data e km)
de cada lua.

---

## 2. O que é SPICE, explicado sem jargão

**SPICE** é um sistema da NASA/JPL (o NAIF) usado por praticamente todas as missões
planetárias para responder perguntas de **geometria**: *onde está a sonda? onde está
a lua? a que distância? para onde a câmera está apontando?*

Ele tem duas partes:

1. **Os dados**, guardados em arquivos chamados **kernels**.
2. **O software** (o "toolkit"), que lê os kernels e faz as contas. Em Python ele se
   chama **SpiceyPy** (`import spiceypy as spice`).

Você nunca abre os kernels na mão: você **carrega** os kernels e depois **pergunta**
coisas ao SPICE.

### 2.1 Os tipos de kernel (as pastas dentro de `kernels/`)

Cada pasta guarda um tipo de informação. Para este script, as importantes estão em **negrito**:

| Pasta | Nome | O que guarda | Usamos? |
|---|---|---|---|
| **`spk/`** | SP-Kernel | **Posições ao longo do tempo** (trajetória da JUICE, órbitas dos planetas e das luas). Arquivos `.bsp`. | **Sim, é o principal** |
| **`lsk/`** | Leapseconds | Tabela de segundos intercalares, necessária para converter datas (UTC) para o tempo interno do SPICE. Arquivo `naifNNNN.tls`. | **Sim** |
| **`mk/`** | Meta-kernel | Uma "lista de compras": um arquivo de texto `.tm` que diz quais kernels carregar. | **Sim** |
| `pck/` | Planetary constants | Tamanho, forma e rotação dos corpos (raio de Júpiter etc.). | Carregado, não usado aqui |
| `fk/` | Frames | Definição de sistemas de coordenadas. | Carregado, não usado aqui |
| `ck/` | C-Kernel | Orientação (para onde a sonda aponta). | Não precisa para distância |
| `sclk/` | Spacecraft clock | Relógio de bordo da sonda ↔ tempo na Terra. | Não precisa |
| `ik/` | Instrument | Campo de visão dos instrumentos. | Não precisa |
| `dsk/` | Shape | Modelos 3D de forma. | Não precisa |

**Resumo:** distância = posição da lua − posição da JUICE. As posições estão nos SPKs.

### 2.2 O meta-kernel (`.tm`), ou por que não carregar arquivo por arquivo

São dezenas de kernels. Em vez de carregar um por um, o meta-kernel lista todos.
Por dentro um `.tm` se parece com isto:

```
PATH_VALUES     = ( '..' )
PATH_SYMBOLS    = ( 'KERNELS' )
KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls'
                    '$KERNELS/spk/juice_crema_....bsp'
                    ... )
```

`$KERNELS` é substituído por `PATH_VALUES`, que é `'..'` (a pasta acima de `mk/`).
**Esse `'..'` é relativo à pasta onde o programa está rodando**, por isso o README
oficial da JUICE manda editar o `PATH_VALUES`. O script resolve isso sozinho: ele
entra na pasta `mk/` só durante o carregamento e depois volta. **Você não precisa
editar nada no meta-kernel.**

Qual meta-kernel usar (variável `META_KERNEL` no script):

- **`juice_plan.tm`** (padrão): trajetória **planejada** da missão inteira, ou seja, a
  trajetória real já voada mais a prevista (cenário "crema") até o fim. Ideal para
  "a missão toda".
- `juice_ops.tm`: só o que foi voado mais a previsão de curto prazo. Ainda não cobre a
  fase em Júpiter.
- `juice_crema_<versão>.tm`: um cenário de trajetória específico do *Mission Analysis*.
  Use se o seu grupo trabalha com uma versão crema específica.

Se você colocar um nome que não existe, o script lista todos os `.tm` disponíveis.

### 2.3 Códigos NAIF: cada corpo tem um número

O SPICE identifica tudo por um número inteiro (e aceita também o nome):

| Corpo | Nome SPICE | Código NAIF |
|---|---|---|
| JUICE (sonda) | `JUICE` | **-28** (sondas têm código negativo) |
| Júpiter | `JUPITER` | 599 |
| Amalthea | `AMALTHEA` | 505 |
| Thebe | `THEBE` | 514 |
| Adrastea | `ADRASTEA` | 515 |
| Metis | `METIS` | 516 |

Repare: as luas de Júpiter são 5xx (Júpiter é o 5º planeta). Io é 501, Europa 502 etc.

### 2.4 Tempo no SPICE: ET

Internamente o SPICE usa **ET (Ephemeris Time)**: **segundos desde 1 jan 2000, 12:00
TDB**. É só um número, o que facilita a conta. Para converter:

- `spice.str2et("2031-07-01")` → texto para ET (precisa do kernel de leapseconds!)
- `spice.et2utc(et, "C", 0)` → ET para texto legível
- `spice.et2datetime(et)` → ET para `datetime` do Python (bom para o matplotlib)

### 2.5 "Cobertura" e "janelas"

Um SPK só tem dados para um certo período (a **cobertura**). Se você pedir a posição
fora dele, o SPICE dá erro (`SPKINSUFFDATA`). Por isso o script **primeiro pergunta**
quais períodos têm dados da JUICE e de cada lua, e só calcula onde os dois existem.

O SPICE representa períodos como **janelas** (*windows*): uma lista de intervalos
`[início, fim]`. Funções úteis:

- `spkcov(arquivo, id, janela)` → cobertura de um corpo num SPK
- `wnintd(a, b)` → interseção de duas janelas (onde **as duas** têm dados)
- `wncard(janela)` → quantos intervalos; `wnfetd(janela, k)` → o k-ésimo intervalo

### 2.6 A conta da distância: `spkpos`

```python
pos, lt = spice.spkpos("METIS", et, "J2000", "NONE", "JUICE")
distancia = spice.vnorm(pos)   # km
```

Leia como: *"posição de METIS, no instante `et`, no referencial J2000, sem
correção, vista da JUICE"*.

- **`"METIS"`** = alvo (*target*); **`"JUICE"`** = observador.
- **`"J2000"`** = sistema de coordenadas (*frame*) inercial padrão. Para **distância**
  o frame não importa, porque o tamanho do vetor é o mesmo em qualquer sistema.
- **`"NONE"`** = correção de aberração. `"NONE"` dá a distância **geométrica**, ou
  seja, onde os corpos realmente estão no mesmo instante. A alternativa `"LT+S"` daria
  onde a lua *parece* estar por causa do tempo que a luz leva para viajar. Para
  distância, `"NONE"` é o certo.
- Retorna `pos` = vetor (x, y, z) em **km** e `lt` = tempo-luz em segundos.

### 2.7 Achar a menor distância exata: `gfdist`

Um gráfico com pontos a cada 30 min pode "pular" o momento exato do sobrevoo.
O **Geometry Finder** (`gfdist`) do SPICE busca o mínimo com precisão:

```python
spice.gfdist("METIS", "NONE", "JUICE", "ABSMIN", 0, 0, passo, 1000, janela, resultado)
```

`"ABSMIN"` = mínimo absoluto dentro da janela. O `passo` (600 s) precisa ser menor que
o tempo entre dois mínimos/máximos seguidos da distância.

### 2.8 Carregar e descarregar

- `spice.furnsh(arquivo)` → carrega um kernel (ou meta-kernel)
- `spice.kclear()` → descarrega tudo (boa prática no fim)

---

## 3. Configuração no topo do script

```python
PASTA_KERNELS = "/pho/figaro/estec/juice/kernels"
META_KERNEL   = "juice_plan.tm"
INICIO = None          # None = missão inteira
FIM    = None          # ou ex.: INICIO = "2031-07-01", FIM = "2035-12-31"
PASSO_GRAFICO_S = 1800 # um ponto a cada 30 min no gráfico
PASSO_BUSCA_S   = 600  # passo da busca do mínimo
```

Dicas:

- **Ver só a fase em Júpiter:** a chegada da JUICE a Júpiter está prevista para
  **julho de 2031**. Use `INICIO = "2031-07-01"`. Antes disso a sonda está no
  cruzeiro, a centenas de milhões de km, e as quatro curvas ficam quase iguais.
- **Ver um sobrevoo de perto:** coloque um intervalo curto (ex.: 2 dias) e diminua
  `PASSO_GRAFICO_S` para 60.
- O eixo y está em **escala log**, porque a distância vai de bilhões de km (cruzeiro)
  até milhares de km. Para escala linear, troque `semilogy` por `plot` na função
  `grafico`.

---

## 4. O que o script faz, passo a passo

1. **`carrega_meta_kernel`**: entra em `kernels/mk/`, faz `furnsh` do `.tm` e volta.
2. **`cobertura_spk(-28)`**: percorre todos os SPKs carregados (`ktotal`/`kdata`), vê
   quais têm a JUICE (`spkobj`) e junta a cobertura (`spkcov`). Isso define o
   "tempo da missão".
3. Para **cada lua** (Metis → Adrastea → Amalthea → Thebe):
   - acha a cobertura da lua e cruza com a da JUICE (`wnintd`);
   - **`serie_distancias`**: gera instantes a cada `PASSO_GRAFICO_S` e chama
     `spkpos` para obter a distância. Onde há buraco nos dados, insere `NaN` para o
     gráfico não ligar os pontos;
   - **`distancia_minima`**: usa `gfdist` para achar a aproximação mínima exata;
   - salva o PNG e o CSV.
4. Faz o gráfico com as 4 luas juntas e chama `kclear()`.

---

## 5. Problemas comuns

| Mensagem | Causa / solução |
|---|---|
| `meta-kernel não encontrado` | Nome errado em `META_KERNEL`. O script lista os disponíveis; escolha um. |
| `NENHUM SPK carregado tem a efeméride desta lua` | O meta-kernel não inclui as luas internas. Procure em `kernels/spk/` um arquivo `jupNNN*.bsp` ou `noe-5-*.bsp` (efemérides das luas de Júpiter) e use um meta-kernel que o inclua, ou adicione um `spice.furnsh(".../spk/<arquivo>.bsp")` depois de `carrega_meta_kernel`. |
| `Sem período em comum com a JUICE` | A efeméride da lua não cobre as datas da trajetória (ou o `INICIO`/`FIM` escolhido). |
| `SPICE(NOLEAPSECONDS)` | O kernel `lsk/naif*.tls` não foi carregado. Verifique o meta-kernel. |
| `SPICE(SPKINSUFFDATA)` | Pediu posição fora da cobertura. O script evita isso; se aparecer, confira `INICIO`/`FIM`. |
| `SPICE(FILENOTFOUND)` | Algum arquivo listado no `.tm` não existe na sua pasta (repositório incompleto). Atualize os kernels ou use outro meta-kernel. |
| Script muito lento | Aumente `PASSO_GRAFICO_S` (ex.: 3600) ou restrinja `INICIO`/`FIM`. |

---

## 6. Para aprender mais

- Tutoriais do NAIF (em inglês, muito bons, com slides): https://naif.jpl.nasa.gov/naif/tutorials.html
  (comece por "Overview", "SPK" e "Time")
- Documentação do SpiceyPy: https://spiceypy.readthedocs.io
- ESA SPICE Service (kernels da JUICE): https://spice.esac.esa.int
