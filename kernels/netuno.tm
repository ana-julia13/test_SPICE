KPL/MK

Meta-kernel para Netuno e suas luas.

Ordem importa: quando dois arquivos tem o mesmo corpo no mesmo instante,
vale o carregado por ULTIMO. Os "xl" (cobertura longa) vem antes, entao
nep097/nep105 tem prioridade dentro da cobertura deles e os xl so sao
usados fora dela.

Ajuste PATH_VALUES para a pasta onde estao os kernels no seu PC.

\begindata

   PATH_VALUES     = ( 'C:/SPICE/kernels' )
   PATH_SYMBOLS    = ( 'K' )

   KERNELS_TO_LOAD = ( '$K/naif0012.tls'
                       '$K/pck00011.tpc'
                       '$K/de440s.bsp'
                       '$K/nep097xl-899.bsp'
                       '$K/nep101xl-802.bsp'
                       '$K/nep097.bsp'
                       '$K/nep104.bsp'
                       '$K/nep105.bsp'
                       '$K/nep098_part-1.bsp' )

\begintext
