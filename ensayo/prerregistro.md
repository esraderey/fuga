# Pre-registro del ensayo FUGA (CPU) — fijado el 2026-10-24 antes de ver resultados

Corpus: music21 core, dominio público, 14,955 piezas (train 14,102 / val 375 / test 478, partición por archivo).
Modelo idéntico en todos los brazos: preset `nano` (2.50M parámetros), ctx 512, AdamW, WSD, lr 2e-3, dropout 0.0,
360 pasos × 8,192 tokens por lote (≈2.9M tokens predichos o de contexto por brazo: igual cómputo), 2 transposiciones
deterministas por pieza en brazos absolutos, ninguna en el brazo relativo.

Brazos: `base` (literal), `eco` (ECO sin materializar), `fant` (ECO + contenido materializado sin pérdida),
`eco_rel` (ECO en alturas relativas a la tónica; ablación residual de MARCO).

| Id | Hipótesis | Criterio | Consecuencia si falla |
|---|---|---|---|
| K1 | ECO modela mejor que literal | IC95 pareado por pieza de bits/nota (eco − base) o (fant − base) < 0 | ECO cae como representación |
| K2 | ECO acerca la estructura a la real | IC95 pareado por prompt de D (eco|fant − base) < 0 | Se retira la reclamación sobre F1 (queda solo compresión) |
| K3 | Materializar el contenido ayuda | IC95 de bits/nota (fant − eco) < 0 | Informativo: decide si el RFC exige fantasmas |
| K4 | Coordenadas relativas ahorran datos | bits/nota (eco_rel − eco) < 0 y adherencia exacta ≥ | Informativo (MARCO ya cayó en triaje como solución de control) |
| K5 | La gramática garantiza decodificación | 0 fallos de decodificación en todas las generaciones | Defecto de implementación: bloquea el RFC |
| K6 | La perilla controla la forma | max P_trans crece de sesgo −3 a +3 con IC95 disjuntos | La perilla se retira |

Límites declarados: D se basa en repetición por lag y un operador de copia puede ganarla sin función musical
(falta el destructor D7 de la escalera metamórfica); 2.5M parámetros y ~3M tokens es un régimen subentrenado:
un "sobrevive" habilita la escalera de GPU, no confirma a 150–400M. Juez de tonalidad: Temperley–Kostka–Payne
(music21), distinto del etiquetador del corpus; se reporta su techo sobre música real.

Desviación registrada antes de ver resultados: dropout 0.1 → 0.0 en todos los brazos. Con dropout la atención en CPU cae al kernel sin fusión y cada proceso pasa de ~2 GB a ~4 GB de pico; tres de cuatro procesos murieron por memoria sin llegar a evaluar. Con <1 época no hay repetición de datos, así que el dropout no tenía función.
