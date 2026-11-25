# Entrenamiento final de FUGA: resumen (2026-11-25 16:34)

- Plan `final`, preset `fuga-s`, dropout 0.3, dispositivo cuda.
- Sonda: 0.164 s/paso. Duración real de la cadena hasta aquí: 117 min (~2.35 USD a 1.20 USD/h).
- Regla de elección: menor bits por nota en validación completa entre modelo_mejor.pt y modelo.pt de todas las corridas.

| Corrida | Checkpoint | Paso | Val bits/nota | Test bits/nota | |
|---|---|---|---|---|---|
| completo | modelo.pt | 18380 / 18380 | 8.186 | 8.591 |  |
| completo | modelo_mejor.pt | 4250 / 18380 | 3.442 | 3.745 |  |
| wsd_4500 | modelo.pt | 4500 / 4500 | 3.374 | 3.655 |  |
| wsd_4500 | modelo_mejor.pt | 4250 / 4500 | 3.362 | 3.645 | **elegido** |
| wsd_6000 | modelo.pt | 6000 / 6000 | 3.600 | 3.874 |  |
| wsd_6000 | modelo_mejor.pt | 4250 / 6000 | 3.480 | 3.778 |  |
| wsd_8000 | modelo.pt | 8000 / 8000 | 4.498 | 4.769 |  |
| wsd_8000 | modelo_mejor.pt | 4250 / 8000 | 3.448 | 3.746 |  |

- KF1 (mejor corto − mejor largo, pareado por pieza en test, 475 piezas): -0.0897 [-0.1035; -0.0758] → pasa (IC95 < 0).
- Z-Space (bf16): 37 versiones, 8.02 GB = 71.5 % del equivalente bf16 y 35.7 % del fp32; exacto final: True, exacto mejor: None; guardado medio 9.1 s, suma 328 s.
- Muestras (12 con notas): copia de 4 compases máx. 0.000, repetición media 0.177 (real 0.243); ilustran la ficha: [1, 8].
