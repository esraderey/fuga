# Informe: barrido corto de dropout en FUGA-S eco_rel (3 000 pasos)

- Pre-registro: `ensayo/prerregistro_regularizacion.md` (fijado el 2026-10-30, antes de entrenar)
- Resultados crudos: `ensayo/regularizacion_resultados.json`; registros en `runs/reg/{d01,d02,d03}/log.json` y `runs/reg/cadena.log`
- Máquina: RTX 4060 (8,6 GB), Windows 11, torch 2.6+cu124, bf16, checkpointing de activaciones; una semilla por brazo

## Resultado en una frase

A 3 000 pasos con decaimiento completo de la tasa, subir el dropout no mejora el test de forma detectable (0,3) o lo
empeora un poco (0,2); lo que sí mejora al modelo, frente a la corrida de 3 h, es el presupuesto corto con decaimiento.
La hipótesis declarada (0,2 mejora, 0,3 no) cae.

## Cifras (validación completa: 617 trozos; test completo: 631 trozos de 475 piezas)

| Brazo | Dropout | Mejor paso | Val bits/nota | Test bits/nota | KR1: test pareado por pieza frente al control, IC95 | Veredicto |
|---|---|---|---|---|---|---|
| d01 (control) | 0,1 | 3 000 (final) | 3,486 | 3,766 | — | referencia |
| d02 | 0,2 | 3 000 (final) | 3,484 | 3,771 | +0,022 [+0,011; +0,033] | perjudica |
| d03 | 0,3 | 3 000 (final) | 3,443 | 3,737 | −0,003 [−0,015; +0,009] | sin efecto detectable |
| corrida de 3 h, mejor checkpoint | 0,1 | 3 250 de 9 436 | 3,533 | 3,848 | +0,067 [+0,049; +0,085] | peor que el control |

- **KR2.** En los tres brazos el mínimo de validación parcial es el último paso: a 3 000 pasos no hay memorización con
  ningún dropout, así que el barrido no puede decir si el dropout la retrasa.
- **KR3.** `modelo.pt` y `modelo_mejor.pt` coinciden en los tres brazos (sobreajuste 0,000); en la corrida de 3 h la
  distancia es 3,31 bits por nota.

## Lectura

- La ganancia grande no es del dropout sino del calendario: el control a 3 000 pasos gana 0,067 bits por nota por pieza
  al mejor checkpoint de la corrida de 3 h, que era una instantánea de la meseta sin decaimiento.
- El dropout 0,3 gana 0,04 en validación completa y 0,03 en el test agregado, pero el contraste pareado por pieza no lo
  distingue del control. La regla del pre-registro de 6 h (menor validación completa) lo eligió para la corrida larga.

## Desviaciones

- El control se entrenó tres veces: un corte de luz (2026-10-30) y una parada pedida por el usuario lo interrumpieron; la
  corrida válida es la del 2026-11-08. El brazo truncado se conserva en `runs/reg/d01_cortado_2034`.
- El bootstrap de KR1 usó 2 000 remuestreos (valor por defecto de `bootstrap`) y no los 10 000 pre-registrados. Los
  intervalos quedan lejos de 0 o lo contienen con holgura; el veredicto no depende de ello.

## Límites

Una semilla por brazo; un solo presupuesto (3 000 pasos); sin barrido conjunto con el decaimiento de pesos.
