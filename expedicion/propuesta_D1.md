# Propuesta D1 — ECO: operadores de referencia con transformación dentro del vocabulario del LM

## (1) Mecanismo
Representación por compás y parte (orden parte-mayor dentro del compás). Cada segmento (compás b, parte p) se codifica de una de tres formas, elegidas por un parseo canónico determinista en el preprocesado:
- literal: PART_p (POS PITCH DUR | PITCH DUR)*
- ECO (copia exacta transformada): PART_p ECO DIST_d FROM_q (TRC_k | TRD_k) — "este segmento es el de la parte q, d compases atrás (d=0: parte anterior del mismo compás, para doblajes), transpuesto k semitonos (TRC, k∈[−12,12]) o k grados diatónicos de la tonalidad declarada (TRD, k∈[−7,7])".
- ECOR (eco rítmico): PART_p ECOR DIST_d FROM_q PITCH×n — copia posiciones y duraciones de la fuente; el modelo solo emite las n alturas nuevas.
Parseo canónico: para cada segmento se busca en d∈[0,32] y en todas las partes la referencia más cercana (prefiere misma parte y TRC_0); ECO solo si ahorra tokens (literal >4 tokens), ECOR si 3+n < literal. El LM es un decoder causal estándar entrenado con NLL sobre esta derivación canónica. Decodificación con gramática: un autómata incremental mantiene el contenido resuelto y solo permite DIST/FROM que existan y TRC/TRD que no saquen notas de rango ni rompan la escala; toda salida es decodificable al 100%. Un sesgo aditivo sobre los logits de ECO/ECOR es una perilla de forma en inferencia (más o menos repetición). Para que la atención resuelva "d compases atrás", una cabeza kv de cada capa rota su RoPE por índice de compás en vez de por índice de token (adopción de la familia de posiciones musicales ya existente, no novedad por sí sola).

## (2) Novedad contra el mapa
Casilla F1. El mapa localiza atención selectiva a compases (Museformer), parcheo por compás (NotaGen), condicionamiento por tema (Theme Transformer), BPE (compresión estadística sin transformación ni referencia a compases) y, como adyacentes, LZMidi (árbol LZ78 no neuronal, monofónico, sin transposición) y R-Transformer (generador separado de variaciones de motivo). No se localizó un operador explícito de referencia-con-transformación (cromática, diatónica, rítmica, entre partes) dentro del vocabulario de un LM autorregresivo. ECO convierte la repetición musical —el rasgo que los LMs de audio declaran no resolver— en una decisión discreta, barata y controlable.

## (3) Coste
Cero parámetros extra (≈50 tokens nuevos en el vocabulario). Preprocesado O(32 × partes × n) por segmento, en CPU. Medido en una muestra de 1,186 piezas folk de dominio público: tokens/nota 3.48 → 2.49 (−28.5%), ida y vuelta exacta en las 1,186. En 4096 tokens de contexto cabe ~40% más música.

## (4) Riesgos
- Degeneración por copia excesiva (el modelo aprende que copiar es barato) → música en bucle.
- La NLL de la derivación canónica es solo una cota superior de −log p(música) (hay derivaciones alternativas): la comparación con la base puede desfavorecer a ECO, no favorecerlo.
- En repertorios poco repetitivos o con microtiming expresivo (MIDI interpretado) las coincidencias exactas escasean y la ganancia cae.
- Error en cascada: una referencia mal elegida copia un compás entero equivocado.

## (5) Falsación
Ensayo CPU: dos modelos idénticos de ~2.5M parámetros, mismo presupuesto de tiempo, mismos datos y mismos rangos de compases de evaluación; uno con ECO, otro literal. Muere si (a) los bits por nota en test de ECO no son menores que los de la base, o (b) la tasa de repetición de las muestras generadas se aleja más de la del conjunto real que la de la base.
