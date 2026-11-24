# Notas de la exploración — Etapa 2 y comienzo de la Etapa 3 (SEM, BSE)

Notas de trabajo, no documentación formal. Resumen de lo probado entre el 22 y el 24/11/2026 para
pasarlo después a `docs/sem.md` y a funciones en `src/sem/`.

## Dónde está cada cosa

```
pruebas_sem/
    NOTAS_EXPLORACION.md          este archivo
    respaldo_exploracion/
        suavizado/                figuras del suavizado (01-06)
        sombras/                  figuras de corrección de manchas (01-10)
        ecualizacion/             figuras de ecualización (01-04)
        segmentacion/             figuras de segmentación y paisajes (01-56)
        scripts/                  scripts que generaron las figuras
```

- Las figuras (`.png`) **no se versionan** (el `.gitignore` ignora `*.png`): quedan solo en la
  máquina donde se generaron. Las notas y los scripts sí se versionan.
- Los scripts se corren desde la raíz del proyecto
  (`uv run python pruebas_sem/respaldo_exploracion/scripts/<script>.py`). Guardan las figuras en
  las rutas viejas (`pruebas_sem/<etapa>/...`), no en el respaldo.
- Los scripts que empiezan con `_` son auxiliares: `_componer_roi3.py` arma
  `roi3_regiones_variante_c.py` a partir de `_roi3_cuerpo.py` y de funciones de otros scripts.
- Las primeras pruebas del suavizado se corrieron como comandos sueltos y no tienen script; el
  código definitivo está en `src/sem/suavizado.py` y `notebooks/sem_suavizado.ipynb`.

ROIs usadas: `data/sem/roi/BSE_1/BSE_1_{1,2,3,4}` (generadas con la Etapa 1 desde
`imagenes/SEM/BSE/BSE_1.tif`). Al final se trabajó solo con la **ROI 1** (todo patrón) y la
**ROI 3** (patrón, zona lisa, franja negra y zona gris).

## Vocabulario

| Término | Significado |
|---|---|
| población / fase | grupo de píxeles con un gris propio (en BSE, un compuesto) |
| actividad | `gaussiana(sobel(imagen), σ)`: cuántos bordes hay alrededor de cada píxel. σ grande = mide zonas; σ chico = distingue borde de interior |
| paisaje gris-actividad | histograma conjunto (2D): cuántos píxeles hay con cada par (gris, actividad) |
| región de análisis | parte de la ROI que se analiza por separado (por gris y actividad) |
| degradé | mezcla de grises en los bordes entre fases (desenfoque); "come" el gris propio de las estructuras finas |
| persistencia | fracción de filas del paisaje en las que aparece un pico en un mismo gris |

## 1. Suavizado (implementado y commiteado)

`src/sem/suavizado.py`: non-local means con parámetros automáticos.
- σ del ruido estimado con `estimate_sigma` (wavelets).
- `factor_h` por criterio de discrepancia: lo quitado tiene la desviación estándar del ruido.
- Parche 5×5: menor autocorrelación de lo quitado (figuras `suavizado/05`, `06`).

## 2. Corrección de manchas (descartada por ahora)

Objetivo: quitar manchas difusas de la adquisición (tipo suciedad en una lente).
- Método por fases (comparar cada píxel con el gris de su fase): el mapa encuentra zonas, pero la
  corrección casi no cambia la imagen (`sombras/01-09`).
- Problema de fondo: las manchas oscurecen unos pocos %, lo mismo que separa fases parecidas.
- `blur_resta` (restar un desenfoque de ~5 px) deja el patrón fino y quita lo grande, pero aplana
  las zonas negras. `white_tophat` (r=50) conserva lo negro y separa regiones con distinto
  "comportamiento" (franja diagonal de la ROI 3) (`sombras/10`).
- Esa observación llevó a los mapas de actividad.

## 3. Ecualización (descartada como paso previo)

- CLAHE: con `clip_limit` ≥ 0.01 funde los dos picos de la ROI 3; el tamaño de región casi no
  influye (`ecualizacion/01-03`).
- Ecualización global con límite: conserva mejor las poblaciones (`ecualizacion/04`), pero no
  agrega información para separar (es monótona).
- CLAHE también fue el peor realce antes de la persistencia (muchas poblaciones falsas).

## 4. Regiones de análisis (funciona)

Procedimiento (`segmentacion/28-30`, script `segmentacion_por_region.py`):
1. **Zonas por gris**: picos y valles del histograma (criterios tomados del módulo TEM:
   prominencia 5 %, distancia 8 bins, masa 1 %, valle < 75 % del pico menor); las manchitas
   < 1 % del área se reasignan a la zona vecina.
2. **Franja de borde** entre zonas: ancho = distancia donde el gris recorre el 90 % del camino
   hasta su valor interior (sale de la imagen: 2-9 px en la ROI 3). No se usa para buscar regiones
   ni poblaciones; al final toma la etiqueta del vecino más cercano.
3. **Actividad sin cruzar límites de zona** (σ = 25 px).
4. Dentro de cada zona, **clases de actividad** por picos y valles.

ROI 3 → 4 regiones: franja negra, zona gris de abajo, zona lisa, patrón. ROI 1 → 1 región.

Descartado para esto: mezcla de gaussianas con BIC (siempre elige el máximo de nubes,
`segmentacion/13`); picos 2D en el paisaje con watershed (sobrefragmenta lo negro,
`segmentacion/17`).

## 5. Separar las fases dentro del patrón (en curso)

### Lo que no funcionó
- Picos/valles del gris: la fase oscura del patrón no forma un pico propio (`segmentacion/03`).
- Multi-Otsu directo: toma el lado sombreado del relieve como fase (`segmentacion/01`).
- Triángulo sobre el contraste local: "bastante" bien en la ROI 1 (14 %), pero no general
  (`segmentacion/05`).
- Random walker con semillas automáticas (`segmentacion/08-10`, `23`): el relieve funciona como
  pared y el lado en sombra de las islas cae en la fase oscura.
- Filtros de crestas (`sato`): dibujan bien la red, pero el histograma no tiene valle
  (`segmentacion/15-16`).
- Transformada de Fourier del paisaje: muestra cuántas poblaciones conviven en una fila y a qué
  distancia, pero no su gris (`segmentacion/19-21`, `31-32`).

### Lo que aprendimos con imágenes simuladas
Simuladas: burbujas y gusanos de 5 poblaciones, patrón tipo ROI 1, laberinto conectado + matriz
(`segmentacion/20-24`, `31-34`, `43-46`).
- Con bordes nítidos, **cada población es una línea vertical en el paisaje**; la **persistencia**
  las encuentra todas sin decir cuántas son (`segmentacion/34`: 5/5 y 3/3).
- El **degradé** (≈ 1.5-2 px) borra las estructuras de 3-6 px: sus píxeles no llegan a su gris.
  En el paisaje, las dos poblaciones se unen en un **arco o "L"** (laberinto con degradé,
  `segmentacion/46`) — es la nube diagonal de la ROI 1 real.
- **Sensibilidad de la actividad**: con σ chico (~ espesor de las estructuras finas, 3 px) la
  actividad distingue el interior de las islas (actividad baja) de los canales (actividad alta)
  (`segmentacion/44-46`).

### Deshacer el degradé
- Realces comparados antes de la persistencia: máscara de enfoque, deconvolución
  (Richardson-Lucy), CLAHE (`segmentacion/35-39`). El orden cambia según detalles: resultado frágil.
- Estimación del degradé: ancho 10-90 % de bordes **aislados** (sin otro borde cerca); da
  2.15 px para 2 reales y 1.5-1.8 px en las ROIs reales. Falla con paredes muy finas.
- **Filtro de choque** (Osher-Rudin, esquema minmod): empuja cada píxel hacia el lado del borde
  al que pertenece. En el laberinto separa el arco en dos columnas, pero engorda las paredes
  (`segmentacion/48-49`). En la ROI 1 real es el único que hace aparecer **dos picos** en el
  histograma (`segmentacion/50`).

### Variante c (la mejor hasta ahora, aprobada visualmente en la ROI 1)
`suavizado → máscara de enfoque (radio = degradé estimado, cantidad 1) → filtro de choque
(σ interno 1 px, frena con cambio < 0.1 × ruido) → corte`.
- ROI 1: 22.9 % de fase oscura, red continua (`segmentacion/51-53`).
- ROI 3 por región (`segmentacion/54-56`): el patrón tiene dos picos (0.565 y 0.72, fase oscura
  18 %), pero el criterio del valle < 75 % (de TEM) lo rechaza; ese mismo criterio rechazaría la
  ROI 1 (valle al 79 %).

## Parámetros

| Parámetro | Valor | ¿Sale de la imagen? |
|---|---|---|
| ruido, `factor_h` (suavizado) | — | sí |
| ancho del degradé | 1.5-1.8 px en las ROIs | sí (bordes aislados) |
| fin del filtro de choque | cambio < 0.1 × ruido | sí |
| franja de borde | 2-9 px | sí |
| cantidad de la máscara de enfoque | 1.0 | no |
| σ interno del filtro de choque | 1 px (0.5 anduvo peor) | no |
| σ de la actividad | 25 px (regiones), 3 px (paisaje) | no (idea: ~ espesor de lo más fino) |
| criterios de picos | 5 %, 8 bins, 1-5 %, 75 % | no (vienen de TEM) |

## Problemas abiertos
1. **Encontrar todas las poblaciones**, no dos: usar la persistencia en el paisaje de cada región
   sobre la imagen procesada con la variante c (todavía no se combinaron).
2. El criterio de decisión depende de la región (en el patrón no siempre hay valle).
3. Halos de la máscara de enfoque en lo negro (ignorados por ahora).
4. Las regiones 1, 2 y 3 de la ROI 3 deberían mostrar patrón; hoy quedan como una población.

## Próximo paso acordado
Simulada con una matriz y **dos patrones oscuros de distinta intensidad** (tres fases mezcladas),
con y sin degradé, para ver si variante c + persistencia encuentra las tres fases.
