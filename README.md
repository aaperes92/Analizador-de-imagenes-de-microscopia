# Analizador de imágenes de microscopía

Herramientas en Python para analizar imágenes de tres técnicas de
microscopía:

- **MOKE** (microscopía magneto-óptica Kerr): detección y seguimiento de
  dominios magnéticos cuadro a cuadro.
- **SEM** (microscopía electrónica de barrido): carga, selección de la
  región de interés, suavizado y segmentación de fases en imágenes de
  electrones retrodispersados (BSE).
- **TEM** (microscopía electrónica de transmisión): preprocesamiento,
  segmentación de fases, morfología, análisis de fases y patrones de
  difracción.

Se usa desde una **interfaz gráfica** donde se elige la técnica, la
entrada y las opciones, o desde scripts y desde Python.

## Instalación

Requiere **Python ≥ 3.14** y [uv](https://docs.astral.sh/uv/). Desde la
raíz del proyecto:

```bash
uv sync
```

`uv sync` instala todas las dependencias declaradas en `pyproject.toml`:
`numpy`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`,
`tifffile`, `exifread`, `pywavelets`, `mrcfile` y `ncempy`. La interfaz
usa Tkinter, que viene con Python.

## Uso: interfaz gráfica

```bash
uv run python main.py
```

En la ventana se elige la técnica (MOKE, SEM o TEM) y se muestran solo
sus opciones. La entrada (carpeta o archivo) se elige con un diálogo y
**Analizar** corre el análisis en segundo plano: la ventana no se congela
y el registro muestra los mensajes de cada módulo, el avance (imagen i de
N), los avisos, los errores y la carpeta donde quedaron los resultados.
Un error en una imagen se informa y, con una carpeta, se sigue con la
siguiente.

| Técnica | Entrada | Opciones | Resultados |
|---|---|---|---|
| MOKE | carpeta con `0.png`, `1.png`, … y la imagen de fondo | escala y unidad | `data/moke/resultados/` |
| SEM (solo BSE) | una imagen `.tif` (abre el selector de ROI) o una carpeta con ROIs guardadas | sufijo de la ROI, intensidad de pozos | `data/sem/roi/`, junto a cada ROI |
| TEM | una imagen o una carpeta (`.dm3`, `.mrc`, `.tif`, `.tiff`, `.png`, `.jpg`) | filtro, umbral, fases, fondo, difracción, ancho instrumental | `data/samples/` |

Las figuras se guardan siempre; con una sola imagen, además se abren en
ventanas aparte. Documentación completa: [docs/interfaz.md](docs/interfaz.md).

## Datos de entrada

- **MOKE**: secuencia de imágenes numeradas (`0.png`, `1.png`, … o
  `.jpg`) del mismo tamaño, más una imagen de fondo tomada sin el
  dominio.
- **SEM**: imágenes `.tif` del microscopio (FEI, Zeiss u otros con
  EXIF), con o sin barra de datos. Se analiza una región de interés
  (ROI) elegida a mano. La segmentación es solo para imágenes BSE.
- **TEM**: imágenes de campo claro u oscuro y patrones de difracción en
  `.dm3`, `.mrc`, `.tif`, `.tiff`, `.png` o `.jpg`. Si a un patrón le
  faltan voltaje, longitud de cámara o tamaño de píxel en los metadatos,
  se piden al usuario.

Las imágenes originales y los resultados no se versionan en git.

## Resultados

- **MOKE**: por imagen, el contorno del dominio y el círculo ajustado
  (centro, radio, área y contraste), en una figura por imagen y en la
  tabla `medidas_moke.csv`.
- **SEM**: la ROI guardada con su metadata y, para BSE, la etiqueta de
  población (fase) de cada píxel, una tabla de poblaciones y una figura
  de la segmentación.
- **TEM**: imágenes preprocesadas y segmentadas, propiedades
  morfológicas de cada región, fracciones de área y volumen, intensidad
  relativa y distribución espacial de cada fase, y, en los patrones de
  difracción, espaciados d, tamaño de cristalito y red de spots. Las
  fases se identifican por intensidad (Fase 1, Fase 2…): el programa no
  asigna elementos químicos.

## Módulo MOKE
Detección de dominios magnéticos en secuencias de imágenes de microscopía
Kerr (`src/moke/`): lee imágenes numeradas (`0.png`, `1.png`, …, también
`.jpg`) y una imagen de fondo tomada sin el dominio. Con ella corrige la
iluminación despareja de la apertura y calcula el contraste relativo
(imagen − fondo) / fondo, de modo que −0.10 significa un 10 % más oscuro.

En cada imagen encuentra la apertura iluminada y descarta su borde. Ubica
el dominio a partir de una semilla, que sale de la última imagen (la del
dominio más grande), y lo separa del fondo con un umbral iterativo a mitad
de camino entre ambos niveles. Después extrae el contorno y le ajusta un
círculo por mínimos cuadrados. Guarda centro, radio, área y contraste, y
marca como dudosas las imágenes con contraste menor que 3 veces el ruido.
Si no hay imagen de fondo, la estima con un polinomio 2D robusto.

Desde la interfaz: técnica MOKE, carpeta, fondo, escala y unidad. Desde
Python:

```python
from moke import analizadorMOKE, detectar_dominio, leer_gris
filas = analizadorMOKE("background.png", "evolucion_dominio",
                       escala=75, unidad="nm")           # carpeta completa
img = leer_gris("evolucion_dominio/10.png")
fondo = leer_gris("evolucion_dominio/background.png")
res = detectar_dominio(img, referencia=fondo)            # una imagen
res["radio"], res["centro"], res["area"], res["contraste"]
```

Desde Python, los resultados quedan por defecto en `resultados/` (se
cambia con `salida=`); desde la interfaz, en `data/moke`. En
ambos casos: una figura `dominio_<n>.png` por imagen, con el contorno y
el círculo ajustado, y la tabla `medidas_moke.csv` con la evolución del
dominio cuadro a cuadro.

Documentación completa: [docs/moke.md](docs/moke.md).

## Módulo SEM
Carga, suavizado y segmentación de imágenes SEM (`src/sem/`). La carga
lee el `.tif` del microscopio con su metadata (FEI, Zeiss o EXIF), quita
la barra de datos y permite seleccionar y guardar una región de interés
(ROI). El suavizado reduce el ruido con non-local means, con parámetros
que se fijan automáticamente a partir de la imagen. Carga y suavizado
sirven para imágenes BSE y SE.

La segmentación es **solo para imágenes de electrones retrodispersados
(BSE)** (`src/sem/bse/`): divide la ROI en regiones de análisis y, en
cada una, encuentra las poblaciones (fases) sin necesidad de decir
cuántas hay ni de que el histograma tenga un valle entre ellas. Para
eso estima el desenfoque de los bordes, realza la imagen, aplica un
filtro de choque que devuelve a cada píxel el gris de su fase, y separa
las poblaciones en el paisaje gris × actividad.

```bash
uv run python scripts/sem_carga.py imagenes/SEM/BSE/BSE_1.tif                       # carga y ROI
uv run python scripts/sem_bse_segmentacion.py data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy   # segmentación BSE
```

```python
from sem import suavizar            # BSE y SE
from sem.bse import segmentar_roi   # solo BSE
s = segmentar_roi(roi)              # suavizado, regiones y poblaciones
s["etiquetas"], s["tabla"]          # etiqueta de cada píxel y tabla de poblaciones
```

Documentación completa: [docs/sem.md](docs/sem.md).

## Módulo TEM
Carga y preprocesamiento de imágenes TEM (`src/tem/`): lee `.tif`,
`.tiff`, `.jpg`, `.png`, `.dm3` y `.mrc`, normaliza a [0, 1] por
percentiles y reduce el ruido con un filtro gaussiano, de mediana,
bilateral o de Wiener, elegido a mano o automáticamente.

Además segmenta la imagen en fases de intensidad (sin asignar elementos
químicos): detecta cuántas fases hay, elige el umbral (Otsu, GMM o
triángulo para campo oscuro), separa partículas que se tocan y calcula
área, intensidad y número de regiones de cada fase. Por último mide la
morfología de cada región (tamaño, forma y orientación, en píxeles).
Finalmente analiza las fases: fracciones de área y volumen, intensidad
relativa e inhomogeneidades, y distribución espacial (RDF, Hopkins,
DBSCAN, gradientes). Las fases se llaman Fase 1, Fase 2… y el fondo.

```python
from tem import (pipeline_tem, pipeline_segmentation,
                 pipeline_morphology, pipeline_phase_analysis)
pre = pipeline_tem("imagen.dm3", show=False)        # Etapa 1: preprocesamiento
seg = pipeline_segmentation(pre, show=False)        # Etapa 2: segmentación
mor = pipeline_morphology(seg, show=False)          # Etapa 3: morfología
res = pipeline_phase_analysis(mor)                  # Etapa 4: análisis de fases
```

Si la imagen es un patrón de difracción (SAED de anillos o de puntos,
NBED, amorfo o perfil 1D), la Etapa 4.5 detecta el centro del haz
aunque lo tape el beam stopper, mide los espaciados d de los anillos,
el tamaño de cristalito (Scherrer) y la red de los spots:

```python
from tem import pipeline_diffraction
dif = pipeline_diffraction("patron.dm3")             # Etapa 4.5: difracción
```

Para correr todo sobre una imagen o una carpeta (detecta solo cuáles
son patrones de difracción):

```bash
uv run python scripts/analizar_tem.py "RUTA"
```

Con `--difraccion` se fuerza el análisis de difracción.

Documentación completa: [docs/tem.md](docs/tem.md).

## Estructura del proyecto

```
main.py          abre la interfaz gráfica
src/interfaz/    interfaz gráfica (Tkinter)
src/moke/        módulo MOKE
src/sem/         módulo SEM (src/sem/bse/: segmentación BSE)
src/tem/         módulo TEM
scripts/         scripts de línea de comandos y de prueba por etapa
tests/           tests (pytest), una carpeta por módulo
docs/            documentación de cada módulo y de la interfaz
data/            resultados y datos generados (no se versionan)
```

## Tests

```bash
uv run pytest
```

## Autores

| Integrante | Parte | Contacto |
|---|---|---|
| Mauricio Romano | Módulo SEM | [mauricio.romano@ib.edu.ar](mailto:mauricio.romano@ib.edu.ar) |
| Daniel Romero | Módulo TEM | [daniel.romero@ib.edu.ar](mailto:daniel.romero@ib.edu.ar) |
| Ángel A. Pérez | Módulo MOKE e interfaz gráfica | [angel.perez@ib.edu.ar](mailto:angel.perez@ib.edu.ar) |

Para consultas sobre el programa, escribí a cualquiera de los autores.

## Licencia

MIT. Ver [LICENSE](LICENSE).
