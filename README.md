## Titulo
"[Analizador de imágenes de Microscopía]"

## Overview

EL programa contiene herramientas para analizar imágenes obtenidas mediante técnicas de microscopía, incluyendo TEM, SEM y Microscopía MOKE.
Comprende funciones para procesar imágenes, segmentarlas y extraer información relevante sobre la composición de las muestras.
Además, permite seleccionar funciones en dependencia si estas analizando imágenes de MOKE, o de TEM/SEM.

#Requeriments
El código se desarrollará en Python y puede requerir los siguientes paquetes:

numpy
scipy
matplotlib
scikit-image
exifread

puedes instalar los paquetes necesarios ejecutando el siguiente comando en tu terminal o línea de comandos:

pip install numpy scipy matplotlib scikit-image

# Input data
- Las entradas del programa son imágenes de microscopía en formatos estándar (por ejemplo, .tif) de datos experimentales.
- Es importante conocer que tipo de imagen se está analizando, ya que el programa ofrece diferentes funciones según si se trata de imágenes de MOKE o de TEM/SEM.
- Para el caso de composiciones habría que entrarle la composición de la muestra, por ejemplo: Fe y Ni.

# Output data
- Los resultados del análisis se presentan en forma de gráficos y tablas, que incluyen ajustes de curvas
- Poblaciones relativas de los compuestos presentes en la muestra, obtenidas a partir de las imágenes de microscopía y los datos experimentales.
- Imágenes segmentadas y procesadas, que muestren cada compuesto por separado.
- Para imágenes de MOKE, se pueden obtener mapas de contraste magnético, contornos y tamaños de dominio.

# Additional information

## Author
- [Mauricio Romano]
- [Daniel Romero]
- [Angel A. Perez]

## Contact
Para consultas relacionadas con el conjunto de datos, puede comunicarse con los autores a través de los siguientes correos electrónicos:  
- [mauricio.romano@ib.edu.ar]
- [daniel.romero@ib.edu.ar]
- [angel.perez@ib.edu.ar]

## Cómo usarlo: interfaz gráfica
El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz se instalan
las dependencias y se abre la interfaz:

```bash
uv sync
uv run python main.py
```

En la ventana se elige la técnica (MOKE, SEM o TEM) y se muestran solo sus
opciones. La entrada (carpeta o archivo) se elige con un diálogo y
"Analizar" corre el análisis en segundo plano: la ventana no se congela y
el registro muestra los mensajes de cada módulo, el avance (imagen i de N),
los avisos, los errores y la carpeta donde quedaron los resultados. Un error
en una imagen se informa y, con una carpeta, se sigue con la siguiente.

| Técnica | Entrada | Opciones | Resultados |
|---|---|---|---|
| MOKE | carpeta `0.png`, `1.png`, … y la imagen de fondo | escala y unidad | `resultados/` |
| SEM (solo BSE) | una imagen `.tif` (abre el selector de ROI) o una carpeta con ROIs guardadas | sufijo de la ROI, intensidad de pozos | `data/sem/roi/` |
| TEM | una imagen o una carpeta (`.dm3`, `.mrc`, `.tif`, `.tiff`, `.png`, `.jpg`) | filtro, umbral, fases, fondo, difracción, ancho instrumental | `data/samples/` |

Las figuras se guardan siempre; con una sola imagen, además se abren en
ventanas aparte. El código de la interfaz está en `src/interfaz/`. Los
scripts de `scripts/` siguen funcionando igual que antes.

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

Los resultados quedan en `resultados/`: una figura `dominio_<n>.png` por
imagen, con el contorno y el círculo ajustado, y la tabla `medidas_moke.csv`
con la evolución del dominio cuadro a cuadro.

Documentación completa: [docs/moke.md](docs/moke.md).

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
son patrones de difracción): `uv run python scripts/analizar_tem.py
"RUTA"`; con `--difraccion` se fuerza el análisis de difracción.

Documentación completa: [docs/tem.md](docs/tem.md).

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
