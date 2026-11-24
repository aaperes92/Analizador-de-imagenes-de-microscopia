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
