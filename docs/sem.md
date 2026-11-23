# Módulo SEM

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Mauricio Romano (mauricio.romano@ib.edu.ar).

El módulo `sem` trabaja con imágenes de microscopía electrónica de barrido (SEM):

1. **[Carga](#carga)**: lee la imagen, interpreta su metadata de adquisición, quita la barra
   de datos y permite seleccionar y guardar una región de interés (ROI).
   Código: [`src/sem/carga.py`](../src/sem/carga.py).
2. **[Suavizado](#suavizado)**: reduce el ruido de la ROI preservando los bordes entre fases,
   con parámetros que se fijan automáticamente a partir de la imagen.
   Código: [`src/sem/suavizado.py`](../src/sem/suavizado.py).

Desde la etapa 2, el procesamiento está pensado para imágenes de **electrones retrodispersados
(BSE)**, donde cada fase aparece como una población de intensidad. Las imágenes de electrones
secundarios (SE) requieren un análisis distinto, todavía no implementado.

Al final hay un [glosario](#glosario) con los términos técnicos.

---

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz del proyecto:

```bash
uv sync
```

Librerías que usa el módulo SEM:

| Librería | Para qué se usa |
|---|---|
| `numpy` | Arrays y cálculo numérico |
| `scikit-image` | Lectura de imágenes y todo el procesamiento (filtros, morfología, ecualización) |
| `tifffile` | Lectura de la metadata propietaria de los TIFF de FEI y Zeiss |
| `exifread` | Lectura de metadata EXIF estándar (respaldo) |
| `pywavelets` | Requerida por `skimage.restoration.estimate_sigma` (estimación del ruido) |
| `matplotlib` | Figuras y ventana interactiva del selector de ROI |

El módulo no usa OpenCV.

## Organización del código

```
src/sem/
    __init__.py           expone las funciones públicas del módulo
    carga.py              lectura, metadata, barra de datos y ROI
    suavizado.py          estimación del ruido y suavizado non-local means
scripts/
    sem_carga.py          script de línea de comandos para la carga
notebooks/
    sem_suavizado.ipynb   prueba del suavizado sobre ROIs reales, paso a paso
data/sem/roi/             ROIs guardadas (no se versionan en git)
```

Las funciones se pueden importar desde el paquete:

```python
import sys
sys.path.insert(0, "src")          # desde la raíz del proyecto

from sem import cargar_imagen, leer_metadata, recortar_barra_datos
```

Todas las funciones tienen docstring con sus parámetros, retorno y errores: esa es la
referencia completa. Este documento explica qué hace cada una y cómo funciona.

Convenciones comunes a todo el módulo:

- Las imágenes se manejan como arrays 2D `float32`.
- Ninguna función modifica su entrada: siempre retornan un array nuevo.
- Las coordenadas de una ROI son `(x1, y1, x2, y2)` en píxeles, con `x` = columna e `y` = fila.

---

## Imágenes de entrada

### Formatos

Cualquier formato que lea `scikit-image` (`.tif`, `.png`, `.jpg`, ...). El formato esperado
es el `.tif` que guarda el microscopio, porque es el único que conserva la metadata.

Se aceptan imágenes en escala de grises (uint8, uint16, con o sin paleta) y RGB/RGBA, que se
convierten a escala de grises.

### Metadata de adquisición

La metadata de los microscopios SEM no está en los campos EXIF estándar sino en **tags TIFF
propietarios** de cada fabricante:

| Fabricante | Tag TIFF | Formato | Cómo se lee |
|---|---|---|---|
| FEI / Thermo Fisher | 34682 | texto tipo INI (`[Beam] HV=...`) | `tifffile` (`fei_metadata`) |
| Zeiss (SmartSEM) | 34118 | binario | `tifffile` (`sem_metadata`) |
| Otros | EXIF estándar | — | `exifread` (solo ancho, alto y fecha) |

El dato más importante es el **tamaño de píxel**: permite pasar de píxeles a unidades reales.
Una imagen sin metadata se puede procesar igual, pero las medidas quedan en píxeles.

### Barra de datos

Ambos fabricantes incrustan una barra de datos (escala, voltaje, magnificación) en las filas
inferiores de la imagen. Hay que quitarla antes de analizar, porque sus píxeles saturados
deforman el histograma.

| Fabricante | Barra | Cómo se detecta |
|---|---|---|
| FEI | fondo negro, texto blanco | La metadata indica las filas útiles (`[Image] ResolutionY`) |
| Zeiss | fondo blanco, texto negro | No figura en la metadata: se detecta en la imagen (ver [Recorte de la barra de datos](#recorte-de-la-barra-de-datos)) |

---

## Carga

### Uso

```bash
uv run python scripts/sem_carga.py <ruta_imagen> [--salida CARPETA] [--sufijo SUFIJO]
```

| Opción | Descripción | Por defecto |
|---|---|---|
| `imagen` | Ruta a la imagen SEM | (obligatorio) |
| `--salida` | Carpeta raíz donde se guardan las ROIs | `data/sem/roi` |
| `--sufijo` | Sufijo que identifica la ROI (ej. `_zona_grano`) | primer `_N` libre |

El script:

1. Carga la imagen, lee la metadata, quita la barra de datos e imprime un resumen.
2. Abre una ventana con la imagen y su histograma. Se cierra para continuar.
3. Abre el selector de ROI: se arrastra un rectángulo con el botón izquierdo (se puede ajustar)
   y se confirma con **Enter** o cerrando la ventana.
4. Guarda la ROI. Si no se seleccionó nada, no guarda y lo avisa.

### Salidas

Cada ROI se guarda en `<salida>/<nombre_imagen>/<nombre_imagen><sufijo>/`:

| Archivo | Contenido | Uso |
|---|---|---|
| `roi_<nombre><sufijo>.npy` | recorte `float32` en [0, 1] | entrada del preprocesamiento |
| `roi_<nombre><sufijo>.tif` | recorte `uint8` | solo para visualizar |
| `roi_<nombre><sufijo>.json` | imagen de origen, coordenadas, tamaños y metadata | calibración en unidades reales |

Sin `--sufijo` se usa el primer `_N` libre, así que nunca se sobrescribe una ROI anterior.

### Funcionamiento

#### Lectura y normalización — `cargar_imagen`

Lee la imagen con `skimage.io.imread`, convierte RGB/RGBA a escala de grises y la lleva a
`float32` en [0, 1] con `img_as_float32`.

La normalización **depende del tipo de dato, no del contenido**: uint8 se divide por 255 y
uint16 por 65535. No se estira el contraste. Así, dos imágenes del mismo equipo siguen siendo
comparables entre sí, y una imagen que no usa todo el rango del detector no llega a 1.0.

#### Metadata — `leer_metadata`

Intenta, en orden, FEI, Zeiss y EXIF, y retorna los valores con **claves y unidades comunes**
a todos los fabricantes:

| Clave | Unidad / formato |
|---|---|
| `fabricante` | `'FEI'`, `'Zeiss'` o `'EXIF'` |
| `ancho_px`, `alto_px` | píxeles (imagen completa, con barra) |
| `alto_util_px` | filas sin barra de datos (solo FEI) |
| `tamano_pixel_m` | metros por píxel |
| `ancho_campo_um` | campo de visión horizontal, µm |
| `magnificacion` | número (ej. `2020.0`) |
| `voltaje_kV` | kV |
| `distancia_trabajo_mm` | mm |
| `detector` | texto (ej. `'ETD'`, `'BSD4 A'`) |
| `fecha_adquisicion` | `'AAAA-MM-DD HH:MM:SS'` |
| `crudo` | diccionario original completo del fabricante |

Detalles de interpretación:

- **Tamaño de píxel en Zeiss.** Zeiss guarda dos valores: `ap_pixel_size`, referido a un
  barrido de 1024 px de ancho, y `ap_image_pixel_size`, referido a la imagen guardada. Se usa
  el segundo, que cumple `tamaño de píxel × ancho = campo de visión`. Usar el primero daría
  medidas 2 o 4 veces mayores en imágenes de 2048 o 4096 px.
- **Magnificación en FEI.** FEI no la guarda; se calcula como `MagCanvasRealWidth / HFW`
  (ancho de pantalla de referencia sobre campo de visión).
- **Voltaje.** FEI lo guarda en volts y Zeiss en kV; se unifica a kV.

Nunca lanza una excepción por metadata ausente o ilegible: retorna `{}` y avisa. También avisa
si no encontró el tamaño de píxel.

#### Recorte de la barra de datos — `recortar_barra_datos`

- **FEI:** se queda con las primeras `alto_util_px` filas.
- **Zeiss:** busca, en el 20 % inferior de la imagen, la primera fila completamente blanca
  cuya fila anterior sea mayormente no blanca: ese es el borde superior de la barra. Las filas
  de texto dentro de la barra siguen siendo mayormente blancas y no generan falsos bordes.
  Si no encuentra el borde, no recorta.

#### Visualización — `mostrar_imagen_con_histograma`

Figura con la imagen (escala de grises fija en [0, 1], sin estirar el contraste) y su
histograma de 256 niveles. Si hay tamaño de píxel, agrega una barra de escala de longitud
redonda (1, 2 o 5 × 10ⁿ µm) de alrededor del 20 % del ancho. Debajo muestra la metadata.
Retorna la figura; no la abre (eso lo hace `plt.show()`).

#### Selección de ROI — `seleccionar_roi`

Ventana con un `RectangleSelector` de matplotlib. Bloquea hasta Enter o hasta cerrar la
ventana, y retorna `(x1, y1, x2, y2)` ordenado y recortado a los bordes de la imagen, o `None`
si no se seleccionó nada. Requiere un backend de matplotlib con ventanas (por ejemplo TkAgg).

#### Guardado — `guardar_roi`

Recorta la imagen y escribe los tres archivos descritos en [Salidas](#salidas). El `.json` no
incluye la metadata `crudo` del fabricante.

---

## Suavizado

Reduce el ruido de la ROI antes de corregir la iluminación y separar las poblaciones. Al bajar
el ruido, las poblaciones de intensidad se angostan en el histograma y se distinguen mejor.

### Uso

```python
import numpy as np
from sem.suavizado import suavizar

roi = np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy")
roi_suave = suavizar(roi)          # todos los parámetros se fijan automáticamente
```

El notebook [`notebooks/sem_suavizado.ipynb`](../notebooks/sem_suavizado.ipynb) recorre el
proceso paso a paso sobre las ROIs de una imagen, con figuras e histogramas de antes y después.

### Funcionamiento

El suavizado usa **non-local means** (`skimage.restoration.denoise_nl_means`). Para cada
píxel busca, en una ventana alrededor, parches parecidos al suyo y promedia sus centros, con más
peso cuanto más parecidos son. En un borde entre fases los parches de la otra fase no se
parecen, así que casi no se mezclan y **el borde se conserva**. Esto es importante en BSE: un
filtro que borronea los bordes (como el gaussiano) crea píxeles de intensidad intermedia que
después parecen una población más.

La intensidad del filtrado es `h = factor_h × σ`, donde σ es la desviación estándar del ruido.
Ningún parámetro se ajusta a mano:

| Parámetro | Cómo se fija | Función |
|---|---|---|
| σ (ruido) | Se estima de la propia ROI | `estimar_ruido` |
| `factor_h` | Criterio de discrepancia | `buscar_factor_h` |
| `tamano_parche` | 5×5, elegido con la prueba de parches (ver abajo) | — |
| `distancia_parche` | 6 píxeles (valor típico de skimage) | — |

#### Estimación del ruido — `estimar_ruido`

Usa `skimage.restoration.estimate_sigma`: descompone la imagen con una transformada wavelet y
toma los coeficientes de detalle más finos, donde casi todo es ruido. La desviación estándar se
calcula con la mediana del valor absoluto (`σ = mediana(|c|) / 0.6745`). La mediana hace que los
bordes entre fases, que también generan coeficientes grandes pero ocupan una fracción chica de la
imagen, casi no afecten la estimación (Donoho y Johnstone, 1994).

Las ROIs de una misma imagen dan un σ parecido, porque el ruido depende del equipo y de la
adquisición, no de la zona.

#### Intensidad del filtrado — `buscar_factor_h`

Aplica el **criterio de discrepancia**: el filtrado es adecuado cuando lo que se quita tiene la
misma desviación estándar que el ruido,

```
desviación estándar(original − suavizada) = σ
```

Si se quita menos, queda ruido; si se quita más, se borra señal. Como lo quitado crece con
`factor_h`, el valor se busca por bisección entre 0.3 y 3.0 y se corta cuando la diferencia es
menor al 2 % (en general alcanza con 5 o 6 filtrados). En las imágenes probadas da
`factor_h ≈ 1.1`.

#### Suavizado — `suavizar`

Combina las dos anteriores: si no se le pasan `sigma_ruido` ni `factor_h`, los calcula, y
después filtra. Si se le pasan, usa esos valores (por ejemplo, para usar el mismo σ en todas las
ROIs de una imagen). Devuelve una imagen nueva `float32` con el mismo tamaño.

#### Elección del tamaño de parche

La discrepancia fija **cuánto** se quita, pero no elige el parche: para cada tamaño hay un
`factor_h` que la cumple. Para compararlos se mira **qué** se quitó, con la autocorrelación a
1 píxel de lo quitado (la correlación entre lo quitado y lo quitado corrido un píxel). Si lo
quitado es ruido, los vecinos no se parecen y la autocorrelación es baja; si el filtro se llevó
parte de la imagen, lo quitado tiene forma y la autocorrelación sube.

Con las ROIs de prueba, 5×5 dio la menor autocorrelación en todas; 3×3 se lleva parte de la
microestructura. La autocorrelación horizontal sale siempre algo mayor que la vertical por el
barrido del haz, que hace que el ruido de una misma fila esté algo relacionado. Para imágenes muy
distintas (otra magnificación u otro equipo) conviene repetir esta prueba, que está en el anexo
del notebook.

---

## Glosario

| Término | Significado |
|---|---|
| **SEM** | Microscopía electrónica de barrido |
| **SE / BSE** | Detector de electrones secundarios (contraste topográfico) / retrodispersados (contraste por número atómico) |
| **ROI** | Región de interés: el recorte de la imagen que se analiza |
| **Tamaño de píxel** | Distancia real que representa un píxel (m/píxel) |
| **HFW** | Ancho horizontal del campo de visión |
| **WD** | Distancia de trabajo entre la lente y la muestra |
| **Barra de datos** | Franja inferior de la imagen con escala y datos de adquisición |
| **Histograma** | Cantidad de píxeles en cada nivel de intensidad |
| **Población** | Grupo de píxeles con intensidad parecida (un pico del histograma); en BSE, una fase |
| **Non-local means** | Filtro que promedia píxeles cuyos vecindarios (parches) se parecen, aunque estén lejos |
| **Criterio de discrepancia** | Regla para fijar la intensidad de un filtro: lo quitado debe tener la desviación estándar del ruido |
| **Autocorrelación** | Correlación de una imagen consigo misma corrida unos píxeles; mide si los vecinos se parecen |
| **Backend de matplotlib** | Motor que dibuja las figuras; uno "con ventanas" (TkAgg, QtAgg) es necesario para las herramientas interactivas |
