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
3. **[Segmentación BSE](#segmentación-bse)** (solo electrones retrodispersados): divide la ROI
   en regiones de análisis y, dentro de cada una, encuentra las poblaciones (fases) y asigna
   cada píxel a una. Código: subpaquete [`src/sem/bse/`](../src/sem/bse/).

La carga y el suavizado sirven para BSE y SE. Desde la segmentación, el procesamiento es **solo
para imágenes de electrones retrodispersados (BSE)**, donde cada fase aparece como una población
de intensidad; por eso vive en el subpaquete `sem.bse`. Las imágenes de electrones secundarios
(SE), donde el gris refleja el relieve, requieren un análisis distinto, todavía no implementado.

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
    __init__.py           expone las funciones públicas de carga y suavizado
    carga.py              lectura, metadata, barra de datos y ROI (BSE y SE)
    suavizado.py          estimación del ruido y suavizado non-local means (BSE y SE)
    bse/                  segmentación, solo electrones retrodispersados
        __init__.py       expone las funciones públicas de la segmentación BSE
        regiones.py       regiones de análisis: zonas por gris, franja de borde, actividad
        realce.py         degradé, máscara de enfoque, filtro de choque (solo y con pozos)
        poblaciones.py    paisaje gris × actividad, núcleos y pozos
        segmentacion.py   la cadena completa: segmentar_roi
        resultados.py     imagen coloreada y archivos junto a la ROI: segmentar_y_guardar
scripts/
    sem_carga.py              script de línea de comandos para la carga
    sem_bse_segmentacion.py   script de línea de comandos para la segmentación BSE
tests/sem/
    test_bse.py           tests de la segmentación BSE con imágenes simuladas
pruebas_sem/              exploración: notas, scripts y figuras de las pruebas
data/sem/roi/             ROIs guardadas (no se versionan en git)
```

Las funciones se pueden importar desde el paquete:

```python
import sys
sys.path.insert(0, "src")          # desde la raíz del proyecto

from sem import cargar_imagen, leer_metadata, recortar_barra_datos
from sem.bse import segmentar_roi   # segmentación, solo BSE
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
distintas (otra magnificación u otro equipo) conviene repetir esta prueba.

---

## Segmentación BSE

**Solo para imágenes de electrones retrodispersados (BSE).** En BSE el gris de cada píxel indica
la composición: cada fase tiene un gris propio y forma una **población**. La segmentación busca
esas poblaciones y asigna cada píxel a una, sin que haya que decir cuántas hay y sin exigir que el
histograma de gris tenga un valle entre ellas (en las ROIs reales, el desenfoque suele borrarlo).

### Uso

Desde la línea de comandos, sobre una ROI guardada por la carga:

```bash
uv run python scripts/sem_bse_segmentacion.py <ruta_roi.npy> [--intensidad-pozos 2]
```

Guarda, junto a la ROI, `segmentacion_<nombre>.npy` (etiqueta de cada píxel), `.csv` (tabla de
poblaciones) y `.png` (ROI suavizada y segmentada lado a lado). El script solo lee los argumentos y
llama a `sem.bse.segmentar_y_guardar(ruta_roi, intensidad_pozos)`
([`src/sem/bse/resultados.py`](../src/sem/bse/resultados.py)), que también usa la interfaz.

Desde la **interfaz gráfica** (`uv run python main.py`, técnica SEM) hay dos modos:

- **Imagen y ROI**: se elige una imagen `.tif`; se carga (metadata y recorte de la barra de datos),
  se abre directamente el selector de ROI, la ROI se guarda en `data/sem/roi/` y se segmenta. La
  figura de la segmentación se abre en una ventana aparte. Opciones: sufijo de la ROI e intensidad
  de pozos.
- **ROIs guardadas**: se elige una carpeta y se segmentan todas las `roi_*.npy` que haya adentro,
  también en subcarpetas, sin abrir ventanas. Si una ROI falla, se sigue con la siguiente.

Desde Python, toda la cadena se aplica en seguidilla con una sola función:

```python
import numpy as np
from sem.bse import segmentar_roi

roi = np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")
s = segmentar_roi(roi)             # suaviza, divide en regiones y busca las poblaciones
s["etiquetas"]                     # etiqueta de cada píxel (una por par región-población)
s["tabla"]                         # región, población, gris central y fracciones
```

### La cadena

`segmentar_roi` ([`segmentacion.py`](../src/sem/bse/segmentacion.py)) aplica, en orden:

| Paso | Operación | Función | Módulo |
|---|---|---|---|
| 1 | Suavizado non-local means | `suavizar` | `sem.suavizado` |
| 2 | Regiones de análisis | `regiones_de_analisis` | `sem.bse.regiones` |
| 3 | Aislar cada región | `aislar_region` | `sem.bse.regiones` |
| 4 | Ruido y degradé de la región | `estimar_ruido`, `estimar_degrade` | `sem.suavizado`, `sem.bse.realce` |
| 5 | Realce (máscara de enfoque) | `realzar` | `sem.bse.realce` |
| 6 | Actividad de partida (σ = 3 px) | `actividad` | `sem.bse.poblaciones` |
| 7 | Filtro de choque con pozos | `filtro_choque_con_pozos` | `sem.bse.realce` |
| 8 | Paisaje, núcleos y pozos → poblaciones | `poblaciones_de_region` | `sem.bse.poblaciones` |
| 9 | La franja de borde toma la etiqueta del vecino más cercano | — | `sem.bse.segmentacion` |

Los pasos 3 a 8 se hacen **por separado en cada región** (`procesar_region`).

El filtro de choque se aplica **una sola vez**. `filtro_choque_con_pozos` trabaja en dos etapas
seguidas sobre la misma imagen: primero el choque solo (reutiliza `filtro_choque`) hasta que deja
de cambiar, y desde ese punto, sin volver a empezar, el choque con la atracción hacia los pozos.
Con `intensidad_pozos=0` queda solo la primera etapa.

### Resumen del procesamiento

Sea cual sea la ROI BSE, después del suavizado se procesa así:

1. **Se divide en regiones de análisis**, partes de la ROI que se comportan distinto (por gris y
   por cantidad de bordes). Una ROI homogénea queda como una sola región.
2. **Cada región se aísla** del resto, para que sus características no se mezclen con las de las
   demás.
3. **En cada región se recupera el gris propio de cada fase**: se estima cuánto desenfoque hay en
   sus bordes, se realza y se aplica el filtro de choque con pozos, que lleva cada píxel al gris de
   la fase a la que pertenece.
4. **En cada región se buscan las poblaciones** en el paisaje gris × actividad: los núcleos dicen
   cuántas hay y su gris; los pozos dicen a cuál va cada píxel.
5. **Se arma el resultado**: una etiqueta por par (región, población), con su gris central y su
   fracción de área.

### Funciones y su justificación

Para cada función: qué hace, por qué se hace así y qué puede decidir el usuario.

#### `segmentar_roi(roi, suavizada=False, intensidad_pozos=2.0)`

- **Qué hace:** aplica toda la cadena a una ROI y devuelve las etiquetas, la tabla de poblaciones
  y los resultados intermedios de cada región.
- **Por qué:** el orden de las operaciones importa (cada una prepara la entrada de la siguiente);
  una sola función evita aplicarlas en otro orden o saltear alguna.
- **El usuario decide:** si la ROI ya está suavizada (para no suavizar dos veces) y el peso de la
  atracción hacia los pozos (0 = choque solo).

#### `procesar_region(suavizada, mascara, intensidad_pozos=2.0)`

- **Qué hace:** aísla una región y le aplica, solo a ella, la estimación de ruido y degradé, el
  realce, la actividad, el choque con pozos y la búsqueda de poblaciones.
- **Por qué:** cada región tiene su propio ruido, su propio desenfoque y sus propias fases. Si se
  procesa toda la ROI junta, una zona muy contrastada (por ejemplo, una franja negra) domina la
  estimación del degradé, el realce y el paisaje, y tapa las fases de las zonas suaves.
- **El usuario decide:** la región (cualquier máscara, no solo las de `regiones_de_analisis`).

#### `regiones_de_analisis(imagen, sigma_actividad=25)`

- **Qué hace:** divide la ROI en zonas por gris y, dentro de cada zona, en clases de actividad.
  Devuelve la región de cada píxel y la franja de borde.
- **Por qué:** en una misma ROI puede haber zonas de composición distinta (gris distinto) y zonas
  de igual gris pero distinta textura (lisa o con patrón). Las dos cosas cambian qué poblaciones
  hay, así que se separan antes de buscarlas.
- **El usuario decide:** la escala de la actividad. Grande (25 px) mide zonas enteras, no bordes
  individuales.

#### `cortes_por_valles(valores, ...)`

- **Qué hace:** encuentra los cortes entre los picos reales de un histograma.
- **Por qué:** un histograma real tiene picos chicos de ruido. Exigir prominencia, masa mínima y
  un valle profundo entre picos evita cortar donde no hay dos grupos distintos.
- **El usuario decide:** los criterios (prominencia, distancia, masa y profundidad del valle).

#### `zonas_por_gris(gris, area_minima=0.01)`

- **Qué hace:** asigna cada píxel a una zona según los cortes del histograma de gris, y pasa las
  manchitas chicas a la zona vecina.
- **Por qué:** una zona de análisis tiene que ser una parte extensa de la ROI; una mancha aislada
  de otro gris es un detalle de la zona que la rodea, no una zona aparte.
- **El usuario decide:** el área mínima de una mancha para conservar su zona.

#### `franja_de_borde(gris, zonas)`

- **Qué hace:** marca los píxeles de transición entre zonas.
- **Por qué:** por el desenfoque, los píxeles junto al límite entre dos zonas tienen un gris
  intermedio que no es de ninguna. Si se los incluyera, aparecerían como una población falsa. Su
  ancho se mide en la propia imagen, porque depende del desenfoque de cada adquisición.

#### `actividad_sin_cruzar(imagen, zonas, borde, sigma=25)`

- **Qué hace:** promedia el gradiente de cada zona usando solo píxeles de esa zona.
- **Por qué:** el salto de gris entre dos zonas no es textura de ninguna de ellas. Si se lo
  promediara, el límite entre zonas aparecería como una región de mucha actividad.

#### `aislar_region(imagen, mascara, margen=20)`

- **Qué hace:** recorta la región y reemplaza lo que queda fuera de ella por el píxel más cercano
  de la región.
- **Por qué:** los filtros miran el vecindario de cada píxel. En el límite de una región, ese
  vecindario incluye píxeles de otras regiones; al reemplazarlos, la región se procesa como si
  continuara con sus propios valores.

#### `estimar_degrade(imagen, mascara=None)`

- **Qué hace:** mide el ancho del desenfoque en los bordes (σ de una gaussiana).
- **Por qué:** el realce tiene que deshacer exactamente el desenfoque que hay: si se realza con
  un radio menor no alcanza, y con uno mayor crea halos. Se miden solo bordes aislados porque un
  borde con otro cerca se ve más angosto o más ancho de lo que es. Con menos de 10 bordes
  aislados no hay datos suficientes y se usa 1 px.
- **El usuario decide:** con `mascara`, de qué parte de la imagen se toman los bordes.

#### `realzar(imagen, sigma_degrade, cantidad=1.0)`

- **Qué hace:** máscara de enfoque: suma a la imagen su diferencia con una versión desenfocada.
- **Por qué:** con radio igual al degradé, lo que se suma es el detalle que el desenfoque borró.
  Los bordes quedan más empinados y el filtro de choque los puede cortar en el lugar correcto.
- **El usuario decide:** la cantidad de detalle que se suma.

#### `filtro_choque(imagen, sigma_ruido, sigma_interno=1.0)`

- **Qué hace:** lleva cada píxel de un borde hacia el gris del lado al que pertenece, hasta que
  quedan mesetas de gris constante separadas por un salto.
- **Por qué:** en BSE cada fase tiene un gris propio, pero el desenfoque lo mezcla en los bordes y
  en las estructuras finas. Un filtro lineal (como la deconvolución) amplifica esa mezcla y crea
  anillos; el choque, en cambio, decide de qué lado está cada píxel y lo manda ahí, así que
  devuelve mesetas. Se detiene cuando el cambio de un paso ya no se distingue del ruido.
- **El usuario decide:** la escala con la que se estima la dirección del borde (`sigma_interno`).

#### `filtro_choque_con_pozos(imagen, act, sigma_ruido, intensidad=2.0)`

- **Qué hace:** primero el choque solo; cuando frena, sigue con el choque más una atracción de
  cada píxel hacia el gris central de su pozo en el paisaje.
- **Por qué:** después del choque, cada fase forma una masa en el paisaje, pero los píxeles de
  borde quedan en un gris algo distinto que los del interior (la masa es una mancha y no una
  línea). La atracción alinea a todos los píxeles de una masa en el mismo gris, en cualquier
  nivel de actividad, y deja las fases mejor separadas. Empieza recién cuando el choque frenó
  porque, antes, la imagen desenfocada tiene una sola masa y la atracción juntaría todos los
  grises.
- **El usuario decide:** la intensidad de la atracción (0 = choque solo).

#### `actividad(imagen, sigma=3)`

- **Qué hace:** promedio local del gradiente, a la escala de las estructuras más finas.
- **Por qué:** permite distinguir, para un mismo gris, el interior de una estructura (poca
  actividad) de su borde (mucha). Es el segundo eje del paisaje. Se calcula sobre la imagen
  **antes** del choque porque el choque afila todos los bordes y subiría la actividad de todas las
  fases por igual.
- **El usuario decide:** la escala (del orden del espesor de lo más fino que se quiere separar).

#### `paisaje(gris, act, bins=128)`

- **Qué hace:** cuenta cuántos píxeles hay con cada par (gris, actividad).
- **Por qué:** en el histograma de gris solo, las fases finas no forman un pico propio porque sus
  píxeles están casi todos en bordes. Separando por actividad, cada fase forma una masa propia
  aunque en el histograma de gris no se vea.

#### `normalizar_por_fila(conteos)`

- **Qué hace:** divide cada fila del paisaje por su máximo.
- **Por qué:** una fase con mucha área (la matriz) domina el paisaje y tapa a las demás. Dividiendo
  por fila, cada nivel de actividad pesa lo mismo y una fase escasa pero bien definida se ve tanto
  como una abundante. Las filas casi vacías se dejan en cero para no amplificar ruido.

#### `nucleos_y_pozos(normalizado, conteos, media_altura=0.5, masa_minima=0.01)`

- **Qué hace:** encuentra los núcleos (partes densas del paisaje con masa suficiente) e inunda el
  paisaje desde ellos para formar los pozos.
- **Por qué:** los núcleos responden **cuántas poblaciones hay y qué gris tienen**, sin exigir que
  haya un valle en el histograma; la masa mínima evita contar como población un grupo de pocos
  píxeles. Los pozos responden **a qué población va cada píxel**: la frontera entre dos pozos queda
  donde hay menos píxeles, así que no depende de cómo se escalen los ejes del paisaje.
- **El usuario decide:** qué fracción de la fila define un núcleo y la masa mínima de una
  población.

#### `poblaciones_de_region(gris, act)`

- **Qué hace:** une paisaje, normalización, núcleos y pozos, y devuelve la población de cada
  píxel, el gris central y la fracción de cada una, numeradas de la más oscura a la más clara.
- **Por qué:** es la unidad que se aplica a cada región; numerar por gris hace que la población 0
  sea siempre la más oscura, en cualquier región y cualquier ROI.

### Regiones de análisis — `regiones.py`

Una ROI puede tener zonas con comportamientos muy distintos (una franja negra, una zona lisa,
una zona con patrón). Si se buscan poblaciones en toda la ROI a la vez, las zonas grandes o de
mucho contraste dominan y tapan a las demás. Por eso primero se divide en regiones:

1. **Zonas por gris** (`zonas_por_gris`): cortes en los valles del histograma de gris
   (`cortes_por_valles`, con los criterios del módulo TEM: prominencia 5 %, distancia 8 casillas,
   masa 1 %, valle < 75 % del pico menor). Las manchitas de menos del 1 % del área pasan a la zona
   vecina.
2. **Franja de borde** (`franja_de_borde`): los píxeles de transición entre zonas. Su ancho sale de
   la imagen: es la distancia a la que el gris recorre el 90 % del camino hasta su valor interior.
3. **Actividad sin cruzar límites** (`actividad_sin_cruzar`): promedio del gradiente con σ = 25 px,
   usando solo píxeles de la misma zona y fuera de la franja.
4. **Clases de actividad**: dentro de cada zona, cortes en los valles del histograma de actividad.

Cada región se **aísla** (`aislar_region`) antes de procesarla: se recorta y todo lo que queda
fuera de ella se reemplaza por el píxel más cercano de la región. Así el realce, el choque y la
actividad no ven el gris ni los bordes de las otras regiones.

### Realce — `realce.py`

El desenfoque del microscopio mezcla el gris de dos fases alrededor de cada borde (**degradé**).
En las estructuras finas, ningún píxel llega a su gris propio. El realce lo deshace en parte:

- **`estimar_degrade`**: ancho del degradé como σ de una gaussiana. Mide el ancho 10-90 % de
  bordes aislados (perfil monótono con mesetas lejos del borde) y lo divide por 2.563. Si hay
  menos de 10 bordes aislados, devuelve 1.0 px (valor fijo, no una estimación).
- **`realzar`**: máscara de enfoque con radio igual al degradé y cantidad 1.
- **`filtro_choque`**: filtro de choque de Osher y Rudin. Cada píxel se mueve hacia el gris del
  lado del borde al que pertenece, según el signo de la segunda derivada en la dirección del
  gradiente; el degradé se parte en dos y quedan mesetas. Frena cuando el cambio medio es menor
  que 0.1 × ruido.
- **`filtro_choque_con_pozos`**: primero el choque solo, hasta que frena; después le suma una
  atracción hacia el gris central del pozo de cada píxel, pesada por la densidad del paisaje en su
  casilla. Cada masa del paisaje tiende a una línea vertical (el mismo gris en el interior y en el
  borde). Se activa recién cuando el choque frenó: antes, el paisaje de una imagen con degradé
  tiene un solo pozo y la atracción comprimiría todos los grises.

### Poblaciones — `poblaciones.py`

1. **`actividad`**: promedio del gradiente con σ = 3 px (del orden del espesor de lo más fino),
   calculado sobre la imagen **de partida**. Separa el interior de una estructura (actividad baja)
   de su borde (alta). Se usa la de partida porque el choque afila los bordes y subiría la
   actividad de todas las fases por igual.
2. **`paisaje`**: histograma conjunto gris × actividad (128 × 128 casillas). Cada población forma
   una masa; con bordes nítidos, una línea vertical.
3. **`normalizar_por_fila`**: cada fila dividida por su máximo, para que cada nivel de actividad
   pese lo mismo y una fase con mucha masa no tape a las demás.
4. **`nucleos_y_pozos`**:
   - **núcleo**: casillas sobre la media altura de su fila, conexas, con al menos el 1 % de los
     píxeles. **Cada núcleo es una población**: dice cuántas hay y da su gris central (mediana de
     sus píxeles);
   - **pozo**: lo que se inunda (watershed) desde cada núcleo; la frontera entre pozos queda donde
     hay menos píxeles. **Cada píxel toma la población del pozo en el que cae.**
5. **`poblaciones_de_region`**: une los pasos anteriores y numera las poblaciones por gris (0 = la
   más oscura).

### Parámetros

| Parámetro | Valor | ¿Sale de la imagen? |
|---|---|---|
| ruido (por región) | — | sí (`estimar_ruido`) |
| degradé (por región) | — | sí, si hay ≥ 10 bordes aislados; si no, 1.0 px fijo |
| ancho de la franja de borde | — | sí |
| fin del filtro de choque | cambio < 0.1 × ruido | sí |
| criterios de picos y valles | 5 %, 8 casillas, 1 %, 75 % | no (vienen de TEM) |
| σ de la actividad | 25 px (regiones), 3 px (paisaje) | no |
| cantidad del realce | 1 | no |
| σ interno del choque | 1 px | no |
| intensidad de los pozos | 2 | no |
| núcleo: media altura y masa mínima | 0.5 y 1 % | no |

Los parámetros que no salen de la imagen son argumentos con esos valores por defecto. Falta
trabajar criterios objetivos para fijarlos.

### Resultados en las ROIs de prueba

| ROI | Regiones | Poblaciones |
|---|---|---|
| BSE_1_1 (todo patrón) | 1 | 2: grises 0.549 y 0.718; fase oscura 25 % |
| BSE_1_3 (patrón, zona lisa, franja negra, zona gris) | 4 | patrón: 2 (0.559 y 0.692; fase oscura 19 %); las otras tres, 1 cada una |

### Limitaciones conocidas

- **Fases con un valle claro en el gris.** Si dos fases tienen un valle marcado en el histograma
  (imagen de mucho contraste), `zonas_por_gris` las separa como si fueran zonas y las clases de
  actividad se fragmentan. En las ROIs reales no pasa; en simuladas de alto contraste sí (hay dos
  tests marcados como falla conocida en `tests/sem/test_bse.py`).
- **Zona lisa de BSE_1_3.** Su paisaje tiene forma de V: dos ramas (estrías oscuras y fondo
  claro) que se tocan con actividad baja. La media altura las toma como un solo núcleo, así que
  sale una sola población aunque probablemente haya dos.
- **Degradé en regiones lisas o chicas.** Con pocos bordes aislados se usa el valor fijo de 1 px.
- **Normalización por fila en regiones chicas.** Con pocos píxeles, algunas filas quedan casi
  vacías y la normalización las infla.

La exploración que llevó a este método (lo probado, lo descartado y por qué) está en
[`pruebas_sem/NOTAS_EXPLORACION.md`](../pruebas_sem/NOTAS_EXPLORACION.md).

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
| **Degradé** | Mezcla de grises en la franja alrededor de un borde, causada por el desenfoque; borra el gris propio de las estructuras finas |
| **Región de análisis** | Parte de la ROI que se procesa por separado (zona de gris × clase de actividad) |
| **Franja de borde** | Píxeles de transición entre dos zonas de gris; no pertenecen a ninguna región |
| **Actividad** | Promedio local del gradiente: cuántos bordes hay alrededor de un píxel |
| **Paisaje gris × actividad** | Histograma conjunto: cuántos píxeles hay con cada par (gris, actividad) |
| **Núcleo** | Parte más densa de una masa del paisaje; cada núcleo es una población y da su gris central |
| **Pozo** | Todo lo que se inunda desde un núcleo; decide a qué población va cada píxel |
| **Filtro de choque** | Filtro que lleva cada píxel al gris del lado del borde al que pertenece y deja mesetas |
| **Backend de matplotlib** | Motor que dibuja las figuras; uno "con ventanas" (TkAgg, QtAgg) es necesario para las herramientas interactivas |
