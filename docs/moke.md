# Módulo MOKE

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Ángel A. Pérez (angel.perez@ib.edu.ar).

El módulo `moke` trabaja con secuencias de imágenes de **microscopía
magneto-óptica Kerr (MOKE)** en las que un **dominio magnético** crece
(o se achica) cuadro a cuadro. En cada imagen detecta el dominio, traza
su contorno, le ajusta un círculo y guarda centro, radio, área y
contraste en una tabla.

La técnica se apoya en una **imagen de fondo** (`background`), tomada
sin el dominio, con la que se corrige la iluminación despareja de la
apertura del microscopio:

1. **[Carga](#paso-1--carga-de-las-imágenes)**: lee las imágenes
   numeradas y el fondo, y las pasa a escala de grises en [0, 1].
2. **[Apertura e interior](#paso-2--apertura-iluminada-e-interior)**:
   encuentra la zona iluminada y descarta su borde.
3. **[Corrección por el fondo](#paso-3--corrección-por-el-fondo)**:
   calcula el contraste relativo (imagen − fondo) / fondo.
4. **[Zona inicial](#paso-4--zona-inicial-semilla)**: ubica el dominio,
   con una semilla o automáticamente.
5. **[Umbral iterativo](#paso-5--umbral-entre-dominio-y-fondo)**:
   separa el dominio del fondo con un umbral a mitad de camino entre
   ambos niveles.
6. **[Contorno y círculo](#paso-6--contorno-y-círculo-ajustado)**:
   extrae el borde y ajusta un círculo por mínimos cuadrados.

Código: [`moke.py`](../moke.py). Se llama desde
[`main.py`](../main.py).

---

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz:

```bash
uv sync
```

El módulo solo necesita `numpy`, `scipy`, `scikit-image` y
`matplotlib`.

## Uso rápido

Desde la raíz del proyecto:

```bash
uv run python main.py moke background.png evolucion_dominio --escala 75 --unidad nm
```

| Argumento | Significado |
|---|---|
| `moke` | técnica |
| `background.png` | imagen de fondo: un nombre dentro de la carpeta o una ruta |
| `evolucion_dominio` | carpeta con las imágenes `0.png`, `1.png`, `2.png`, … |
| `--escala` | tamaño físico de un píxel (por defecto 1) |
| `--unidad` | unidad de `--escala` (por defecto `px`) |

Los resultados quedan en `resultados/`: una figura
`dominio_<n>.png` por imagen y la tabla `medidas_moke.csv`.

Desde Python, con más control:

```python
import moke

filas = moke.analizadorMOKE("background.png", "evolucion_dominio",
                            salida="resultados", escala=75, unidad="nm",
                            graficar=False)

# Una sola imagen
img = moke.leer_gris("evolucion_dominio/10.png")
fondo = moke.leer_gris("evolucion_dominio/background.png")
res = moke.detectar_dominio(img, referencia=fondo)
res["radio"], res["centro"], res["contraste"]
moke.mostrar_dominio(img, res)
```

## Requisitos de las imágenes

- **Nombres**: las imágenes a analizar se llaman `0.png`, `1.png`, … (o
  `.jpg`). El orden es numérico (`10` va después de `9`). Cualquier otro
  PNG o JPG en la carpeta que no sea el fondo da `ValueError`.
- **El fondo** puede llamarse como sea (`background.png`) y estar en la
  carpeta o en otra parte. Se excluye de la lista **antes** de leer los
  números, así no rompe el `int(nombre)`.
- **Mismo tamaño y alineadas**: imagen y fondo tienen que coincidir
  píxel a píxel. No se corrige la deriva de la muestra.
- **La última imagen debe tener el dominio más grande**: de ella sale la
  semilla para todas las demás (ver
  [Paso 4](#paso-4--zona-inicial-semilla)).
- **El dominio es más oscuro que el fondo**. Un dominio más claro no se
  detecta.
- Color (RGB) se convierte a gris; enteros de 8 o 16 bits se llevan a
  float en [0, 1] (`img_as_float`).

---

## Cómo funciona, paso a paso

Todo el cálculo se hace sobre una **copia reducida** de la imagen
(factor `escala = 4`: una imagen de 2048 px pasa a 512 px) para que sea
rápido. Los parámetros se expresan en **píxeles de la imagen original**
y se dividen por 4 internamente; el contorno y el área se devuelven en
píxeles originales.

> **Dos "escalas" distintas.** En `detectar_dominio`, `escala` es el
> factor de reducción (4). En `analizadorMOKE` y `main.py`, `escala` es
> la calibración física (nm/px, µm/px). No tienen relación.

### Paso 1 — Carga de las imágenes

`leer_gris` lee con `skimage.io.imread(as_gray=True)` y convierte con
`img_as_float`. No se normaliza por percentiles: como después se divide
por el fondo, solo importan las intensidades **relativas**.

Se valida que la imagen sea 2D, que no tenga valores no finitos (NaN,
inf) y que el fondo tenga el mismo tamaño.

### Paso 2 — Apertura iluminada e interior

En MOKE solo una zona de la cámara está iluminada (la apertura); el
resto es negro y no tiene información.

1. Se umbraliza con **Otsu** y se toma la región conectada más grande,
   con los huecos rellenos: esa es la apertura.
2. Se **erosiona** con un disco de radio `margen` (80 px): el
   **interior**. El borde de la apertura tiene un gradiente fuerte de
   iluminación y reflejos; fuera del interior no se mide nada.

Si hay fondo, **la apertura se busca en el fondo** y no en la imagen: en
el fondo no hay dominio que pueda confundir a Otsu. Si la erosión borra
toda la apertura, se pide reducir `margen`.

### Paso 3 — Corrección por el fondo

Imagen y fondo se suavizan con el **mismo** filtro gaussiano
(σ = `sigma` = 12 px) y se calcula el contraste relativo:

```
plana = (imagen − fondo) / fondo        (solo en el interior; 0 afuera)
```

- **Por qué dividir y no solo restar**: la iluminación de la apertura es
  despareja (más fuerte en el centro) y el contraste Kerr es
  **proporcional** a la luz que llega. Restando, el mismo dominio se ve
  con más contraste en el centro que en los bordes; dividiendo, el
  contraste es el mismo en toda la apertura.
- **Lectura directa**: el fondo queda en ~0 y el dominio es negativo.
  `plana = −0.10` significa **un 10 % más oscuro** que el fondo.
- La resta se hace en float: los valores negativos no se recortan ni
  desbordan (con `uint8` o `uint16` sí pasaría).
- Si el fondo vale 0 o menos en algún punto del interior, se lanza
  `ValueError`.

**Sin imagen de fondo** (`referencia=None`, solo desde Python) el fondo
se **estima** con un polinomio 2D de orden `orden` = 2 ajustado sobre el
interior. El ajuste es **robusto**: en 15 iteraciones, los píxeles que
quedan por debajo del polinomio (el dominio, que es oscuro) pesan 0.2
en lugar de 1, así el dominio no arrastra el fondo hacia abajo. Es una
alternativa cuando no se tomó el fondo, pero un fondo medido es siempre
mejor.

### Paso 4 — Zona inicial (semilla)

Hace falta una primera idea de dónde está el dominio:

- **Con semilla** `(fila, columna)`: un círculo de radio `suavizado`
  (24 px) alrededor del punto, dentro del interior.
- **Sin semilla** (automático): se umbraliza `plana` con Otsu, se aplica
  una apertura morfológica (disco de 24 px) para borrar manchas chicas,
  se descartan las regiones que tocan el borde del interior
  (`clear_border`) y se toma la más grande.

En `analizadorMOKE` se usan las dos: primero se detecta el dominio de la
**última imagen** en modo automático y su **centro** pasa a ser la
semilla de **todas** las imágenes, incluida la última. Así, en los
cuadros donde el dominio es chico (y Otsu podría elegir una mancha
cualquiera) se busca siempre en el lugar correcto.

### Paso 5 — Umbral entre dominio y fondo

El umbral se coloca **a mitad de camino entre el nivel del dominio y el
del fondo**, y ambos niveles se refinan juntos:

1. Nivel inicial del fondo: percentil 95 de `plana` en el interior.
2. Nivel del dominio: **mediana** de `plana` en la zona actual.
3. Umbral = (fondo + dominio) / 2. El fondo pasa a ser la mediana de los
   píxeles por encima del umbral (3 veces, hasta que se estabiliza).
4. Se marcan los píxeles por debajo del umbral, se aplica una apertura
   (disco de 24 px) y se elige la región que más se superpone con la
   zona; se rellenan sus huecos. Esa es la nueva zona.
5. Se repite desde 2 (3 veces en total).

Al final se aplica un cierre morfológico (disco de 24 px) que suaviza
el borde.

- **Por qué la mitad y no Otsu**: el dominio ocupa a veces una fracción
  chica del interior (cuadros iniciales) y a veces casi todo (finales).
  Otsu depende de esa proporción; el punto medio entre los niveles no.
- **Por qué medianas**: no se mueven con píxeles sueltos (polvo, defectos
  de la cámara).
- **No supone que el fondo valga 0**: si la imagen tiene otra exposición
  que el fondo, todo `plana` se corre, pero el umbral se corre con él.

**Ruido**: se estima como la dispersión robusta (MAD × 1.4826) de los
píxeles de fondo **después del suavizado** del Paso 3.

**Avisos** (con `warnings.warn`; no detienen el análisis):

| Aviso | Cuándo |
|---|---|
| "El contraste es bajo comparado con el ruido" | \|dominio − fondo\| < 3 × ruido |
| "El dominio llega al borde de la apertura" | la máscara toca el borde del interior: el contorno puede estar cortado |

### Paso 6 — Contorno y círculo ajustado

- **Contorno**: `skimage.measure.find_contours` sobre la máscara (nivel
  0.5); si hay varios, el más largo. Se pasa a coordenadas de la imagen
  original.
- **Círculo**: ajuste algebraico por mínimos cuadrados (método de Kåsa).
  Para cada punto (f, c) del contorno se pide
  `f² + c² = 2·f₀·f + 2·c₀·c + k`, que es **lineal** en f₀, c₀ y k; el
  radio es `√(k + f₀² + c₀²)`. No necesita valores iniciales ni
  iteraciones.
- **Área**: píxeles de la máscara (en la imagen reducida) × 4², es decir
  en píxeles originales. Es independiente del círculo: si el dominio no
  es circular, área y radio dejan de concordar (√(área/π) ≠ radio).

---

## Referencia de funciones

### `leer_gris(ruta) -> np.ndarray`

Lee una imagen y la devuelve en escala de grises, float en [0, 1].

### `detectar_dominio(img, semilla=None, referencia=None, escala=4, sigma=12, margen=80, suavizado=24, orden=2) -> dict`

Detecta un dominio oscuro en una imagen.

- **img** (`np.ndarray`): imagen 2D.
- **semilla** (`(fila, col)` o `None`): punto dentro del dominio, en px
  de la imagen original. `None`: búsqueda automática.
- **referencia** (`np.ndarray` o `None`): imagen de fondo sin el dominio,
  del mismo tamaño. `None`: fondo polinomial.
- **escala**: factor de reducción para el cálculo.
- **sigma**, **margen**, **suavizado**: en px de la imagen original (ver
  [parámetros](#parámetros-ajustables)).
- **orden**: grado del polinomio de fondo (solo sin `referencia`).
- **Devuelve**:

| Clave | Contenido |
|---|---|
| `contorno` | array (N, 2) de puntos (fila, columna), en px |
| `centro` | (fila, columna) del círculo ajustado, en px |
| `radio` | radio del círculo ajustado, en px |
| `area` | área de la máscara, en px² |
| `contraste` | nivel del dominio − nivel del fondo, en unidades de (I − I₀)/I₀ (negativo) |
| `ruido` | dispersión robusta del fondo suavizado, mismas unidades |
| `mascara` | máscara booleana del dominio, del tamaño de `img` |

- **Errores** (`ValueError`): imagen no 2D o con NaN/inf, tamaños
  distintos, sin apertura, margen demasiado grande, fondo ≤ 0, dominio no
  encontrado, semilla fuera de una zona oscura, dominio y fondo
  inseparables.

### `mostrar_dominio(img, res, ax=None) -> Axes`

Muestra la imagen con el contorno (rojo), el círculo ajustado (cian,
punteado) y su centro. El contraste de la imagen se ajusta a los
percentiles 1–99 de la parte iluminada (sin la zona negra fuera de la
apertura). El título indica centro y radio en px.

### `analizadorMOKE(background, carpeta, salida="resultados", escala=1.0, unidad="px", graficar=False) -> list`

Analiza todas las imágenes numeradas de una carpeta.

- **background**: nombre del fondo dentro de `carpeta`, o una ruta.
- **carpeta**: carpeta con `0.png`, `1.png`, …
- **salida**: carpeta de resultados (se crea si no existe).
- **escala**, **unidad**: calibración física de un píxel.
- **graficar**: si es `True`, abre cada figura con `plt.show()` (además
  de guardarla).
- **Devuelve**: las filas de la tabla (la primera es el encabezado).
- Si una imagen da `ValueError`, imprime el motivo y sigue con la
  siguiente (esa imagen no aparece en la tabla).
- **Errores**: `NotADirectoryError` si no existe la carpeta,
  `FileNotFoundError` si no existe el fondo, `ValueError` si no hay
  imágenes o tienen nombres no numéricos.

## Salidas

**`dominio_<n>.png`** (150 dpi): la figura de `mostrar_dominio` para la
imagen `<n>`.

**`medidas_moke.csv`**, una fila por imagen analizada:

| Columna | Contenido |
|---|---|
| `archivo` | nombre de la imagen |
| `fila_centro_px`, `col_centro_px` | centro del círculo ajustado (px) |
| `radio_px` | radio (px) |
| `radio_<unidad>` | radio × `escala` |
| `area_<unidad>2` | área × `escala`² |
| `contraste` | (I − I₀)/I₀ del dominio respecto del fondo |
| `dudoso` | `True` si \|contraste\| < 3 × ruido |

---

## Parámetros ajustables

Son argumentos de `detectar_dominio` (desde `main.py` se usan los
valores por defecto):

| Parámetro | Valor | Qué controla |
|---|---|---|
| `escala` | 4 | Factor de reducción para el cálculo. Más chico: más preciso y más lento |
| `sigma` | 12 px | Suavizado gaussiano de imagen y fondo. Sube si hay mucho ruido; baja para dominios chicos |
| `margen` | 80 px | Borde de la apertura que se descarta |
| `suavizado` | 24 px | Radio de la semilla y del disco de apertura/cierre: manchas más chicas que esto se ignoran |
| `orden` | 2 | Grado del polinomio de fondo (solo sin imagen de fondo) |

Constantes fijas dentro del código: 15 iteraciones y peso 0.2 del ajuste
robusto del polinomio, percentil 95 para el nivel inicial del fondo, 3×3
iteraciones del umbral, límite de 3 × ruido para `dudoso`.

---

## Resultados

### Secuencia real (`resultados/medidas_moke.csv`)

Imágenes `6.png` a `15.png`, calibración 75 nm/px:

| Imagen | Radio (px) | Radio (µm) | Contraste |
|---|---|---|---|
| 6 | 211.1 | 15.8 | −0.097 |
| 8 | 275.2 | 20.6 | −0.098 |
| 10 | 338.5 | 25.4 | −0.099 |
| 12 | 403.0 | 30.2 | −0.099 |
| 15 | 498.6 | 37.4 | −0.100 |

- **El radio crece linealmente**: 31.96 px por cuadro (2.40 µm por
  cuadro), con un apartamiento máximo de la recta de 0.4 px. Con el
  intervalo entre cuadros, esto da directamente la velocidad de la pared
  de dominio.
- **El dominio es circular**: el radio equivalente al área,
  √(área/π), coincide con el radio ajustado en más del 99.7 % en todos
  los cuadros.
- **El contraste es constante** (−0.097 a −0.100, ~10 % más oscuro) y
  ningún cuadro queda `dudoso`.
- **El centro se corre** ~9 px hacia abajo y ~12 px hacia la izquierda
  en 9 cuadros (~1 µm): el dominio no crece exactamente igual en todas
  las direcciones, o la muestra deriva.

### Imágenes sintéticas

Prueba de control con 6 imágenes de 2048 × 1536 (16 bits): apertura
elíptica, iluminación gaussiana despareja (60 % a 85 %), dominio
circular 10 % más oscuro con radios conocidos y ruido gaussiano del 3 %.
Tarda ~1.4 s por imagen.

| Radio real (px) | Radio medido (px) | Centro (real: 760, 1100) | Contraste | Dudoso |
|---|---|---|---|---|
| 0 (sin dominio) | 74.5 | 1124, 550 | −0.001 | **True** |
| 60 | 60.2 | 760.1, 1100.0 | −0.090 | False |
| 120 | 119.8 | 760.3, 1100.0 | −0.099 | False |
| 200 | 199.7 | 759.8, 1099.9 | −0.099 | False |
| 300 | 299.8 | 760.0, 1100.0 | −0.100 | False |
| 420 | 420.0 | 760.0, 1100.0 | −0.100 | False |

- Radio y centro con error **menor que 0.3 px**, a pesar de la
  iluminación despareja: la división por el fondo la elimina.
- En el dominio de 60 px el contraste baja a −0.090: el suavizado de
  σ = 12 px mezcla el borde con el fondo y se nota en dominios chicos.
- **Sin dominio, igual se informa uno**: una mancha de ruido lejos de la
  semilla. Lo delata `dudoso = True` y el contraste casi nulo.

---

## Limitaciones y próximos pasos

- **Una imagen sin dominio no se descarta**: aparece en la tabla con un
  "dominio" de ruido. Hay que filtrar por `dudoso` (o por contraste)
  antes de usar los datos.
- **Los avisos se muestran una sola vez**: Python muestra cada
  `warnings.warn` una sola vez por ejecución. Si varias imágenes tocan el
  borde de la apertura, solo se avisa en la primera, y la tabla no tiene
  una columna que lo indique.
- **Dominio cortado por la apertura**: si el dominio sale del interior,
  el contorno incluye el borde recto del interior y el círculo ajustado
  queda sesgado. El área tampoco es la real.
- **Un solo dominio, oscuro y más o menos circular**: no detecta varios
  dominios a la vez, dominios claros (inversión de contraste) ni formas
  dendríticas, donde el radio del círculo pierde sentido (el área sigue
  siendo válida).
- **Semilla fija**: el centro de la última imagen se usa para todas. Si
  el dominio se desplaza mucho o se nuclea en otro lugar en los primeros
  cuadros, la semilla puede caer fuera de él.
- **Sin corrección de deriva**: imagen y fondo tienen que estar
  alineados; un corrimiento de la muestra deja bordes falsos en
  `plana`.
- **Resolución**: con `escala = 4` el borde se determina en una grilla
  de 4 px (el ajuste del círculo, que usa todo el contorno, lo afina a
  menos de 1 px). Manchas y entrantes de menos de ~24 px se borran con la
  apertura y el cierre.
- **El ruido se mide sobre la imagen suavizada**, así que es menor que el
  ruido por píxel: el criterio de 3 × ruido es permisivo.
- **Detalles menores**: con `--unidad px` la tabla tiene dos columnas
  `radio_px`; los parámetros de detección no se pueden cambiar desde
  `main.py`.
- **Próximos pasos**: velocidad de la pared (radio vs. tiempo, con el
  intervalo entre cuadros), una columna `toca_borde` en la tabla,
  descartar automáticamente las imágenes sin dominio, y tests en
  `tests/moke/` con imágenes sintéticas como las de arriba.
