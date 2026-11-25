# Interfaz gráfica

Parte del proyecto **Analizador de Imágenes de Microscopía**.

La interfaz gráfica reúne en una sola ventana los tres módulos del
proyecto: **[MOKE](moke.md)**, **[SEM](sem.md)** y **[TEM](tem.md)**. Se
elige la técnica, la entrada (una imagen o una carpeta) y las opciones, y
el botón **Analizar** corre el análisis de ese módulo. La interfaz no
analiza nada por su cuenta: llama a las mismas funciones que los scripts
de `scripts/` y que el uso desde Python, con los mismos valores por
defecto, y guarda los resultados en las mismas carpetas.

1. **[Uso](#uso)**: cómo se abre y qué hace cada panel.
2. **[MOKE](#moke)**, **[SEM](#sem)** y **[TEM](#tem)**: entrada,
   opciones y salidas de cada técnica, y a qué función del módulo se
   conectan.
3. **[Cómo funciona](#cómo-funciona)**: el análisis en segundo plano, el
   registro, los datos que se piden durante el análisis y las figuras.
4. **[Referencia de funciones](#referencia-de-funciones)**.

Código: [`src/interfaz/`](../src/interfaz/). Punto de entrada:
[`main.py`](../main.py). Tests: [`tests/interfaz/`](../tests/interfaz/).

---

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz:

```bash
uv sync
```

La interfaz usa **Tkinter**, que viene con Python; no agrega
dependencias propias. En Linux, si Python se instaló desde el sistema y
no con uv, puede hacer falta el paquete de Tk (por ejemplo `python3-tk`
en Debian/Ubuntu). Cada técnica necesita además las librerías de su
módulo (ver la documentación de cada una).

## Uso

Desde la raíz del proyecto:

```bash
uv run python main.py
```

`main.py` agrega `src/` al `sys.path` y llama a `abrir_interfaz()`. La
ventana tiene, de arriba hacia abajo:

1. **Técnica**: MOKE, SEM o TEM. Al cambiarla se muestra solo el panel
   de esa técnica; lo cargado en las otras se conserva.
2. **Panel de la técnica**: la entrada, con botones "Elegir carpeta…" /
   "Elegir archivo…" que abren un diálogo, y las opciones. También se
   puede escribir o pegar la ruta en el campo.
3. **Analizar** (a la derecha) y **Limpiar registro** (a la izquierda).
   Entre ambos, el estado: "Analizando…", "Elegí la ROI…", "Listo" o
   "Terminó con errores".
4. **Registro**: los mensajes de cada módulo, el avance, los avisos (en
   naranja), los errores (en rojo) y la carpeta de resultados (en
   verde).

Al presionar Analizar:

- Se **validan las opciones**. Si algo está mal (una carpeta que no
  existe, un número inválido, un formato no soportado) se informa en el
  registro y no se corre nada.
- Mientras corre el análisis, **los controles se deshabilitan** y la
  ventana sigue respondiendo: se puede mover, redimensionar y leer el
  registro.
- **Un error no cierra la interfaz.** Con una carpeta, el error de una
  imagen se informa y se sigue con la siguiente; al final se dice
  cuántas fallaron.
- Las **figuras se guardan siempre**. Si se analizó **una sola imagen**,
  además se abren en ventanas aparte, con la barra de zoom de
  matplotlib.

Los números aceptan coma o punto decimal (`0,5` o `0.5`).

### Resumen por técnica

| Técnica | Entrada | Opciones | Función del módulo | Resultados |
|---|---|---|---|---|
| MOKE | carpeta con `0.png`, `1.png`, … y la imagen de fondo | escala y unidad | `moke.analizadorMOKE` | `data/moke/resultados/` |
| SEM (solo BSE) | una imagen `.tif` (abre el selector de ROI) o una carpeta con ROIs guardadas | sufijo de la ROI, intensidad de pozos | `sem.carga.*` y `sem.bse.resultados.segmentar_y_guardar` | `data/sem/roi/<imagen>/<roi>/` |
| TEM | una imagen o una carpeta (`.dm3`, `.mrc`, `.tif`, `.tiff`, `.png`, `.jpg`) | filtro, umbral, fases, fondo, difracción, ancho instrumental | `tem.analizar_ruta` | `data/samples/` |

---

## MOKE

Detección del dominio magnético cuadro a cuadro. Detalle del método en
[docs/moke.md](moke.md).

**Entrada**

| Campo | Contenido |
|---|---|
| Carpeta (0.png, 1.png, …) | carpeta con las imágenes numeradas (`.png` o `.jpg`) |
| Fondo | imagen de fondo tomada sin el dominio; puede estar dentro o fuera de la carpeta |

**Opciones**

| Campo | Por defecto | Contenido |
|---|---|---|
| Escala | `1` | tamaño físico de un píxel; tiene que ser mayor que 0 |
| Unidad | `px` | unidad de la escala, por ejemplo `nm` o `um`; no puede quedar vacía |

**Qué hace**: `correr_moke` llama a
`analizadorMOKE(fondo, carpeta, salida=data/moke, escala, unidad)`
con el resto de los parámetros de `detectar_dominio` en sus valores por
defecto. Al terminar informa cuántas imágenes tuvieron dominio detectado
y lista las marcadas como **dudosas** (contraste menor que 3 veces el
ruido).

**Salidas** (en `data/moke/resultados/`): una figura
`dominio_<n>.png` por imagen y la tabla `medidas_moke.csv` (columnas en
[docs/moke.md — Salidas](moke.md#salidas)). Como MOKE siempre analiza una
carpeta, las figuras no se abren.


## SEM

Carga, selección de ROI y segmentación de imágenes de **electrones
retrodispersados (BSE)**. Las imágenes SE todavía no tienen análisis, y
el panel lo avisa. Detalle del método en [docs/sem.md](sem.md).

El panel tiene dos **modos**:

### Modo "Imagen y ROI"

| Campo | Por defecto | Contenido |
|---|---|---|
| Imagen BSE (.tif) | — | imagen del microscopio, `.tif` o `.tiff` |
| Sufijo de la ROI | vacío | nombre de la ROI, por ejemplo `_1`; vacío = el primer `_N` libre. No puede tener `/ \ : * ? " < > \|` |

Al presionar Analizar, en el **hilo principal** (porque abre ventanas):

1. `preparar_imagen_sem` carga la imagen (`cargar_imagen`), lee la
   metadata (`leer_metadata`: FEI, Zeiss o EXIF) y quita la barra de
   datos (`recortar_barra_datos`). En el registro quedan el fabricante,
   el tamaño antes y después del recorte y el tamaño de píxel.
2. `elegir_y_guardar_roi` abre el **selector de ROI**
   (`seleccionar_roi`): se arrastra el rectángulo y se confirma con
   Enter o cerrando la ventana. La ROI se guarda con `guardar_roi` en
   `data/sem/roi/<imagen>/<imagen><sufijo>/`. Si se cierra sin elegir,
   no se guarda nada y no se segmenta.

Después, en **segundo plano**, `correr_sem_roi` segmenta la ROI con
`segmentar_y_guardar` y abre la figura de la segmentación.

### Modo "ROIs guardadas"

| Campo | Contenido |
|---|---|
| Carpeta | carpeta donde buscar ROIs; el diálogo arranca en `data/sem/roi/` |

`correr_sem_rois` busca todos los `roi_*.npy` de la carpeta **y sus
subcarpetas** y los segmenta uno por uno, informando `[i/N]`. Si una ROI
falla, se sigue con la siguiente. Si no hay ninguna, es un error. No se
abren figuras.

### Opción común

| Campo | Por defecto | Contenido |
|---|---|---|
| Intensidad de pozos | `2` | peso del término de pozos del filtro de choque (≥ 0); `0` = filtro de choque solo. Ver [`filtro_choque_con_pozos`](sem.md#filtro_choque_con_pozosimagen-act-sigma_ruido-intensidad20) |

**Salidas**: junto a cada ROI, `segmentacion_<nombre>.npy` (etiqueta de
cada píxel), `.csv` (tabla de poblaciones) y `.png` (figura).

## TEM

Análisis completo de imágenes TEM o de patrones de difracción, igual que
[`scripts/analizar_tem.py`](../scripts/analizar_tem.py). Detalle del
método en [docs/tem.md](tem.md).

**Entrada**: una imagen o una carpeta ("Elegir imagen…" o "Elegir
carpeta…"). Formatos: `.dm3`, `.mrc`, `.tif`, `.tiff`, `.png`, `.jpg`.

**Opciones** (las mismas que la sección CONFIGURACIÓN del script):

| Campo | Por defecto | Valores | Etapa |
|---|---|---|---|
| Filtro | `auto` | `auto`, `gaussian`, `median`, `bilateral`, `wiener` | 1 — preprocesamiento |
| Umbral | `auto` | `auto`, `otsu`, `gmm`, `triangle`, `adaptive` | 2 — segmentación |
| Fases | vacío | vacío = todas menos el fondo; o una lista de etiquetas, ej. `1,2` | 3 — morfología |
| Fondo | vacío | vacío = fase mayoritaria; o una etiqueta entera, ej. `0` | 3 y 4 |
| Difracción | Detectar sola | Detectar sola (`None`), Analizar siempre como difracción (`True`), Nunca (`False`) | 4.5 — difracción |
| Ancho instrumental (px) | `0` | ≥ 0; se resta en Scherrer; `0` = no se resta | 4.5 |

Si se indican **Fondo** y **Fases** a la vez, Fondo manda: se miden
todas las demás fases y se avisa en el registro.

**Qué hace**: `correr_tem` llama a `analizar_ruta` con esas opciones,
`mostrar=False` y `pedir_dato` (ver
[Datos que faltan](#datos-que-se-piden-durante-el-análisis)). Para cada
imagen, si es un patrón de difracción corre la Etapa 4.5 y si no, las
Etapas 1 a 4.

**Salidas** (en `data/samples/`): las figuras de cada etapa y los CSV de
resumen (el de difracción aparte). Con una sola imagen se abren las
figuras **creadas en esta corrida** (los PNG de la carpeta modificados
después de presionar Analizar). La corrección manual del centro del haz
no está en la interfaz, solo en el script.

---

## Cómo funciona

### Organización del código

```
main.py                 punto de entrada: agrega src/ al path y abre la interfaz
src/interfaz/
    __init__.py         expone Ventana y abrir_interfaz
    ventana.py          la ventana: paneles, diálogos, registro y botón Analizar
    opciones.py         validación de los campos (sin Tkinter)
    tecnicas.py         conexión con cada módulo: una función por análisis
    ejecucion.py        hilo de fondo, cola de mensajes y preguntas al usuario
    figuras.py          ventana de figura guardada y diálogo para pedir un número
tests/interfaz/
    test_opciones.py    validación de opciones
    test_ejecucion.py   captura de print, warnings y errores; preguntas
    test_tecnicas.py    MOKE de punta a punta con imágenes sintéticas, búsqueda de ROIs
    test_ventana.py     paneles y valores por defecto (se saltean sin pantalla)
```

La separación en capas permite probar casi todo sin abrir ventanas:
`opciones.py`, `tecnicas.py` y `ejecucion.py` no tocan Tkinter.

### Validación de opciones — `opciones.py`

`validar_opciones(tecnica, campos)` recibe el texto de cada campo y
devuelve los parámetros ya convertidos (`Path`, `float`, `list[int]`,
`None`…) para la función de la técnica, o lanza `ValueError` con un
mensaje para el usuario. Los valores por defecto de cada campo están en
`POR_DEFECTO` y coinciden con los de los módulos y scripts. Las listas de
filtros, umbrales y formatos TEM (`FILTROS_TEM`, `UMBRALES_TEM`,
`FORMATOS_TEM`, `DIFRACCION_TEM`) se definen acá; si un módulo agrega una
opción, hay que sumarla también en estas constantes.

### Análisis en segundo plano — `ejecucion.py`

El análisis corre en un **hilo aparte** para que la ventana no se
congele. Como Tkinter solo se puede usar desde el hilo principal, los dos
hilos se comunican por una **cola** (`queue.Queue`) con tres tipos de
mensaje:

| Mensaje | Contenido | Lo manda |
|---|---|---|
| `TEXTO` | estilo y texto | todo `print`, `warning` y error del análisis |
| `PREGUNTA` | un `Pregunta` | el análisis cuando necesita un número del usuario |
| `FIN` | un `Resultado` | el hilo al terminar |

`correr_capturando(tarea, cola)`:

- Redirige `stdout` y `stderr` a la cola (`EscritorCola`), así los
  mensajes que imprimen los módulos aparecen en el registro **sin
  cambiar los módulos**.
- Intercepta los `warnings` y los manda como avisos (`⚠ Categoría:
  mensaje`).
- Atrapa cualquier excepción: el error y su traza van al registro y el
  `Resultado` lleva el mensaje en `error`. Nunca lanza.

La ventana revisa la cola cada 100 ms (`INTERVALO_COLA_MS`) desde el
hilo principal (`_revisar_cola`): escribe el texto, responde las
preguntas y, con `FIN`, rehabilita los controles, informa la carpeta de
resultados y abre las figuras.

El registro colorea por estilo y, para el texto normal, por contenido
(`_estilo_por_contenido`): rojo si empieza con `❌`/`✖` o contiene
`ERROR`, naranja si empieza con `⚠`/`Warning`, verde si empieza con
`✅`/`✔`.

### Datos que se piden durante el análisis

Si a un patrón de difracción TEM le faltan en los metadatos el
**voltaje**, la **longitud de cámara** o el **tamaño de píxel del
detector**, el módulo llama a `pedir_dato(texto, valor_por_defecto)`. En
la interfaz esa función (`ejecucion.pedir_dato`) manda una `Pregunta` a
la cola y **espera** (`threading.Event`) a que la ventana la responda.
La ventana muestra un diálogo modal (`figuras.pedir_numero`) con:

- el valor propuesto, que se acepta con "Usar el valor propuesto";
- la casilla **"Usar este valor para las demás imágenes"**, que guarda
  la respuesta y no vuelve a preguntar lo mismo en esa corrida (se
  olvida al presionar Analizar de nuevo).

El valor cargado a mano queda en el registro. Solo se aceptan números
mayores que 0.

### Matplotlib y las figuras — `figuras.py`

pyplot no se puede usar desde un hilo que no es el principal. Por eso:

- `abrir_interfaz` fija el backend **`Agg`** (sin ventanas): los módulos
  guardan sus figuras en PNG y nunca abren ventanas.
- Las figuras se abren **desde el PNG guardado** (`abrir_figura`), con
  una `Figure` de matplotlib incrustada en una ventana Tkinter y la
  barra de zoom, sin pyplot. La ventana se escala para entrar en
  1100 × 750 px.
- El **selector de ROI** de SEM sí necesita pyplot interactivo: mientras
  está abierto se cambia a `TkAgg` y al cerrarse se vuelve a `Agg`.
- Después de cerrar una ventana se llama a `gc.collect()` en el hilo
  principal, para que los objetos Tk se liberen ahí y no desde el hilo
  de fondo (que daría `RuntimeError`).

### Carpetas de salida

Las rutas se calculan desde la raíz del proyecto (`tecnicas.RAIZ`), así
no dependen de la carpeta desde donde se corre `main.py`:

| Constante | Carpeta |
|---|---|
| `SALIDA_MOKE` | `data/moke/resultados/` |
| `SALIDA_SEM` | `data/sem/roi/` |
| TEM (`analizar_ruta`) | `data/samples/` |

Todo lo que está bajo `data/` está en el `.gitignore`.

---

## Referencia de funciones

### `ventana.py`

| Función / clase | Qué hace |
|---|---|
| `abrir_interfaz()` | Fija el backend `Agg`, crea la ventana y espera a que se cierre. |
| `Ventana(raiz)` | Ventana principal. `campos(tecnica)` devuelve el texto de los campos; `analizar()` valida y arranca; `escribir(texto, estilo)` agrega al registro. |

### `opciones.py`

| Función | Qué hace |
|---|---|
| `validar_opciones(tecnica, campos) -> dict` | Convierte los campos en parámetros; `ValueError` si alguno es inválido. |

### `tecnicas.py`

| Función | Hilo | Qué hace |
|---|---|---|
| `correr_moke(parametros) -> Resultado` | fondo | `analizadorMOKE` sobre la carpeta. |
| `correr_tem(parametros, pedir_dato) -> Resultado` | fondo | `analizar_ruta`; devuelve las figuras nuevas si fue una sola imagen. |
| `preparar_imagen_sem(imagen) -> (recortada, metadata)` | principal | Carga, metadata y recorte de la barra de datos. |
| `elegir_y_guardar_roi(imagen, recortada, metadata, sufijo) -> Path \| None` | principal | Selector de ROI y guardado; `None` si no se eligió. |
| `correr_sem_roi(ruta_roi, intensidad_pozos) -> Resultado` | fondo | Segmenta una ROI y devuelve su figura. |
| `buscar_rois(carpeta) -> list[Path]` | — | `roi_*.npy` de la carpeta y subcarpetas, ordenadas. |
| `correr_sem_rois(carpeta, intensidad_pozos) -> Resultado` | fondo | Segmenta todas las ROIs; sigue si una falla. |

### `ejecucion.py`

| Función / clase | Qué hace |
|---|---|
| `Resultado(salida, figuras, error)` | Lo que devuelve una tarea: carpeta de salida, figuras a abrir y mensaje de error. |
| `Pregunta(texto, por_defecto)` | Un número que el análisis le pide al usuario. |
| `EscritorCola(cola, estilo)` | Objeto tipo archivo que manda cada `write` a la cola. |
| `correr_capturando(tarea, cola)` | Corre la tarea capturando print, warnings y errores. |
| `iniciar_en_segundo_plano(tarea, cola) -> Thread` | Lo anterior en un hilo `daemon`. |
| `pedir_dato(cola, texto, por_defecto) -> float` | Pregunta desde el hilo de fondo y espera la respuesta. |
| `pngs_nuevos(carpeta, desde, recursivo=False) -> list[Path]` | PNG modificados después de `desde`: las figuras de esta corrida. |

### `figuras.py`

| Función | Qué hace |
|---|---|
| `abrir_figura(padre, ruta) -> Toplevel` | Abre un PNG en una ventana con zoom. |
| `pedir_numero(padre, titulo, texto, por_defecto, nota="") -> (float \| None, bool)` | Diálogo modal; devuelve el número (o `None`) y si se marcó "usar para las demás". |

---

## Tests

```bash
uv run pytest tests/interfaz
```

30 tests: validación de opciones (15), captura y preguntas en segundo
plano (5), conexión con los módulos (6, incluye MOKE de punta a punta con
imágenes sintéticas) y la ventana (4). Los de la ventana se saltean si
no hay pantalla (por ejemplo en un servidor o en CI).

## Limitaciones y próximos pasos

- **SEM solo BSE**: el análisis de imágenes SE no está implementado en el
  módulo, así que tampoco en la interfaz.
- **Opciones avanzadas**: solo se exponen las opciones principales de
  cada técnica. Los parámetros de `detectar_dominio` (MOKE) y de la
  segmentación SEM usan sus valores por defecto; para cambiarlos hay que
  usar Python.
- **Centro del haz en difracción**: la corrección manual solo está en
  `scripts/analizar_tem.py`.
- **Carpeta de salida fija**: no se puede elegir desde la ventana.
- **No se puede cancelar** un análisis en curso; hay que esperar a que
  termine o cerrar la ventana (el hilo es `daemon` y se corta con ella).
- **Figuras de TEM con una imagen**: se detectan por fecha de
  modificación; si otro proceso escribe PNG en `data/samples/` al mismo
  tiempo, también se abrirían.
