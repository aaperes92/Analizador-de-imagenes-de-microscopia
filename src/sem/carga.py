"""Carga de imágenes SEM: lectura, metadata de adquisición y selección de la región de interés.

Lee imágenes SEM (.tif de FEI/Thermo y Zeiss), interpreta su metadata (tamaño de píxel,
voltaje, detector...), quita la barra de datos, permite seleccionar una ROI y la guarda.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import json
import re
from datetime import datetime
from pathlib import Path

import exifread
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import tifffile
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Backends de matplotlib que no abren ventanas (no sirven para seleccionar el ROI).
_BACKENDS_SIN_VENTANA = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}

# Factores de conversión a metros para las unidades que usan los equipos SEM.
_A_METROS = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "mm": 1e-3, "m": 1.0}


def cargar_imagen(ruta: Path) -> np.ndarray:
    """
    Carga una imagen SEM desde disco como array 2D float32 normalizado en [0, 1].

    Las imágenes RGB/RGBA se convierten a escala de grises; las imágenes con paleta,
    uint8 o uint16 se reescalan según el rango de su dtype (no según su min/max).
    No recorta la barra de datos: eso lo hace `recortar_barra_datos` usando la metadata.

    Parameters
    ----------
    ruta : Path
        Ruta completa al archivo de imagen (.tif, .png, .jpg, ...).

    Returns
    -------
    np.ndarray
        Array 2D float32 con valores en [0, 1].

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta especificada.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta.resolve()}")

    imagen = ski.io.imread(ruta)

    # Color -> escala de grises. rgb2gray devuelve float en [0, 1].
    if imagen.ndim == 3:
        if imagen.shape[-1] == 4:
            imagen = ski.color.rgba2rgb(imagen)
        imagen = ski.color.rgb2gray(imagen)

    if imagen.ndim != 2:
        raise ValueError(
            f"Se esperaba una imagen 2D o RGB(A); {ruta.name} tiene shape {imagen.shape}"
        )

    # Reescala según el rango del dtype (uint8 -> /255, uint16 -> /65535) sin modificar el original.
    return ski.util.img_as_float32(imagen)


def leer_metadata(ruta: Path) -> dict:
    """
    Lee los metadatos de adquisición de una imagen SEM y los retorna normalizados.

    Intenta, en orden: metadata FEI/Thermo (tag TIFF 34682), Zeiss SmartSEM (tag 34118)
    —ambas vía `tifffile`— y, como respaldo, EXIF estándar vía `exifread`.
    Nunca lanza excepción por metadata ausente o ilegible.

    Parameters
    ----------
    ruta : Path
        Ruta completa al archivo de imagen.

    Returns
    -------
    dict
        Claves normalizadas (solo las que se encuentren):
        'fabricante' ('FEI', 'Zeiss' o 'EXIF'), 'ancho_px', 'alto_px',
        'alto_util_px' (filas sin barra de datos, solo FEI),
        'tamano_pixel_m' (float, metros/píxel), 'ancho_campo_um' (campo de visión horizontal),
        'magnificacion', 'voltaje_kV', 'distancia_trabajo_mm', 'detector',
        'fecha_adquisicion' (ISO 'AAAA-MM-DD HH:MM:SS'),
        y 'crudo' con el diccionario original completo del fabricante.
        Retorna {} si no hay metadata (e imprime un warning).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta.resolve()}")

    metadata = _leer_metadata_tiff(ruta) or _leer_metadata_exif(ruta)

    if not metadata:
        print(f"⚠️ No se encontraron metadatos en {ruta.name}")
    elif "tamano_pixel_m" not in metadata:
        print(f"⚠️ {ruta.name}: sin tamaño de píxel, las medidas quedarán en píxeles")
    return metadata


def recortar_barra_datos(imagen: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Elimina la barra de datos inferior (escala, kV, magnificación) de una imagen SEM.

    - FEI: usa 'alto_util_px' de la metadata.
    - Zeiss: la metadata no indica la barra, así que se detecta en la imagen buscando
      su borde superior (ver `_detectar_barra_zeiss`).
    - Si no hay información ni se detecta barra, retorna una copia sin recortar.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D tal como la devuelve `cargar_imagen`.
    metadata : dict
        Diccionario devuelto por `leer_metadata`.

    Returns
    -------
    np.ndarray
        Copia de la imagen con solo las filas útiles: imagen[:alto_util_px].

    Raises
    ------
    ValueError
        Si 'alto_util_px' no es positivo o es mayor que el alto de la imagen.
    """
    alto_util = metadata.get("alto_util_px")
    if alto_util is None and metadata.get("fabricante") == "Zeiss":
        alto_util = _detectar_barra_zeiss(imagen)
    if alto_util is None:
        return imagen.copy()

    alto_util = int(alto_util)
    if not 0 < alto_util <= imagen.shape[0]:
        raise ValueError(
            f"alto_util_px={alto_util} fuera de rango para una imagen de {imagen.shape[0]} filas"
        )
    return imagen[:alto_util].copy()


def mostrar_imagen_con_histograma(
    imagen: np.ndarray, titulo: str = "", metadata: dict | None = None
) -> plt.Figure:
    """
    Muestra la imagen SEM junto a su histograma de intensidades y, opcionalmente, su metadata.

    Panel izquierdo: imagen (cmap='gray', vmin=0, vmax=1) con barra de escala en µm si la
    metadata trae 'tamano_pixel_m'. Panel derecho: histograma de 256 bins en [0, 1].
    El título incluye `titulo`, shape, dtype y min/max. Si hay metadata, se muestran
    fabricante, detector, kV, WD, magnificación, tamaño de píxel y fecha debajo de la figura.

    La figura se crea pero no se bloquea la ejecución: el script que la usa llama a
    `plt.show()` cuando quiere abrir las ventanas.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en [0, 1], idealmente ya sin barra de datos (`recortar_barra_datos`).
    titulo : str
        Título para la figura (por ejemplo el nombre del archivo).
    metadata : dict, optional
        Diccionario devuelto por `leer_metadata`.

    Returns
    -------
    plt.Figure
        La figura creada, para poder guardarla con `fig.savefig(...)`.
    """
    metadata = metadata or {}
    fig, (ax_imagen, ax_hist) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # Imagen con rango fijo [0, 1]: no se estira el contraste.
    ax_imagen.imshow(imagen, cmap="gray", vmin=0, vmax=1)
    ax_imagen.set_axis_off()
    tamano_pixel = metadata.get("tamano_pixel_m")
    if tamano_pixel:
        _agregar_barra_escala(ax_imagen, imagen.shape[1], tamano_pixel)

    # Histograma de intensidades.
    conteos, bordes = np.histogram(imagen.ravel(), bins=256, range=(0.0, 1.0))
    centros = (bordes[:-1] + bordes[1:]) / 2
    ax_hist.plot(centros, conteos, color="0.2", linewidth=1)
    ax_hist.fill_between(centros, conteos, color="0.6", alpha=0.5)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylim(bottom=0)
    ax_hist.set_xlabel("Intensidad normalizada")
    ax_hist.set_ylabel("Cantidad de píxeles")
    ax_hist.set_title("Histograma de intensidades")
    ax_hist.grid(alpha=0.3)

    estadisticas = (
        f"shape={imagen.shape}  dtype={imagen.dtype}  "
        f"min={imagen.min():.3f}  max={imagen.max():.3f}"
    )
    fig.suptitle(f"{titulo}\n{estadisticas}" if titulo else estadisticas, fontsize=11)

    texto_metadata = _formatear_metadata(metadata)
    if texto_metadata:
        fig.text(0.5, 0.02, texto_metadata, ha="center", va="bottom", fontsize=9, family="monospace")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    return fig


def seleccionar_roi(imagen: np.ndarray, titulo: str = "") -> tuple[int, int, int, int] | None:
    """
    Abre una ventana interactiva para seleccionar un rectángulo (ROI) sobre la imagen SEM.

    Se arrastra con el botón izquierdo; el rectángulo se puede ajustar antes de cerrar.
    La función bloquea hasta que se cierra la ventana (o se presiona Enter) y retorna
    la última selección.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en [0, 1], ya sin barra de datos.
    titulo : str
        Texto para el título de la ventana (por ejemplo el nombre del archivo).

    Returns
    -------
    tuple[int, int, int, int] | None
        Coordenadas (x1, y1, x2, y2) en píxeles, con x1 < x2 e y1 < y2, recortadas
        a los límites de la imagen. None si se cerró sin seleccionar nada.

    Raises
    ------
    RuntimeError
        Si matplotlib usa un backend sin ventanas (por ejemplo 'Agg').
    """
    if plt.get_backend().lower() in _BACKENDS_SIN_VENTANA:
        raise RuntimeError(
            f"El backend de matplotlib '{plt.get_backend()}' no abre ventanas; "
            "usá uno interactivo (por ejemplo TkAgg) para seleccionar el ROI."
        )
    fig, _selector, seleccion = _preparar_selector_roi(imagen, titulo)
    plt.show(block=True)  # _selector debe seguir vivo mientras la ventana esté abierta
    plt.close(fig)
    return seleccion[0]


def _preparar_selector_roi(imagen: np.ndarray, titulo: str):
    """
    Crea la figura con el RectangleSelector. Retorna (fig, selector, seleccion), donde
    `seleccion[0]` se actualiza con (x1, y1, x2, y2) en cada selección válida.
    Separada de `seleccionar_roi` para poder probarla sin abrir ventanas.
    """
    alto, ancho = imagen.shape[:2]
    seleccion: list[tuple[int, int, int, int] | None] = [None]

    fig, ax = plt.subplots(figsize=(10, 10 * alto / ancho))
    ax.imshow(imagen, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{titulo}\nArrastrá para seleccionar el ROI · Enter o cerrar la ventana para confirmar",
        fontsize=10,
    )

    def al_seleccionar(inicio, fin) -> None:
        if inicio.xdata is None or fin.xdata is None:
            return
        x1, x2 = sorted((round(inicio.xdata), round(fin.xdata)))
        y1, y2 = sorted((round(inicio.ydata), round(fin.ydata)))
        x1, x2 = int(np.clip(x1, 0, ancho)), int(np.clip(x2, 0, ancho))
        y1, y2 = int(np.clip(y1, 0, alto)), int(np.clip(y2, 0, alto))
        if x2 > x1 and y2 > y1:  # se ignoran clics sin área
            seleccion[0] = (x1, y1, x2, y2)
            ax.set_xlabel(f"ROI: x={x1}:{x2}, y={y1}:{y2}  ({x2 - x1} × {y2 - y1} px)")
            fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, al_seleccionar, useblit=True, button=[1], interactive=True, minspanx=2, minspany=2
    )

    def al_presionar_tecla(evento) -> None:
        if evento.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", al_presionar_tecla)
    fig.tight_layout()
    return fig, selector, seleccion


def guardar_roi(
    imagen: np.ndarray,
    nombre_archivo: str,
    coordenadas: tuple[int, int, int, int],
    ruta_mascaras: Path,
    metadata: dict | None = None,
    sufijo: str | None = None,
) -> Path:
    """
    Recorta una ROI de la imagen SEM y la guarda en .npy (float32), .tif (uint8) y .json.

    Estructura: ruta_mascaras/<nombre_base>/<nombre_base><sufijo>/roi_<nombre_base><sufijo>.{npy,tif,json}
    El .json guarda las coordenadas, el archivo de origen y, si hay metadata, el tamaño de
    píxel y los datos de adquisición, para que el análisis posterior mida en unidades reales.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en [0, 1] sobre la que se seleccionó el ROI (ya sin barra de datos).
    nombre_archivo : str
        Nombre del archivo original; se usa su nombre sin extensión como base.
    coordenadas : tuple[int, int, int, int]
        (x1, y1, x2, y2) devuelto por `seleccionar_roi`.
    ruta_mascaras : Path
        Carpeta raíz donde se guardan los ROIs.
    metadata : dict, optional
        Diccionario devuelto por `leer_metadata` (no se guarda el campo 'crudo').
    sufijo : str, optional
        Sufijo de versión (ej. "_1"). Si es None, se usa el primer "_N" libre,
        así nunca se sobrescribe un ROI anterior.

    Returns
    -------
    Path
        Carpeta donde se guardaron los archivos.

    Raises
    ------
    ValueError
        Si las coordenadas no definen un rectángulo válido dentro de la imagen.
    FileExistsError
        Si se pasa un `sufijo` explícito que ya existe.
    """
    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = (int(c) for c in coordenadas)
    if not (0 <= x1 < x2 <= ancho and 0 <= y1 < y2 <= alto):
        raise ValueError(
            f"Coordenadas {coordenadas} inválidas para una imagen de {ancho}×{alto} px "
            "(se espera 0 <= x1 < x2 <= ancho y 0 <= y1 < y2 <= alto)"
        )

    nombre_base = Path(nombre_archivo).stem
    carpeta_base = Path(ruta_mascaras) / nombre_base
    if sufijo is None:
        n = 1
        while (carpeta_base / f"{nombre_base}_{n}").exists():
            n += 1
        sufijo = f"_{n}"
    carpeta_roi = carpeta_base / f"{nombre_base}{sufijo}"
    if carpeta_roi.exists():
        raise FileExistsError(f"Ya existe un ROI con sufijo '{sufijo}': {carpeta_roi}")
    carpeta_roi.mkdir(parents=True)

    recorte = imagen[y1:y2, x1:x2].copy()
    nombre_salida = f"roi_{nombre_base}{sufijo}"
    np.save(carpeta_roi / f"{nombre_salida}.npy", recorte)
    ski.io.imsave(
        carpeta_roi / f"{nombre_salida}.tif", ski.util.img_as_ubyte(recorte), check_contrast=False
    )

    info = {
        "archivo_origen": Path(nombre_archivo).name,
        "coordenadas": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "shape_roi": list(recorte.shape),
        "shape_imagen": [alto, ancho],
        "metadata": {k: v for k, v in (metadata or {}).items() if k != "crudo"},
    }
    with open(carpeta_roi / f"{nombre_salida}.json", "w", encoding="utf-8") as archivo:
        json.dump(info, archivo, indent=2, ensure_ascii=False)

    return carpeta_roi


def _agregar_barra_escala(ax: plt.Axes, ancho_px: int, tamano_pixel_m: float) -> None:
    """Dibuja una barra de escala de longitud 'redonda' (1, 2 o 5 × 10^n µm) de ~20% del ancho."""
    objetivo_um = 0.2 * ancho_px * tamano_pixel_m * 1e6
    potencia = 10 ** np.floor(np.log10(objetivo_um))
    longitud_um = max(m * potencia for m in (1, 2, 5) if m * potencia <= objetivo_um)
    etiqueta = f"{longitud_um:g} µm" if longitud_um >= 1 else f"{longitud_um * 1000:g} nm"
    barra = AnchoredSizeBar(
        ax.transData,
        longitud_um * 1e-6 / tamano_pixel_m,
        etiqueta,
        loc="lower right",
        color="white",
        frameon=False,
        size_vertical=max(1, ancho_px // 200),
        pad=0.6,
    )
    ax.add_artist(barra)


def _formatear_metadata(metadata: dict) -> str:
    """Arma una línea de texto con los campos normalizados de la metadata que estén presentes."""
    partes = []
    if "fabricante" in metadata:
        partes.append(metadata["fabricante"])
    if "detector" in metadata:
        partes.append(f"Detector: {metadata['detector']}")
    if "voltaje_kV" in metadata:
        partes.append(f"{metadata['voltaje_kV']:g} kV")
    if "distancia_trabajo_mm" in metadata:
        partes.append(f"WD: {metadata['distancia_trabajo_mm']:.1f} mm")
    if "magnificacion" in metadata:
        partes.append(f"Mag: {metadata['magnificacion']:,.0f}×")
    if "tamano_pixel_m" in metadata:
        partes.append(f"Píxel: {metadata['tamano_pixel_m'] * 1e9:.3g} nm")
    if "fecha_adquisicion" in metadata:
        partes.append(metadata["fecha_adquisicion"])
    return "  |  ".join(partes)


def _detectar_barra_zeiss(
    imagen: np.ndarray, fraccion_busqueda: float = 0.2, umbral_blanco: float = 0.98
) -> int | None:
    """
    Detecta la fila donde empieza la barra de datos Zeiss (fondo blanco, texto negro).

    El borde superior de la barra es una fila completamente blanca justo debajo de una
    fila de imagen (mayormente no blanca). Las filas de texto dentro de la barra siguen
    siendo mayormente blancas, así que no generan falsos bordes. Solo se busca en la
    `fraccion_busqueda` inferior de la imagen.

    Returns
    -------
    int | None
        Índice de la primera fila de la barra (= alto útil), o None si no se encuentra.
    """
    fraccion_blanca = (imagen >= umbral_blanco).mean(axis=1)
    inicio_busqueda = max(1, int(imagen.shape[0] * (1 - fraccion_busqueda)))
    for fila in range(inicio_busqueda, imagen.shape[0]):
        if fraccion_blanca[fila] >= 0.99 and fraccion_blanca[fila - 1] < 0.5:
            return fila
    return None


def _leer_metadata_tiff(ruta: Path) -> dict:
    """Lee metadata FEI o Zeiss con tifffile. Retorna {} si no es TIFF o no trae esos tags."""
    try:
        with tifffile.TiffFile(ruta) as tif:
            pagina = tif.pages[0]
            ancho_px, alto_px = int(pagina.imagewidth), int(pagina.imagelength)
            if tif.is_fei and tif.fei_metadata:
                return _normalizar_fei(tif.fei_metadata, ancho_px, alto_px)
            if tif.is_sem and tif.sem_metadata:
                return _normalizar_zeiss(tif.sem_metadata, ancho_px, alto_px)
    except tifffile.TiffFileError:
        pass  # no es un TIFF: se sigue con el respaldo EXIF sin avisar
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"⚠️ No se pudo leer la metadata TIFF de {ruta.name}: {error}")
    return {}


def _normalizar_fei(crudo: dict, ancho_px: int, alto_px: int) -> dict:
    """Extrae los campos normalizados de la metadata INI de FEI/Thermo (unidades SI en origen)."""
    haz = crudo.get("EBeam") or crudo.get("Beam") or {}
    barrido = crudo.get("Scan") or crudo.get("EScan") or {}
    imagen = crudo.get("Image", {})
    usuario = crudo.get("User", {})

    metadata = {"fabricante": "FEI", "ancho_px": ancho_px, "alto_px": alto_px}
    if "ResolutionY" in imagen:
        metadata["alto_util_px"] = int(imagen["ResolutionY"])
    if barrido.get("PixelWidth"):
        metadata["tamano_pixel_m"] = float(barrido["PixelWidth"])
    if haz.get("HFW"):
        metadata["ancho_campo_um"] = float(haz["HFW"]) * 1e6
        # FEI no guarda la magnificación; se calcula respecto del ancho de pantalla de referencia.
        if imagen.get("MagCanvasRealWidth"):
            metadata["magnificacion"] = float(imagen["MagCanvasRealWidth"]) / float(haz["HFW"])
    if haz.get("HV"):
        metadata["voltaje_kV"] = float(haz["HV"]) / 1000
    if haz.get("WD"):
        metadata["distancia_trabajo_mm"] = float(haz["WD"]) * 1000
    if crudo.get("Detectors", {}).get("Name"):
        metadata["detector"] = str(crudo["Detectors"]["Name"])
    fecha = _fecha_iso(f"{usuario.get('Date', '')} {usuario.get('Time', '')}", "%m/%d/%Y %I:%M:%S %p")
    if fecha:
        metadata["fecha_adquisicion"] = fecha
    metadata["crudo"] = crudo
    return metadata


def _normalizar_zeiss(crudo: dict, ancho_px: int, alto_px: int) -> dict:
    """
    Extrae los campos normalizados de la metadata Zeiss SmartSEM.

    Cada entrada es una tupla (descripción, valor[, unidad]). Se usa 'ap_image_pixel_size'
    (tamaño de píxel de la imagen guardada) y no 'ap_pixel_size', que está referido a un
    barrido de 1024 px de ancho.
    """
    metadata = {"fabricante": "Zeiss", "ancho_px": ancho_px, "alto_px": alto_px}

    tamano_pixel = _valor_en_metros(crudo.get("ap_image_pixel_size"))
    ancho_campo = _valor_en_metros(crudo.get("ap_width"))
    if tamano_pixel is None and ancho_campo is not None:
        tamano_pixel = ancho_campo / ancho_px
    if tamano_pixel is not None:
        metadata["tamano_pixel_m"] = tamano_pixel
    if ancho_campo is not None:
        metadata["ancho_campo_um"] = ancho_campo * 1e6

    magnificacion = _magnificacion_zeiss(crudo.get("ap_mag"))
    if magnificacion is not None:
        metadata["magnificacion"] = magnificacion
    voltaje = _valor_numerico(crudo.get("ap_actualkv"))
    if voltaje is not None:
        metadata["voltaje_kV"] = voltaje
    distancia_trabajo = _valor_en_metros(crudo.get("ap_wd"))
    if distancia_trabajo is not None:
        metadata["distancia_trabajo_mm"] = distancia_trabajo * 1000
    detector = crudo.get("dp_detector_type")
    if detector and len(detector) > 1:
        metadata["detector"] = str(detector[1])

    fecha, hora = crudo.get("ap_date"), crudo.get("ap_time")
    if fecha and hora:
        fecha_iso = _fecha_iso(f"{fecha[1]} {hora[1]}", "%d %b %Y %H:%M:%S")
        if fecha_iso:
            metadata["fecha_adquisicion"] = fecha_iso
    metadata["crudo"] = crudo
    return metadata


def _leer_metadata_exif(ruta: Path) -> dict:
    """Respaldo: EXIF/TIFF estándar con exifread. Retorna {} si no hay tags."""
    try:
        with open(ruta, "rb") as archivo:
            tags = exifread.process_file(archivo, details=False)
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"⚠️ No se pudo leer EXIF de {ruta.name}: {error}")
        return {}
    if not tags:
        return {}

    metadata = {"fabricante": "EXIF"}
    for tag, clave in (("Image ImageWidth", "ancho_px"), ("Image ImageLength", "alto_px")):
        if tag in tags:
            metadata[clave] = int(tags[tag].values[0])
    if "EXIF ExposureTime" in tags:
        metadata["tiempo_exposicion"] = str(tags["EXIF ExposureTime"])
    fecha = _fecha_iso(str(tags.get("Image DateTime", "")), "%Y:%m:%d %H:%M:%S")
    if fecha:
        metadata["fecha_adquisicion"] = fecha
    metadata["crudo"] = {clave: str(valor) for clave, valor in tags.items()}
    return metadata


def _valor_numerico(entrada) -> float | None:
    """Devuelve el valor numérico de una tupla Zeiss (descripción, valor, unidad), o None."""
    if not entrada or len(entrada) < 2:
        return None
    try:
        return float(entrada[1])
    except (TypeError, ValueError):
        return None


def _valor_en_metros(entrada) -> float | None:
    """Convierte una tupla Zeiss (descripción, valor, unidad) de longitud a metros, o None."""
    valor = _valor_numerico(entrada)
    if valor is None or len(entrada) < 3:
        return None
    unidad = str(entrada[2]).strip()
    # tifffile puede decodificar 'µ' de forma incorrecta: una unidad de 2 letras terminada
    # en 'm' que no sea nm/pm/mm se interpreta como micrómetros.
    if unidad not in _A_METROS and len(unidad) == 2 and unidad.endswith("m"):
        unidad = "um"
    factor = _A_METROS.get(unidad)
    return valor * factor if factor is not None else None


def _magnificacion_zeiss(entrada) -> float | None:
    """Convierte una magnificación Zeiss tipo '2.02 K X' o '500 X' a número."""
    if not entrada or len(entrada) < 2:
        return None
    coincidencia = re.match(r"\s*([\d.]+)\s*([KM]?)\s*X", str(entrada[1]), re.IGNORECASE)
    if not coincidencia:
        return None
    multiplicador = {"": 1, "K": 1e3, "M": 1e6}[coincidencia.group(2).upper()]
    return float(coincidencia.group(1)) * multiplicador


def _fecha_iso(texto: str, formato: str) -> str | None:
    """Convierte una fecha según `formato` a 'AAAA-MM-DD HH:MM:SS'; None si no se puede."""
    try:
        return datetime.strptime(texto.strip(), formato).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
