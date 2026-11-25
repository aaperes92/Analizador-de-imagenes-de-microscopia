"""Carga y preprocesamiento de imágenes de microscopía TEM.

Etapa 1 del módulo TEM del proyecto "Analizador de Imágenes de
Microscopía".

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import color, filters, io, morphology, restoration

# Formatos soportados, agrupados según la librería que los lee
_SKIMAGE_FORMATS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".jpg", ".png"}
)
_DM3_FORMATS: frozenset[str] = frozenset({".dm3"})
_MRC_FORMATS: frozenset[str] = frozenset({".mrc"})
SUPPORTED_FORMATS: frozenset[str] = (
    _SKIMAGE_FORMATS | _DM3_FORMATS | _MRC_FORMATS
)

# Carpeta por defecto para las figuras: <raíz del proyecto>/data/samples
# (este archivo está en <raíz>/src/tem/loader.py)
DEFAULT_OUTPUT_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "tem"
)

# Percentiles (inferior, superior) que se llevan a 0 y 1 al normalizar
NORM_PERCENTILES: tuple[float, float] = (0.1, 99.9)

# Métodos de preprocesamiento disponibles
PREPROCESS_METHODS: tuple[str, ...] = (
    "gaussian", "median", "bilateral", "wiener", "auto",
)

# Parámetros por defecto de los filtros (imágenes normalizadas a [0, 1])
GAUSSIAN_SIGMA: float = 1.0
MEDIAN_RADIUS: int = 2
BILATERAL_SIGMA_COLOR: float = 0.05
BILATERAL_SIGMA_SPATIAL: float = 2.0
WIENER_WINDOW: int = 5

# Umbrales del modo 'auto', calibrados con imágenes sintéticas
# (partículas + ruido). Conviene ajustarlos con imágenes TEM reales.
# Fracción de píxeles impulsivos (extremos y distintos de su entorno)
AUTO_IMPULSE_FRACTION: float = 0.002
# Dispersión robusta de los gradientes Sobel (MAD escalada). Los bordes
# ocupan pocos píxeles y casi no la afectan: mide sobre todo el ruido.
# Una imagen sin ruido da ~0 y ruido puro da ~0.13. En 13 imágenes
# reales de campo claro HR dio entre 0.010 y 0.052.
AUTO_GRAD_LOW: float = 0.04  # por debajo: ruido bajo
AUTO_GRAD_HIGH: float = 0.09  # por encima: ruido alto
# Factor que convierte la MAD en un desvío estándar (ruido gaussiano)
_MAD_TO_STD: float = 1.4826


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normaliza un array al rango [0, 1] usando percentiles.

    Toma como 0 y 1 los percentiles NORM_PERCENTILES y recorta lo que
    queda afuera. Así unos pocos píxeles extremos (píxeles calientes,
    bordes de la rejilla) no comprimen el contraste del resto.

    Args:
        image: Array numérico de cualquier dtype.

    Returns:
        Array en float32 con valores entre 0 y 1. Si la imagen es
        constante, devuelve un array de ceros con el mismo shape.
    """
    data = image.astype(np.float64)
    vmin, vmax = (float(v) for v in np.percentile(data, NORM_PERCENTILES))
    # Si los percentiles coinciden (imagen casi constante), se usa el
    # mínimo y el máximo
    if vmax == vmin:
        vmin, vmax = float(data.min()), float(data.max())
    # Imagen constante: se evita la división por cero
    if vmax == vmin:
        return np.zeros(image.shape, dtype=np.float32)
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _read_raw(path: Path, ext: str) -> np.ndarray:
    """Lee el archivo con la librería que corresponde a su formato.

    Args:
        path: Ruta al archivo de imagen.
        ext: Extensión en minúsculas (por ejemplo, ".dm3").

    Returns:
        Array con los datos tal como vienen en el archivo.
    """
    if ext in _SKIMAGE_FORMATS:
        return np.asarray(io.imread(path))
    if ext in _DM3_FORMATS:
        # Import diferido: ncempy solo se necesita para .dm3
        from ncempy.io import dm

        return np.asarray(dm.dmReader(str(path))["data"])
    # Formato .mrc; permissive tolera encabezados irregulares
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        return np.array(mrc.data)


def _to_2d(data: np.ndarray, ext: str) -> np.ndarray:
    """Reduce la imagen a 2D: escala de grises y un solo cuadro.

    Args:
        data: Array leído del archivo (2D, color o pila de cuadros).
        ext: Extensión en minúsculas, para distinguir color de pila.

    Returns:
        Array 2D.

    Raises:
        ValueError: Si la imagen no se puede interpretar como 2D.
    """
    # Imágenes en color (RGB/RGBA) de formatos raster
    if (
        ext in _SKIMAGE_FORMATS
        and data.ndim == 3
        and data.shape[-1] in (3, 4)
    ):
        if data.shape[-1] == 4:
            data = color.rgba2rgb(data)
        return color.rgb2gray(data)
    # Pila de cuadros (por ejemplo, series .mrc o .dm3)
    while data.ndim > 2:
        print(
            f"  Aviso: la imagen tiene {data.shape[0]} cuadros; "
            "se usa el primero."
        )
        data = data[0]
    if data.ndim != 2:
        raise ValueError(
            f"No se pudo interpretar la imagen como 2D "
            f"(shape {data.shape})."
        )
    return data


def load_tem_image(filepath: str | Path) -> np.ndarray:
    """Carga una imagen TEM y la normaliza al rango [0, 1].

    Formatos soportados: .tif, .tiff, .jpg y .png (con scikit-image),
    .dm3 (con ncempy) y .mrc (con mrcfile). Las imágenes en color se
    pasan a escala de grises y de las pilas se toma el primer cuadro.

    Args:
        filepath: Ruta al archivo de imagen.

    Returns:
        Array 2D en float32 con valores entre 0 y 1.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no está soportado o la imagen no
            se puede interpretar como 2D.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        soportados = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Formato '{ext or '(sin extensión)'}' no soportado. "
            f"Formatos válidos: {soportados}."
        )

    image = _normalize(_to_2d(_read_raw(path, ext), ext))

    print(f"Archivo: {path.name}")
    print(f"  Formato detectado: {ext}")
    print(f"  Shape: {image.shape} | dtype: {image.dtype}")
    return image


def _select_method(image: np.ndarray) -> tuple[str, str]:
    """Elige el filtro según el nivel y el tipo de ruido de la imagen.

    Primero busca ruido impulsivo (sal y pimienta): píxeles en los
    extremos del rango que difieren mucho de la mediana de su entorno.
    Si no lo hay, mide el ruido con la dispersión robusta (MAD) de los
    gradientes Sobel horizontal y vertical. A diferencia de la
    varianza, la MAD casi no se ve afectada por los bordes de las
    partículas, que son pocos píxeles, y refleja sobre todo el ruido.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Tupla (método, justificación).
    """
    # Ruido impulsivo: píxel extremo y muy distinto de sus vecinos
    data = image.astype(np.float64)
    local_median = filters.median(data, footprint=morphology.disk(1))
    is_extreme = (data <= 0.02) | (data >= 0.98)
    is_outlier = np.abs(data - local_median) > 0.3
    impulse = float(np.mean(is_extreme & is_outlier))
    if impulse > AUTO_IMPULSE_FRACTION:
        return "median", (
            f"{impulse:.2%} de píxeles impulsivos (> "
            f"{AUTO_IMPULSE_FRACTION:.1%}): ruido sal y pimienta, la "
            "mediana lo elimina sin difuminar bordes."
        )

    # Dispersión robusta de los gradientes: crece con el ruido
    grads = np.concatenate(
        [filters.sobel_h(data).ravel(), filters.sobel_v(data).ravel()]
    )
    noise = _MAD_TO_STD * float(
        np.median(np.abs(grads - np.median(grads)))
    )
    if noise < AUTO_GRAD_LOW:
        return "gaussian", (
            f"dispersión de los gradientes {noise:.4f} < "
            f"{AUTO_GRAD_LOW}: ruido bajo, alcanza con un suavizado "
            "gaussiano leve."
        )
    if noise <= AUTO_GRAD_HIGH:
        return "bilateral", (
            f"dispersión de los gradientes {noise:.4f} entre "
            f"{AUTO_GRAD_LOW} y {AUTO_GRAD_HIGH}: ruido moderado, el "
            "bilateral suaviza conservando los bordes."
        )
    return "wiener", (
        f"dispersión de los gradientes {noise:.4f} > "
        f"{AUTO_GRAD_HIGH}: ruido alto, el filtro de Wiener se adapta "
        "al ruido local."
    )


def _apply_filter(image: np.ndarray, method: str) -> np.ndarray:
    """Aplica uno de los filtros con sus parámetros por defecto.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        method: 'gaussian', 'median', 'bilateral' o 'wiener'.

    Returns:
        Imagen filtrada (float64, sin recortar).
    """
    data = image.astype(np.float64)
    if method == "gaussian":
        return filters.gaussian(data, sigma=GAUSSIAN_SIGMA)
    if method == "median":
        return filters.median(
            data, footprint=morphology.disk(MEDIAN_RADIUS)
        )
    if method == "bilateral":
        return restoration.denoise_bilateral(
            data,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_spatial=BILATERAL_SIGMA_SPATIAL,
        )
    # Wiener: en zonas sin varianza local puede dar NaN; se reemplazan
    # por el valor original
    with np.errstate(divide="ignore", invalid="ignore"):
        result = signal.wiener(data, mysize=WIENER_WINDOW)
    return np.where(np.isfinite(result), result, data)


def preprocess_tem_image(
    image: np.ndarray, method: str = "auto"
) -> np.ndarray:
    """Reduce el ruido de una imagen TEM con el filtro indicado.

    Filtros disponibles y sus parámetros por defecto:
    - 'gaussian': suavizado gaussiano, sigma = 1.0.
    - 'median': mediana con un disco de radio 2.
    - 'bilateral': sigma_color = 0.05, sigma_spatial = 2.
    - 'wiener': filtro adaptativo con ventana de 5x5.
    - 'auto': elige uno de los anteriores según el ruido medido e
      imprime la justificación.

    Args:
        image: Imagen 2D normalizada a [0, 1], como la devuelve
            load_tem_image.
        method: Nombre del filtro o 'auto'.

    Returns:
        Imagen filtrada en float32, recortada a [0, 1], con el mismo
        shape que la de entrada.

    Raises:
        ValueError: Si el método no existe o la imagen no es 2D.
    """
    if method not in PREPROCESS_METHODS:
        raise ValueError(
            f"Método '{method}' no válido. Opciones: "
            f"{', '.join(PREPROCESS_METHODS)}."
        )
    if image.ndim != 2:
        raise ValueError(
            f"Se esperaba una imagen 2D y llegó shape {image.shape}."
        )

    if method == "auto":
        method, reason = _select_method(image)
        print(f"Modo auto: se elige '{method}' porque {reason}")

    processed = _apply_filter(image, method)
    return np.clip(processed, 0.0, 1.0).astype(np.float32)


def show_preprocessing_results(
    original: np.ndarray,
    processed: np.ndarray,
    filepath: str | Path,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la comparación antes y después del filtrado.

    La figura tiene 4 paneles: imagen original, imagen procesada,
    histograma de la original (azul) e histograma de la procesada
    (rojo). Se guarda como '<nombre>_preprocessing.png'.

    Args:
        original: Imagen original normalizada a [0, 1].
        processed: Imagen preprocesada normalizada a [0, 1].
        filepath: Ruta del archivo de origen; se usa para el título y
            el nombre de la figura.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_preprocessing.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Preprocesamiento TEM: {path.name}", fontsize=14)

    # Fila superior: imágenes en escala de grises, misma escala [0, 1]
    for ax, img, title in (
        (axes[0, 0], original, "Imagen original"),
        (axes[0, 1], processed, "Imagen procesada"),
    ):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

    # Fila inferior: histogramas de intensidad
    for ax, img, title, col in (
        (axes[1, 0], original, "Histograma de la original", "blue"),
        (axes[1, 1], processed, "Histograma de la procesada", "red"),
    ):
        ax.hist(img.ravel(), bins=256, range=(0.0, 1.0), color=col)
        ax.set_title(title)
        ax.set_xlabel("Intensidad normalizada")
        ax.set_ylabel("Cantidad de píxeles")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    # Se cierra para no acumular figuras abiertas en memoria
    plt.close(fig)
    return out_path


def pipeline_tem(
    filepath: str | Path,
    method: str = "auto",
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, np.ndarray | str | Path]:
    """Carga, preprocesa y grafica una imagen TEM en un solo paso.

    Encadena load_tem_image, preprocess_tem_image y
    show_preprocessing_results.

    Args:
        filepath: Ruta al archivo de imagen.
        method: Filtro a aplicar o 'auto' (ver preprocess_tem_image).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Diccionario con las claves 'original' (array), 'processed'
        (array), 'filepath' (la ruta recibida) y 'method_used' (el
        filtro aplicado; en modo 'auto', el que se eligió).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato o el método no son válidos.
    """
    if method not in PREPROCESS_METHODS:
        raise ValueError(
            f"Método '{method}' no válido. Opciones: "
            f"{', '.join(PREPROCESS_METHODS)}."
        )

    original = load_tem_image(filepath)

    # En modo 'auto' se elige acá el filtro para poder informarlo
    method_used = method
    if method == "auto":
        method_used, reason = _select_method(original)
        print(f"Modo auto: se elige '{method_used}' porque {reason}")

    processed = preprocess_tem_image(original, method=method_used)
    show_preprocessing_results(
        original, processed, filepath, output_dir=output_dir, show=show
    )
    return {
        "original": original,
        "processed": processed,
        "filepath": filepath,
        "method_used": method_used,
    }
