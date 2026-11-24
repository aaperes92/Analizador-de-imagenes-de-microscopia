"""Análisis de patrones de difracción de electrones.

Etapa 4.5 del módulo TEM del proyecto "Analizador de Imágenes de
Microscopía". Se activa solo cuando la imagen es un patrón de
difracción (SAED con anillos o puntos, NBED o un perfil 1D) y no una
imagen de campo claro u oscuro.

Convenciones:
    - Las coordenadas son (fila, columna) = (y, x), en píxeles de la
      imagen completa.
    - La calibración se expresa como 1/Å por píxel: el espaciado de un
      anillo de radio r (en píxeles) es d = 1 / (r · calibración).

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import optimize, signal
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max
from skimage.measure import block_reduce

# ====================================================================
# Constantes
# ====================================================================

# --- Detección del tipo de imagen (is_diffraction_pattern) ---
# Las heurísticas trabajan sobre una copia reducida a ~512 px de lado
ANALYSIS_SIZE: int = 512
# Ganancia de la vista logarítmica log(1 + g·I) con I en [0, 1]
LOG_GAIN: float = 100.0
# Fracción de píxeles más oscuros que no se usan (beam stopper y zonas
# sin señal)
DARK_EXCLUDE_PCT: float = 10.0
# Erosión (px) de la máscara de señal: descarta el borde del stopper
MASK_EROSION: int = 3
# Suavizado para ubicar el halo central (fracción del lado menor)
HALO_SIGMA_FRACTION: float = 0.04
# Ventana de búsqueda del centro de simetría alrededor del halo
SYMMETRY_SEARCH_FRACTION: float = 0.12
# Superposición mínima imagen/rotación para aceptar un centro
SYMMETRY_MIN_OVERLAP: float = 0.1
# Votos: simetría puntual (correlación con la rotación de 180°),
# decaimiento radial y uniformidad azimutal
SYMMETRY_MIN: float = 0.5
DECAY_MIN: float = 0.5
AZIMUTHAL_IQR_MAX: float = 0.26
# El centro del patrón no puede estar pegado al borde (fracción del lado)
CENTER_BORDER_MIN: float = 0.15
# Perfil radial para los votos 2 y 3: de 3 % a 50 % del lado menor
RADIAL_RANGE: tuple[float, float] = (0.03, 0.5)
RADIAL_BINS: int = 30
# Un tramo del perfil "baja" si cae más de 0.5 % del valor inicial
DECAY_STEP: float = 0.005

# --- Tipo de patrón (detect_pattern_type) ---
TYPE_SIZE: int = 1024
# Relación de aspecto a partir de la cual la imagen es un perfil 1D
PROFILE_ASPECT: float = 3.0
# En un perfil 1D, varianza transversal máxima relativa a la longitudinal
PROFILE_CROSS_VAR: float = 0.1
# Anillos: picos del perfil radial con prominencia > RING_PROMINENCE·ruido
RING_PROMINENCE: float = 4.0
RING_HALF_WIDTH: int = 3
AZIMUTHAL_SECTORS: int = 36
# Un sector "ve" el anillo si supera al entorno por RING_SECTOR_SIGMA·error
RING_SECTOR_SIGMA: float = 3.0
# Anillo continuo: cubre >= 75 % de la circunferencia; o dos que cubren
# >= 50 %
RING_STRONG_COVERAGE: float = 0.75
RING_COVERAGE: float = 0.5
# Spots: máximos del residuo (log - fondo suave) por encima de
# SPOT_SIGMA desvíos del ruido
SPOT_SIGMA: float = 12.0
SPOT_MIN_DISTANCE: int = 3
MIN_SPOTS: int = 10
# NBED: spots anchos (FWHM medio > 5 % de su distancia al centro)
NBED_WIDTH_FRACTION: float = 0.05
# Escala del fondo (halo) que se resta para medir el ancho de los spots
# (fracción del lado menor)
SPOT_BACKGROUND: float = 0.05
# Radio mínimo de análisis (fracción del lado menor): excluye el haz
# directo
INNER_RADIUS_FRACTION: float = 0.03
# Radio mínimo para refinar el centro por nitidez de anillos: más cerca
# quedan restos del halo y de la punta del beam stopper
SHARPNESS_INNER_FRACTION: float = 0.05

# --- Centro del haz (find_beam_center) ---
# El centro por nitidez de anillos se acepta si no empeora la
# uniformidad azimutal del halo más de este factor (sin anillos, la
# nitidez puede llevarlo lejos)
CENTER_UNIFORMITY_TOL: float = 1.2
# Centro de masa y gaussiana solo ganan si dejan el halo claramente más
# uniforme (con beam stopper fallan por 200 px)
CENTER_ALTERNATIVE_GAIN: float = 0.8
# Métodos "aceptables": uniformidad dentro de este factor de la mejor
CENTER_ACCEPTABLE: float = 1.1
# Dispersión (px) entre métodos aceptables a partir de la cual se avisa
CENTER_SPREAD_WARN: float = 5.0
# Umbral del centro de masa (percentil) y recorte de la gaussiana
CENTER_MASS_PERCENTILE: float = 95.0
CENTER_GAUSS_CROP: float = 0.2

# --- Beam stopper y preprocesamiento ---
# Radio del haz directo que se enmascara (fracción del lado menor)
BEAM_MASK_FRACTION: float = 0.02
# Un píxel es del stopper si está STOPPER_SIGMA desvíos (MAD) por debajo
# de la mediana de su anillo y al menos STOPPER_MIN_DROP del contraste
STOPPER_SIGMA: float = 5.0
STOPPER_MIN_DROP: float = 0.1
# La zona oscura tiene que llegar a STOPPER_REACH del lado desde el
# centro y ocupar al menos STOPPER_MIN_AREA de la imagen
STOPPER_REACH: float = 0.15
STOPPER_MIN_AREA: float = 0.001
# Otras zonas oscuras cerca del centro (la punta) se suman si miden al
# menos esta fracción de la más grande
STOPPER_PIECE_FRACTION: float = 0.05
STOPPER_DILATION: int = 2
# Suavizado (px de la copia de ~1024 px) antes de buscar el stopper
STOPPER_SMOOTH: float = 6.0
# Suavizado del preprocesamiento (reduce ruido sin perder spots)
PREPROCESS_SIGMA: float = 0.5

# --- Anillos (Bloque 3) ---
# Suavizado del perfil radial (px)
PROFILE_SMOOTH: float = 1.5
# Radio mínimo del perfil (distancia del centro al borde), fracción del
# lado menor
PROFILE_MIN_RADIUS: float = 0.1
# Ventana del fondo del perfil (apertura morfológica): fracción de r_max,
# con un mínimo en píxeles
RING_BACKGROUND_FRACTION: float = 0.08
RING_BACKGROUND_MIN_PX: int = 41
# Base local de cada anillo: puntos a más de 1.5 FWHM del pico (al menos
# 4 de cada lado) para ajustar una cuadrática
RING_BASE_EXCLUDE: float = 1.5
RING_BASE_MIN_POINTS: int = 4
# Prominencia mínima de un anillo, en veces el ruido del perfil
RING_SIGNIFICANCE: float = 8.0
# Distancia mínima entre anillos (fracción de r_max)
RING_MIN_DISTANCE: float = 0.03
# Scherrer: factor de forma (esferas) y rango de validez en nm
SCHERRER_K: float = 0.94
SCHERRER_RANGE_NM: tuple[float, float] = (2.0, 100.0)

# --- Spots (Bloque 4) ---
# Fondo (halo) que se resta antes de buscar spots (sigma en px)
SPOT_BG_SIGMA: float = 8.0
# Suavizado del tamaño de un spot (filtro adaptado, sigma en px)
SPOT_SMOOTH: float = 2.0
# Umbral: veces el ruido local (medido en bloques de SPOT_NOISE_BLOCK px)
SPOT_DETECT_SIGMA: float = 6.0
SPOT_NOISE_BLOCK: int = 64
# Distancia mínima entre spots (px) y exclusión del haz directo
# (fracción del lado menor)
SPOT_DETECT_DISTANCE: int = 5
SPOT_CENTER_EXCLUDE: float = 0.03
# Ventana (px) para la intensidad máxima y el centroide de cada spot
SPOT_WINDOW: int = 5
# Par de Friedel: tolerancia en px; índice de simetría mínimo
SPOT_PAIR_TOL: float = 3.0
SPOT_SYMMETRY_MIN: float = 0.7
# Geometría: tolerancia relativa para agrupar spots por radio, ancho de
# los bins del histograma de ángulos y tolerancia para indexar un spot
# con la red (fracción de |a*|, con un mínimo en px)
FAMILY_TOLERANCE: float = 0.05
ANGLE_BIN_DEG: float = 2.0
LATTICE_TOLERANCE: float = 0.1
LATTICE_TOLERANCE_MIN_PX: float = 3.0
# Iteraciones del ajuste de la red por mínimos cuadrados
LATTICE_REFINE_STEPS: int = 3

# --- Perfil 1D (Bloque 5) ---
# Perfil de dos lados: simetría espejo con correlación >= 0.9 y al menos
# 30 % del perfil a cada lado del origen
PROFILE_SYMMETRY_MIN: float = 0.9
PROFILE_SYMMETRY_OVERLAP: float = 0.3
# Suavizado de visualización (Savitzky-Golay)
PROFILE_SAVGOL_WINDOW: int = 11
PROFILE_SAVGOL_ORDER: int = 3

# --- Figura (Bloque 6) ---
# Filas máximas en las tablas de la figura
MAX_TABLE_ROWS: int = 14

# --- Metadatos (read_camera_metadata) ---
DEFAULT_VOLTAGE_KV: float = 200.0
DEFAULT_CAMERA_LENGTH_MM: float = 1000.0
DEFAULT_DETECTOR_PIXEL_UM: float = 14.0
# Constantes de la longitud de onda relativista de los electrones:
# λ[Å] = 12.2643 / sqrt(V · (1 + 0.97848e-6 · V)), con V en voltios
_WAVELENGTH_H: float = 12.2643
_WAVELENGTH_REL: float = 0.97848e-6

_MAD_TO_STD: float = 1.4826


# ====================================================================
# Utilidades internas
# ====================================================================


def _downsample(image: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    """Reduce la imagen promediando bloques hasta ~size px de lado.

    Args:
        image: Imagen 2D.
        size: Lado máximo deseado.

    Returns:
        (imagen reducida en float64, factor de reducción).
    """
    factor = max(1, int(np.ceil(max(image.shape) / size)))
    data = image.astype(np.float64)
    if factor > 1:
        data = block_reduce(data, (factor, factor), np.mean)
    return data, factor


def _to_full(
    point: tuple[float, float], factor: int
) -> tuple[float, float]:
    """Pasa un punto de la imagen reducida a la imagen completa.

    El píxel i de la copia reducida promedia los píxeles i·f a
    i·f + f - 1, así que su centro está en i·f + (f - 1)/2.
    """
    off = (factor - 1) / 2.0
    return (float(point[0]) * factor + off, float(point[1]) * factor + off)


def _to_small(
    point: tuple[float, float], factor: int
) -> tuple[float, float]:
    """Pasa un punto de la imagen completa a la reducida (inversa de
    _to_full)."""
    off = (factor - 1) / 2.0
    return ((float(point[0]) - off) / factor,
            (float(point[1]) - off) / factor)


def _log_view(image: np.ndarray) -> np.ndarray:
    """Vista logarítmica log(1 + g·I) que comprime los spots intensos."""
    return np.log1p(LOG_GAIN * np.clip(image, 0.0, None))


def _signal_mask(log_image: np.ndarray) -> np.ndarray:
    """Píxeles con señal: excluye el DARK_EXCLUDE_PCT % más oscuro.

    Así quedan afuera el beam stopper y las zonas sin electrones, que
    romperían la simetría del patrón.
    """
    smooth = ndi.gaussian_filter(log_image, 4)
    return smooth > np.percentile(smooth, DARK_EXCLUDE_PCT)


def _masked_smooth(
    image: np.ndarray, mask: np.ndarray, sigma: float
) -> np.ndarray:
    """Suavizado gaussiano que solo promedia píxeles válidos.

    Convolución normalizada: G*(I·m) / G*m. Evita que el beam stopper
    "se derrame" sobre los píxeles vecinos al restar el fondo.
    """
    m = mask.astype(np.float64)
    num = ndi.gaussian_filter(image * m, sigma)
    den = ndi.gaussian_filter(m, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def _detail(
    log_image: np.ndarray, mask: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Detalle fino (imagen menos fondo suave) y máscara erosionada.

    La máscara se erosiona MASK_EROSION px para descartar el borde del
    beam stopper, donde la resta del fondo deja artefactos.

    Returns:
        (detalle, máscara erosionada).
    """
    inner = ndi.binary_erosion(mask, iterations=MASK_EROSION)
    detail = (log_image - _masked_smooth(log_image, mask, sigma)) * inner
    return detail, inner


def _is_1d_profile(image: np.ndarray) -> bool:
    """True si la imagen es un perfil 1D (o una franja que varía en un
    solo eje)."""
    if image.ndim == 1 or min(image.shape) <= 3:
        return True
    h, w = image.shape
    if max(h, w) / min(h, w) < PROFILE_ASPECT:
        return False
    # Eje largo: el perfil; eje corto: debería ser casi constante
    axis_long = 1 if w >= h else 0
    along = image.mean(axis=1 - axis_long)
    across = image.mean(axis=axis_long)
    var_along = float(np.var(along))
    return var_along > 0 and np.var(across) <= PROFILE_CROSS_VAR * var_along


def _symmetry_center(
    log_image: np.ndarray,
    mask: np.ndarray,
    start: tuple[float, float],
    window: float,
) -> tuple[float, tuple[float, float]]:
    """Busca el centro de simetría puntual cerca de start.

    Para cada centro candidato t calcula la correlación entre la imagen
    y su rotación de 180° alrededor de t. Con FFT se obtienen todos los
    centros de una vez: la convolución de la imagen consigo misma en
    2t es la suma de I(x)·I(2t - x).

    Args:
        log_image: Imagen (ya suavizada) en escala logarítmica.
        mask: Píxeles válidos.
        start: Centro inicial (y, x).
        window: Semiancho de la ventana de búsqueda en píxeles.

    Returns:
        (correlación máxima, centro (y, x)). Correlación -1 si no hay
        centros válidos.
    """
    m = mask.astype(np.float64)
    z = (log_image - log_image[mask].mean()) * m
    num = fftconvolve(z, z)
    energy = fftconvolve(z**2, m)
    overlap = fftconvolve(m, m)
    ty, tx = np.mgrid[0:num.shape[0], 0:num.shape[1]] / 2.0
    zone = (
        (np.abs(ty - start[0]) <= window)
        & (np.abs(tx - start[1]) <= window)
        & (overlap > SYMMETRY_MIN_OVERLAP * m.sum())
        & (energy > 1e-12)
    )
    if not zone.any():
        return -1.0, (float(start[0]), float(start[1]))
    corr = np.where(zone, num / np.maximum(energy, 1e-12), -1.0)
    k = np.unravel_index(np.argmax(corr), corr.shape)
    return float(corr[k]), (k[0] / 2.0, k[1] / 2.0)


def _coarse_center(
    small: np.ndarray,
) -> tuple[float, tuple[float, float], np.ndarray, np.ndarray]:
    """Centro aproximado del patrón en la imagen reducida.

    Primero ubica el halo (máximo de la imagen muy suavizada) y después
    refina con el centro de simetría puntual.

    Args:
        small: Imagen reducida (~ANALYSIS_SIZE px), normalizada.

    Returns:
        (correlación de simetría, centro (y, x), vista log, máscara de
        señal).
    """
    log_image = _log_view(small)
    n = min(small.shape)
    halo = ndi.gaussian_filter(log_image, HALO_SIGMA_FRACTION * n)
    start = np.unravel_index(np.argmax(halo), halo.shape)
    mask = _signal_mask(log_image)
    smooth = ndi.gaussian_filter(log_image, 2)
    corr, center = _symmetry_center(
        smooth, mask, (float(start[0]), float(start[1])),
        SYMMETRY_SEARCH_FRACTION * n,
    )
    return corr, center, log_image, mask


def _sharpness_objective(log_image: np.ndarray, mask: np.ndarray):
    """Función objetivo de nitidez de anillos para un centro (y, x).

    Devuelve menos el promedio del perfil radial al cuadrado del detalle
    fino (imagen menos fondo suave, sin el beam stopper). Con el centro
    correcto los anillos caen en un solo radio y el valor es más
    negativo. Se ignora el centro (r < SHARPNESS_INNER_FRACTION del
    lado), donde quedan restos del halo y de la punta del stopper.
    """
    detail, inner = _detail(log_image, mask, 8)
    yy, xx = np.nonzero(inner)
    values = detail[inner]
    n = min(log_image.shape)
    r_lim = 0.45 * n
    r_min = SHARPNESS_INNER_FRACTION * n

    def objective(p) -> float:
        r = np.hypot(yy - p[0], xx - p[1]).astype(int)
        keep = (r < r_lim) & (r >= r_min)
        if not keep.any():
            return 0.0
        counts = np.bincount(r[keep])
        profile = np.bincount(r[keep], values[keep]) / np.maximum(counts, 1)
        used = counts > 0
        return -float(np.average(profile[used] ** 2, weights=counts[used]))

    return objective


def _ring_sharpness_center(
    log_image: np.ndarray,
    mask: np.ndarray,
    start: tuple[float, float],
) -> tuple[float, float]:
    """Refina el centro maximizando la nitidez de los anillos.

    Con el centro correcto, cada anillo cae en un solo radio y el perfil
    radial tiene picos altos; con un centro corrido, los anillos se
    ensanchan y los picos bajan. Se maximiza con Nelder-Mead desde el
    centro de simetría (que puede estar corrido decenas de píxeles por
    el stopper: en diff1, 36 px). Sin anillos el resultado no es
    confiable: find_beam_center lo controla con la uniformidad del halo.

    Args:
        log_image: Imagen en escala logarítmica.
        mask: Píxeles válidos.
        start: Centro inicial (y, x).

    Returns:
        Centro refinado (y, x).
    """
    objective = _sharpness_objective(log_image, mask)
    y0, x0 = float(start[0]), float(start[1])
    res = optimize.minimize(
        objective, np.array([y0, x0]), method="Nelder-Mead",
        options={
            "initial_simplex": [[y0, x0], [y0 + 6, x0], [y0, x0 + 6]],
            "xatol": 0.2, "fatol": 1e-12,
        },
    )
    return float(res.x[0]), float(res.x[1])


def _radial_quartiles(
    image: np.ndarray, mask: np.ndarray, center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Mediana e IQR/mediana de la intensidad en coronas alrededor del
    centro.

    Returns:
        (mediana por corona, IQR/mediana por corona); NaN en coronas
        con menos de 30 píxeles válidos.
    """
    n = min(image.shape)
    y, x = np.indices(image.shape)
    r = np.hypot(y - center[0], x - center[1])
    edges = np.linspace(RADIAL_RANGE[0] * n, RADIAL_RANGE[1] * n,
                        RADIAL_BINS)
    smooth = ndi.gaussian_filter(image, 2)
    medians, spreads = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (r >= a) & (r < b) & mask
        if sel.sum() < 30:
            medians.append(np.nan)
            spreads.append(np.nan)
            continue
        q1, q2, q3 = np.percentile(smooth[sel], [25, 50, 75])
        medians.append(q2)
        spreads.append((q3 - q1) / max(q2, 1e-4))
    return np.array(medians), np.array(spreads)


# ====================================================================
# BLOQUE 1 — Tipo de imagen y metadatos
# ====================================================================


def is_diffraction_pattern(
    image: np.ndarray,
    metadata: dict | None = None,
    verbose: bool = True,
) -> bool:
    """Detecta si la imagen es un patrón de difracción.

    Si los metadatos dicen que se adquirió en modo difracción, eso
    decide. Si no, se usan tres heurísticas sobre la imagen, alrededor
    del centro de simetría del patrón:

    1. Simetría puntual: correlación de la imagen (log, sin el beam
       stopper ni las zonas sin señal) con su rotación de 180°
       >= SYMMETRY_MIN, y centro lejos del borde.
    2. Decaimiento radial: la intensidad baja con el radio en al menos
       DECAY_MIN de los tramos del perfil (halo central y fondo
       oscuro).
    3. Uniformidad azimutal: en cada radio la intensidad es pareja
       alrededor del centro (IQR/mediana <= AZIMUTHAL_IQR_MAX), como en
       un anillo o un halo.

    Hace falta que voten las tres. Con voto mayoritario, 9 de 22
    imágenes de campo claro y oscuro daban "difracción" (partículas
    grandes y gradientes de iluminación cumplen 1 y 2).

    Args:
        image: Imagen 2D (normalizada a [0, 1]) o perfil 1D.
        metadata: Diccionario de read_camera_metadata (opcional).
        verbose: Si es True, imprime el resultado y los votos.

    Returns:
        True si es un patrón de difracción.
    """
    if metadata and str(metadata.get("modo_operacion", "")).upper() == (
        "DIFFRACTION"
    ):
        if verbose:
            print("¿Patrón de difracción? Sí (metadatos: modo DIFFRACTION)")
        return True
    if _is_1d_profile(image):
        if verbose:
            print("¿Patrón de difracción? Sí (perfil 1D)")
        return True
    if min(image.shape) < 32:
        if verbose:
            print("¿Patrón de difracción? No (imagen demasiado chica)")
        return False

    small, _ = _downsample(image, ANALYSIS_SIZE)
    corr, center, _, mask = _coarse_center(small)
    h, w = small.shape
    border = min(center[0] / h, 1 - center[0] / h,
                 center[1] / w, 1 - center[1] / w)
    vote_sym = corr >= SYMMETRY_MIN and border >= CENTER_BORDER_MIN

    medians, spreads = _radial_quartiles(small, mask, center)
    valid = medians[np.isfinite(medians)]
    if valid.size >= 3:
        decay = float(np.mean(np.diff(valid) < -DECAY_STEP * valid[0]))
        iqr = float(np.nanmedian(spreads))
    else:
        decay, iqr = 0.0, np.inf
    vote_decay = decay >= DECAY_MIN
    vote_uniform = iqr <= AZIMUTHAL_IQR_MAX

    result = bool(vote_sym and vote_decay and vote_uniform)
    if verbose:
        si = {True: "Sí", False: "No"}
        print(f"¿Patrón de difracción? {si[result]}")
        print(f"  1. Simetría puntual:    {si[vote_sym]:2} "
              f"(correlación {corr:.2f}, centro a {100 * border:.0f} % "
              "del borde)")
        print(f"  2. Decaimiento radial:  {si[vote_decay]:2} "
              f"({100 * decay:.0f} % de los tramos bajan)")
        print(f"  3. Uniformidad azimutal: {si[vote_uniform]:2} "
              f"(IQR/mediana {iqr:.2f})")
    return result


def detect_pattern_type(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    verbose: bool = True,
) -> str:
    """Clasifica un patrón de difracción.

    - 'perfil_1d': relación de aspecto > 3:1 o variación en un solo eje.
    - 'anillos' (policristalino): al menos un anillo que cubre >= 75 %
      de la circunferencia, o dos que cubren >= 50 %. Los anillos son
      picos del perfil radial; la cobertura se mide en 36 sectores.
    - 'puntos' (monocristalino o pocos granos): al menos MIN_SPOTS spots
      intensos sin anillos continuos.
    - 'nbed': como 'puntos', pero con spots anchos (FWHM medio > 5 % de
      su distancia al centro).
    - 'amorfo': ni anillos ni spots, solo un halo difuso.

    Solo tiene sentido si is_diffraction_pattern dio True.

    Args:
        image: Imagen 2D normalizada a [0, 1] (o perfil 1D).
        center: Centro del haz (y, x) en píxeles de la imagen completa.
            None para estimarlo.
        verbose: Si es True, imprime el tipo y el criterio.

    Returns:
        'anillos', 'puntos', 'nbed', 'perfil_1d' o 'amorfo'.
    """
    if _is_1d_profile(image):
        if verbose:
            print("Tipo de patrón: perfil_1d (variación en un solo eje)")
        return "perfil_1d"

    info = _pattern_features(image, center)
    strong = sum(c >= RING_STRONG_COVERAGE for c in info["coberturas"])
    good = sum(c >= RING_COVERAGE for c in info["coberturas"])
    if strong >= 1 or good >= 2:
        kind = "anillos"
        reason = (f"{good} anillo(s) con >= {100 * RING_COVERAGE:.0f} % de "
                  "la circunferencia cubierta")
    elif info["n_spots"] >= MIN_SPOTS:
        ratio = info["ancho_relativo"]
        kind = "nbed" if ratio > NBED_WIDTH_FRACTION else "puntos"
        reason = (f"{info['n_spots']} spots intensos sin anillos continuos; "
                  f"ancho medio = {100 * ratio:.1f} % del radio "
                  f"({'>' if kind == 'nbed' else '<='} "
                  f"{100 * NBED_WIDTH_FRACTION:.0f} %)")
    else:
        kind = "amorfo"
        reason = (f"sin anillos continuos y solo {info['n_spots']} spots "
                  f"intensos (< {MIN_SPOTS}): halo difuso")
    if verbose:
        cov = ", ".join(f"r={r:.0f}: {100 * c:.0f} %"
                        for r, c in zip(info["radios_anillos"],
                                        info["coberturas"]))
        print(f"Tipo de patrón: {kind} ({reason})")
        print(f"  Picos radiales y cobertura: {cov or 'ninguno'}")
    return kind


def _pattern_features(
    image: np.ndarray, center: tuple[float, float] | None
) -> dict:
    """Mide anillos (radio y cobertura azimutal) y spots del patrón.

    Trabaja sobre una copia reducida a ~TYPE_SIZE px.

    Args:
        image: Imagen 2D normalizada.
        center: Centro (y, x) en la imagen completa, o None.

    Returns:
        Diccionario con 'centro' (imagen completa), 'radios_anillos'
        (px de la imagen completa), 'coberturas', 'n_spots' y
        'ancho_relativo' (FWHM medio de los spots / su radio medio).
    """
    small, factor = _downsample(image, TYPE_SIZE)
    log_image = _log_view(small)
    mask = _signal_mask(log_image)
    if center is None:
        center = _center_candidates(image)["centro"]
    c = _to_small(center, factor)

    n = min(small.shape)
    y, x = np.indices(small.shape)
    rr = np.hypot(y - c[0], x - c[1])
    ri = rr.astype(int)
    r_max = int(min(c[0], c[1], small.shape[0] - c[0],
                    small.shape[1] - c[1]))
    r0 = int(INNER_RADIUS_FRACTION * n)
    rings: list[int] = []
    coverages: list[float] = []
    if r_max > r0 + 10:
        sel = mask & (ri < r_max)
        counts = np.bincount(ri[sel], minlength=r_max)
        mean = (np.bincount(ri[sel], log_image[sel], minlength=r_max)
                / np.maximum(counts, 1))
        base = ndi.median_filter(mean, size=int(0.08 * n) | 1)
        excess = (mean - base)[r0:r_max]
        noise = _MAD_TO_STD * np.median(np.abs(excess - np.median(excess)))
        peaks, _ = signal.find_peaks(
            excess, prominence=RING_PROMINENCE * max(noise, 1e-9),
            distance=max(3, int(0.03 * r_max)), width=2,
        )
        theta = (np.degrees(np.arctan2(y - c[0], x - c[1])) + 360) % 360
        sector = (theta // (360 / AZIMUTHAL_SECTORS)).astype(int)
        w = RING_HALF_WIDTH
        for p in peaks + r0:
            dist = np.abs(rr - p)
            ring = sel & (dist <= w)
            side = sel & (dist > w + 2) & (dist <= 3 * w + 2)
            hits = total = 0
            for k in range(AZIMUTHAL_SECTORS):
                a = log_image[ring & (sector == k)]
                b = log_image[side & (sector == k)]
                if a.size < 10 or b.size < 10:
                    continue
                total += 1
                err = np.sqrt(a.var() / a.size + b.var() / b.size)
                hits += (a.mean() - b.mean()) > RING_SECTOR_SIGMA * err
            rings.append(int(p))
            coverages.append(hits / total if total else 0.0)

    # Spots: máximos del residuo (log menos fondo suave)
    detail, inner = _detail(log_image, mask, 6)
    resid = ndi.gaussian_filter(detail, 1)
    valid = inner & (rr > r0)
    sigma = _MAD_TO_STD * np.median(
        np.abs(resid[valid] - np.median(resid[valid])))
    spots = peak_local_max(
        np.where(valid, resid, 0.0), min_distance=SPOT_MIN_DISTANCE,
        threshold_abs=SPOT_SIGMA * max(sigma, 1e-9), exclude_border=True,
    )
    # Ancho de cada spot: FWHM sobre la imagen sin el halo (fondo de
    # escala grande), así los discos anchos de NBED no quedan recortados
    flat, _ = _detail(log_image, mask, SPOT_BACKGROUND * n)
    widths, radii = [], []
    for sy, sx in spots:
        fwhm = _spot_fwhm(flat, int(sy), int(sx),
                          max_steps=int(0.25 * rr[sy, sx]) + 2)
        if fwhm is not None:
            widths.append(fwhm)
            radii.append(rr[sy, sx])
    ratio = (float(np.median(widths) / np.median(radii))
             if widths else 0.0)
    return {
        "centro": _to_full(c, factor),
        "radios_anillos": [r * factor for r in rings],
        "coberturas": coverages,
        "n_spots": int(len(spots)),
        "ancho_relativo": ratio,
    }


def _spot_fwhm(
    image: np.ndarray, y: int, x: int, max_steps: int
) -> float | None:
    """FWHM de un spot: distancia a media altura en las 4 direcciones.

    Args:
        image: Imagen sin fondo (el valor lejos de los spots es ~0).
        y, x: Posición del máximo.
        max_steps: Distancia máxima a recorrer en cada dirección.

    Returns:
        FWHM en píxeles (promedio de los anchos en x y en y), o None si
        el spot no es más alto que el fondo.
    """
    peak = image[y, x]
    if peak <= 0:
        return None
    half = peak / 2.0
    h, w = image.shape
    reach = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        step = 1
        while step <= max_steps:
            yy, xx = y + dy * step, x + dx * step
            if not (0 <= yy < h and 0 <= xx < w) or image[yy, xx] < half:
                break
            step += 1
        # Interpolación lineal entre el último punto alto y el primero bajo
        yy, xx = y + dy * step, x + dx * step
        prev = image[y + dy * (step - 1), x + dx * (step - 1)]
        if 0 <= yy < h and 0 <= xx < w and prev > image[yy, xx]:
            frac = (prev - half) / (prev - image[yy, xx])
            reach.append(step - 1 + frac)
        else:
            reach.append(float(step - 1))
    return float(sum(reach) / 2.0)


def electron_wavelength(voltage_kV: float) -> float:
    """Longitud de onda relativista de los electrones.

    λ = 12.2643 / sqrt(V · (1 + 0.97848·10⁻⁶ · V)) Å, con V en voltios.
    A 200 kV da 0.02508 Å.

    Args:
        voltage_kV: Voltaje de aceleración en kV.

    Returns:
        Longitud de onda en Å.
    """
    volts = voltage_kV * 1000.0
    return _WAVELENGTH_H / np.sqrt(volts * (1 + _WAVELENGTH_REL * volts))


def _read_tags(path: Path) -> tuple[dict[str, str], dict]:
    """Lee los tags del archivo como texto.

    Args:
        path: Archivo .dm3/.dm4 (ncempy) o .tif/.tiff (exifread).

    Returns:
        (tags {nombre: valor como texto}, calibración {'escala',
        'unidad'} del archivo o vacía).
    """
    ext = path.suffix.lower()
    tags: dict[str, str] = {}
    calibration: dict = {}
    if ext in (".dm3", ".dm4"):
        from ncempy.io import dm

        with dm.fileDM(str(path)) as fdm:
            fdm.parseHeader()
            dataset = fdm.getDataset(0)
            # ImageList.1 es la miniatura; los datos están en la última
            raw = fdm.allTags
        tags = {k: str(v) for k, v in raw.items()}
        unit = dataset.get("pixelUnit")
        size = dataset.get("pixelSize")
        if unit and size is not None:
            calibration = {"escala": float(np.ravel(size)[-1]),
                           "unidad": str(np.ravel(unit)[-1])}
    elif ext in (".tif", ".tiff"):
        import exifread

        with path.open("rb") as fh:
            raw = exifread.process_file(fh, details=False)
        tags = {k: str(v) for k, v in raw.items()}
    return tags, calibration


def _find_tag(
    tags: dict[str, str],
    names: tuple[str, ...],
    exclude: tuple[str, ...] = (),
) -> str | None:
    """Busca un tag cuyo nombre termine en alguna de las variantes.

    Compara sin espacios ni mayúsculas y prueba las variantes en orden
    (la primera que aparece gana). Si una variante aparece varias
    veces, prefiere la de la imagen principal (el último ImageList).

    Returns:
        Valor como texto, o None si no está.
    """
    def norm(s: str) -> str:
        return re.sub(r"[\s_]", "", s).lower()

    banned = [norm(e) for e in exclude]
    keys = [(norm(k), k) for k in tags if not any(b in norm(k)
                                                    for b in banned)]
    for name in names:
        wanted = norm(name)
        hits = sorted(k for nk, k in keys
                      if nk == wanted or nk.endswith("." + wanted))
        if hits:
            return tags[hits[-1]]
    return None


def _number(text: str | None) -> float | None:
    """Primer número de un texto ('200kV' -> 200.0), o None."""
    if text is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else None


def _ask(prompt: str, default: float) -> float:
    """Pide un número por teclado; Enter (o sin teclado) usa default."""
    try:
        answer = input(f"  {prompt} [{default:g}]: ").strip()
    except EOFError:
        return default
    value = _number(answer.replace(",", "."))
    return default if value is None else value


def read_camera_metadata(
    filepath: str | Path, interactive: bool = True,
    pedir_dato: Callable[[str, float], float] | None = None,
) -> dict:
    """Lee los parámetros de adquisición y calcula la calibración.

    Lee los tags del .dm3/.dm4 (con ncempy) o del .tif (con exifread):
    voltaje, longitud de cámara, magnificación, tamaño de píxel de la
    imagen y del detector, modo de operación y fecha.

    La calibración (1/Å por píxel) sale, en este orden, de:
        1. La escala recíproca del archivo (unidad 1/nm o 1/Å).
        2. λL / píxel del detector, con λ calculada del voltaje y L la
           longitud de cámara. En modo difracción, DigitalMicrograph
           guarda la longitud de cámara en el campo de magnificación.
        3. Valores pedidos al usuario (interactive=True) o por defecto.

    Args:
        filepath: Ruta al archivo.
        interactive: Si es True, pide por teclado los datos que faltan.
            Si es False, usa los valores por defecto (para carpetas y
            tests).
        pedir_dato: Función (texto, valor por defecto) -> número que
            reemplaza a la pregunta por teclado cuando interactive es
            True (la usa la interfaz gráfica). None = por teclado.

    Returns:
        Diccionario con 'camera_length_mm', 'voltage_kV',
        'pixel_size_nm' (de la imagen, None si no hay),
        'pixel_detector_um', 'lambda_angs', 'lambda_L' (Å·mm),
        'calibracion_inv_angs_px', 'metodo_calibracion' ('archivo',
        'lambda_L'), 'magnificacion', 'modo_operacion',
        'fecha_adquisicion', 'fuente' ('metadatos', 'usuario',
        'por_defecto' o 'mixta') y 'fuentes' (fuente de cada campo).
    """
    path = Path(filepath)
    tags, calibration = _read_tags(path) if path.is_file() else ({}, {})
    sources: dict[str, str] = {}

    mode = _find_tag(tags, ("Operation Mode", "OperationMode"))
    mode = mode.strip().upper() if mode else None

    voltage = _number(_find_tag(tags, (
        "Microscope Info.Voltage", "Accelerating Voltage",
        "AcceleratingVoltage", "High Tension", "Voltage")))
    if voltage is not None and voltage > 1000:
        voltage /= 1000.0  # el tag está en voltios
    magnification = _number(_find_tag(tags, (
        "Actual Magnification", "Indicated Magnification",
        "Magnification")))
    camera_length = _number(_find_tag(
        tags, ("Camera Length", "CameraLength"), exclude=("STEM",)))
    if camera_length is None and mode == "DIFFRACTION":
        camera_length = magnification
    detector_um = _number(_find_tag(tags, (
        "CCD.Pixel Size (um)", "Detector.Pixel Size (um)",
        "Pixel Size (um)")))
    binning = _number(_find_tag(tags, ("High Level.Binning",)))
    if detector_um is not None and binning:
        detector_um *= binning
    date = _find_tag(tags, ("Acquisition Date", "DateTimeOriginal",
                            "DateTime"))
    time_ = _find_tag(tags, ("Acquisition Time",))
    if date and time_:
        date = f"{date} {time_}"

    pixel_nm = None
    reciprocal = None  # 1/Å por píxel
    unit = calibration.get("unidad", "")
    scale = calibration.get("escala")
    if scale and unit:
        u = unit.replace("µ", "u").lower()
        if u == "1/nm":
            reciprocal = scale / 10.0
        elif u in ("1/a", "1/å"):
            reciprocal = scale
        elif u == "nm":
            pixel_nm = scale
        elif u == "um":
            pixel_nm = scale * 1000.0
        elif u in ("a", "å"):
            pixel_nm = scale / 10.0

    def fill(name: str, value: float | None, prompt: str,
             default: float) -> float:
        if value is not None:
            sources[name] = "metadatos"
            return float(value)
        if interactive:
            sources[name] = "usuario"
            return (pedir_dato or _ask)(prompt, default)
        sources[name] = "por_defecto"
        return default

    voltage = fill("voltage_kV", voltage, "Voltaje de aceleración (kV)",
                   DEFAULT_VOLTAGE_KV)
    lam = float(electron_wavelength(voltage))

    if reciprocal is not None:
        method = "archivo"
        sources["calibracion"] = "metadatos"
        if detector_um is not None:
            sources["pixel_detector_um"] = "metadatos"
            lambda_l = detector_um * 1e-3 / reciprocal
            if camera_length is None:
                camera_length = lambda_l / lam
                sources["camera_length_mm"] = "calculada"
            else:
                sources["camera_length_mm"] = "metadatos"
        else:
            lambda_l = (lam * camera_length
                        if camera_length is not None else None)
            sources["camera_length_mm"] = (
                "metadatos" if camera_length is not None else "sin_dato")
    else:
        method = "lambda_L"
        camera_length = fill("camera_length_mm", camera_length,
                             "Longitud de cámara (mm)",
                             DEFAULT_CAMERA_LENGTH_MM)
        detector_um = fill("pixel_detector_um", detector_um,
                           "Tamaño de píxel del detector (µm)",
                           DEFAULT_DETECTOR_PIXEL_UM)
        lambda_l = lam * camera_length
        reciprocal = detector_um * 1e-3 / lambda_l
        sources["calibracion"] = sources["camera_length_mm"]

    used = {sources["voltage_kV"], sources["calibracion"]}
    if used == {"metadatos"}:
        source = "metadatos"
    elif len(used) == 1:
        source = used.pop()
    else:
        source = "mixta"

    result = {
        "camera_length_mm": camera_length,
        "voltage_kV": voltage,
        "pixel_size_nm": pixel_nm,
        "pixel_detector_um": detector_um,
        "lambda_angs": lam,
        "lambda_L": lambda_l,
        "calibracion_inv_angs_px": float(reciprocal),
        "metodo_calibracion": method,
        "magnificacion": magnification,
        "modo_operacion": mode,
        "fecha_adquisicion": date or "",
        "fuente": source,
        "fuentes": sources,
    }
    _print_metadata(path, result)
    return result


def _print_metadata(path: Path, meta: dict) -> None:
    """Imprime la tabla de parámetros de read_camera_metadata."""
    src = meta["fuentes"]

    def fmt(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    rows = [
        ("Modo de operación", meta["modo_operacion"] or "-", "metadatos"),
        ("Voltaje (kV)", fmt(meta["voltage_kV"], ".0f"),
         src.get("voltage_kV", "-")),
        ("Longitud de onda λ (Å)", fmt(meta["lambda_angs"], ".5f"),
         "calculada"),
        ("Longitud de cámara L (mm)", fmt(meta["camera_length_mm"], ".1f"),
         src.get("camera_length_mm", "-")),
        ("Constante de cámara λL (Å·mm)", fmt(meta["lambda_L"], ".3f"),
         "calculada"),
        ("Píxel del detector (µm)", fmt(meta["pixel_detector_um"], ".2f"),
         src.get("pixel_detector_um", "metadatos")),
        ("Calibración (1/Å por píxel)",
         fmt(meta["calibracion_inv_angs_px"], ".6f"),
         f"{src.get('calibracion', '-')} ({meta['metodo_calibracion']})"),
        ("Píxel de la imagen (nm)", fmt(meta["pixel_size_nm"], ".4f"),
         "metadatos"),
        ("Magnificación", fmt(meta["magnificacion"], ".1f"), "metadatos"),
        ("Fecha de adquisición", meta["fecha_adquisicion"] or "-",
         "metadatos"),
    ]
    print(f"Metadatos de {path.name}:")
    for name, value, origin in rows:
        print(f"  {name:31} {value:>14}   {origin}")
    if meta["modo_operacion"] == "IMAGING":
        print("  Aviso: los metadatos son de modo IMAGEN; si la imagen es "
              "un patrón de difracción, la calibración del archivo no "
              "sirve y hay que indicar la longitud de cámara.")
    if meta["fuente"] != "metadatos":
        print("  Aviso: parte de la calibración no viene del archivo "
              f"({meta['fuente']}); los d calculados dependen de esos "
              "valores.")


# ====================================================================
# BLOQUE 2 — Centro del haz y preprocesamiento
# ====================================================================


def _halo_uniformity(
    small: np.ndarray, mask: np.ndarray, center: tuple[float, float]
) -> float:
    """Uniformidad azimutal del halo alrededor de un centro.

    Mediana de IQR/mediana de la intensidad en coronas: con el centro
    correcto el halo y los anillos son parejos en cada radio y el valor
    es bajo. Es independiente de cómo se calculó el centro.

    Returns:
        Valor (más bajo = más uniforme); inf si no se puede medir.
    """
    _, spreads = _radial_quartiles(small, mask, center)
    if not np.isfinite(spreads).any():
        return float("inf")
    return float(np.nanmedian(spreads))


def _mass_center(small: np.ndarray) -> tuple[float, float]:
    """Centro de masa de la región brillante más grande (> percentil
    CENTER_MASS_PERCENTILE), en la imagen reducida."""
    smooth = ndi.gaussian_filter(small, 2)
    bright = smooth > np.percentile(smooth, CENTER_MASS_PERCENTILE)
    labels, count = ndi.label(bright)
    if count == 0:
        return (small.shape[0] / 2.0, small.shape[1] / 2.0)
    biggest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    cy, cx = ndi.center_of_mass(smooth, labels, biggest)
    return float(cy), float(cx)


def _gaussian_center(small: np.ndarray) -> tuple[float, float] | None:
    """Centro de una gaussiana 2D ajustada al spot central.

    Recorta CENTER_GAUSS_CROP del lado alrededor del máximo de la imagen
    suavizada y ajusta con curve_fit.

    Returns:
        Centro (y, x) en la imagen reducida, o None si el ajuste falla.
    """
    smooth = ndi.gaussian_filter(small, 3)
    py, px = np.unravel_index(np.argmax(smooth), smooth.shape)
    half = max(3, int(CENTER_GAUSS_CROP * min(small.shape) / 2))
    y0, x0 = max(py - half, 0), max(px - half, 0)
    crop = small[y0:py + half, x0:px + half]
    yy, xx = np.indices(crop.shape)

    def gauss(coords, amp, cy, cx, sigma, base):
        y, x = coords
        return (amp * np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2)
                             / sigma**2) + base).ravel()

    p0 = [float(crop.max() - crop.min()), py - y0, px - x0, 5.0,
          float(crop.min())]
    try:
        params, _ = optimize.curve_fit(gauss, (yy, xx), crop.ravel(),
                                       p0=p0, maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    cy, cx = params[1] + y0, params[2] + x0
    if not (0 <= cy < small.shape[0] and 0 <= cx < small.shape[1]):
        return None
    return float(cy), float(cx)


def _center_candidates(image: np.ndarray) -> dict:
    """Calcula el centro con los cuatro métodos y elige uno.

    Métodos (todos en píxeles de la imagen completa):
        - 'simetria': centro de simetría puntual del halo.
        - 'nitidez': refina el anterior maximizando la nitidez de los
          anillos.
        - 'centro_masa': centro de masa de los píxeles más brillantes.
        - 'gaussiana': gaussiana 2D ajustada al spot central.

    Elección: 'nitidez' si no empeora la uniformidad del halo más de
    CENTER_UNIFORMITY_TOL veces respecto de 'simetria' (si no, la
    simetría). Centro de masa o gaussiana reemplazan a ese centro solo
    si dejan el halo claramente más uniforme: con el haz tapado por el
    beam stopper caen sobre el halo, a ~200 px del centro real.

    Returns:
        Diccionario con 'centro', 'metodo', 'candidatos', 'uniformidad',
        'aceptables' (métodos con uniformidad dentro de
        CENTER_ACCEPTABLE de la mejor) y 'dispersion_px' (distancia
        máxima entre el centro elegido y los aceptables).
    """
    coarse, f_coarse = _downsample(image, ANALYSIS_SIZE)
    _, c_sym, _, mask_coarse = _coarse_center(coarse)
    small, factor = _downsample(image, TYPE_SIZE)
    log_image = _log_view(small)
    mask = _signal_mask(log_image)

    candidates: dict[str, tuple[float, float]] = {}
    candidates["simetria"] = _to_full(c_sym, f_coarse)
    start = _to_small(candidates["simetria"], factor)
    candidates["nitidez"] = _to_full(
        _ring_sharpness_center(log_image, mask, start), factor)
    candidates["centro_masa"] = _to_full(_mass_center(small), factor)
    gauss = _gaussian_center(small)
    if gauss is not None:
        candidates["gaussiana"] = _to_full(gauss, factor)

    scores = {
        name: _halo_uniformity(coarse, mask_coarse,
                               _to_small(point, f_coarse))
        for name, point in candidates.items()
    }
    method = ("nitidez"
              if scores["nitidez"] <= CENTER_UNIFORMITY_TOL
              * scores["simetria"] else "simetria")
    for alt in ("centro_masa", "gaussiana"):
        if alt in scores and (scores[alt]
                              < CENTER_ALTERNATIVE_GAIN * scores[method]):
            method = alt
    center = candidates[method]
    best = min(scores.values())
    acceptable = [name for name, s in scores.items()
                  if s <= CENTER_ACCEPTABLE * best]
    spread = max(
        (float(np.hypot(candidates[n][0] - center[0],
                        candidates[n][1] - center[1]))
         for n in acceptable),
        default=0.0,
    )
    return {
        "centro": center,
        "metodo": method,
        "candidatos": candidates,
        "uniformidad": scores,
        "aceptables": acceptable,
        "dispersion_px": spread,
    }


def find_beam_center(
    image: np.ndarray, verbose: bool = True
) -> tuple[float, float]:
    """Detecta el centro del haz directo (aunque esté tapado).

    Calcula el centro con cuatro métodos (simetría puntual del halo,
    nitidez de los anillos, centro de masa y gaussiana 2D) y elige el
    que deja el halo más uniforme alrededor (ver _center_candidates).
    No usa la mediana de los métodos: con beam stopper, el centro de
    masa y la gaussiana caen a ~200 px y arrastran la mediana.

    Si los métodos aceptables (los que dejan el halo casi igual de
    uniforme) difieren en más de CENTER_SPREAD_WARN px, avisa que el
    centro puede ser impreciso.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        verbose: Si es True, imprime los métodos y el centro elegido.

    Returns:
        Centro (y, x) en píxeles de la imagen.

    Raises:
        ValueError: Si la imagen es un perfil 1D.
    """
    if _is_1d_profile(image):
        raise ValueError("Un perfil 1D no tiene centro de haz.")
    info = _center_candidates(image)
    center = info["centro"]
    if verbose:
        names = {"simetria": "Simetría del halo",
                 "nitidez": "Nitidez de anillos",
                 "centro_masa": "Centro de masa (p95)",
                 "gaussiana": "Gaussiana 2D"}
        print(f"Centro del haz: (y, x) = ({center[0]:.1f}, "
              f"{center[1]:.1f}) px — método: {names[info['metodo']]}")
        print(f"  {'método':22} {'y':>8} {'x':>8} {'dist.':>7} "
              f"{'uniformidad':>12}")
        for name, label in names.items():
            if name not in info["candidatos"]:
                print(f"  {label:22} {'(no ajustó)':>17}")
                continue
            py, px = info["candidatos"][name]
            dist = np.hypot(py - center[0], px - center[1])
            mark = " *" if name == info["metodo"] else ""
            print(f"  {label:22} {py:8.1f} {px:8.1f} {dist:7.1f} "
                  f"{info['uniformidad'][name]:12.3f}{mark}")
        print("  (uniformidad = IQR/mediana del halo en cada radio: "
              "más bajo, mejor centrado)")
        if info["dispersion_px"] > CENTER_SPREAD_WARN:
            print(f"  Aviso: métodos casi igual de buenos difieren en "
                  f"{info['dispersion_px']:.1f} px; el centro puede ser "
                  "impreciso (conviene revisarlo a mano).")
    return center


class _CenterPicker:
    """Ventana para aceptar o corregir el centro con el mouse.

    Botones: "Aceptar centro" cierra y devuelve el centro actual;
    "Corregir manualmente" activa el modo click (cada click mueve la
    cruz) y muestra "Confirmar".
    """

    def __init__(self, image: np.ndarray, center: tuple[float, float],
                 guide_radii: list[float] | None = None) -> None:
        from matplotlib.widgets import Button

        self._plt = plt
        small, self.factor = _downsample(image, TYPE_SIZE)
        self.center = (float(center[0]), float(center[1]))
        self.manual = False
        self.radii = list(guide_radii or [])

        self.fig, self.ax = plt.subplots(figsize=(8, 8.6))
        self.fig.subplots_adjust(bottom=0.12)
        self.ax.imshow(_log_view(small), cmap="gray")
        sy, sx = _to_small(self.center, self.factor)
        (self.cross,) = self.ax.plot(sx, sy, "r+", markersize=26,
                                     markeredgewidth=2)
        self.circles = []
        for r in self.radii:
            circle = plt.Circle((sx, sy), r / self.factor, fill=False,
                                color="lime", lw=1, ls="--")
            self.ax.add_patch(circle)
            self.circles.append(circle)
        self.ax.set_axis_off()
        self._title()

        self.btn_ok = Button(self.fig.add_axes([0.08, 0.02, 0.26, 0.06]),
                             "Aceptar centro")
        self.btn_fix = Button(self.fig.add_axes([0.37, 0.02, 0.30, 0.06]),
                              "Corregir manualmente")
        self.ax_confirm = self.fig.add_axes([0.70, 0.02, 0.22, 0.06])
        self.btn_confirm = Button(self.ax_confirm, "Confirmar")
        self.ax_confirm.set_visible(False)
        self.btn_ok.on_clicked(lambda _event: self.accept())
        self.btn_fix.on_clicked(lambda _event: self.start_manual())
        self.btn_confirm.on_clicked(lambda _event: self.accept())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def _title(self) -> None:
        text = (f"Centro: (y, x) = ({self.center[0]:.1f}, "
                f"{self.center[1]:.1f}) px")
        if self.manual:
            text += "\nHacé click en el centro correcto y después Confirmar"
        self.ax.set_title(text)

    def accept(self) -> None:
        """Cierra la ventana con el centro actual."""
        self._plt.close(self.fig)

    def start_manual(self) -> None:
        """Activa el modo click y muestra el botón Confirmar."""
        self.manual = True
        self.ax_confirm.set_visible(True)
        self._title()
        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        """En modo manual, mueve el centro al punto clickeado."""
        if not self.manual or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.center = _to_full((event.ydata, event.xdata), self.factor)
        self.cross.set_data([event.xdata], [event.ydata])
        for circle in self.circles:
            circle.center = (event.xdata, event.ydata)
        self._title()
        self.fig.canvas.draw_idle()


def correct_center_interactive(
    image: np.ndarray,
    center: tuple[float, float],
    guide_radii: list[float] | None = None,
) -> tuple[float, float]:
    """Muestra el centro detectado y permite aceptarlo o corregirlo.

    Abre una ventana con la imagen (log), una cruz roja en el centro y,
    si se pasan, círculos guía en los radios de los anillos (si el
    centro es correcto, caen sobre los anillos). Botones:
        - "Aceptar centro": devuelve el centro actual.
        - "Corregir manualmente": cada click en la imagen mueve la cruz;
          "Confirmar" acepta la posición.
    Si se cierra la ventana sin confirmar, se devuelve el último centro
    marcado. Con un backend no interactivo (por ejemplo, en los tests)
    devuelve el centro sin cambios.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        center: Centro detectado (y, x) en píxeles de la imagen.
        guide_radii: Radios (px de la imagen) de los círculos guía.

    Returns:
        Centro (y, x) aceptado o corregido.
    """
    import warnings

    picker = _CenterPicker(image, center, guide_radii)
    with warnings.catch_warnings():
        # Sin pantalla (backend no interactivo) no hay ventana: se
        # devuelve el centro sin cambios
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
        plt.show()
    plt.close(picker.fig)
    if picker.center != (float(center[0]), float(center[1])):
        print(f"Centro corregido a mano: (y, x) = "
              f"({picker.center[0]:.1f}, {picker.center[1]:.1f}) px")
    return picker.center


def detect_beam_stopper(
    image: np.ndarray, center: tuple[float, float]
) -> np.ndarray:
    """Detecta el beam stopper (la "aguja" que tapa el haz directo).

    Un píxel es candidato si es mucho más oscuro que la mediana de su
    anillo (STOPPER_SIGMA desvíos y al menos STOPPER_MIN_DROP del
    contraste). Se toma la zona candidata más grande que llega a
    STOPPER_REACH del centro y se completa con la envolvente convexa
    junto con el disco del haz directo: cerca de la punta el stopper
    ocupa más de medio anillo y la mediana ya no lo distingue.

    Lejos del centro, donde el fondo es tan oscuro como el stopper, puede
    quedar sin marcar; ahí no hay señal que proteger.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        center: Centro del haz (y, x) en píxeles de la imagen.

    Returns:
        Máscara booleana (shape de la imagen): True en el stopper.
    """
    from skimage.morphology import convex_hull_image

    small, factor = _downsample(image, TYPE_SIZE)
    view = ndi.gaussian_filter(_log_view(small), STOPPER_SMOOTH)
    cy, cx = _to_small(center, factor)
    y, x = np.indices(view.shape)
    rr = np.hypot(y - cy, x - cx)
    ri = rr.astype(int)
    index = np.arange(1, ri.max() + 2)
    median = np.asarray(ndi.median(view, labels=ri + 1, index=index))
    mad = np.asarray(ndi.median(np.abs(view - median[ri]), labels=ri + 1,
                                index=index))
    drop = median[ri] - view
    contrast = np.maximum(median[ri] - view.min(), 1e-6)
    dark = drop > np.maximum(STOPPER_SIGMA * _MAD_TO_STD * mad[ri],
                             STOPPER_MIN_DROP * contrast)

    n = min(view.shape)
    stopper = np.zeros(view.shape, dtype=bool)
    labels, _ = ndi.label(dark)
    near = np.unique(labels[dark & (rr < STOPPER_REACH * n)])
    near = near[near > 0]
    if near.size:
        sizes = np.asarray(ndi.sum(dark, labels, near))
        if sizes.max() >= STOPPER_MIN_AREA * view.size:
            body = labels == near[int(np.argmax(sizes))]
            # La punta suele pasar del centro y queda separada del cuerpo
            keep = near[sizes >= STOPPER_PIECE_FRACTION * sizes.max()]
            pieces = np.isin(labels, keep)
            disk = rr <= BEAM_MASK_FRACTION * n
            # El stopper es una varilla recta: se prolonga el cuerpo hasta
            # el borde en la dirección centro -> cuerpo (lejos del centro
            # es tan oscuro como el fondo y la mediana no lo separa)
            # (con el eje principal de la zona detectada no sirve: a veces
            # es un trozo corto y ancho y su eje sale perpendicular)
            by, bx = np.nonzero(body)
            axis = np.array([by.mean() - cy, bx.mean() - cx])
            axis /= max(np.hypot(*axis), 1e-9)
            # Se traslada el cuerpo una diagonal entera: al recortarlo a la
            # imagen queda su sección apoyada sobre el borde
            shift = np.round(axis * np.hypot(*view.shape)).astype(int)
            far = np.zeros_like(body)
            far[np.clip(by + shift[0], 0, view.shape[0] - 1),
                np.clip(bx + shift[1], 0, view.shape[1] - 1)] = True
            stopper = convex_hull_image(pieces | disk | far)
            stopper = ndi.binary_dilation(stopper,
                                          iterations=STOPPER_DILATION)
    # De vuelta a la resolución completa
    full = np.repeat(np.repeat(stopper, factor, axis=0), factor, axis=1)
    return full[:image.shape[0], :image.shape[1]]


def preprocess_diffraction(
    image: np.ndarray,
    center: tuple[float, float],
    verbose: bool = True,
) -> np.ndarray:
    """Preprocesamiento específico para patrones de difracción.

    1. Vista logarítmica log(1 + g·I) (g = LOG_GAIN): con la imagen en
       [0, 1], log(1 + I) sería casi lineal; con g = 100 se ven a la vez
       los spots intensos y los débiles.
    2. Suavizado gaussiano suave (sigma PREPROCESS_SIGMA) para bajar el
       ruido sin perder spots.
    3. NaN en el haz directo (disco de radio BEAM_MASK_FRACTION del lado
       menor) y en el beam stopper (detect_beam_stopper), para que no
       distorsionen el análisis.

    No se traslada la imagen para centrarla: interpolar desplaza
    intensidades y pierde los bordes. Los análisis siguientes usan el
    centro detectado.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        center: Centro del haz (y, x) en píxeles de la imagen.
        verbose: Si es True, imprime cuánto se enmascaró.

    Returns:
        Imagen preprocesada en float32, con NaN en las zonas
        enmascaradas.
    """
    data = np.clip(image.astype(np.float64), 0.0, None)
    out = ndi.gaussian_filter(np.log1p(LOG_GAIN * data), PREPROCESS_SIGMA)
    y, x = np.indices(image.shape)
    radius = BEAM_MASK_FRACTION * min(image.shape)
    beam = np.hypot(y - center[0], x - center[1]) <= radius
    stopper = detect_beam_stopper(image, center)
    out[beam | stopper] = np.nan
    if verbose:
        print(f"Preprocesamiento: log(1 + {LOG_GAIN:g}·I), suavizado "
              f"sigma {PREPROCESS_SIGMA}")
        print(f"  Haz directo enmascarado: radio {radius:.0f} px")
        if stopper.any():
            print(f"  Beam stopper enmascarado: {100 * stopper.mean():.2f} "
                  "% de la imagen")
        else:
            print("  Beam stopper: no detectado")
    return out.astype(np.float32)


# ====================================================================
# BLOQUE 3 — Anillos (policristalino)
# ====================================================================


def compute_radial_profile(
    image: np.ndarray,
    center: tuple[float, float],
    metadata: dict,
) -> dict:
    """Perfil de intensidad radial I(r) (integración azimutal).

    Para cada radio r (en pasos de 1 px, de 0 a r_max) promedia los
    píxeles a distancia r ± 0.5 px del centro, ignorando los NaN (haz
    directo y beam stopper). r_max es la distancia del centro al borde
    más cercano. La integración se hace con np.bincount, que suma todos
    los píxeles de cada radio de una vez.

    El eje se pasa a espaciado d con la calibración de los metadatos:
    d = 1 / (r · calibración), que equivale a d = λL / R con R el radio
    sobre el detector (r · píxel del detector). q = 1/d.

    Args:
        image: Imagen 2D. Se recomienda la imagen lineal (normalizada)
            con NaN en lo enmascarado por preprocess_diffraction, así el
            FWHM de los anillos no se deforma con el logaritmo.
        center: Centro del haz (y, x) en píxeles.
        metadata: Diccionario de read_camera_metadata.

    Returns:
        Diccionario con:
            'r_px': radios (0, 1, ..., r_max).
            'r_mm': radio sobre el detector en mm (NaN si no se conoce
                el píxel del detector).
            'd_angs': espaciado d en Å (inf en r = 0; decrece con r).
            'q_inv_angs': q = 1/d en 1/Å.
            'intensity': I(r) media (NaN en radios sin píxeles válidos).
            'intensity_smooth': I(r) suavizado (gaussiana, sigma
                PROFILE_SMOOTH px), sin huecos.
            'error': error estándar de la media en cada radio.
            'n_pixeles': píxeles promediados en cada radio.
            'r_inicio': primer radio útil (fuera del haz directo).
            'centro', 'calibracion_inv_angs_px', 'lambda_angs'.
    """
    data = np.asarray(image, dtype=np.float64)
    h, w = data.shape
    cy, cx = float(center[0]), float(center[1])
    r_max = int(min(cy, cx, h - 1 - cy, w - 1 - cx))
    if r_max < PROFILE_MIN_RADIUS * min(h, w):
        raise ValueError(
            f"El centro ({cy:.0f}, {cx:.0f}) está a {r_max} px del borde: "
            "no hay radio para un perfil. ¿Es un patrón de difracción? Si "
            "lo es, corregí el centro a mano (interactive=True).")
    y, x = np.indices(data.shape)
    r = np.rint(np.hypot(y - cy, x - cx)).astype(int)
    valid = np.isfinite(data) & (r <= r_max)
    size = r_max + 1
    counts = np.bincount(r[valid], minlength=size)
    sums = np.bincount(r[valid], data[valid], minlength=size)
    sums2 = np.bincount(r[valid], data[valid] ** 2, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        var = np.maximum(sums2 / np.maximum(counts, 1) - mean**2, 0.0)
        error = np.where(counts > 1, np.sqrt(var / np.maximum(counts, 1)),
                         np.nan)

    radii = np.arange(size, dtype=np.float64)
    good = np.isfinite(mean)
    if good.sum() < 5:
        raise ValueError("Casi no hay píxeles válidos para el perfil.")
    filled = np.interp(radii, radii[good], mean[good])
    smooth = ndi.gaussian_filter1d(filled, PROFILE_SMOOTH)

    cal = float(metadata["calibracion_inv_angs_px"])
    q = radii * cal
    with np.errstate(divide="ignore"):
        d = np.where(radii > 0, 1.0 / np.maximum(q, 1e-300), np.inf)
    pixel_um = metadata.get("pixel_detector_um")
    r_mm = (radii * pixel_um * 1e-3 if pixel_um is not None
            else np.full(size, np.nan))
    first = int(np.ceil(BEAM_MASK_FRACTION * min(h, w))) + 3
    return {
        "r_px": radii,
        "r_mm": r_mm,
        "d_angs": d,
        "q_inv_angs": q,
        "intensity": mean,
        "intensity_smooth": smooth,
        "error": error,
        "n_pixeles": counts,
        "r_inicio": min(first, r_max - 1),
        "centro": (cy, cx),
        "calibracion_inv_angs_px": cal,
        "lambda_angs": float(metadata["lambda_angs"]),
    }


def _profile_background(profile: dict) -> np.ndarray:
    """Fondo suave del perfil radial (el halo), para DETECTAR anillos.

    Mediana móvil del log del perfil con una ventana de
    RING_BACKGROUND_FRACTION de r_max (al menos RING_BACKGROUND_MIN_PX):
    los anillos son más angostos que la ventana y la mediana los ignora.
    Sirve para encontrar los picos; la altura y el FWHM se miden después
    contra una línea de base local (_measure_peak), porque sobre la
    pendiente del halo este fondo no es exacto.

    Returns:
        Fondo en escala lineal (misma longitud que el perfil).
    """
    smooth = profile["intensity_smooth"]
    start = profile["r_inicio"]
    log_p = np.log(np.maximum(smooth, 1e-9))
    base = log_p.copy()
    window = max(int(RING_BACKGROUND_FRACTION * len(smooth)),
                 RING_BACKGROUND_MIN_PX) | 1
    base[start:] = ndi.median_filter(log_p[start:], size=window,
                                     mode="nearest")
    return np.exp(base)


def _peak_on_baseline(
    smooth: np.ndarray, left: int, right: int, baseline: np.ndarray
) -> dict | None:
    """Posición, altura y FWHM del pico sobre una línea de base dada.

    Args:
        smooth: Perfil suavizado.
        left, right: Extremos del tramo (inclusive).
        baseline: Línea de base en ese tramo (len = right - left + 1).

    Returns:
        Diccionario con 'r', 'altura' y 'fwhm', o None si no hay pico.
    """
    excess = smooth[left:right + 1] - baseline
    k = int(np.argmax(excess))
    height = float(excess[k])
    if height <= 0 or k in (0, len(excess) - 1):
        return None
    a, b, c = excess[k - 1], excess[k], excess[k + 1]
    denom = a - 2 * b + c
    shift = 0.0
    if denom < 0:
        shift = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    half = height / 2.0
    # Cruces a media altura, interpolados linealmente
    i = k
    while i > 0 and excess[i] >= half:
        i -= 1
    j = k
    while j < len(excess) - 1 and excess[j] >= half:
        j += 1
    x_left = float(i)
    if excess[i] < half:
        x_left = i + (half - excess[i]) / (excess[i + 1] - excess[i])
    x_right = float(j)
    if excess[j] < half:
        x_right = j - (half - excess[j]) / (excess[j - 1] - excess[j])
    return {"r": float(left + k + shift), "altura": height,
            "fwhm": float(x_right - x_left)}


def _measure_peak(
    smooth: np.ndarray, peak: int, low: int, high: int
) -> dict | None:
    """Mide un pico contra una línea de base local.

    1. Base recta entre el mínimo a la izquierda (entre low y el pico)
       y el mínimo a la derecha (entre el pico y high).
    2. Como el halo es convexo, esa recta queda por encima del fondo
       real y angosta el pico. Si hay al menos RING_BASE_MIN_POINTS
       puntos a cada lado a más de RING_BASE_EXCLUDE FWHM del pico, se
       ajusta una cuadrática (en log) solo con esos puntos y se vuelve a
       medir.
    Sobre el exceso (perfil - base) se miden la posición (parábola por
    el máximo), la altura y el FWHM (cruces a media altura).

    Args:
        smooth: Perfil suavizado.
        peak: Índice del pico.
        low, high: Límites de búsqueda de las bases.

    Returns:
        Diccionario con 'r', 'altura', 'fwhm', 'base_izq', 'base_der' y
        'base' ('recta' o 'cuadratica'), o None si el pico no sobresale.
    """
    left = low + int(np.argmin(smooth[low:peak + 1]))
    right = peak + int(np.argmin(smooth[peak:high + 1]))
    if right - left < 3:
        return None
    idx = np.arange(left, right + 1)
    line = np.interp(idx, [left, right], [smooth[left], smooth[right]])
    result = _peak_on_baseline(smooth, left, right, line)
    if result is None:
        return None
    kind = "recta"
    # Refinamiento: cuadrática en log con los puntos lejos del pico
    lo, hi = max(low, 0), min(high, len(smooth) - 1)
    span = np.arange(lo, hi + 1)
    far = np.abs(span - result["r"]) > RING_BASE_EXCLUDE * result["fwhm"]
    n_left = int(np.sum(far & (span < result["r"])))
    n_right = int(np.sum(far & (span > result["r"])))
    if min(n_left, n_right) >= RING_BASE_MIN_POINTS:
        log_s = np.log(np.maximum(smooth[span[far]], 1e-9))
        coef = np.polyfit(span[far] - result["r"], log_s, 2)
        curve = np.exp(np.polyval(coef, idx - result["r"]))
        refined = _peak_on_baseline(smooth, left, right, curve)
        if refined is not None:
            result, kind = refined, "cuadratica"
    result.update({"base_izq": int(left), "base_der": int(right),
                   "base": kind})
    return result


def detect_diffraction_rings(radial_profile: dict,
                             verbose: bool = True) -> dict:
    """Detecta los anillos de difracción como picos de I(r).

    1. Detección: scipy.signal.find_peaks en el log del perfil
       suavizado menos su fondo suave (_profile_background), con
        - prominencia >= RING_SIGNIFICANCE veces el ruido del perfil
          (error estándar relativo de cada radio). El criterio de la
          especificación (10 % del rango del perfil) no encontraba
          ningún anillo en diff1 ni en diff3: el halo central ocupa casi
          todo el rango.
        - distancia mínima entre picos: 3 % de r_max.
        - ancho mínimo: 2 px.
    2. Medición de cada pico contra una recta de base local entre los
       mínimos a cada lado (_measure_peak), limitada por los anillos
       vecinos: posición, altura sobre el fondo y FWHM.

    Args:
        radial_profile: Diccionario de compute_radial_profile.
        verbose: Si es True, imprime la tabla de anillos.

    Returns:
        Diccionario con 'n_anillos', 'anillos' (lista con 'id', 'r_px',
        'd_angs', 'q_inv_angs', 'intensidad' (altura sobre el fondo),
        'intensidad_total', 'fwhm_px', 'fwhm_d_angs', 'fwhm_q_inv_angs',
        'significancia' (prominencia / ruido), 'bases_px' y
        'q2_relativo' ((d1/d)²: en una red cúbica sigue la serie
        h² + k² + l²)), 'ratios_d' (matriz n×n de d_i/d_j), 'fondo'
        (fondo usado para detectar) y la calibración y λ.
    """
    smooth = radial_profile["intensity_smooth"]
    start = radial_profile["r_inicio"]
    cal = radial_profile["calibracion_inv_angs_px"]
    r_max = len(smooth) - 1
    background = _profile_background(radial_profile)
    excess_log = np.log(np.maximum(smooth, 1e-9)) - np.log(background)

    rel_error = radial_profile["error"] / np.maximum(
        np.abs(radial_profile["intensity"]), 1e-9)
    valid_error = rel_error[start:][np.isfinite(rel_error[start:])]
    noise = float(np.median(valid_error)) if valid_error.size else np.nan
    if not np.isfinite(noise) or noise <= 0:
        # Sin error por punto (perfil 1D de una sola fila): ruido
        # relativo del perfil sin suavizar respecto del suavizado
        raw = radial_profile["intensity"][start:]
        rel = (raw - smooth[start:]) / np.maximum(np.abs(smooth[start:]),
                                                  1e-9)
        rel = rel[np.isfinite(rel)]
        noise = (_MAD_TO_STD * float(np.median(np.abs(rel - np.median(rel))))
                 if rel.size else 0.0) or 1e-6

    peaks, props = signal.find_peaks(
        excess_log[start:], prominence=RING_SIGNIFICANCE * noise,
        distance=max(1, int(RING_MIN_DISTANCE * r_max)), width=2,
    )
    peaks = peaks + start
    reach = max(int(RING_BACKGROUND_FRACTION * len(smooth)),
                RING_BACKGROUND_MIN_PX) // 2

    rings = []
    for k, p in enumerate(peaks):
        # Las bases se buscan hasta el anillo vecino (o media ventana)
        low = max(start, p - reach,
                  peaks[k - 1] if k > 0 else start)
        high = min(r_max, p + reach,
                   peaks[k + 1] if k + 1 < len(peaks) else r_max)
        peak = _measure_peak(smooth, int(p), int(low), int(high))
        if peak is None:
            continue
        r = peak["r"]
        d = 1.0 / (r * cal)
        rings.append({
            "id": len(rings) + 1,
            "r_px": r,
            "d_angs": d,
            "q_inv_angs": 1.0 / d,
            "intensidad": peak["altura"],
            "intensidad_total": float(smooth[int(round(r))]),
            "fwhm_px": peak["fwhm"],
            # d = 1/(cal·r) => Δd = d·Δr/r
            "fwhm_d_angs": float(d * peak["fwhm"] / r),
            "fwhm_q_inv_angs": float(peak["fwhm"] * cal),
            "significancia": float(props["prominences"][k] / noise),
            "bases_px": (peak["base_izq"], peak["base_der"]),
        })
    if rings:
        d1 = rings[0]["d_angs"]
        for ring in rings:
            ring["q2_relativo"] = (d1 / ring["d_angs"]) ** 2
    d_values = np.array([ring["d_angs"] for ring in rings])
    ratios = (d_values[:, None] / d_values[None, :] if rings
              else np.zeros((0, 0)))
    result = {
        "n_anillos": len(rings),
        "anillos": rings,
        "ratios_d": ratios,
        "fondo": background,
        "calibracion_inv_angs_px": cal,
        "lambda_angs": radial_profile["lambda_angs"],
    }
    if verbose:
        _print_rings(result)
    return result


def _print_rings(rings: dict, label: str = "Anillos") -> None:
    """Imprime la tabla de anillos (o picos) de detect_diffraction_rings."""
    print(f"{label} detectados: {rings['n_anillos']}")
    if not rings["anillos"]:
        return
    print(f"  {'id':>3} {'r (px)':>8} {'d (Å)':>8} {'q (1/Å)':>8} "
          f"{'FWHM px':>8} {'FWHM d (Å)':>10} {'altura':>8} "
          f"{'signif.':>8} {'(d1/d)²':>8}")
    for ring in rings["anillos"]:
        print(f"  {ring['id']:>3} {ring['r_px']:>8.1f} "
              f"{ring['d_angs']:>8.3f} {ring['q_inv_angs']:>8.4f} "
              f"{ring['fwhm_px']:>8.1f} {ring['fwhm_d_angs']:>10.4f} "
              f"{ring['intensidad']:>8.4f} {ring['significancia']:>8.0f} "
              f"{ring['q2_relativo']:>8.3f}")
    print("  (d1/d)²: en una red cúbica es proporcional a h² + k² + l² "
          "(por ejemplo FCC: 3, 4, 8, 11, 12 -> 1, 1.33, 2.67, 3.67, 4)")


def estimate_crystallite_size(
    rings: dict, instrumental_fwhm_px: float = 0.0, verbose: bool = True
) -> dict:
    """Tamaño de cristalito con la fórmula de Scherrer.

    D = K·λ / (β·cos θ), con K = SCHERRER_K (cristalitos esféricos),
    θ = arcsin(λ / 2d) y β el ancho del anillo en 2θ (radianes),
    calculado a partir del FWHM en q: 2θ = 2·arcsin(λq/2). Como en
    electrones θ es muy chico, D ≈ K / Δq.

    El ancho medido incluye el del instrumento (convergencia del haz,
    tamaño del spot, resolución del detector). Si se conoce
    (instrumental_fwhm_px), se resta en cuadratura; si no, D es un
    tamaño MÍNIMO. Se avisa si D queda fuera del rango en que Scherrer
    es razonable (SCHERRER_RANGE_NM).

    Args:
        rings: Diccionario de detect_diffraction_rings.
        instrumental_fwhm_px: Ancho instrumental en píxeles (0: no se
            corrige).
        verbose: Si es True, imprime la tabla.

    Returns:
        Diccionario con 'por_anillo' (lista con 'id', 'd_angs',
        'fwhm_px', 'fwhm_corregido_px', 'beta_rad', 'theta_rad', 'D_nm'
        (None si el anillo no es más ancho que el instrumento) y
        'fuera_de_rango'), 'D_medio_nm' (ponderado por la altura de los
        anillos), 'K', 'instrumental_fwhm_px', 'es_minimo' y
        'advertencias'.
    """
    lam = rings["lambda_angs"]
    cal = rings["calibracion_inv_angs_px"]
    low, high = SCHERRER_RANGE_NM
    rows, warnings = [], []
    for ring in rings["anillos"]:
        fwhm = ring["fwhm_px"]
        corrected = (np.sqrt(fwhm**2 - instrumental_fwhm_px**2)
                     if fwhm > instrumental_fwhm_px else 0.0)
        theta = float(np.arcsin(lam / (2 * ring["d_angs"])))
        size = beta = None
        if corrected > 0:
            q = ring["q_inv_angs"]
            dq = corrected * cal
            # Ancho en 2θ entre q - Δq/2 y q + Δq/2
            beta = float(2 * np.arcsin(lam * (q + dq / 2) / 2)
                         - 2 * np.arcsin(lam * max(q - dq / 2, 0) / 2))
            size = float(SCHERRER_K * lam / (beta * np.cos(theta)) / 10)
        outside = size is not None and not (low <= size <= high)
        if size is None:
            warnings.append(f"Anillo {ring['id']}: no es más ancho que el "
                            "instrumento; no se puede estimar D.")
        elif outside:
            warnings.append(f"Anillo {ring['id']}: D = {size:.1f} nm, "
                            f"fuera del rango válido de Scherrer "
                            f"({low:g}-{high:g} nm).")
        rows.append({
            "id": ring["id"],
            "d_angs": ring["d_angs"],
            "fwhm_px": fwhm,
            "fwhm_corregido_px": float(corrected),
            "beta_rad": beta,
            "theta_rad": theta,
            "D_nm": size,
            "fuera_de_rango": bool(outside),
        })
    weights = [(r["D_nm"], ring["intensidad"])
               for r, ring in zip(rows, rings["anillos"])
               if r["D_nm"] is not None and ring["intensidad"] > 0]
    mean = (float(np.average([s for s, _ in weights],
                             weights=[w for _, w in weights]))
            if weights else None)
    result = {
        "por_anillo": rows,
        "D_medio_nm": mean,
        "K": SCHERRER_K,
        "instrumental_fwhm_px": float(instrumental_fwhm_px),
        "es_minimo": instrumental_fwhm_px <= 0,
        "advertencias": warnings,
    }
    if verbose:
        _print_crystallite(result)
    return result


def _print_crystallite(result: dict) -> None:
    """Imprime la tabla de estimate_crystallite_size."""
    print(f"Tamaño de cristalito (Scherrer, K = {result['K']}):")
    print(f"  {'id':>3} {'d (Å)':>8} {'FWHM px':>8} {'β (mrad)':>9} "
          f"{'D (nm)':>8}")
    for row in result["por_anillo"]:
        beta = "-" if row["beta_rad"] is None else (
            f"{1000 * row['beta_rad']:.3f}")
        size = "-" if row["D_nm"] is None else f"{row['D_nm']:.1f}"
        mark = " *" if row["fuera_de_rango"] else ""
        print(f"  {row['id']:>3} {row['d_angs']:>8.3f} "
              f"{row['fwhm_px']:>8.1f} {beta:>9} {size:>8}{mark}")
    if result["D_medio_nm"] is not None:
        print(f"  Tamaño medio (ponderado por la altura de los anillos): "
              f"{result['D_medio_nm']:.1f} nm")
    if result["es_minimo"]:
        print("  Aviso: no se restó el ancho instrumental; los D son "
              "tamaños MÍNIMOS (el cristalito puede ser más grande).")
    for text in result["advertencias"]:
        print(f"  Aviso: {text}")


# ====================================================================
# BLOQUE 4 — Spots (monocristalino / NBED)
# ====================================================================


def _local_noise(values: np.ndarray, valid: np.ndarray,
                 block: int) -> tuple[np.ndarray, np.ndarray]:
    """Nivel y ruido locales en bloques de block × block px.

    En cada bloque: nivel = mediana; ruido = MAD alrededor de esa
    mediana (escalada a desvío estándar). La mediana ignora los spots,
    que ocupan pocos píxeles de cada bloque. Se interpola entre bloques
    para tener un valor por píxel.

    Returns:
        (nivel, ruido), mapas con el shape de values.
    """
    import warnings

    h, w = values.shape
    data = np.where(valid, values, np.nan)
    data = np.pad(data, ((0, -h % block), (0, -w % block)),
                  constant_values=np.nan)
    bh, bw = data.shape[0] // block, data.shape[1] // block
    blocks = (data.reshape(bh, block, bw, block).transpose(0, 2, 1, 3)
              .reshape(bh, bw, -1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(blocks, axis=2)
        noise = np.nanmedian(np.abs(blocks - level[..., None]),
                             axis=2) * _MAD_TO_STD

    def to_full(grid: np.ndarray, default: float) -> np.ndarray:
        good = np.isfinite(grid)
        fill = np.nanmedian(grid) if good.any() else default
        grid = np.where(good, grid, fill)
        full = ndi.zoom(grid, block, order=1)
        full = np.pad(full, ((0, max(0, h - full.shape[0])),
                             (0, max(0, w - full.shape[1]))), mode="edge")
        return full[:h, :w]

    noise = np.where(noise > 0, noise, np.nan)
    return to_full(level, 0.0), to_full(noise, 1.0)


def detect_diffraction_spots(
    image: np.ndarray,
    center: tuple[float, float],
    metadata: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Detecta los spots discretos del patrón.

    1. Se resta el halo (suavizado SPOT_BG_SIGMA sin los NaN) y se
       suaviza con el tamaño de un spot (SPOT_SMOOTH): filtro adaptado.
    2. Cada píxel se divide por el ruido local (bloques de
       SPOT_NOISE_BLOCK px): cerca del centro el halo es más brillante
       y más ruidoso. El umbral de la especificación (percentil 90 de
       la imagen) daba miles de "spots" de ruido (31 608 en Frame9).
    3. skimage.feature.peak_local_max con min_distance = 5 px,
       umbral SPOT_DETECT_SIGMA y exclude_border=True. Se descartan los
       spots a menos de 3 % del lado menor del centro (haz directo).
    4. Por spot: posición sub-píxel (centroide en 5 × 5), r, ángulo
       φ = arctan2(y - cy, x - cx) en 0-360°, intensidad (máximo en
       5 × 5 de la imagen) y d = 1 / (r · calibración) en Å.
    5. Simetría (ley de Friedel): cada spot debería tener otro en la
       posición opuesta respecto del centro (tolerancia 3 px). Solo
       cuentan los spots cuyo opuesto cae dentro de la imagen y fuera
       del stopper. Si el índice es < 0.7, se avisa.

    Args:
        image: Imagen 2D lineal normalizada, con NaN en lo enmascarado
            (haz directo y stopper), como para compute_radial_profile.
        center: Centro del haz (y, x) en píxeles.
        metadata: Diccionario de read_camera_metadata, para pasar r a
            d. Si es None, d queda en None.
        verbose: Si es True, imprime el resumen.

    Returns:
        Diccionario con 'n_spots', 'spots' (lista con 'id', 'y_px',
        'x_px', 'r_px', 'phi_deg', 'd_angs', 'intensidad', 'snr' y
        'tiene_par_simetrico'), 'indice_simetria' (fracción de spots
        verificables con par; None si no hay verificables),
        'n_verificables' y 'centro'.
    """
    data = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    cy, cx = float(center[0]), float(center[1])
    resid = np.where(valid,
                     filled - _masked_smooth(filled, valid, SPOT_BG_SIGMA),
                     0.0)
    smooth = ndi.gaussian_filter(resid, SPOT_SMOOTH)
    inner = ndi.binary_erosion(valid, iterations=MASK_EROSION)
    level, noise = _local_noise(smooth, inner, SPOT_NOISE_BLOCK)
    snr = np.where(inner, (smooth - level) / np.maximum(noise, 1e-12), 0.0)
    peaks = peak_local_max(snr, min_distance=SPOT_DETECT_DISTANCE,
                           threshold_abs=SPOT_DETECT_SIGMA,
                           exclude_border=True)
    r_peaks = np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx)
    peaks = peaks[r_peaks > SPOT_CENTER_EXCLUDE * min(data.shape)]

    cal = metadata["calibracion_inv_angs_px"] if metadata else None
    half = SPOT_WINDOW // 2
    h, w = data.shape
    spots = []
    for py, px in peaks:
        y0, y1 = max(py - half, 0), min(py + half + 1, h)
        x0, x1 = max(px - half, 0), min(px + half + 1, w)
        win = np.clip(smooth[y0:y1, x0:x1], 0, None)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        total = win.sum()
        y = float((gy * win).sum() / total) if total > 0 else float(py)
        x = float((gx * win).sum() / total) if total > 0 else float(px)
        r = float(np.hypot(y - cy, x - cx))
        spots.append({
            "id": len(spots) + 1,
            "y_px": y,
            "x_px": x,
            "r_px": r,
            "phi_deg": float(np.degrees(np.arctan2(y - cy, x - cx)) % 360),
            "d_angs": (1.0 / (r * cal)) if cal else None,
            "intensidad": float(np.nanmax(data[y0:y1, x0:x1])),
            "snr": float(snr[py, px]),
        })

    # Pares de Friedel
    checkable = 0
    paired = 0
    if spots:
        pos = np.array([[s["y_px"], s["x_px"]] for s in spots])
        for k, s in enumerate(spots):
            oy, ox = 2 * cy - s["y_px"], 2 * cx - s["x_px"]
            iy, ix = int(round(oy)), int(round(ox))
            margin = SPOT_DETECT_DISTANCE
            inside = (margin <= iy < h - margin and margin <= ix < w - margin
                      and inner[iy, ix])
            dist = np.hypot(pos[:, 0] - oy, pos[:, 1] - ox)
            dist[k] = np.inf
            s["tiene_par_simetrico"] = bool(dist.min() <= SPOT_PAIR_TOL)
            if inside:
                checkable += 1
                paired += s["tiene_par_simetrico"]
    index = paired / checkable if checkable else None
    result = {
        "n_spots": len(spots),
        "spots": spots,
        "indice_simetria": index,
        "n_verificables": checkable,
        "centro": (cy, cx),
    }
    if verbose:
        print(f"Spots detectados: {len(spots)} (umbral "
              f"{SPOT_DETECT_SIGMA:g} veces el ruido local)")
        if index is None:
            print("  Simetría: no hay spots con el opuesto dentro de la "
                  "imagen")
        else:
            print(f"  Índice de simetría (ley de Friedel): {index:.2f} "
                  f"({paired} de {checkable} spots verificables tienen "
                  "par opuesto)")
            if index < SPOT_SYMMETRY_MIN:
                print("  Aviso: índice < 0.7: el patrón puede estar "
                      "descentrado, ser de baja calidad o venir de varios "
                      "cristales (policristal con pocos granos).")
    return result


def _reduce_basis(g1: np.ndarray,
                  g2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reducción de Lagrange-Gauss de una base 2D.

    Devuelve la base más corta y casi ortogonal de la misma red, con el
    ángulo entre los vectores en [60°, 90°].
    """
    a, b = g1.astype(float), g2.astype(float)
    for _ in range(50):
        if b @ b < a @ a:
            a, b = b, a
        m = round(float(a @ b) / float(a @ a))
        if m == 0:
            break
        b = b - m * a
    if a @ b < 0:  # ángulo agudo
        b = -b
    return a, b


def analyze_spot_geometry(spots: dict, center: tuple[float, float],
                          metadata: dict | None = None,
                          verbose: bool = True) -> dict:
    """Geometría del patrón de puntos: ángulos, familias y red.

    - Ángulos entre spots: para cada par, el ángulo entre sus vectores
      desde el centro (0-180°), en un histograma (bins de
      ANGLE_BIN_DEG). Los ángulos característicos se toman de los pares
      dentro de una misma familia (mismo d), que son los equivalentes
      por simetría: 60° y 120° en una red hexagonal, 90° en una
      cuadrada. Con todos los pares el histograma se aplana.
    - Familias de planos: spots agrupados por distancia al centro
      (tolerancia 5 %); cada familia tiene un mismo d. Para cada una:
      d, radio, cantidad de spots y separación angular media entre
      spots vecinos.
    - Red recíproca: a* y b* salen de los dos vectores más cortos no
      paralelos (reducidos con Lagrange-Gauss) y se ajustan por mínimos
      cuadrados con todos los spots que indexan (h·a* + k·b*); γ* es el
      ángulo entre ellos. Red real (sección 2D de la zona):
      a = 1/(a*·sen γ*), b = 1/(b*·sen γ*), γ = 180° - γ*. (La
      especificación usaba a = 1/a*, que solo vale si γ* = 90°.)
    - Fracción indexada: spots que caen en h·a* + k·b* (h, k enteros)
      dentro de LATTICE_TOLERANCE·|a*|. Cerca de 1 en un monocristal;
      baja si el patrón viene de varios cristales.

    Args:
        spots: Diccionario de detect_diffraction_spots.
        center: Centro del haz (y, x).
        metadata: Para pasar a Å (si es None, se usa el d de los spots o
            se informa en píxeles).
        verbose: Si es True, imprime el resumen.

    Returns:
        Diccionario con 'angulos_caracteristicos' (lista de (ángulo,
        cantidad de pares)), 'histograma_angulos' (conteos y bordes),
        'familias' (lista con 'id', 'r_px', 'd_angs', 'n_spots',
        'phi_deg', 'separacion_angular_deg'), 'red' ('a_estrella_px',
        'b_estrella_px', 'gamma_estrella_deg', 'a_estrella_inv_angs',
        'b_estrella_inv_angs', 'a_angs', 'b_angs', 'gamma_deg',
        'vectores_px'; None con menos de 2 spots) y 'fraccion_indexada'.
    """
    items = spots["spots"]
    cal = metadata["calibracion_inv_angs_px"] if metadata else None
    if cal is None and items and items[0].get("d_angs"):
        cal = 1.0 / (items[0]["d_angs"] * items[0]["r_px"])
    cy, cx = float(center[0]), float(center[1])
    vecs = np.array([[s["y_px"] - cy, s["x_px"] - cx] for s in items])

    # Familias por radio
    families = []
    groups: list[list[int]] = []
    if items:
        order_r = np.argsort([s["r_px"] for s in items])
        for k in order_r:
            r = items[k]["r_px"]
            if groups and r <= (1 + FAMILY_TOLERANCE) * np.mean(
                    [items[j]["r_px"] for j in groups[-1]]):
                groups[-1].append(k)
            else:
                groups.append([k])
        for g in groups:
            r = float(np.mean([items[j]["r_px"] for j in g]))
            phis = np.sort([items[j]["phi_deg"] for j in g])
            gaps = (np.diff(np.concatenate([phis, [phis[0] + 360]]))
                    if len(phis) > 1 else np.array([]))
            families.append({
                "id": len(families) + 1,
                "r_px": r,
                "d_angs": (1.0 / (r * cal)) if cal else None,
                "n_spots": len(g),
                "phi_deg": [float(p) for p in phis],
                "separacion_angular_deg": (float(np.mean(gaps))
                                           if gaps.size else None),
            })

    # Ángulos entre todos los pares de spots (histograma)
    bins = np.arange(0, 180 + ANGLE_BIN_DEG, ANGLE_BIN_DEG)

    def pair_angles(index: list[int]) -> np.ndarray:
        if len(index) < 2:
            return np.array([])
        unit = vecs[index] / np.linalg.norm(vecs[index], axis=1)[:, None]
        iu = np.triu_indices(len(index), 1)
        return np.degrees(np.arccos(np.clip((unit @ unit.T)[iu], -1, 1)))

    counts, edges = np.histogram(pair_angles(list(range(len(items)))),
                                 bins=bins)
    # Ángulos característicos: entre spots de una misma familia (mismo
    # d, reflexiones equivalentes por simetría). Con todos los pares el
    # histograma se aplana: 100 spots dan ~5000 ángulos.
    within = np.concatenate([pair_angles(g) for g in groups]
                            or [np.array([])])
    fam_counts, _ = np.histogram(within, bins=bins)
    fam_counts[0] = fam_counts[-1] = 0  # 0° y 180°: spots opuestos
    found, _ = signal.find_peaks(np.concatenate([[0], fam_counts, [0]]))
    found = found - 1
    best = found[np.argsort(fam_counts[found])[::-1]][:6]
    characteristic = [(float((edges[k] + edges[k + 1]) / 2),
                       int(fam_counts[k])) for k in best
                      if fam_counts[k] >= 2]

    # Red recíproca: dos vectores más cortos no paralelos, reducidos
    lattice = None
    indexed = None
    if len(items) >= 2:
        norms = np.linalg.norm(vecs, axis=1)
        order_n = np.argsort(norms)
        g1 = vecs[order_n[0]]
        g2 = None
        for k in order_n[1:]:
            cos = abs(vecs[k] @ g1) / (norms[k] * np.linalg.norm(g1))
            if cos < np.cos(np.radians(15)):
                g2 = vecs[k]
                break
        if g2 is not None:
            a_vec, b_vec = _reduce_basis(g1, g2)
            # Ajuste por mínimos cuadrados con todos los spots indexados
            # (dos spots solos dan la red con el error de su posición)
            for _ in range(LATTICE_REFINE_STEPS):
                basis = np.column_stack([a_vec, b_vec])
                hk = np.rint(np.linalg.solve(basis, vecs.T)).T
                err = np.linalg.norm(vecs - hk @ basis.T, axis=1)
                tol = max(LATTICE_TOLERANCE * np.linalg.norm(a_vec),
                          LATTICE_TOLERANCE_MIN_PX)
                use = (err <= tol) & np.any(hk != 0, axis=1)
                if use.sum() < 3 or np.linalg.matrix_rank(hk[use]) < 2:
                    break
                fitted, *_ = np.linalg.lstsq(hk[use], vecs[use], rcond=None)
                a_vec, b_vec = fitted[0], fitted[1]
            a_vec, b_vec = _reduce_basis(a_vec, b_vec)
            la, lb = np.linalg.norm(a_vec), np.linalg.norm(b_vec)
            gamma_star = float(np.degrees(np.arccos(
                np.clip(a_vec @ b_vec / (la * lb), -1, 1))))
            sin_g = np.sin(np.radians(gamma_star))
            lattice = {
                "a_estrella_px": float(la),
                "b_estrella_px": float(lb),
                "gamma_estrella_deg": gamma_star,
                "a_estrella_inv_angs": float(la * cal) if cal else None,
                "b_estrella_inv_angs": float(lb * cal) if cal else None,
                "a_angs": float(1 / (la * cal * sin_g)) if cal else None,
                "b_angs": float(1 / (lb * cal * sin_g)) if cal else None,
                "gamma_deg": 180.0 - gamma_star,
                "vectores_px": (a_vec.tolist(), b_vec.tolist()),
            }
            basis = np.column_stack([a_vec, b_vec])
            hk = np.linalg.solve(basis, vecs.T)
            fit = basis @ np.rint(hk)
            err = np.linalg.norm(vecs.T - fit, axis=0)
            tol = max(LATTICE_TOLERANCE * la, LATTICE_TOLERANCE_MIN_PX)
            indexed = float(np.mean(err <= tol))
    result = {
        "angulos_caracteristicos": characteristic,
        "histograma_angulos": (counts, edges),
        "familias": families,
        "red": lattice,
        "fraccion_indexada": indexed,
    }
    if verbose:
        _print_geometry(result)
    return result


def _print_geometry(geo: dict) -> None:
    """Imprime el resumen de analyze_spot_geometry."""
    angles = ", ".join(f"{a:.0f}° ({n})"
                       for a, n in geo["angulos_caracteristicos"])
    print(f"Ángulos más repetidos entre spots: {angles or '-'}")
    print(f"Familias de planos (spots con el mismo d, ±"
          f"{100 * FAMILY_TOLERANCE:.0f} %): {len(geo['familias'])}")
    print(f"  {'id':>3} {'r (px)':>8} {'d (Å)':>8} {'spots':>6} "
          f"{'separación media':>17}")
    for fam in geo["familias"][:15]:
        d = "-" if fam["d_angs"] is None else f"{fam['d_angs']:.3f}"
        sep = ("-" if fam["separacion_angular_deg"] is None
               else f"{fam['separacion_angular_deg']:.1f}°")
        print(f"  {fam['id']:>3} {fam['r_px']:>8.1f} {d:>8} "
              f"{fam['n_spots']:>6} {sep:>17}")
    if len(geo["familias"]) > 15:
        print(f"  ... ({len(geo['familias']) - 15} familias más)")
    red = geo["red"]
    if red is None:
        print("Red: hacen falta al menos dos spots no alineados.")
        return
    print(f"Red recíproca: |a*| = {red['a_estrella_px']:.1f} px, "
          f"|b*| = {red['b_estrella_px']:.1f} px, "
          f"γ* = {red['gamma_estrella_deg']:.1f}°")
    if red["a_angs"] is not None:
        print(f"Red real estimada (sección 2D): a = {red['a_angs']:.3f} Å, "
              f"b = {red['b_angs']:.3f} Å, γ = {red['gamma_deg']:.1f}°")
    print(f"Spots indexados con esa red: "
          f"{100 * geo['fraccion_indexada']:.0f} %"
          + ("  (bajo: el patrón no es de un solo cristal)"
             if geo["fraccion_indexada"] < 0.5 else ""))


# ====================================================================
# BLOQUE 5 — Perfil 1D
# ====================================================================


def _profile_origin(profile: np.ndarray) -> tuple[float, str]:
    """Busca dónde está r = 0 (el haz directo) en un perfil 1D.

    1. Perfil que cruza el centro (dos lados): es simétrico como un
       espejo respecto del origen. Se prueban todos los puntos con al
       menos PROFILE_SYMMETRY_OVERLAP del perfil a cada lado; si la
       mejor correlación entre los dos lados es >= PROFILE_SYMMETRY_MIN,
       ese es el origen.
    2. Si no, el perfil es de un solo lado (como una integración
       azimutal) y el origen está en el extremo más brillante: el halo
       decae hacia afuera. No se usa el máximo del perfil: con beam
       stopper el máximo es el anillo más intenso.

    Returns:
        (origen en px, descripción del criterio).
    """
    n = len(profile)
    smooth = ndi.gaussian_filter1d(np.asarray(profile, float), 1.5)
    min_half = max(5, int(PROFILE_SYMMETRY_OVERLAP * n))
    best, best_corr = None, -np.inf
    for c in range(min_half, n - min_half):
        half = min(c, n - 1 - c)
        left = smooth[c - half:c][::-1]
        right = smooth[c + 1:c + 1 + half]
        if left.std() == 0 or right.std() == 0:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if corr > best_corr:
            best, best_corr = c, corr
    if best is not None and best_corr >= PROFILE_SYMMETRY_MIN:
        return float(best), (f"perfil de dos lados, simétrico respecto de "
                             f"{best} px (correlación {best_corr:.2f})")
    half = n // 2
    if np.mean(smooth[:half]) >= np.mean(smooth[half:]):
        return 0.0, ("perfil de un lado; el origen es el extremo inicial "
                     "(el más brillante)")
    return float(n - 1), ("perfil de un lado; el origen es el extremo "
                          "final (el más brillante)")


def analyze_1d_profile(
    image: np.ndarray,
    metadata: dict,
    origin: float | None = None,
    instrumental_fwhm_px: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Analiza un perfil 1D de difracción (intensidad vs posición).

    1. Eje principal: el de mayor varianza del perfil medio (horizontal
       si la variación es a lo largo de las columnas). Se colapsa con la
       media sobre el eje perpendicular; el error de cada punto es el
       desvío entre filas / sqrt(n).
    2. Origen (r = 0, haz directo): si no se indica, se busca
       (_profile_origin). Si el haz está en el medio, se promedian los
       dos lados (más señal, y los picos de Friedel coinciden).
    3. Posición -> d en Å con la calibración de los metadatos
       (d = 1 / (r · calibración)), igual que en compute_radial_profile.
    4. Picos con los mismos criterios que detect_diffraction_rings y
       tamaño de cristalito con estimate_crystallite_size (Scherrer).
    5. Perfil suavizado con scipy.signal.savgol_filter (ventana 11,
       orden 3) para visualización.

    Args:
        image: Perfil 1D (vector) o imagen angosta que varía en un eje.
        metadata: Diccionario de read_camera_metadata (calibración y λ).
        origin: Posición del haz directo en px (None: se detecta).
        instrumental_fwhm_px: Ancho instrumental para Scherrer.
        verbose: Si es True, imprime el resumen y las tablas.

    Returns:
        Diccionario con 'eje' ('horizontal', 'vertical' o 'vector'),
        'origen_px', 'criterio_origen', 'perfil' (mismas claves que
        compute_radial_profile), 'intensity_savgol', 'picos' (como
        detect_diffraction_rings) y 'cristalito' (como
        estimate_crystallite_size).
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 1:
        axis, line, error, counts = "vector", data, None, None
    else:
        if data.ndim != 2:
            raise ValueError("Se espera un vector o una imagen 2D.")
        along_x = float(np.nanvar(np.nanmean(data, axis=0)))
        along_y = float(np.nanvar(np.nanmean(data, axis=1)))
        axis = "horizontal" if along_x >= along_y else "vertical"
        rows = data if axis == "horizontal" else data.T
        line = np.nanmean(rows, axis=0)
        counts = np.sum(np.isfinite(rows), axis=0)
        error = (np.nanstd(rows, axis=0) / np.sqrt(np.maximum(counts, 1))
                 if rows.shape[0] > 1 else None)
    n = len(line)
    if n < 20:
        raise ValueError("El perfil es demasiado corto (< 20 puntos).")

    if origin is None:
        origin, reason = _profile_origin(np.nan_to_num(line))
    else:
        reason = "indicado por el usuario"
    o = int(round(origin))

    # Lado(s) del perfil desde el origen hacia afuera
    right = line[o:]
    left = line[o::-1]
    err_r = error[o:] if error is not None else None
    err_l = error[o::-1] if error is not None else None
    size = max(len(right), len(left))
    if 0 < o < n - 1:
        stack = np.full((2, size), np.nan)
        stack[0, :len(right)] = right
        stack[1, :len(left)] = left
        mean = np.nanmean(stack, axis=0)
        sides = np.sum(np.isfinite(stack), axis=0)
        if error is not None:
            e_stack = np.full((2, size), np.nan)
            e_stack[0, :len(err_r)] = err_r
            e_stack[1, :len(err_l)] = err_l
            err = np.sqrt(np.nansum(e_stack**2, axis=0)) / sides
        else:
            err = np.full(size, np.nan)
    else:
        mean = right if o == 0 else left
        err = (err_r if o == 0 else err_l) if error is not None else \
            np.full(len(mean), np.nan)
        sides = np.ones(len(mean), dtype=int)

    radii = np.arange(len(mean), dtype=np.float64)
    good = np.isfinite(mean)
    filled = np.interp(radii, radii[good], mean[good])
    cal = float(metadata["calibracion_inv_angs_px"])
    q = radii * cal
    with np.errstate(divide="ignore"):
        d = np.where(radii > 0, 1.0 / np.maximum(q, 1e-300), np.inf)
    pixel_um = metadata.get("pixel_detector_um")
    profile = {
        "r_px": radii,
        "r_mm": (radii * pixel_um * 1e-3 if pixel_um is not None
                 else np.full(len(radii), np.nan)),
        "d_angs": d,
        "q_inv_angs": q,
        "intensity": mean,
        "intensity_smooth": ndi.gaussian_filter1d(filled, PROFILE_SMOOTH),
        "error": err,
        "n_pixeles": (sides if counts is None else sides * int(
            np.median(counts))),
        "r_inicio": max(3, int(INNER_RADIUS_FRACTION * len(radii))),
        "centro": (float(origin),),
        "calibracion_inv_angs_px": cal,
        "lambda_angs": float(metadata["lambda_angs"]),
    }
    window = min(PROFILE_SAVGOL_WINDOW, len(filled) - (len(filled) + 1) % 2)
    savgol = (signal.savgol_filter(filled, window, PROFILE_SAVGOL_ORDER)
              if window > PROFILE_SAVGOL_ORDER else filled.copy())

    peaks = detect_diffraction_rings(profile, verbose=False)
    sizes = estimate_crystallite_size(
        peaks, instrumental_fwhm_px=instrumental_fwhm_px, verbose=False)
    result = {
        "eje": axis,
        "origen_px": float(origin),
        "criterio_origen": reason,
        "perfil": profile,
        "intensity_savgol": savgol,
        "picos": peaks,
        "cristalito": sizes,
    }
    if verbose:
        print(f"Perfil 1D: eje {axis}, {n} puntos; origen (r = 0) en "
              f"{origin:.1f} px — {reason}")
        _print_rings(peaks, label="Picos")
        _print_crystallite(sizes)
    return result


# ====================================================================
# BLOQUE 6 — Visualización y pipeline
# ====================================================================

_TYPE_TITLES = {
    "anillos": "SAED policristalino (anillos)",
    "puntos": "SAED de puntos",
    "nbed": "NBED (spots anchos)",
    "perfil_1d": "perfil 1D",
    "amorfo": "amorfo (halo difuso)",
}


def _plot_image(ax, image: np.ndarray, center: tuple | None,
                title: str) -> int:
    """Imagen en escala log (reducida a ~1024 px) con el centro.

    Returns:
        Factor de reducción usado.
    """
    small, factor = _downsample(np.nan_to_num(image, nan=0.0), TYPE_SIZE)
    ax.imshow(_log_view(small), cmap="gray")
    if center is not None:
        cy, cx = _to_small(center, factor)
        ax.plot(cx, cy, "r+", markersize=16, markeredgewidth=2)
    ax.set_title(title)
    ax.axis("off")
    return factor


def _plot_profile(ax, profile: dict, peaks: list[dict], x_axis: str,
                  smooth: np.ndarray | None = None) -> None:
    """Perfil I vs d o I vs q, con los picos marcados y rotulados."""
    start = profile["r_inicio"]
    y = profile["intensity_smooth"] if smooth is None else smooth
    x = profile["d_angs"] if x_axis == "d" else profile["q_inv_angs"]
    ax.plot(x[start:], y[start:], color="0.25", lw=1)
    top = np.nanmax(y[start:])
    for k, p in enumerate(peaks):
        pos = p["d_angs"] if x_axis == "d" else p["q_inv_angs"]
        color = plt.cm.tab10(k % 10)
        ax.axvline(pos, color=color, lw=0.8)
        ax.plot(pos, y[int(round(p["r_px"]))], "x", color=color,
                markersize=8, markeredgewidth=2)
        ax.text(pos, top * (1.02 + 0.05 * (k % 2)), f"{p['d_angs']:.2f}",
                color=color, ha="center", fontsize=8)
    if x_axis == "d":
        ax.set_xlim(x[-1], x[start])
        ax.invert_xaxis()
        ax.set_xlabel("d (Å)")
        ax.set_title("Perfil I vs d (rótulos: d en Å)")
    else:
        ax.set_xlabel("q = 1/d (1/Å)")
        ax.set_title("Perfil I vs q (espacio recíproco)")
    ax.set_ylabel("intensidad media")
    ax.set_ylim(top=top * 1.15)


def _plot_table(ax, header: list[str], rows: list[list[str]],
                title: str, note: str = "") -> None:
    """Tabla visual en un eje vacío."""
    ax.axis("off")
    ax.set_title(title)
    if not rows:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return
    shown = rows[:MAX_TABLE_ROWS]
    table = ax.table(cellText=shown, colLabels=header, loc="upper center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    if len(rows) > MAX_TABLE_ROWS:
        note = (f"(se muestran {MAX_TABLE_ROWS} de {len(rows)} filas) "
                + note)
    if note:
        ax.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8,
                transform=ax.transAxes, wrap=True)


def show_diffraction_results(
    image: np.ndarray,
    center: tuple[float, float] | None,
    pattern_type: str,
    results: dict,
    radial_profile: dict | None,
    metadata: dict,
    filepath: str | Path,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Figura de 4 paneles adaptada al tipo de patrón.

    'anillos' y 'amorfo': imagen con los anillos (círculos y d), perfil
    I vs d con los picos, perfil I vs q y tabla (d, q, intensidad, FWHM
    y tamaño de Scherrer).
    'puntos' y 'nbed': imagen con los spots coloreados por familia de
    planos, red recíproca con a* y b*, histograma de d y tabla de
    familias y parámetros de red.
    'perfil_1d': imagen o perfil original, I vs d, I vs q y tabla de
    picos.
    En todos: título con archivo, tipo y cantidad de picos o spots, y la
    constante de cámara λL y la calibración en una esquina. Se guarda
    como <nombre>_difraccion.png.

    Args:
        image: Imagen (o perfil 1D) original.
        center: Centro del haz (None en un perfil 1D).
        pattern_type: Tipo de patrón (detect_pattern_type).
        results: 'results' de pipeline_diffraction.
        radial_profile: Perfil radial (None en un perfil 1D).
        metadata: Diccionario de read_camera_metadata.
        filepath: Ruta de la imagen (para el nombre y el título).
        output_dir: Carpeta de salida. Por defecto, data/samples/.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del PNG guardado.
    """
    from .loader import DEFAULT_OUTPUT_DIR

    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_difraccion.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    ax1, ax2, ax3, ax4 = axes.ravel()

    if pattern_type in ("anillos", "amorfo"):
        rings = results["anillos"]
        sizes = results.get("cristalito")
        peaks = rings["anillos"]
        count = f"{len(peaks)} anillos" if pattern_type == "anillos" \
            else f"{len(peaks)} máximos del halo"
        what = "anillos" if pattern_type == "anillos" else "máximos del halo"
        factor = _plot_image(ax1, image, center,
                             f"Imagen (log) con los {what} detectados")
        cy, cx = _to_small(center, factor)
        for k, p in enumerate(peaks):
            color = plt.cm.tab10(k % 10)
            radius = p["r_px"] / factor
            ax1.add_patch(plt.Circle((cx, cy), radius, fill=False,
                                     color=color, lw=1.2))
            ax1.text(cx + radius * 0.71, cy - radius * 0.71,
                     f"{p['d_angs']:.2f} Å", color=color, fontsize=9,
                     fontweight="bold")
        _plot_profile(ax2, radial_profile, peaks, "d")
        _plot_profile(ax3, radial_profile, peaks, "q")
        by_id = ({r["id"]: r["D_nm"] for r in sizes["por_anillo"]}
                 if sizes else {})
        rows = [[str(p["id"]), f"{p['d_angs']:.3f}",
                 f"{p['q_inv_angs']:.4f}", f"{p['intensidad']:.4f}",
                 f"{p['fwhm_px']:.1f}",
                 "-" if by_id.get(p["id"]) is None
                 else f"{by_id[p['id']]:.1f}"] for p in peaks]
        note = ""
        if sizes and sizes["es_minimo"]:
            note = "Scherrer sin restar el ancho instrumental: tamaños " \
                   "mínimos."
        if pattern_type == "amorfo":
            note = "Amorfo: los máximos del halo no son anillos de Bragg " \
                   "(no se aplica Scherrer)."
        _plot_table(ax4, ["id", "d (Å)", "q (1/Å)", "altura", "FWHM (px)",
                          "D (nm)"], rows, what.capitalize(), note)
    elif pattern_type in ("puntos", "nbed"):
        spots = results["spots"]
        geo = results["geometria"]
        items = spots["spots"]
        count = f"{spots['n_spots']} spots"
        families = geo["familias"]
        family_of = {}
        for fam in families:
            for phi in fam["phi_deg"]:
                family_of[round(phi, 6)] = fam["id"]
        cmap = plt.get_cmap("tab20")
        colors = [cmap((family_of.get(round(s["phi_deg"], 6), 1) - 1) % 20)
                  for s in items]
        factor = _plot_image(ax1, image, center,
                             "Spots (color = familia de planos, mismo d)")
        pts = [_to_small((s["y_px"], s["x_px"]), factor) for s in items]
        ax1.scatter([p[1] for p in pts], [p[0] for p in pts], s=50,
                    facecolors="none", edgecolors=colors, linewidths=1.3)
        dx = [s["x_px"] - center[1] for s in items]
        dy = [center[0] - s["y_px"] for s in items]
        ax2.scatter(dx, dy, c=colors, s=25)
        ax2.plot(0, 0, "r+", markersize=14)
        red = geo["red"]
        if red is not None:
            for vec, label, col in zip(red["vectores_px"], ("a*", "b*"),
                                       ("red", "blue")):
                ax2.annotate("", xy=(vec[1], -vec[0]), xytext=(0, 0),
                             arrowprops={"arrowstyle": "->", "color": col,
                                         "lw": 2})
                ax2.text(vec[1] * 1.15, -vec[0] * 1.15, label, color=col,
                         fontsize=12, fontweight="bold")
        ax2.set_aspect("equal")
        ax2.set_xlabel("x - cx (px)")
        ax2.set_ylabel("cy - y (px)")
        sym = spots["indice_simetria"]
        ax2.set_title("Spots en espacio recíproco"
                      + (f" — simetría {sym:.2f}" if sym is not None
                         else ""))
        d_values = [s["d_angs"] for s in items if s["d_angs"]]
        ax3.hist(d_values, bins=40, color="0.5")
        ax3.set_xlabel("d (Å)")
        ax3.set_ylabel("spots")
        ax3.set_title("Distancias d de los spots")
        rows = [[str(f["id"]), "-" if f["d_angs"] is None
                 else f"{f['d_angs']:.3f}", str(f["n_spots"]),
                 "-" if f["separacion_angular_deg"] is None
                 else f"{f['separacion_angular_deg']:.1f}"]
                for f in families]
        note = ""
        if red is not None:
            note = (f"Red: |a*| = {red['a_estrella_px']:.1f} px, "
                    f"|b*| = {red['b_estrella_px']:.1f} px, γ* = "
                    f"{red['gamma_estrella_deg']:.1f}°")
            if red["a_angs"] is not None:
                note += (f" -> a = {red['a_angs']:.2f} Å, b = "
                         f"{red['b_angs']:.2f} Å, γ = "
                         f"{red['gamma_deg']:.1f}°")
            note += (f"\nIndexados: {100 * geo['fraccion_indexada']:.0f} %"
                     + (" (no es un solo cristal)"
                        if geo["fraccion_indexada"] < 0.5 else ""))
        _plot_table(ax4, ["familia", "d (Å)", "spots", "separación (°)"],
                    rows, "Familias de planos", note)
    elif pattern_type == "perfil_1d":
        prof = results["perfil_1d"]
        peaks = prof["picos"]["anillos"]
        count = f"{len(peaks)} picos"
        if np.ndim(image) == 2:
            ax1.imshow(image, cmap="gray", aspect="auto")
        else:
            ax1.plot(np.asarray(image), color="0.3")
            ax1.axvline(prof["origen_px"], color="r", lw=1)
        ax1.set_title(f"Perfil original ({prof['eje']}; origen en "
                      f"{prof['origen_px']:.0f} px)")
        _plot_profile(ax2, prof["perfil"], peaks, "d",
                      prof["intensity_savgol"])
        _plot_profile(ax3, prof["perfil"], peaks, "q",
                      prof["intensity_savgol"])
        sizes = prof["cristalito"]
        by_id = {r["id"]: r["D_nm"] for r in sizes["por_anillo"]}
        rows = [[str(p["id"]), f"{p['d_angs']:.3f}",
                 f"{p['q_inv_angs']:.4f}", f"{p['fwhm_px']:.1f}",
                 "-" if by_id.get(p["id"]) is None
                 else f"{by_id[p['id']]:.1f}"] for p in peaks]
        _plot_table(ax4, ["id", "d (Å)", "q (1/Å)", "FWHM (px)", "D (nm)"],
                    rows, "Picos",
                    "Scherrer sin restar el ancho instrumental: tamaños "
                    "mínimos." if sizes["es_minimo"] else "")
    else:
        raise ValueError(f"Tipo de patrón desconocido: {pattern_type}")

    lam_l = metadata.get("lambda_L")
    fig.text(0.99, 0.005,
             (f"λL = {lam_l:.3f} Å·mm" if lam_l else "λL = -")
             + f" | calibración {metadata['calibracion_inv_angs_px']:.6f} "
             f"1/Å por px ({metadata['fuente']})",
             ha="right", va="bottom", fontsize=9, color="0.3")
    fig.suptitle(f"{path.name} — {_TYPE_TITLES.get(pattern_type, '')} — "
                 f"{count}", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def pipeline_diffraction(
    filepath: str | Path,
    image: np.ndarray | None = None,
    force: bool = False,
    interactive: bool = True,
    instrumental_fwhm_px: float = 0.0,
    output_dir: str | Path | None = None,
    show: bool = True,
    pedir_dato: Callable[[str, float], float] | None = None,
    corregir_centro: bool | None = None,
) -> dict:
    """Análisis completo de un patrón de difracción (Etapa 4.5).

    Secuencia:
        1. Carga la imagen (load_tem_image) si no se pasa.
        2. read_camera_metadata: primero, porque el modo de operación
           del archivo también sirve para decidir si es difracción.
        3. is_diffraction_pattern (salvo force=True): si no es
           difracción, devuelve {'es_difraccion': False} y termina.
        4. find_beam_center y, si interactive, correct_center_interactive
           (con los anillos como círculos guía).
        5. detect_pattern_type, con el centro ya confirmado.
        6. preprocess_diffraction (log, suavizado, haz y stopper en NaN).
        7. Según el tipo:
            'anillos': compute_radial_profile, detect_diffraction_rings,
                estimate_crystallite_size.
            'puntos' / 'nbed': compute_radial_profile,
                detect_diffraction_spots, analyze_spot_geometry.
            'amorfo': compute_radial_profile y detect_diffraction_rings
                (máximos del halo; sin Scherrer).
            'perfil_1d': analyze_1d_profile.
        8. show_diffraction_results.

    Args:
        filepath: Ruta del archivo.
        image: Imagen ya cargada (normalizada a [0, 1]); None para
            cargarla.
        force: Si es True, se analiza como difracción sin preguntarle a
            is_diffraction_pattern (opción --difraccion).
        interactive: Si es True, pide los metadatos que faltan y abre la
            ventana para corregir el centro. False para carpetas y tests.
        instrumental_fwhm_px: Ancho instrumental para Scherrer.
        output_dir: Carpeta de salida de la figura.
        show: Si es True, muestra la figura.
        pedir_dato: Función (texto, valor por defecto) -> número para
            pedir los metadatos que faltan sin usar el teclado (ver
            read_camera_metadata). None = por teclado.
        corregir_centro: Si se abre la ventana para corregir el centro.
            None = igual que interactive.

    Returns:
        {'es_difraccion': False, 'filepath'} si no es difracción; si no,
        {'es_difraccion': True, 'pattern_type', 'metadata', 'center',
        'image_preprocessed', 'radial_profile' (None en perfil_1d),
        'results', 'figura', 'filepath'}.
    """
    from .loader import load_tem_image

    path = Path(filepath)
    if image is None:
        image = load_tem_image(path)
    print("=" * 72 + "\nETAPA 4.5 — Análisis de difracción\n" + "=" * 72)
    meta = read_camera_metadata(path, interactive=interactive,
                                pedir_dato=pedir_dato)
    if not force and not is_diffraction_pattern(image, meta):
        print("No es un patrón de difracción: se sigue con las Etapas 2-4.")
        return {"es_difraccion": False, "filepath": str(path)}

    one_d = _is_1d_profile(image)
    center = None
    if not one_d:
        center = find_beam_center(image)
        if interactive if corregir_centro is None else corregir_centro:
            info = _pattern_features(image, center)
            guides = [r for r, c in zip(info["radios_anillos"],
                                        info["coberturas"])
                      if c >= RING_COVERAGE]
            center = correct_center_interactive(image, center, guides)
    pattern_type = detect_pattern_type(image, center=center)

    prep = None
    profile = None
    if pattern_type == "perfil_1d":
        results = {"perfil_1d": analyze_1d_profile(
            image, meta, instrumental_fwhm_px=instrumental_fwhm_px)}
    else:
        prep = preprocess_diffraction(image, center)
        masked = np.where(np.isnan(prep), np.nan, image)
        profile = compute_radial_profile(masked, center, meta)
        if pattern_type == "anillos":
            rings = detect_diffraction_rings(profile)
            results = {"anillos": rings,
                       "cristalito": estimate_crystallite_size(
                           rings, instrumental_fwhm_px)}
        elif pattern_type == "amorfo":
            print("Patrón amorfo: se informan los máximos del halo (no son "
                  "anillos de Bragg; no se aplica Scherrer).")
            results = {"anillos": detect_diffraction_rings(profile),
                       "cristalito": None}
        else:
            spots = detect_diffraction_spots(masked, center, meta)
            results = {"spots": spots,
                       "geometria": analyze_spot_geometry(spots, center,
                                                          meta)}

    figure = show_diffraction_results(image, center, pattern_type, results,
                                      profile, meta, path, output_dir,
                                      show)
    return {
        "es_difraccion": True,
        "pattern_type": pattern_type,
        "metadata": meta,
        "center": center,
        "image_preprocessed": prep,
        "radial_profile": profile,
        "results": results,
        "figura": figure,
        "filepath": str(path),
    }
