"""Regiones de análisis de una ROI SEM de electrones retrodispersados (BSE).

Solo para BSE. Una ROI puede tener zonas con comportamientos muy distintos (una franja negra, una
zona lisa, una zona con patrón). Si se buscan poblaciones en toda la ROI a la vez, las zonas
grandes o de mucho contraste dominan y tapan a las demás. Por eso la ROI se divide primero en
regiones de análisis, que después se procesan por separado:

1. zonas por gris: picos y valles del histograma de gris;
2. franja de borde entre zonas: los píxeles de transición, que no pertenecen a ninguna;
3. actividad sin cruzar límites de zona: cuántos bordes hay alrededor de cada píxel (σ = 25 px);
4. dentro de cada zona, clases de actividad por picos y valles.

Cada región se aísla (`aislar_region`) para que los filtros posteriores no vean las demás.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import numpy as np
import skimage as ski
from scipy import ndimage, signal


def _escalar(valores: np.ndarray) -> np.ndarray:
    """Lleva los valores al rango 0-1 (mínimo → 0, máximo → 1)."""
    return (valores - valores.min()) / max(valores.max() - valores.min(), 1e-12)


def _valles_entre(h: np.ndarray, picos: list[int]) -> list[int]:
    """Índice del mínimo de h entre cada par de picos consecutivos."""
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def _masas(h: np.ndarray, picos: list[int]) -> list[float]:
    """Fracción del total de h que queda entre los valles alrededor de cada pico."""
    bordes = [0, *_valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def cortes_por_valles(
    valores: np.ndarray,
    bins: int = 256,
    sigma_histograma: float = 2.0,
    prominencia: float = 0.05,
    distancia: int = 8,
    masa_minima: float = 0.01,
    valle_maximo: float = 0.75,
) -> np.ndarray:
    """
    Cortes entre los picos reales del histograma de unos valores (escalados a 0-1).

    1. Histograma de `bins` casillas, suavizado con una gaussiana de `sigma_histograma` casillas.
    2. Picos con prominencia ≥ `prominencia` × máximo y separados al menos `distancia` casillas.
    3. Se descartan, uno por vez, los picos cuya masa (lo que queda entre sus valles) es menor
       que `masa_minima`, y los pares cuyo valle es mayor que `valle_maximo` × el pico menor
       (no hay un valle real entre ellos): se quita el pico más bajo del par.
    4. Los cortes son los valles entre los picos que quedan.

    Los criterios son los del módulo TEM. Son valores fijos, no salen de la imagen.

    Parameters
    ----------
    valores : np.ndarray
        Valores a separar, escalados a 0-1 (cualquier forma; se aplanan).
    bins, sigma_histograma, prominencia, distancia, masa_minima, valle_maximo
        Criterios del histograma y de los picos (ver arriba).

    Returns
    -------
    np.ndarray
        Cortes (en 0-1), ordenados. Vacío si hay un solo pico.
    """
    conteos, bordes = np.histogram(np.ravel(valores), bins=bins, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), sigma_histograma)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])), prominence=prominencia * h.max(),
                                 distance=distancia)
    picos = [int(p) - 1 for p in picos]
    while len(picos) > 1:
        m = _masas(h, picos)
        peor = int(np.argmin(m))
        if m[peor] < masa_minima:
            picos.pop(peor)
            continue
        valles = _valles_entre(h, picos)
        relativo = [h[v] / min(h[a], h[b]) for v, a, b in zip(valles, picos, picos[1:])]
        i = int(np.argmax(relativo))
        if relativo[i] <= valle_maximo:
            break
        picos.pop(i if h[picos[i]] < h[picos[i + 1]] else i + 1)
    return centros[_valles_entre(h, picos)]


def zonas_por_gris(gris: np.ndarray, area_minima: float = 0.01) -> np.ndarray:
    """
    Divide la imagen en zonas por gris: cortes en los valles del histograma de gris.

    Las manchitas de una zona con menos de `area_minima` del área de la imagen se reasignan a la
    zona vecina más cercana; así quedan solo las zonas grandes (por ejemplo, una franja negra).

    Parameters
    ----------
    gris : np.ndarray
        Imagen 2D en gris, escalada a 0-1.
    area_minima : float
        Fracción mínima del área para que una mancha conserve su zona. Valor fijo (1 %).

    Returns
    -------
    np.ndarray
        Etiqueta de zona de cada píxel (0, 1, ...; 0 = la más oscura).
    """
    zonas = np.digitize(gris, cortes_por_valles(gris))
    limpia = np.full(zonas.shape, -1)
    for z in np.unique(zonas):
        grandes = ski.morphology.remove_small_objects(zonas == z, max_size=int(area_minima * zonas.size) - 1)
        limpia[grandes] = z
    _, (fi, co) = ndimage.distance_transform_edt(limpia < 0, return_indices=True)
    limpia = limpia[fi, co]
    _, renumerada = np.unique(limpia, return_inverse=True)
    return renumerada.reshape(zonas.shape)


def franja_de_borde(gris: np.ndarray, zonas: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    Píxeles de transición junto al límite entre zonas.

    Para cada zona: distancia de cada píxel al límite con otra zona y mediana del gris a cada
    distancia. El ancho de la franja es la primera distancia en la que el gris ya recorrió el
    90 % del camino desde el valor en el límite hasta el valor típico del interior de la zona.
    Sale de la imagen.

    Parameters
    ----------
    gris : np.ndarray
        Imagen 2D en gris, escalada a 0-1.
    zonas : np.ndarray
        Etiqueta de zona de cada píxel (de `zonas_por_gris`).

    Returns
    -------
    borde : np.ndarray
        Máscara booleana de la franja de borde.
    anchos : dict[int, int]
        Ancho de la franja (px) de cada zona.
    """
    borde = np.zeros(zonas.shape, bool)
    anchos = {}
    for z in np.unique(zonas):
        mascara = zonas == z
        dist = ndimage.distance_transform_edt(mascara)
        dmax = int(dist.max())
        interior = gris[dist >= max(3, 0.5 * dmax)]
        referencia = np.median(interior) if interior.size else np.median(gris[mascara])
        en_limite = np.median(gris[mascara & (dist <= 1)])
        ancho = 1
        for d in range(1, min(dmax, 40) + 1):
            anillo = mascara & (dist > d - 1) & (dist <= d)
            if anillo.any() and abs(np.median(gris[anillo]) - referencia) <= 0.1 * abs(en_limite - referencia):
                ancho = d
                break
        anchos[int(z)] = ancho
        borde |= mascara & (dist <= ancho)
    return borde, anchos


def actividad_sin_cruzar(
    imagen: np.ndarray,
    zonas: np.ndarray,
    borde: np.ndarray,
    sigma: float = 25,
) -> np.ndarray:
    """
    Actividad (promedio local del gradiente) de cada zona, sin mezclar con las vecinas.

    El gradiente (Sobel) se promedia con una gaussiana de σ = `sigma`, usando solo píxeles de la
    misma zona, lejos de su límite (erosión 3 × 3) y fuera de la franja de borde. Así el salto
    de gris entre dos zonas no cuenta como actividad de ninguna.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris (la ROI suavizada).
    zonas : np.ndarray
        Etiqueta de zona de cada píxel.
    borde : np.ndarray
        Máscara de la franja de borde.
    sigma : float
        Escala del promedio (px). σ grande mide zonas. Valor fijo (25 px), no sale de la imagen.

    Returns
    -------
    np.ndarray
        Actividad de cada píxel, en unidades de gradiente.
    """
    gradiente = ski.filters.sobel(imagen)
    actividad = np.zeros(imagen.shape)
    for z in np.unique(zonas):
        mascara = zonas == z
        valido = (ski.morphology.erosion(mascara, ski.morphology.footprint_rectangle((3, 3))) & ~borde)
        valido = valido.astype(float)
        promedio = ski.filters.gaussian(gradiente * valido, sigma) / np.maximum(ski.filters.gaussian(valido, sigma),
                                                                                1e-6)
        actividad[mascara] = promedio[mascara]
    return actividad


def regiones_de_analisis(
    imagen: np.ndarray,
    sigma_actividad: float = 25,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[int, int]]:
    """
    Divide la ROI en regiones de análisis: zonas por gris × clases de actividad.

    1. Zonas por gris (`zonas_por_gris`).
    2. Franja de borde entre zonas (`franja_de_borde`); no pertenece a ninguna región.
    3. Actividad sin cruzar límites de zona (`actividad_sin_cruzar`).
    4. Dentro de cada zona, clases de actividad por los valles de su histograma
       (`cortes_por_valles`).

    Parameters
    ----------
    imagen : np.ndarray
        ROI 2D suavizada.
    sigma_actividad : float
        Escala de la actividad (px). Valor fijo (25 px).

    Returns
    -------
    regiones : np.ndarray
        Etiqueta de región de cada píxel (0, 1, ...); -1 en la franja de borde.
    descripcion : list[str]
        Zona de gris y clase de actividad de cada región.
    borde : np.ndarray
        Máscara de la franja de borde.
    anchos : dict[int, int]
        Ancho de la franja (px) de cada zona.

    Raises
    ------
    ValueError
        Si la imagen no es 2D.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    gris = _escalar(imagen)
    zonas = zonas_por_gris(gris)
    borde, anchos = franja_de_borde(gris, zonas)
    actividad = actividad_sin_cruzar(imagen, zonas, borde, sigma_actividad)
    regiones = np.full(imagen.shape, -1)
    descripcion = []
    for z in np.unique(zonas):
        mascara = (zonas == z) & ~borde
        if not mascara.any():
            continue
        a = _escalar(actividad[mascara])
        clases = np.digitize(a, cortes_por_valles(a))
        for c in np.unique(clases):
            sub = np.zeros(imagen.shape, bool)
            sub[mascara] = clases == c
            regiones[sub] = len(descripcion)
            descripcion.append(f"zona de gris {z}, actividad {c}")
    return regiones, descripcion, borde, anchos


def aislar_region(
    imagen: np.ndarray,
    mascara: np.ndarray,
    margen: int = 20,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    """
    Recorte de una región con lo de afuera reemplazado por el píxel más cercano de la región.

    Así los filtros que se aplican después (realce, choque, actividad) no ven el gris ni los
    bordes de las otras regiones ni de la franja de borde: fuera de la región la imagen continúa
    con los valores de su orilla.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D completa.
    mascara : np.ndarray
        Máscara booleana de la región.
    margen : int
        Píxeles de margen alrededor de la región en el recorte.

    Returns
    -------
    aislada : np.ndarray
        Recorte relleno.
    caja : tuple[slice, slice]
        Posición del recorte en la imagen completa (`imagen[caja]`).

    Raises
    ------
    ValueError
        Si la máscara está vacía o no tiene el tamaño de la imagen.
    """
    if mascara.shape != imagen.shape:
        raise ValueError(f"La máscara {mascara.shape} no tiene el tamaño de la imagen {imagen.shape}")
    if not mascara.any():
        raise ValueError("La máscara de la región está vacía")
    filas, cols = np.nonzero(mascara)
    caja = (slice(max(filas.min() - margen, 0), filas.max() + margen + 1),
            slice(max(cols.min() - margen, 0), cols.max() + margen + 1))
    sub, m = imagen[caja], mascara[caja]
    _, (fi, co) = ndimage.distance_transform_edt(~m, return_indices=True)
    return sub[fi, co], caja
