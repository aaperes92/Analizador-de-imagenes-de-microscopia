"""Poblaciones (fases) dentro de una región de una ROI SEM de electrones retrodispersados (BSE).

Solo para BSE: cada fase tiene un gris propio. El método no necesita que el histograma de gris
tenga un valle entre fases ni que se diga cuántas hay:

1. paisaje gris × actividad: histograma conjunto del gris (tras el realce y el choque) y de la
   actividad de la imagen de partida (σ = 3 px), que separa el interior de las estructuras (poca
   actividad) de sus bordes (mucha). Cada población forma una masa en el paisaje;
2. normalización por fila: cada nivel de actividad pesa lo mismo, para que una fase con mucha
   masa (la matriz) no tape a las demás;
3. núcleos: la parte más densa de cada masa (bins sobre la media altura de su fila, conexos, con
   al menos el 1 % de los píxeles). Cada núcleo es una población: dice cuántas hay y su gris
   central;
4. pozos: inundación (watershed) del paisaje desde los núcleos; la frontera entre dos pozos queda
   donde hay menos píxeles. Cada píxel toma la población del pozo en el que cae.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import numpy as np
import skimage as ski
from scipy import ndimage


def actividad(imagen: np.ndarray, sigma: float = 3) -> np.ndarray:
    """
    Actividad local: promedio gaussiano del módulo del gradiente (Sobel).

    Con σ del orden del espesor de las estructuras más finas, distingue el interior de una
    estructura (actividad baja) de sus bordes (actividad alta).

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris (la de partida, antes del realce y del choque: el choque afila los
        bordes y subiría la actividad en todas las fases).
    sigma : float
        Escala del promedio (px). Valor fijo (3 px), no sale de la imagen.

    Returns
    -------
    np.ndarray
        Actividad de cada píxel, en unidades de gradiente.
    """
    return ski.filters.gaussian(ski.filters.sobel(imagen), sigma)


def paisaje(
    gris: np.ndarray,
    act: np.ndarray,
    bins: int = 128,
    rango: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Paisaje gris × actividad: cuántos píxeles hay con cada par (gris, actividad).

    La actividad se normaliza a 0-1 con su percentil 99.9 (los valores mayores se recortan a 1).

    Parameters
    ----------
    gris : np.ndarray
        Gris de cada píxel (cualquier forma; por ejemplo, los píxeles de una región).
    act : np.ndarray
        Actividad de los mismos píxeles.
    bins : int
        Casillas por eje.
    rango : tuple[float, float] | None
        Rango de gris del paisaje. Por defecto, percentiles 0.1 y 99.9 del gris.

    Returns
    -------
    conteos : np.ndarray
        Paisaje (bins × bins); filas = actividad (0 abajo), columnas = gris.
    fila : np.ndarray
        Casilla de actividad de cada píxel.
    columna : np.ndarray
        Casilla de gris de cada píxel.
    rango : tuple[float, float]
        Rango de gris usado.
    """
    gris, act = np.ravel(gris), np.ravel(act)
    if rango is None:
        rango = tuple(float(v) for v in np.percentile(gris, (0.1, 99.9)))
    a = np.clip(act / max(np.percentile(act, 99.9), 1e-12), 0, 1)
    fila = np.clip((a * bins).astype(int), 0, bins - 1)
    columna = np.clip(((gris - rango[0]) / max(rango[1] - rango[0], 1e-12) * bins).astype(int), 0, bins - 1)
    conteos = np.bincount(np.ravel_multi_index((fila, columna), (bins, bins)),
                          minlength=bins ** 2).reshape(bins, bins).astype(float)
    return conteos, fila, columna, rango


def normalizar_por_fila(conteos: np.ndarray, fila_minima: float = 0.005) -> np.ndarray:
    """
    Normaliza el paisaje por fila: cada fila (suavizada 1 casilla) dividida por su máximo.

    Así cada nivel de actividad pesa lo mismo y una población con mucha masa no tapa a las
    demás. Las filas con menos de `fila_minima` × los píxeles de la fila más poblada quedan en
    cero (son casi vacías y su normalización solo amplificaría ruido).

    Parameters
    ----------
    conteos : np.ndarray
        Paisaje (de `paisaje`).
    fila_minima : float
        Fracción mínima de píxeles de una fila respecto de la más poblada. Valor fijo (0.5 %).

    Returns
    -------
    np.ndarray
        Paisaje normalizado (0-1).
    """
    suave = ndimage.gaussian_filter(conteos, 1)
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= fila_minima * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(validas, suave / suave.max(axis=1, keepdims=True), 0.0)


def nucleos_y_pozos(
    normalizado: np.ndarray,
    conteos: np.ndarray,
    media_altura: float = 0.5,
    masa_minima: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Núcleos y pozos del paisaje normalizado.

    - Núcleo: casillas con valor ≥ `media_altura` (la mitad del máximo de su fila), conexas,
      cuya masa (píxeles reales, de `conteos`) es al menos `masa_minima` del total. Cada núcleo
      es una población.
    - Pozo: todo lo que se inunda desde un núcleo (watershed sobre el paisaje invertido) hasta
      encontrarse con el agua de otro. La frontera queda donde hay menos píxeles.

    Parameters
    ----------
    normalizado : np.ndarray
        Paisaje normalizado por fila.
    conteos : np.ndarray
        Paisaje sin normalizar (para medir la masa de cada núcleo).
    media_altura : float
        Umbral relativo de cada fila. Valor fijo (0.5).
    masa_minima : float
        Fracción mínima de los píxeles para que un núcleo sea población. Valor fijo (1 %).

    Returns
    -------
    nucleos : np.ndarray
        Etiqueta de núcleo de cada casilla (0 = ninguno).
    pozos : np.ndarray
        Etiqueta de pozo de cada casilla (la del núcleo desde el que se inundó).
    """
    marcadas = ski.measure.label(normalizado >= media_altura, connectivity=2)
    masas = ndimage.sum(conteos, marcadas, index=np.arange(1, marcadas.max() + 1))
    con_masa = np.flatnonzero(np.asarray(masas) >= masa_minima * conteos.sum()) + 1
    nucleos = np.zeros_like(marcadas)
    for nueva, vieja in enumerate(con_masa, start=1):
        nucleos[marcadas == vieja] = nueva
    pozos = ski.segmentation.watershed(-normalizado, nucleos) if nucleos.max() else nucleos
    return nucleos, pozos


def poblaciones_de_region(
    gris: np.ndarray,
    act: np.ndarray,
    bins: int = 128,
    media_altura: float = 0.5,
    masa_minima: float = 0.01,
) -> dict:
    """
    Poblaciones de una región: paisaje → normalización → núcleos → pozos → población de cada píxel.

    Las poblaciones se numeran por su gris central, de la más oscura (0) a la más clara.

    Parameters
    ----------
    gris : np.ndarray
        Gris de los píxeles de la región, tras el realce y el choque (1D).
    act : np.ndarray
        Actividad de partida de los mismos píxeles (de `actividad`).
    bins, media_altura, masa_minima
        Ver `paisaje` y `nucleos_y_pozos`.

    Returns
    -------
    dict
        - "poblacion": población de cada píxel (0 = la más oscura), mismo orden que `gris`;
        - "centrales": gris central de cada población (mediana de los píxeles de su núcleo);
        - "fracciones": fracción de los píxeles de la región en cada población;
        - "en_nucleo": fracción de los píxeles que cae dentro de un núcleo;
        - "conteos", "normalizado", "nucleos", "pozos", "rango": el paisaje y sus etiquetas,
          con los núcleos y pozos renumerados por gris (1 = la más oscura), para graficar.

    Raises
    ------
    ValueError
        Si `gris` y `act` no tienen la misma cantidad de píxeles.
    """
    gris, act = np.ravel(gris), np.ravel(act)
    if gris.size != act.size:
        raise ValueError(f"gris ({gris.size}) y actividad ({act.size}) deben tener los mismos píxeles")
    conteos, fila, columna, rango = paisaje(gris, act, bins)
    normalizado = normalizar_por_fila(conteos)
    nucleos, pozos = nucleos_y_pozos(normalizado, conteos, media_altura, masa_minima)
    n = int(nucleos.max())
    if n == 0:
        # Sin núcleos (región muy chica o sin estructura): una sola población
        return {"poblacion": np.zeros(gris.size, int), "centrales": np.array([np.median(gris)]),
                "fracciones": np.array([1.0]), "en_nucleo": 0.0, "conteos": conteos,
                "normalizado": normalizado, "nucleos": nucleos, "pozos": np.ones_like(pozos), "rango": rango}

    en_nucleo = nucleos[fila, columna]
    centrales = np.asarray(ndimage.median(gris, en_nucleo, index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    nuevo = np.zeros(n + 1, int)
    nuevo[orden + 1] = np.arange(1, n + 1)
    nucleos, pozos = nuevo[nucleos], nuevo[pozos]
    poblacion = pozos[fila, columna] - 1
    return {"poblacion": poblacion, "centrales": centrales[orden],
            "fracciones": np.bincount(poblacion, minlength=n) / gris.size,
            "en_nucleo": float(np.mean(en_nucleo > 0)), "conteos": conteos, "normalizado": normalizado,
            "nucleos": nucleos, "pozos": pozos, "rango": rango}
