"""Realce de imágenes SEM de electrones retrodispersados (BSE) antes de segmentar.

Solo para BSE: el gris de cada píxel indica la composición (cada fase tiene un gris propio), y
el realce busca recuperar ese gris cerca de los bordes. No se aplica a electrones secundarios
(SE), donde el gris refleja el relieve.

El desenfoque del microscopio mezcla el gris de dos fases en una franja alrededor de cada borde
(degradé). Este módulo estima el ancho de esa franja, realza los bordes con una máscara de enfoque
y los afila con un filtro de choque, que lleva cada píxel hacia el gris del lado del borde al que
pertenece y deja mesetas de gris constante. El filtro de choque con pozos agrega, una vez que el
choque se frena, una atracción hacia el gris central de cada pozo del paisaje gris × actividad.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import numpy as np
import skimage as ski
from scipy import ndimage

from .poblaciones import normalizar_por_fila, nucleos_y_pozos, paisaje

# Ancho 10-90 % de un borde difuminado con una gaussiana de σ = 1 (2 x 1.2816)
ANCHO_10_90_POR_SIGMA = 2.563


def estimar_degrade(
    imagen: np.ndarray,
    mascara: np.ndarray | None = None,
    muestras: int = 1500,
    semilla: int = 0,
) -> tuple[float, int]:
    """
    Estima el ancho del degradé de los bordes, como el σ de una gaussiana, en píxeles.

    Busca los máximos locales del gradiente (Sobel), que son puntos sobre bordes, y en cada uno
    toma el perfil de gris a lo largo de la dirección del gradiente (±10 px). Usa solo bordes
    aislados: el perfil debe ser monótono y llegar a sus dos mesetas (≤ 12 % y ≥ 88 % del
    salto) a más de 5 px del borde, para que otro borde cercano no lo deforme. En cada perfil
    mide el ancho 10-90 % y lo convierte en σ de una gaussiana:

        σ = mediana(ancho 10-90 %) / 2.563

    donde 2.563 es el ancho 10-90 % de un borde difuminado con una gaussiana de σ = 1.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris (por ejemplo, la ROI suavizada, o una región aislada).
    mascara : np.ndarray | None
        Máscara booleana 2D del mismo tamaño. Si se da, solo se usan bordes cuyo máximo de
        gradiente cae dentro de ella (para estimar el degradé de una región). Por defecto,
        toda la imagen.
    muestras : int
        Máximo de bordes a analizar; si hay más, se elige una muestra al azar.
    semilla : int
        Semilla del generador aleatorio que elige la muestra.

    Returns
    -------
    sigma : float
        Ancho del degradé (σ, px). Si hay menos de 10 bordes aislados, devuelve 1.0: no hay
        datos suficientes para estimarlo y se usa un valor fijo.
    usados : int
        Cantidad de bordes aislados con los que se estimó.

    Raises
    ------
    ValueError
        Si la imagen no es 2D, o si la máscara no tiene el mismo tamaño.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    if mascara is not None and mascara.shape != imagen.shape:
        raise ValueError(f"La máscara {mascara.shape} no tiene el tamaño de la imagen {imagen.shape}")

    gy, gx = ndimage.sobel(imagen, axis=0), ndimage.sobel(imagen, axis=1)
    maximos = ski.feature.peak_local_max(np.hypot(gx, gy), min_distance=3, threshold_rel=0.2,
                                         exclude_border=12)
    if mascara is not None:
        maximos = maximos[mascara[maximos[:, 0], maximos[:, 1]]]
    rng = np.random.default_rng(semilla)
    if len(maximos) > muestras:
        maximos = maximos[rng.choice(len(maximos), muestras, replace=False)]

    t = np.linspace(-10, 10, 201)
    anchos = []
    for y, x in maximos:
        n = np.hypot(gy[y, x], gx[y, x])
        perfil = ndimage.map_coordinates(imagen, [y + t * gy[y, x] / n, x + t * gx[y, x] / n], order=1)
        p = (perfil - perfil.min()) / max(perfil.max() - perfil.min(), 1e-9)
        if p[0] > p[-1]:
            p = p[::-1]
        # Borde aislado: mesetas lejos del borde y perfil monótono
        if p[t < -5].max() > 0.12 or p[t > 5].min() < 0.88 or np.min(np.diff(p)) < -0.03:
            continue
        i10, i90 = np.argmax(p >= 0.1), np.argmax(p >= 0.9)
        if i90 > i10:
            anchos.append(t[i90] - t[i10])

    if len(anchos) < 10:
        return 1.0, len(anchos)
    return float(np.median(anchos)) / ANCHO_10_90_POR_SIGMA, len(anchos)


def realzar(imagen: np.ndarray, sigma_degrade: float, cantidad: float = 1.0) -> np.ndarray:
    """
    Realza los bordes con una máscara de enfoque de radio igual al ancho del degradé.

    La máscara de enfoque suma a la imagen su diferencia con una versión desenfocada:

        realzada = imagen + cantidad · (imagen - gaussiana(imagen, radio))

    Con radio = σ del degradé, lo que se suma es justamente el detalle que el desenfoque borró.
    Prepara la imagen para el filtro de choque: los bordes quedan más empinados y el choque
    los afila sin correr las mesetas.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris.
    sigma_degrade : float
        Ancho del degradé (σ, px), de `estimar_degrade`. Se usa como mínimo 0.5 px.
    cantidad : float
        Peso del detalle sumado. Valor fijo (1.0), no sale de la imagen.

    Returns
    -------
    np.ndarray
        Imagen realzada, del mismo tamaño y en la misma escala de gris (puede salir del rango
        original cerca de los bordes).

    Raises
    ------
    ValueError
        Si la imagen no es 2D.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    return ski.filters.unsharp_mask(imagen, radius=max(sigma_degrade, 0.5), amount=cantidad,
                                    preserve_range=True)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pendiente minmod: la de menor módulo si a y b tienen el mismo signo; 0 si no."""
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def filtro_choque(
    imagen: np.ndarray,
    sigma_ruido: float,
    sigma_interno: float = 1.0,
    dt: float = 0.25,
    max_pasos: int = 200,
) -> tuple[np.ndarray, int]:
    """
    Afila los bordes con el filtro de choque de Osher y Rudin.

    En cada paso, cada píxel se mueve en gris según el signo de la segunda derivada en la
    dirección del gradiente (estimada sobre la imagen suavizada con `sigma_interno`):

        I ← I - dt · signo(D2) · |∇I|

    Del lado claro de un borde D2 < 0 y el píxel se aclara; del lado oscuro, se oscurece. Así
    el degradé se parte en dos y cada mitad se lleva al gris de su lado: quedan mesetas de gris
    constante separadas por un salto. |∇I| se calcula con el esquema minmod, que no crea
    oscilaciones nuevas.

    Termina cuando el cambio medio de un paso es menor que 0.1 × ruido (el cambio ya no se
    distingue del ruido) o tras `max_pasos` pasos.

    Referencia: S. Osher y L. I. Rudin, "Feature-oriented image enhancement using shock
    filters", SIAM J. Numer. Anal. 27(4), 919-940 (1990).

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris (en el procesamiento, la realzada).
    sigma_ruido : float
        Desviación estándar del ruido, de `estimar_ruido`; fija el criterio de parada.
    sigma_interno : float
        σ (px) de la gaussiana con la que se estima la dirección del borde. Valor fijo (1 px),
        no sale de la imagen.
    dt : float
        Paso de tiempo (estable para dt ≤ 0.5).
    max_pasos : int
        Máximo de pasos.

    Returns
    -------
    imagen_choque : np.ndarray
        Imagen tras el filtro, del mismo tamaño.
    pasos : int
        Cantidad de pasos realizados.

    Raises
    ------
    ValueError
        Si la imagen no es 2D.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    tolerancia = 0.1 * sigma_ruido
    I = imagen.astype(float).copy()
    for paso in range(max_pasos):
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        # Segunda derivada en la dirección del gradiente
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)
        I += delta
        if np.mean(np.abs(delta)) < tolerancia:
            break
    return I, paso + 1


def filtro_choque_con_pozos(
    imagen: np.ndarray,
    act: np.ndarray,
    sigma_ruido: float,
    intensidad: float = 2.0,
    cada: int = 5,
    sigma_interno: float = 1.0,
    dt: float = 0.25,
    max_pasos: int = 200,
) -> tuple[np.ndarray, int, int | None]:
    """
    Filtro de choque con un término de atracción hacia el gris central de cada pozo del paisaje.

    Piensa a los píxeles como partículas del paisaje gris × actividad: cuando se amontonan en un
    pozo, se alinean en el gris central de ese pozo en todas las actividades. Así cada masa
    tiende a una línea vertical (el mismo gris en el interior y en el borde) en lugar de una
    mancha redonda.

    Dos etapas:
    1. choque solo (`filtro_choque`) hasta que se frena: el término no actúa antes de que el
       choque haya separado las masas (en una imagen con degradé, el paisaje de partida tiene
       un solo pozo y el término comprimiría todos los grises hacia su centro);
    2. choque + atracción, hasta que vuelve a frenarse. Cada `cada` pasos se recalculan el
       paisaje (gris actual × actividad de partida), los núcleos y los pozos
       (`sem.bse.poblaciones`), y en cada paso:

        I ← I - dt · signo(D2) · |∇I| + intensidad · dt · densidad · (gris central del pozo - I)

       donde densidad es el paisaje normalizado (0-1) en la casilla del píxel: cuanto más
       amontonado, más fuerte la atracción.

    Parameters
    ----------
    imagen : np.ndarray
        Imagen 2D en gris (en el procesamiento, la realzada).
    act : np.ndarray
        Actividad de la imagen de partida (`sem.bse.poblaciones.actividad`), mismo tamaño; queda
        fija para que no se realimente.
    sigma_ruido : float
        Desviación estándar del ruido; fija el criterio de parada (cambio medio < 0.1 × ruido).
    intensidad : float
        Peso del término de pozos frente al choque. 0 = choque solo. Valor fijo (2), no sale de
        la imagen.
    cada : int
        Cada cuántos pasos se recalculan los pozos.
    sigma_interno, dt, max_pasos
        Como en `filtro_choque` (`max_pasos` cuenta para cada etapa).

    Returns
    -------
    imagen_choque : np.ndarray
        Imagen tras el filtro.
    pasos : int
        Pasos totales de las dos etapas.
    activacion : int | None
        Paso en que se activó el término de pozos (None si intensidad = 0).

    Raises
    ------
    ValueError
        Si la imagen no es 2D o `act` no tiene su tamaño.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    if act.shape != imagen.shape:
        raise ValueError(f"La actividad {act.shape} no tiene el tamaño de la imagen {imagen.shape}")

    I, pasos = filtro_choque(imagen, sigma_ruido, sigma_interno, dt, max_pasos)
    if intensidad <= 0:
        return I, pasos, None
    activacion = pasos
    tolerancia = 0.1 * sigma_ruido
    objetivo, densidad = I.copy(), np.zeros_like(I)
    for paso in range(max_pasos):
        if paso % cada == 0:
            conteos, fila, columna, _ = paisaje(I.ravel(), act.ravel())
            normalizado = normalizar_por_fila(conteos)
            nucleos, pozos = nucleos_y_pozos(normalizado, conteos)
            n = int(nucleos.max())
            if n:
                centrales = np.asarray(ndimage.median(I.ravel(), nucleos[fila, columna], index=np.arange(1, n + 1)))
                objetivo = centrales[pozos[fila, columna] - 1].reshape(I.shape)
                densidad = normalizado[fila, columna].reshape(I.shape)
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy) + intensidad * dt * densidad * (objetivo - I)
        I += delta
        if np.mean(np.abs(delta)) < tolerancia:
            break
    return I, activacion + paso + 1, activacion
