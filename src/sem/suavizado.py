"""Suavizado de imágenes SEM: reducción de ruido de la ROI preservando los bordes entre fases.

Estima el nivel de ruido de la ROI y lo reduce con non-local means, que promedia parches
parecidos y por eso no mezcla píxeles de fases distintas.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import numpy as np
import skimage as ski


def estimar_ruido(imagen: np.ndarray) -> float:
    """
    Estima la desviación estándar del ruido de la imagen.

    Usa `skimage.restoration.estimate_sigma`, que descompone la imagen con una
    transformada wavelet y toma los coeficientes de detalle más finos (diagonales), donde
    casi todo es ruido. La desviación estándar se calcula con un estimador robusto, la
    mediana del valor absoluto:

        sigma = mediana(|coeficientes|) / 0.6745

    donde 0.6745 es la mediana de |X| para X normal estándar. La mediana hace que los
    bordes entre fases, que también generan coeficientes grandes pero ocupan una fracción
    chica de la imagen, casi no afecten la estimación.

    Referencia: D. L. Donoho e I. M. Johnstone, "Ideal spatial adaptation by wavelet
    shrinkage", Biometrika 81(3), 425-455 (1994).

    Parameters
    ----------
    imagen : np.ndarray
        ROI 2D float32 normalizada (0-1).

    Returns
    -------
    float
        Desviación estándar estimada del ruido, en las mismas unidades que la imagen.

    Raises
    ------
    ValueError
        Si la imagen no es 2D.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    return float(ski.restoration.estimate_sigma(imagen))


def suavizar(
    imagen: np.ndarray,
    factor_h: float | None = None,
    tamano_parche: int = 5,
    distancia_parche: int = 6,
    sigma_ruido: float | None = None,
) -> np.ndarray:
    """
    Reduce el ruido de la imagen con non-local means preservando los bordes entre fases.

    Para cada píxel busca, dentro de una ventana de radio `distancia_parche`, parches de
    `tamano_parche` x `tamano_parche` parecidos al suyo y promedia sus centros, con más
    peso cuanto más parecidos son. En un borde entre fases los parches de la otra fase no
    se parecen, así que casi no se mezclan y el borde se conserva.

    La intensidad del filtrado se fija a partir del ruido de la imagen:

        h = factor_h * sigma_ruido

    donde factor_h se busca por defecto con el criterio de discrepancia
    (ver `buscar_factor_h`). Así una imagen más ruidosa se filtra más, sin ajustar
    parámetros a mano.

    Usa `skimage.restoration.denoise_nl_means` en modo rápido (`fast_mode=True`).

    Parameters
    ----------
    imagen : np.ndarray
        ROI 2D float32 normalizada (0-1).
    factor_h : float | None
        Relación entre la intensidad del filtrado y la desviación estándar del ruido.
        Si es None se busca con `buscar_factor_h` (criterio de discrepancia). Si se pasa
        un número, se usa directamente (skimage recomienda ~0.8 en modo rápido).
        Default: None.
    tamano_parche : int
        Lado de los parches que se comparan, en píxeles. Debe ser más chico que las
        estructuras más finas que se quieren conservar. Default: 5.
    distancia_parche : int
        Radio de la ventana donde se buscan parches parecidos, en píxeles. Más grande
        encuentra más parches pero es más lento. Default: 6.
    sigma_ruido : float | None
        Desviación estándar del ruido. Si es None se estima con `estimar_ruido`.
        Default: None.

    Returns
    -------
    np.ndarray
        Imagen nueva float32 suavizada, con el mismo shape que la entrada.

    Raises
    ------
    ValueError
        Si la imagen no es 2D, si `factor_h` <= 0, o si `tamano_parche` o
        `distancia_parche` son < 1.
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    if factor_h is not None and factor_h <= 0:
        raise ValueError(f"factor_h debe ser > 0, se recibió {factor_h}")
    if tamano_parche < 1 or distancia_parche < 1:
        raise ValueError(
            "tamano_parche y distancia_parche deben ser >= 1, se recibió "
            f"{tamano_parche} y {distancia_parche}"
        )

    if sigma_ruido is None:
        sigma_ruido = estimar_ruido(imagen)
    # Sin ruido no hay nada que filtrar (y h = 0 no es válido para non-local means).
    if sigma_ruido == 0:
        return imagen.astype(np.float32, copy=True)
    if factor_h is None:
        factor_h = buscar_factor_h(imagen, sigma_ruido, tamano_parche, distancia_parche)

    # Pasar sigma además de h hace que la comparación de parches descuente el ruido esperado.
    suavizada = ski.restoration.denoise_nl_means(
        imagen,
        patch_size=int(tamano_parche),
        patch_distance=int(distancia_parche),
        h=factor_h * sigma_ruido,
        sigma=sigma_ruido,
        fast_mode=True,
    )
    return suavizada.astype(np.float32, copy=False)


# Intervalo de búsqueda de factor_h y máximo de pasos de la bisección.
_FACTOR_H_MIN = 0.3
_FACTOR_H_MAX = 3.0
_MAX_PASOS_BISECCION = 20


def buscar_factor_h(
    imagen: np.ndarray,
    sigma_ruido: float,
    tamano_parche: int = 5,
    distancia_parche: int = 6,
    tolerancia: float = 0.02,
) -> float:
    """
    Busca el factor_h de non-local means con el criterio de discrepancia.

    El criterio dice que el filtrado es adecuado cuando lo que se quita de la imagen tiene
    la misma desviación estándar que el ruido:

        desviación estándar(imagen - suavizada) = sigma_ruido

    Si se quita menos, queda ruido; si se quita más, se está borrando señal. Como lo
    quitado crece con factor_h, el valor se encuentra por bisección entre 0.3 y 3.0,
    cortando cuando la diferencia relativa es menor que `tolerancia`.

    Parameters
    ----------
    imagen : np.ndarray
        ROI 2D float32 normalizada (0-1).
    sigma_ruido : float
        Desviación estándar del ruido (ver `estimar_ruido`). Debe ser > 0.
    tamano_parche : int
        Lado de los parches de non-local means, en píxeles. Default: 5.
    distancia_parche : int
        Radio de búsqueda de parches, en píxeles. Default: 6.
    tolerancia : float
        Diferencia relativa aceptada entre lo quitado y sigma_ruido. Default: 0.02 (2 %).

    Returns
    -------
    float
        factor_h que cumple el criterio. Si ni con 3.0 se alcanza sigma_ruido, devuelve
        3.0 y avisa.

    Raises
    ------
    ValueError
        Si la imagen no es 2D, si sigma_ruido <= 0 o si tolerancia no está en (0, 1).
    """
    if imagen.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D; se recibió shape {imagen.shape}")
    if sigma_ruido <= 0:
        raise ValueError(f"sigma_ruido debe ser > 0, se recibió {sigma_ruido}")
    if not 0 < tolerancia < 1:
        raise ValueError(f"tolerancia debe estar en (0, 1), se recibió {tolerancia}")

    minimo, maximo = _FACTOR_H_MIN, _FACTOR_H_MAX
    for _ in range(_MAX_PASOS_BISECCION):
        factor_h = (minimo + maximo) / 2
        suavizada = suavizar(
            imagen,
            factor_h=factor_h,
            tamano_parche=tamano_parche,
            distancia_parche=distancia_parche,
            sigma_ruido=sigma_ruido,
        )
        quitado = float(np.std(imagen - suavizada))

        if abs(quitado - sigma_ruido) / sigma_ruido < tolerancia:
            return factor_h
        # Se quitó poco: hay que filtrar más fuerte. Se quitó de más: más suave.
        if quitado < sigma_ruido:
            minimo = factor_h
        else:
            maximo = factor_h

    # El máximo nunca bajó: ni con el filtrado más fuerte se llega a quitar sigma_ruido.
    if maximo == _FACTOR_H_MAX:
        print(
            f"⚠️ Ni con factor_h = {_FACTOR_H_MAX} se quita sigma_ruido; "
            "se usa ese valor (¿sigma_ruido sobreestimado?)"
        )
        return _FACTOR_H_MAX
    return (minimo + maximo) / 2
