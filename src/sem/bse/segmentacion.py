"""Segmentación completa de una ROI SEM de electrones retrodispersados (BSE).

Solo para BSE. Aplica en seguidilla, sobre la ROI que deja la carga, todas las operaciones de
`sem.suavizado`, `sem.bse.regiones`, `sem.bse.realce` y `sem.bse.poblaciones`:

1. suavizado (non-local means);
2. regiones de análisis (zonas por gris × clases de actividad);
3. en cada región aislada: ruido, degradé, realce, actividad y filtro de choque con pozos;
4. en cada región: paisaje gris × actividad, núcleos y pozos → población de cada píxel;
5. la franja de borde toma la etiqueta del píxel clasificado más cercano.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import numpy as np
from scipy import ndimage

from ..suavizado import estimar_ruido, suavizar
from .poblaciones import actividad, poblaciones_de_region
from .realce import estimar_degrade, filtro_choque_con_pozos, realzar
from .regiones import aislar_region, regiones_de_analisis


def procesar_region(suavizada: np.ndarray, mascara: np.ndarray, intensidad_pozos: float = 2.0) -> dict:
    """
    Realce, choque con pozos y poblaciones de una región, aislada de las demás.

    La región se aísla (`aislar_region`) y sobre ese recorte se estiman el ruido y el degradé
    (solo con bordes de la región), se realza, se calcula la actividad de partida y se aplica
    el filtro de choque con pozos. Después se buscan las poblaciones con solo los píxeles de la
    región.

    Parameters
    ----------
    suavizada : np.ndarray
        ROI 2D suavizada completa.
    mascara : np.ndarray
        Máscara booleana de la región.
    intensidad_pozos : float
        Peso del término de pozos del filtro de choque (0 = choque solo). Valor fijo (2).

    Returns
    -------
    dict
        Lo que devuelve `poblaciones_de_region`, más:
        - "ruido", "degrade", "bordes_usados", "pasos_choque", "activacion_pozos": parámetros
          de la región;
        - "realzada", "choque": gris de los píxeles de la región tras cada paso (1D, en el
          orden de `suavizada[mascara]`).
    """
    aislada, caja = aislar_region(suavizada, mascara)
    m = mascara[caja]
    ruido = estimar_ruido(aislada)
    degrade, usados = estimar_degrade(aislada, mascara=m)
    realzada = realzar(aislada, degrade)
    act = actividad(aislada)
    tras_choque, pasos, activacion = filtro_choque_con_pozos(realzada, act, ruido, intensidad_pozos)
    resultado = poblaciones_de_region(tras_choque[m], act[m])
    resultado.update({"ruido": ruido, "degrade": degrade, "bordes_usados": usados, "pasos_choque": pasos,
                      "activacion_pozos": activacion, "realzada": realzada[m], "choque": tras_choque[m]})
    return resultado


def segmentar_roi(roi: np.ndarray, suavizada: bool = False, intensidad_pozos: float = 2.0) -> dict:
    """
    Segmenta una ROI BSE: regiones de análisis y poblaciones dentro de cada una.

    Parameters
    ----------
    roi : np.ndarray
        ROI 2D en gris (0-1), tal como la deja la carga (`sem.guardar_roi`).
    suavizada : bool
        True si la ROI ya pasó por `sem.suavizar` (no se vuelve a suavizar).
    intensidad_pozos : float
        Peso del término de pozos del filtro de choque (0 = choque solo). Valor fijo (2).

    Returns
    -------
    dict
        - "suavizada": ROI suavizada;
        - "regiones": región de cada píxel (-1 en la franja de borde);
        - "descripcion": zona de gris y clase de actividad de cada región;
        - "borde": máscara de la franja de borde;
        - "por_region": lista con el resultado de `procesar_region` de cada región;
        - "etiquetas": etiqueta final de cada píxel (0, 1, ...), una por par (región,
          población); la franja de borde toma la del píxel clasificado más cercano;
        - "tabla": lista de dicts, uno por etiqueta, con "etiqueta", "region", "poblacion",
          "gris_central", "fraccion_region" y "fraccion_roi".

    Raises
    ------
    ValueError
        Si la ROI no es 2D.
    """
    if roi.ndim != 2:
        raise ValueError(f"Se esperaba una ROI 2D; se recibió shape {roi.shape}")
    imagen = roi.astype(float) if suavizada else suavizar(roi).astype(float)
    regiones, descripcion, borde, _ = regiones_de_analisis(imagen)

    etiquetas = np.full(imagen.shape, -1)
    por_region, tabla = [], []
    for r in range(len(descripcion)):
        mascara = regiones == r
        resultado = procesar_region(imagen, mascara, intensidad_pozos)
        por_region.append(resultado)
        for p, (gris, fraccion) in enumerate(zip(resultado["centrales"], resultado["fracciones"])):
            sub = np.zeros(imagen.shape, bool)
            sub[mascara] = resultado["poblacion"] == p
            etiquetas[sub] = len(tabla)
            tabla.append({"etiqueta": len(tabla), "region": r, "poblacion": p, "gris_central": float(gris),
                          "fraccion_region": float(fraccion), "fraccion_roi": float(sub.mean())})

    _, (fi, co) = ndimage.distance_transform_edt(etiquetas < 0, return_indices=True)
    etiquetas = etiquetas[fi, co]
    for fila in tabla:
        fila["fraccion_roi"] = float(np.mean(etiquetas == fila["etiqueta"]))
    return {"suavizada": imagen, "regiones": regiones, "descripcion": descripcion, "borde": borde,
            "por_region": por_region, "etiquetas": etiquetas, "tabla": tabla}
