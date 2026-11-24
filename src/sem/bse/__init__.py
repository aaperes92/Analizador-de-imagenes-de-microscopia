"""Segmentación de imágenes SEM de electrones retrodispersados (BSE).

Solo para BSE: el gris de cada píxel indica la composición. No se aplica a electrones
secundarios (SE).
"""

# Realce: degradé, máscara de enfoque, filtro de choque (solo y con pozos)
from .realce import estimar_degrade, filtro_choque, filtro_choque_con_pozos, realzar

# Regiones de análisis
from .regiones import (
    actividad_sin_cruzar,
    aislar_region,
    cortes_por_valles,
    franja_de_borde,
    regiones_de_analisis,
    zonas_por_gris,
)

# Poblaciones dentro de una región
from .poblaciones import actividad, normalizar_por_fila, nucleos_y_pozos, paisaje, poblaciones_de_region

# Cadena completa
from .segmentacion import procesar_region, segmentar_roi

__all__ = [
    "actividad",
    "actividad_sin_cruzar",
    "aislar_region",
    "cortes_por_valles",
    "estimar_degrade",
    "filtro_choque",
    "filtro_choque_con_pozos",
    "franja_de_borde",
    "normalizar_por_fila",
    "nucleos_y_pozos",
    "paisaje",
    "poblaciones_de_region",
    "procesar_region",
    "realzar",
    "regiones_de_analisis",
    "segmentar_roi",
    "zonas_por_gris",
]
