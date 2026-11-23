"""Módulo SEM del Analizador de Imágenes de Microscopía."""

# Carga: lectura, metadata, barra de datos y ROI
from .carga import (
    cargar_imagen,
    guardar_roi,
    leer_metadata,
    mostrar_imagen_con_histograma,
    recortar_barra_datos,
    seleccionar_roi,
)

# Suavizado: estimación del ruido y non-local means
from .suavizado import buscar_factor_h, estimar_ruido, suavizar

__all__ = [
    "buscar_factor_h",
    "cargar_imagen",
    "estimar_ruido",
    "guardar_roi",
    "leer_metadata",
    "mostrar_imagen_con_histograma",
    "recortar_barra_datos",
    "seleccionar_roi",
    "suavizar",
]
