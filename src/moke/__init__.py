"""Módulo MOKE del Analizador de Imágenes de Microscopía.

Detección de dominios magnéticos en secuencias de imágenes Kerr, con una imagen de fondo.
"""

from .moke import analizadorMOKE, detectar_dominio, leer_gris, mostrar_dominio

__all__ = ["analizadorMOKE", "detectar_dominio", "leer_gris", "mostrar_dominio"]
