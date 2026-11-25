"""Interfaz gráfica (Tkinter) del Analizador de Imágenes de Microscopía.

Se abre con `uv run python main.py`. Ver `abrir_interfaz`.
"""

from .ventana import Ventana, abrir_interfaz

__all__ = ["Ventana", "abrir_interfaz"]
