"""Analizador de imágenes de microscopía (MOKE, SEM y TEM).

Abre la interfaz gráfica: se elige la técnica, la entrada y las opciones, y se analiza.

    uv run python main.py
"""

import sys
from pathlib import Path

# Permite importar los paquetes moke, sem, tem e interfaz desde src/
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from interfaz import abrir_interfaz  # noqa: E402

if __name__ == "__main__":
    abrir_interfaz()
