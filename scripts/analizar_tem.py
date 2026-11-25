"""Análisis TEM completo: preprocesamiento, segmentación, morfología y
análisis de fases, o análisis de difracción.

Corre las cuatro etapas del módulo TEM sobre una imagen o sobre todas
las imágenes de una carpeta. Antes revisa cada imagen: si es un patrón
de difracción (o se pasa --difraccion), corre la Etapa 4.5 y saltea las
Etapas 2 a 4.

Cómo usarlo (desde la raíz del proyecto):

    1. Editá la sección CONFIGURACIÓN de abajo (la ruta y las opciones).
    2. Corré:
           uv run python scripts/analizar_tem.py
       o pasá la ruta directamente, sin editar el archivo:
           uv run python scripts/analizar_tem.py "C:\\ruta\\imagen.dm3"
       y agregá --difraccion para analizarla como patrón de difracción
       sin detectarlo:
           uv run python scripts/analizar_tem.py "RUTA" --difraccion

Con una imagen: muestra en pantalla las figuras de cada etapa (cerrá
cada ventana para pasar a la siguiente) e imprime todos los
resultados.
Con una carpeta: procesa todas las imágenes sin abrir ventanas, imprime
una tabla y guarda un resumen en CSV (los patrones de difracción van en
un CSV aparte, <carpeta>_resumen_difraccion.csv).

Todas las figuras se guardan en data/samples/ (o en CARPETA_SALIDA).

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ====================================================================
# CONFIGURACIÓN — editá estos valores
# ====================================================================

# Imagen o carpeta a analizar (.dm3, .mrc, .tif, .tiff, .png, .jpg)
RUTA = (
    r"C:\Users\Daniel\Downloads\TEM-20261116T174039Z-1-001\TEM"
    r"\campo-claro-HR\prueba1.dm3"
)

# Etapa 1 — filtro de ruido:
#   'auto', 'gaussian', 'median', 'bilateral' o 'wiener'
FILTRO = "auto"

# Etapa 2 — método de umbralización:
#   'auto', 'otsu', 'gmm', 'triangle' (campo oscuro) o 'adaptive'
UMBRAL = "auto"

# Etapa 3 — fases a medir: None = todas menos la mayoritaria (fondo).
#   Ejemplos: [0] o [1, 2]
FASES = None

# Etapas 3 y 4 — etiqueta de la máscara que es el fondo: None = la fase
#   mayoritaria. Si se indica, la Etapa 3 mide todas las demás (FASES
#   se ignora). Ejemplo: 0 en S041C_0025.dm3, donde las partículas
#   ocupan más de la mitad de la imagen.
FONDO = None

# Etapa 4.5 — patrones de difracción:
#   None = detecta sola cuáles imágenes son de difracción,
#   True = analiza todas como difracción (igual que --difraccion),
#   False = nunca (siempre Etapas 2-4)
DIFRACCION = None

# Etapa 4.5 — ancho instrumental (px) que se resta en Scherrer
#   (0 = no se resta y los tamaños son mínimos)
ANCHO_INSTRUMENTAL = 0.0

# Mostrar las figuras en pantalla (solo con una imagen)
MOSTRAR_FIGURAS = True

# Carpeta donde guardar figuras y CSV: None = data/samples/
CARPETA_SALIDA = None

# ====================================================================
# (No hace falta tocar nada de acá para abajo)
# ====================================================================

# Permite importar el paquete tem desde src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

# La lógica está en src/tem/analisis_completo.py (la usa también la interfaz gráfica).
from tem.analisis_completo import analizar_ruta  # noqa: E402


def main() -> None:
    """Analiza la imagen o la carpeta configurada."""
    args = [a for a in sys.argv[1:] if a != "--difraccion"]
    difraccion = True if "--difraccion" in sys.argv[1:] else DIFRACCION
    ruta = Path(args[0]) if args else Path(RUTA)
    if not ruta.exists():
        print(f"No existe: {ruta}\nRevisá RUTA en la configuración.")
        sys.exit(1)
    try:
        analizar_ruta(ruta, filtro=FILTRO, umbral=UMBRAL, fases=FASES, fondo=FONDO,
                      difraccion=difraccion, ancho_instrumental=ANCHO_INSTRUMENTAL,
                      mostrar=MOSTRAR_FIGURAS, carpeta_salida=CARPETA_SALIDA)
    except ValueError as error:  # carpeta sin imágenes soportadas
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
