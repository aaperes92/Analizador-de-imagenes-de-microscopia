"""Carga SEM: lee una imagen, muestra su diagnóstico y guarda la ROI seleccionada.

Pasos: carga la imagen y su metadata, quita la barra de datos, muestra la imagen con su
histograma, abre el selector de ROI y guarda la ROI (.npy, .tif y .json).

Cómo usarlo (desde la raíz del proyecto):

    uv run python scripts/sem_carga.py <ruta_imagen> [--salida CARPETA] [--sufijo SUFIJO]

Ejemplo:

    uv run python scripts/sem_carga.py imagenes/SEM/BSE/BSE_1.tif

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Permite importar el paquete sem desde src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from sem.carga import (  # noqa: E402
    cargar_imagen,
    guardar_roi,
    leer_metadata,
    mostrar_imagen_con_histograma,
    recortar_barra_datos,
    seleccionar_roi,
)

# Carpeta por defecto donde se guardan las ROIs (data/ no se versiona en git)
CARPETA_ROI: Path = RAIZ / "data" / "sem" / "roi"


def main() -> int:
    """Ejecuta la carga sobre la imagen pasada por línea de comandos. Retorna el código de salida."""
    parser = argparse.ArgumentParser(description="SEM: carga, metadata y selección de ROI.")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen SEM (.tif)")
    parser.add_argument(
        "--salida", type=Path, default=CARPETA_ROI,
        help="Carpeta raíz donde guardar las ROIs (por defecto: data/sem/roi)",
    )
    parser.add_argument(
        "--sufijo", default=None,
        help='Sufijo del ROI, ej. "_1". Por defecto se usa el primer "_N" libre.',
    )
    args = parser.parse_args()
    # Evita errores de codificación al imprimir acentos/emojis en consolas Windows redirigidas.
    sys.stdout.reconfigure(errors="replace")

    # 1. Cargar imagen, leer metadata y quitar la barra de datos.
    try:
        imagen_completa = cargar_imagen(args.imagen)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1
    metadata = leer_metadata(args.imagen)
    imagen = recortar_barra_datos(imagen_completa, metadata)

    print(f"Imagen: {args.imagen.name}")
    print(f"  Fabricante: {metadata.get('fabricante', 'desconocido')}")
    print(f"  Shape original: {imagen_completa.shape} -> sin barra de datos: {imagen.shape}")
    if "tamano_pixel_m" in metadata:
        print(f"  Tamaño de píxel: {metadata['tamano_pixel_m'] * 1e9:.3g} nm")

    # 2. Visualización diagnóstica (cerrar la ventana para continuar).
    mostrar_imagen_con_histograma(imagen, titulo=args.imagen.name, metadata=metadata)
    print("Cerrá la ventana del histograma para pasar a la selección del ROI...")
    plt.show()

    # 3. Selección interactiva del ROI.
    print("Seleccioná el ROI y presioná Enter (o cerrá la ventana) para confirmar.")
    coordenadas = seleccionar_roi(imagen, titulo=args.imagen.name)
    if coordenadas is None:
        print("⚠️ No se seleccionó ningún ROI; no se guardó nada.")
        return 0
    print(f"  ROI (x1, y1, x2, y2): {coordenadas}")

    # 4. Guardado.
    try:
        carpeta = guardar_roi(
            imagen, args.imagen.name, coordenadas, args.salida, metadata, sufijo=args.sufijo
        )
    except (ValueError, FileExistsError) as error:
        print(f"❌ {error}")
        return 1
    print(f"✅ ROI guardada en: {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
