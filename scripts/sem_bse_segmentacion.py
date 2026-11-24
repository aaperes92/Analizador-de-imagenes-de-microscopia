"""Segmentación SEM BSE: separa una ROI de electrones retrodispersados en regiones y poblaciones.

Solo para imágenes BSE. Pasos (todos en `sem.bse.segmentar_roi`): suavizado, regiones de análisis
y, en cada región aislada, realce, filtro de choque con pozos, paisaje gris × actividad, núcleos y
pozos. Guarda, junto a la ROI:

    segmentacion_<nombre>.npy   etiqueta de cada píxel (una por par región-población)
    segmentacion_<nombre>.csv   tabla: región, población, gris central y fracciones
    segmentacion_<nombre>.png   ROI suavizada y segmentada, lado a lado

Cómo usarlo (desde la raíz del proyecto):

    uv run python scripts/sem_bse_segmentacion.py <ruta_roi.npy> [--intensidad-pozos 2]

Ejemplo:

    uv run python scripts/sem_bse_segmentacion.py data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete sem desde src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

# La lógica está en src/sem/bse/resultados.py (la usa también la interfaz gráfica).
from sem.bse.resultados import colorear, segmentar_y_guardar  # noqa: E402, F401


def main() -> int:
    """Segmenta la ROI pasada por línea de comandos. Retorna el código de salida."""
    parser = argparse.ArgumentParser(description="SEM BSE: segmentación en regiones y poblaciones.")
    parser.add_argument("roi", type=Path, help="ROI guardada por la carga (.npy)")
    parser.add_argument("--intensidad-pozos", type=float, default=2.0,
                        help="Peso del término de pozos del filtro de choque (0 = choque solo; por defecto 2)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if not args.roi.is_file():
        print(f"❌ No existe la ROI: {args.roi}")
        return 1
    segmentar_y_guardar(args.roi, intensidad_pozos=args.intensidad_pozos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
