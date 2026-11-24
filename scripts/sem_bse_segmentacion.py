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
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# Permite importar el paquete sem desde src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from sem.bse import segmentar_roi  # noqa: E402

# Un color por región; dentro, la población mayoritaria clara y las demás oscuras
COLORES_REGION = ["#1f3fbf", "#e6a100", "#12a35a", "#c2185b", "#00acc1", "#6d4c41", "#7cb342", "#5e35b1"]


def colorear(segmentacion: dict) -> tuple[np.ndarray, list[Patch]]:
    """Imagen RGB de la segmentación y su leyenda (color = región; claro = población mayoritaria)."""
    rgb = np.zeros((*segmentacion["etiquetas"].shape, 3))
    leyenda = []
    for r, resultado in enumerate(segmentacion["por_region"]):
        base = np.array(to_rgb(COLORES_REGION[r % len(COLORES_REGION)]))
        mayor = int(np.argmax(resultado["fracciones"]))
        menores = [p for p in range(len(resultado["fracciones"])) if p != mayor]
        for fila in (f for f in segmentacion["tabla"] if f["region"] == r):
            p = fila["poblacion"]
            if p == mayor:
                color = base + (1 - base) * 0.35
            else:
                color = base * (0.55 - 0.4 * menores.index(p) / max(len(menores), 1))
            rgb[segmentacion["etiquetas"] == fila["etiqueta"]] = color
            leyenda.append(Patch(color=color, label=f"región {r}: gris {fila['gris_central']:.3f} "
                                                    f"({fila['fraccion_region']:.0%} de la región)"))
    return rgb, leyenda


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
    roi = np.load(args.roi, allow_pickle=False)
    print(f"ROI {args.roi.name}: {roi.shape[1]} × {roi.shape[0]} px. Segmentando (BSE)…")
    segmentacion = segmentar_roi(roi, intensidad_pozos=args.intensidad_pozos)

    for r, (descripcion, resultado) in enumerate(zip(segmentacion["descripcion"], segmentacion["por_region"])):
        print(f"  región {r} ({descripcion}): degradé {resultado['degrade']:.2f} px "
              f"({resultado['bordes_usados']} bordes), choque {resultado['pasos_choque']} pasos, "
              f"{len(resultado['centrales'])} población(es)")
    for fila in segmentacion["tabla"]:
        print(f"    región {fila['region']} · población {fila['poblacion']}: gris {fila['gris_central']:.3f}, "
              f"{fila['fraccion_region']:.1%} de la región, {fila['fraccion_roi']:.1%} de la ROI")

    nombre = args.roi.stem
    carpeta = args.roi.parent
    np.save(carpeta / f"segmentacion_{nombre}.npy", segmentacion["etiquetas"])
    with open(carpeta / f"segmentacion_{nombre}.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(segmentacion["tabla"][0]))
        escritor.writeheader()
        escritor.writerows(segmentacion["tabla"])

    rgb, leyenda = colorear(segmentacion)
    lo, hi = np.percentile(segmentacion["suavizada"], (0.5, 99.5))
    fig, ejes = plt.subplots(1, 2, figsize=(16, 8))
    ejes[0].imshow(segmentacion["suavizada"], cmap="gray", vmin=lo, vmax=hi)
    ejes[0].set_title(f"{nombre} suavizada")
    ejes[1].imshow(rgb, interpolation="nearest")
    ejes[1].set_title("segmentada: color = región; claro = población mayoritaria")
    for ax in ejes:
        ax.set_axis_off()
    fig.legend(handles=leyenda, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(carpeta / f"segmentacion_{nombre}.png", dpi=120)
    plt.close(fig)
    print(f"✅ Guardado en {carpeta}: segmentacion_{nombre}.npy, .csv y .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
