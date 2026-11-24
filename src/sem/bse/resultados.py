"""Resultados de la segmentación BSE: imagen coloreada y archivos junto a la ROI.

Segmenta una ROI guardada por la carga y deja, en la misma carpeta:

    segmentacion_<nombre>.npy   etiqueta de cada píxel (una por par región-población)
    segmentacion_<nombre>.csv   tabla: región, población, gris central y fracciones
    segmentacion_<nombre>.png   ROI suavizada y segmentada, lado a lado

Es la lógica de `scripts/sem_bse_segmentacion.py`, movida acá para usarla también desde la
interfaz gráfica. El script sigue funcionando igual.

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .segmentacion import segmentar_roi

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


def segmentar_y_guardar(ruta_roi: Path, intensidad_pozos: float = 2.0) -> Path:
    """
    Segmenta una ROI BSE guardada (.npy) y guarda la etiqueta, la tabla y la figura a su lado.

    Imprime un resumen por región y por población, igual que el script de segmentación.

    Parameters
    ----------
    ruta_roi : Path
        ROI guardada por la carga (`roi_<nombre>.npy`).
    intensidad_pozos : float
        Peso del término de pozos del filtro de choque (0 = choque solo; por defecto 2).

    Returns
    -------
    Path
        Carpeta donde quedaron segmentacion_<nombre>.npy, .csv y .png (la de la ROI).

    Raises
    ------
    FileNotFoundError
        Si no existe la ROI.
    """
    ruta_roi = Path(ruta_roi)
    if not ruta_roi.is_file():
        raise FileNotFoundError(f"No existe la ROI: {ruta_roi}")
    roi = np.load(ruta_roi, allow_pickle=False)
    print(f"ROI {ruta_roi.name}: {roi.shape[1]} × {roi.shape[0]} px. Segmentando (BSE)…")
    segmentacion = segmentar_roi(roi, intensidad_pozos=intensidad_pozos)

    for r, (descripcion, resultado) in enumerate(zip(segmentacion["descripcion"], segmentacion["por_region"])):
        print(f"  región {r} ({descripcion}): degradé {resultado['degrade']:.2f} px "
              f"({resultado['bordes_usados']} bordes), choque {resultado['pasos_choque']} pasos, "
              f"{len(resultado['centrales'])} población(es)")
    for fila in segmentacion["tabla"]:
        print(f"    región {fila['region']} · población {fila['poblacion']}: gris {fila['gris_central']:.3f}, "
              f"{fila['fraccion_region']:.1%} de la región, {fila['fraccion_roi']:.1%} de la ROI")

    nombre = ruta_roi.stem
    carpeta = ruta_roi.parent
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
    return carpeta
