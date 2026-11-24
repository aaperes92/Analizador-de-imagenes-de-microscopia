"""Comparación de ecualizaciones sobre la ROI suavizada: estiramiento global, global y CLAHE.

Una figura por ROI: fila 1 = imágenes (todas en la escala 0-1), fila 2 = histogramas.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/ecualizacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/ecualizacion")
SALIDA.mkdir(parents=True, exist_ok=True)

for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    p_bajo, p_alto = np.percentile(suave, (0.5, 99.5))
    versiones = {
        "suavizada (sin ecualizar)": suave,
        f"a) estiramiento global\n(percentiles 0.5-99.5: {p_bajo:.2f}-{p_alto:.2f})":
            ski.exposure.rescale_intensity(suave, in_range=(p_bajo, p_alto), out_range=(0.0, 1.0)),
        "b) ecualización global\n(equalize_hist)": ski.exposure.equalize_hist(suave),
        "c) CLAHE, valores por defecto\n(regiones de 1/8 del lado, clip_limit=0.01)":
            ski.exposure.equalize_adapthist(suave),
    }

    fig, ejes = plt.subplots(2, 4, figsize=(24, 11), gridspec_kw={"height_ratios": [1.4, 1]})
    print(f"ROI_{k}")
    for j, (titulo, img) in enumerate(versiones.items()):
        ejes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        ejes[0, j].set_title(f"ROI {k} — {titulo}" if j == 0 else titulo, fontsize=14)
        ejes[0, j].set_axis_off()
        ax = ejes[1, j]
        ax.hist(img.ravel(), bins=256, range=(0, 1), color="0.3")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("intensidad", fontsize=12)
        p1, p99 = np.percentile(img, (1, 99))
        ax.set_title(f"rango usado (percentiles 1-99): {p1:.2f} - {p99:.2f}", fontsize=12)
        print(f"  {titulo.splitlines()[0]:<30} percentiles 1-99 = {p1:.3f} - {p99:.3f}")
    fig.tight_layout()
    fig.savefig(SALIDA / f"01_comparacion_ecualizaciones_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
