"""Ecualización GLOBAL con límite de recorte: CLAHE con una sola región del tamaño de la ROI.

Mismos clip_limit que la prueba de CLAHE, más equalize_hist (global sin límite) como referencia.
Una figura por ROI: fila 1 = imágenes (escala 0-1), fila 2 = histogramas.
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
CLIPS = (0.002, 0.005, 0.01, 0.02)

for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    versiones = {"suavizada (sin ecualizar)": suave}
    for clip in CLIPS:
        versiones[f"global, clip_limit = {clip}\n(una sola región = toda la ROI)"] = \
            ski.exposure.equalize_adapthist(suave, kernel_size=suave.shape, clip_limit=clip)
    versiones["global sin límite\n(equalize_hist)"] = ski.exposure.equalize_hist(suave)

    fig, ejes = plt.subplots(2, len(versiones), figsize=(5.2 * len(versiones), 10.5),
                             gridspec_kw={"height_ratios": [1.4, 1]})
    print(f"ROI_{k}")
    for j, (titulo, img) in enumerate(versiones.items()):
        ejes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        ejes[0, j].set_title(f"ROI {k} — {titulo}" if j == 0 else titulo, fontsize=13)
        ejes[0, j].set_axis_off()
        ax = ejes[1, j]
        ax.hist(img.ravel(), bins=256, range=(0, 1), color="0.3")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("intensidad", fontsize=12)
        p1, p99 = np.percentile(img, (1, 99))
        ax.set_title(f"rango usado (percentiles 1-99): {p1:.2f} - {p99:.2f}", fontsize=11)
        print(f"  {titulo.splitlines()[0]:<32} percentiles 1-99 = {p1:.3f} - {p99:.3f}")
    fig.tight_layout()
    fig.savefig(SALIDA / f"04_global_clip_limit_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
