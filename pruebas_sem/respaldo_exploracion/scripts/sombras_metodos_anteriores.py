"""Corrección de sombras con los dos métodos de la versión anterior (commit fb9ac2c).

- blur_resta: fondo = gaussiano con sigma = fraccion * sqrt(alto*ancho) / sqrt(12), fraccion 0.03;
  corregida = imagen - fondo.
- white_tophat: imagen - apertura con un disco de radio 50 px (descomposición en cruces).
Una figura por ROI: fila 1 = imágenes, fila 2 = histogramas.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/sombras")
FRACCION = 0.03
RADIO = 50


def blur_resta(imagen, fraccion=FRACCION):
    sigma_fondo = fraccion * np.sqrt(imagen.size) / np.sqrt(12)
    fondo = ski.filters.gaussian(imagen, sigma=sigma_fondo, preserve_range=True)
    return (imagen - fondo).astype(np.float32), sigma_fondo


def white_tophat(imagen, radio=RADIO):
    disco = ski.morphology.disk(radio, decomposition="crosses")
    return ski.morphology.white_tophat(imagen, disco).astype(np.float32)


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    resta, sigma_fondo = blur_resta(suave)
    tophat = white_tophat(suave)
    versiones = {
        "suavizada (sin corregir)": suave,
        f"blur_resta (fracción {FRACCION}, σ fondo = {sigma_fondo:.1f} px)": resta,
        f"white_tophat (disco de radio {RADIO} px)": tophat,
    }

    fig, ejes = plt.subplots(2, 3, figsize=(20, 11.5), gridspec_kw={"height_ratios": [1.4, 1]})
    print(f"ROI_{k}")
    for j, (titulo, img) in enumerate(versiones.items()):
        lo, hi = np.percentile(img, (0.5, 99.5))
        ejes[0, j].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ejes[0, j].set_title(f"ROI {k} — {titulo}" if j == 0 else titulo, fontsize=13)
        ejes[0, j].set_axis_off()
        ax = ejes[1, j]
        ax.hist(img.ravel(), bins=256, range=(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)), color="0.3")
        ax.set_yscale("log")
        ax.set_xlabel("intensidad", fontsize=12)
        ax.set_title(f"histograma (rango mostrado: percentiles 0.5-99.5 = {lo:.3f} a {hi:.3f})", fontsize=11)
        print(f"  {titulo:<55} percentiles 0.5-99.5 = {lo:.3f} a {hi:.3f}")
    fig.tight_layout()
    fig.savefig(SALIDA / f"10_metodos_anteriores_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
