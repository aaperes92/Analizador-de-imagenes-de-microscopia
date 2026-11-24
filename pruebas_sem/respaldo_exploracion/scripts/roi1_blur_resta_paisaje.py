"""ROI 1: paisaje gris-actividad de la imagen suavizada y de la suavizada con blur_resta.

blur_resta igual que en 10_metodos_anteriores (fracción 0.03): fondo = gaussiana con
sigma = 0.03 * sqrt(alto * ancho) / sqrt(12); resultado = imagen - fondo.
actividad = gaussiana(sobel, 25 px) a [0, 1]. Paisaje = log(1 + histograma 2D) de 128 x 128, con el
eje del gris entre los percentiles 0.1 y 99.9 de cada imagen. Solo mirar, sin detectar nada.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SIGMA_ACTIVIDAD = 25
BINS = 128
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])

suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
sigma_fondo = 0.03 * np.sqrt(suave.size) / np.sqrt(12)
resta = suave - ski.filters.gaussian(suave, sigma_fondo, preserve_range=True)
print(f"blur_resta: sigma del fondo = {sigma_fondo:.1f} px")

casos = {"ROI 1 suavizada": suave, f"ROI 1 con blur_resta (σ fondo = {sigma_fondo:.1f} px)": resta}
fig, ejes = plt.subplots(len(casos), 3, figsize=(27, 8.5 * len(casos)))
for fila, (nombre, img) in enumerate(casos.items()):
    bajo, alto = np.percentile(img, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))

    ejes[fila, 0].imshow(img, cmap="gray", vmin=bajo, vmax=alto)
    ejes[fila, 0].set_title(nombre, fontsize=19)
    ejes[fila, 0].set_axis_off()
    im = ejes[fila, 1].imshow(actividad, cmap="magma")
    ejes[fila, 1].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=19)
    ejes[fila, 1].set_axis_off()
    fig.colorbar(im, ax=ejes[fila, 1], fraction=0.046)
    ax = ejes[fila, 2]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + cantidad de píxeles)")
    ax.set_xlabel("gris", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("paisaje gris-actividad", fontsize=19)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/40_ROI1_suavizada_y_blur_resta_paisaje.png"), dpi=40)
print("ok")
