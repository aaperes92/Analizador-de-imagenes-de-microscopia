"""ROI 1 real: paisaje gris-actividad con distintas sensibilidades de la actividad.

actividad = gaussiana(sobel(suavizada), sigma), con sigma = 25, 12, 6, 3 y 1 px (más chico = más
local = más sensible), llevada a [0, 1] por mínimo y máximo. Paisaje = log(1 + histograma 2D) de
128 x 128, eje del gris entre percentiles 0.1 y 99.9. Por fila: recorte del mapa de actividad
(200 x 200 px, para ver el detalle), mapa completo y paisaje. Solo mirar.
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

SIGMAS = (25, 12, 6, 3, 1)
BINS = 128
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])

suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
bajo, alto = np.percentile(suave, (0.1, 99.9))
gradiente = ski.filters.sobel(suave)

fig, ejes = plt.subplots(len(SIGMAS) + 1, 3, figsize=(27, 8.2 * (len(SIGMAS) + 1)))
ejes[0, 0].imshow(suave[150:350, 150:350], cmap="gray", vmin=bajo, vmax=alto)
ejes[0, 0].set_title("ROI 1 suavizada (recorte 200 x 200 px)", fontsize=20)
ejes[0, 1].imshow(suave, cmap="gray", vmin=bajo, vmax=alto)
ejes[0, 1].set_title("ROI 1 suavizada (completa)", fontsize=20)
for ax in ejes[0, :2]:
    ax.set_axis_off()
ax = ejes[0, 2]
ax.hist(suave.ravel(), bins=BINS, range=(bajo, alto), color="0.4")
ax.set_xlabel("gris", fontsize=16)
ax.set_title("histograma de gris (lineal)", fontsize=20)

for fila, sigma in enumerate(SIGMAS, start=1):
    actividad = ski.filters.gaussian(gradiente, sigma)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), suave.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    ejes[fila, 0].imshow(actividad[150:350, 150:350], cmap="magma")
    ejes[fila, 0].set_title(f"actividad σ = {sigma} px (recorte)", fontsize=20)
    im = ejes[fila, 1].imshow(actividad, cmap="magma")
    ejes[fila, 1].set_title(f"actividad σ = {sigma} px (completa)", fontsize=20)
    fig.colorbar(im, ax=ejes[fila, 1], fraction=0.046)
    for ax in ejes[fila, :2]:
        ax.set_axis_off()
    ax = ejes[fila, 2]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + píxeles)")
    ax.set_xlabel("gris", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title(f"paisaje gris-actividad, σ = {sigma} px", fontsize=20)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/44_ROI1_sensibilidad_actividad.png"), dpi=32)
print("ok")
