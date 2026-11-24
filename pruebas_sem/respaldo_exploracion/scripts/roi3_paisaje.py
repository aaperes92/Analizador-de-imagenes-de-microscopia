"""ROI 3: imagen suavizada, mapa de actividad y paisaje gris-actividad. Nada más.

actividad = gaussiana(sobel(suavizada), 25 px). Gris y actividad llevados a [0, 1] (mín-máx).
Paisaje = histograma 2D de 128 x 128 (cantidad de píxeles con cada par gris-actividad), en
escala logarítmica para ver también las zonas con pocos píxeles.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SIGMA_ACTIVIDAD = 25
BINS = 128


def escalar_min_max(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


roi = np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")
suave = suavizar(roi)
gris = escalar_min_max(suave)
actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))

fig, ejes = plt.subplots(1, 3, figsize=(25, 8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
ejes[0].imshow(gris, cmap="gray")
ejes[0].set_title("ROI 3 suavizada (gris)", fontsize=16)
ejes[0].set_axis_off()

im = ejes[1].imshow(actividad, cmap="gray")
ejes[1].set_title(f"mapa de actividad (σ = {SIGMA_ACTIVIDAD} px)\nclaro = muchos bordes alrededor", fontsize=16)
ejes[1].set_axis_off()
fig.colorbar(im, ax=ejes[1], fraction=0.046, label="actividad (0-1)")

ax = ejes[2]
_, _, _, im = ax.hist2d(gris.ravel(), actividad.ravel(), bins=BINS, range=((0, 1), (0, 1)),
                        cmap="Greys", norm="log")
fig.colorbar(im, ax=ax, fraction=0.046, label="cantidad de píxeles (escala log)")
ax.set_xlabel("gris (0-1)", fontsize=14)
ax.set_ylabel("actividad (0-1)", fontsize=14)
ax.set_title("paisaje gris-actividad\n(cada píxel es un punto; más oscuro = más píxeles)", fontsize=16)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/18_ROI3_suavizada_actividad_paisaje.png"), dpi=55)
print("ok")
