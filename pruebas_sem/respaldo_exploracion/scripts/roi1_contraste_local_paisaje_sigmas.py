"""ROI 1 real: contraste local del paisaje gris-actividad para distintas sensibilidades.

Para sigma de actividad = 25, 12, 6, 3 y 1 px:
- paisaje = log(1 + histograma 2D) de 256 x 256 (eje del gris entre percentiles 0.1 y 99.9);
- contraste local del paisaje = paisaje - gaussiana(paisaje, 4 bins), solo donde hay píxeles
  (rojo = cresta, más píxeles que alrededor; azul = valle, menos píxeles que alrededor).
Solo mirar.
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
BINS = 256
SIGMA_PAISAJE = 4
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])

suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
bajo, alto = np.percentile(suave, (0.1, 99.9))
gradiente = ski.filters.sobel(suave)

fig, ejes = plt.subplots(len(SIGMAS), 2, figsize=(22, 8.4 * len(SIGMAS)))
for fila, sigma in enumerate(SIGMAS):
    actividad = ski.filters.gaussian(gradiente, sigma)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), suave.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    paisaje = np.log1p(conteos)
    local = paisaje - ski.filters.gaussian(paisaje, SIGMA_PAISAJE, preserve_range=True)
    hay = conteos > 0
    lim = np.percentile(np.abs(local[hay]), 99)
    extent = (bajo, alto, 0, 1)

    ax = ejes[fila, 0]
    im = ax.imshow(paisaje, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + píxeles)")
    ax.set_title(f"paisaje gris-actividad, actividad σ = {sigma} px", fontsize=19)
    ax = ejes[fila, 1]
    im = ax.imshow(np.where(hay, local, np.nan), origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-lim, vmax=lim, extent=extent)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"contraste local del paisaje, σ = {sigma} px\n(rojo = cresta, azul = valle)", fontsize=19)
    for ax in ejes[fila]:
        ax.set_xlabel("gris", fontsize=16)
        ax.set_ylabel("actividad (0-1)", fontsize=16)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/47_ROI1_contraste_local_paisaje_por_sigma.png"), dpi=34)
print("ok")
