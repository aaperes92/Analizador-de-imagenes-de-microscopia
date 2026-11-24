"""Realce del PROPIO paisaje gris-actividad (tratado como imagen), ROI 1.

Paisaje = log(1 + histograma 2D) de 256 x 256 (filas = actividad, columnas = gris; eje del gris
entre percentiles 0.1 y 99.9). Imágenes: ROI 1 suavizada y ROI 1 con blur_resta (fracción 0.03).
Realces del paisaje (solo donde hay píxeles):
1. CLAHE: skimage.exposure.equalize_adapthist sobre el paisaje llevado a [0, 1] (clip 0.01).
2. Contraste local: paisaje - gaussiana(paisaje, 4 bins).
3. Crestas: skimage.filters.sato(paisaje, sigmas 1-3 bins, crestas claras).
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
BINS = 256
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
CMAP_DIVERGENTE = "RdBu_r"

suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
resta = suave - ski.filters.gaussian(suave, 0.03 * np.sqrt(suave.size) / np.sqrt(12), preserve_range=True)
casos = {"ROI 1 suavizada": suave, "ROI 1 con blur_resta": resta}

fig, ejes = plt.subplots(len(casos), 4, figsize=(36, 8.6 * len(casos)))
for fila, (nombre, img) in enumerate(casos.items()):
    bajo, alto = np.percentile(img, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    paisaje = np.log1p(conteos)
    hay = conteos > 0
    extent = (bajo, alto, 0, 1)

    clahe = ski.exposure.equalize_adapthist(paisaje / paisaje.max(), clip_limit=0.01)
    local = paisaje - ski.filters.gaussian(paisaje, 4, preserve_range=True)
    crestas = ski.filters.sato(paisaje, sigmas=(1, 2, 3), black_ridges=False)

    paneles = [
        (paisaje, CMAP_PAISAJE, None, "paisaje ORIGINAL (log, 256 x 256)"),
        (np.where(hay, clahe, np.nan), CMAP_PAISAJE, None, "1) CLAHE sobre el paisaje"),
        (np.where(hay, local, np.nan), CMAP_DIVERGENTE, np.nanpercentile(np.abs(local[hay]), 99),
         "2) contraste local del paisaje\n(rojo = cresta, azul = valle)"),
        (np.where(hay, crestas, np.nan), "magma", None, "3) filtro de crestas (sato) sobre el paisaje"),
    ]
    for col, (datos, cmap, lim, titulo) in enumerate(paneles):
        ax = ejes[fila, col]
        if lim is None:
            im = ax.imshow(datos, origin="lower", aspect="auto", cmap=cmap, extent=extent)
        else:
            im = ax.imshow(datos, origin="lower", aspect="auto", cmap=cmap, extent=extent, vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("gris", fontsize=16)
        ax.set_ylabel("actividad (0-1)", fontsize=16)
        ax.set_title(f"{nombre}\n{titulo}" if col == 0 else titulo, fontsize=18)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/42_ROI1_realce_del_paisaje.png"), dpi=38)
print("ok")
