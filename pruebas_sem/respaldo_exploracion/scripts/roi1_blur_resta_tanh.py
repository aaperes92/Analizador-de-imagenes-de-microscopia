"""ROI 1: blur_resta seguido de tangente hiperbólica, con su paisaje gris-actividad.

blur_resta igual que en 40_ (fracción 0.03, sigma del fondo 4.7 px).
tanh como la normalizar_tanh de la versión anterior del módulo:
    salida = tanh(factor * (x - mediana) / max|x - mediana|)
con factor 1, 2 y 4 (2 era el del ejemplo de la documentación anterior).
Actividad = gaussiana(sobel, 25 px) calculada sobre cada imagen. Eje del gris entre percentiles
0.1 y 99.9. Solo mirar.
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


def tanh_normalizada(x, factor):
    centrada = x - np.median(x)
    return np.tanh(factor * centrada / (np.max(np.abs(centrada)) + 1e-8))


suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
resta = suave - ski.filters.gaussian(suave, 0.03 * np.sqrt(suave.size) / np.sqrt(12), preserve_range=True)
casos = {"blur_resta (referencia)": resta}
for factor in (1, 2, 4):
    casos[f"blur_resta + tanh, factor {factor}"] = tanh_normalizada(resta, factor)

fig, ejes = plt.subplots(len(casos), 3, figsize=(27, 8.3 * len(casos)))
for fila, (nombre, img) in enumerate(casos.items()):
    bajo, alto = np.percentile(img, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))

    ejes[fila, 0].imshow(img, cmap="gray", vmin=bajo, vmax=alto)
    ejes[fila, 0].set_title(nombre, fontsize=19)
    ejes[fila, 0].set_axis_off()
    ax = ejes[fila, 1]
    ax.hist(img.ravel(), bins=BINS, range=(bajo, alto), color="0.4")
    ax.set_xlabel("gris", fontsize=16)
    ax.set_title("histograma de gris (lineal)", fontsize=19)
    ax = ejes[fila, 2]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + cantidad de píxeles)")
    ax.set_xlabel("gris", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("paisaje gris-actividad", fontsize=19)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/41_ROI1_blur_resta_tanh_paisaje.png"), dpi=38)
print("ok")
