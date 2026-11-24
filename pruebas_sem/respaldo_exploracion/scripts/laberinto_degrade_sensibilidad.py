"""Laberinto + matriz (con degradé de 1.5 px): paisaje gris-actividad con distintas sensibilidades.

Mismo patrón que 43_ (laberinto conectado 0.58 sobre matriz 0.70, ruido 0.003, sin degradé).
actividad = gaussiana(sobel, sigma) con sigma = 25, 12, 6, 3 y 1 px, a [0, 1].
El paisaje se colorea por población verdadera (azul = laberinto, naranja = matriz).
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage

ALTO, ANCHO = 520, 560
GRISES = np.array([0.58, 0.70])
COLORES = ["#1f77b4", "#ff7f0e"]
SIGMAS = (25, 12, 6, 3, 1)
BINS = 128


def laberinto(semilla=11):
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * max(ALTO, ANCHO))
    campo = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 8)), 60,
                           reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    campo = campo[y0:y0 + ALTO, x0:x0 + ANCHO]
    semillas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=4)
    semillas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, semillas, watershed_line=True)
    paredes = ndimage.binary_dilation(islas == 0, structure=ski.morphology.disk(1))
    return np.where(paredes, 0, 1), rng


def componer(capas):
    maximo = max(c.max() for c in capas) or 1
    rgb = np.ones(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        rgb -= (c / maximo)[..., None] * (1 - np.array(to_rgb(color)))
    return np.clip(rgb, 0, 1)


etiquetas, rng = laberinto()
img = ski.filters.gaussian(GRISES[etiquetas], 1.5) + rng.normal(0, 0.003, etiquetas.shape)
bajo, alto = np.percentile(img, (0.1, 99.9))
gradiente = ski.filters.sobel(img)

fig, ejes = plt.subplots(len(SIGMAS) + 1, 3, figsize=(27, 8.2 * (len(SIGMAS) + 1)))
ejes[0, 0].imshow(img[150:350, 150:350], cmap="gray", vmin=bajo, vmax=alto)
ejes[0, 0].set_title("laberinto + matriz, con degradé σ=1.5px (recorte 200 x 200 px)", fontsize=20)
ejes[0, 1].imshow(etiquetas, cmap=ListedColormap(COLORES), vmin=-0.5, vmax=1.5, interpolation="nearest")
ejes[0, 1].set_title("población verdadera", fontsize=20)
for ax in ejes[0, :2]:
    ax.set_axis_off()
ax = ejes[0, 2]
ax.hist(img.ravel(), bins=BINS, range=(bajo, alto), color="0.4")
ax.set_xlabel("gris", fontsize=16)
ax.set_title("histograma de gris (lineal)", fontsize=20)

for fila, sigma in enumerate(SIGMAS, start=1):
    actividad = ski.filters.gaussian(gradiente, sigma)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    capas = [np.log1p(np.histogram2d(actividad[etiquetas == p], img[etiquetas == p], bins=BINS,
                                     range=((0, 1), (bajo, alto)))[0]) for p in range(2)]
    ejes[fila, 0].imshow(actividad[150:350, 150:350], cmap="magma")
    ejes[fila, 0].set_title(f"actividad σ = {sigma} px (recorte)", fontsize=20)
    im = ejes[fila, 1].imshow(actividad, cmap="magma")
    ejes[fila, 1].set_title(f"actividad σ = {sigma} px (completa)", fontsize=20)
    fig.colorbar(im, ax=ejes[fila, 1], fraction=0.046)
    for ax in ejes[fila, :2]:
        ax.set_axis_off()
    ax = ejes[fila, 2]
    ax.imshow(componer(capas), origin="lower", aspect="auto", extent=(bajo, alto, 0, 1))
    ax.set_xlabel("gris", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title(f"paisaje gris-actividad, σ = {sigma} px", fontsize=20)

fig.legend(handles=[Patch(color=COLORES[0], label="laberinto (0.58)"), Patch(color=COLORES[1], label="matriz (0.70)")],
           loc="upper center", ncol=2, fontsize=22, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(Path("pruebas_sem/segmentacion/46_laberinto_degrade_sensibilidad_actividad.png"), dpi=32)
print("ok")
