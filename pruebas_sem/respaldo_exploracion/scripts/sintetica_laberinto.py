"""Patrón tipo ROI 1 con DOS poblaciones: laberinto oscuro conectado sobre una matriz clara.

Campo aleatorio suavizado alargado (gaussiana de 2.5 x 7 px, rotada 60°), normalizado. El
laberinto son las líneas donde el campo cruza el cero (|campo| < UMBRAL): forman una red continua
que rodea islas alargadas, como los canales de la ROI 1. Matriz (islas) = el resto.
Grises: laberinto 0.58, matriz 0.70 (rango de la ROI 1). Versiones: nítida y con degradé
(gaussiana 1.5 px); ambas con ruido residual de 0.003.
Gráficos: imagen, población verdadera, histograma de gris, mapa de actividad y paisaje.
Arriba, la ROI 1 real para comparar.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import ndimage

sys.path.insert(0, "src")
from sem.suavizado import suavizar

ALTO, ANCHO = 520, 560
UMBRAL = 0.25
GRISES = np.array([0.58, 0.70])            # 0 = laberinto, 1 = matriz
SIGMA_ACTIVIDAD = 25
BINS = 128
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])


def laberinto(semilla=11):
    """Red conectada: líneas divisorias (watershed) entre las islas de un campo alargado, engrosadas."""
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * max(ALTO, ANCHO))
    campo = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 8)), 60,
                           reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    campo = campo[y0:y0 + ALTO, x0:x0 + ANCHO]
    # Cada máximo del campo es el centro de una isla; las líneas divisorias entre islas son el laberinto
    semillas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=4)
    semillas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, semillas, watershed_line=True)
    paredes = ski.morphology.binary_dilation(islas == 0, ski.morphology.disk(1))
    return np.where(paredes, 0, 1), rng


etiquetas, rng = laberinto()
red = etiquetas == 0
componentes, n_comp = ndimage.label(red, structure=np.ones((3, 3)))
tamanos = np.bincount(componentes.ravel())[1:]
espesor = 2 * np.median(ndimage.distance_transform_edt(red)[ski.morphology.skeletonize(red)])
print(f"laberinto: {red.mean():.1%} del área; {n_comp} pedazos, el más grande tiene "
      f"{tamanos.max() / red.sum():.0%} del laberinto; espesor típico {espesor:.1f} px")

nitida = GRISES[etiquetas] + rng.normal(0, 0.003, etiquetas.shape)
degrade = ski.filters.gaussian(GRISES[etiquetas], 1.5) + rng.normal(0, 0.003, etiquetas.shape)
real = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
casos = {"ROI 1 REAL (suavizada)": (real, None),
         "laberinto + matriz, nítido": (nitida, etiquetas),
         "laberinto + matriz, con degradé (σ=1.5px)": (degrade, etiquetas)}

fig, ejes = plt.subplots(len(casos), 5, figsize=(42, 8.4 * len(casos)))
for fila, (nombre, (img, verdad)) in enumerate(casos.items()):
    bajo, alto = np.percentile(img, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))

    ejes[fila, 0].imshow(img, cmap="gray", vmin=bajo, vmax=alto)
    ejes[fila, 0].set_title(nombre, fontsize=20)
    ejes[fila, 0].set_axis_off()
    ax = ejes[fila, 1]
    if verdad is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "población verdadera:\ndesconocida", ha="center", va="center", fontsize=22)
    else:
        ax.imshow(verdad, cmap=ListedColormap(["#1f77b4", "#e8d8a0"]), vmin=-0.5, vmax=1.5, interpolation="nearest")
        ax.set_title("población verdadera (azul = laberinto)", fontsize=20)
        ax.set_axis_off()
    ax = ejes[fila, 2]
    ax.hist(img.ravel(), bins=BINS, range=(bajo, alto), color="0.4")
    ax.set_xlabel("gris", fontsize=16)
    ax.set_title("histograma de gris (lineal)", fontsize=20)
    im = ejes[fila, 3].imshow(actividad, cmap="magma")
    ejes[fila, 3].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=20)
    ejes[fila, 3].set_axis_off()
    fig.colorbar(im, ax=ejes[fila, 3], fraction=0.046)
    ax = ejes[fila, 4]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + píxeles)")
    for g in (GRISES if verdad is not None else []):
        ax.axvline(g, color="#2ca02c", lw=6, alpha=0.3)
    ax.set_xlabel("gris", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("paisaje gris-actividad" + (" (verde = grises reales)" if verdad is not None else ""), fontsize=20)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/43_laberinto_dos_poblaciones.png"), dpi=34)
print("ok")
