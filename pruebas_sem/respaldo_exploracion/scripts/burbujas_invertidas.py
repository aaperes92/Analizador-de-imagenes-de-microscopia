"""Burbujas de 5 poblaciones con intensidades invertidas: el gris BAJA con el tamaño.

Mismas burbujas que en 31_ (fondo; radios 4, 8, 15 y 30 px), pero:
r=4 -> 0.90, r=8 -> 0.70, fondo -> 0.50, r=15 -> 0.30, r=30 -> 0.10.
Los colores siguen identificando el TAMAÑO (igual que en 31_), así se pueden comparar.
Filas: bordes nítidos y con degradé (gaussiana 2 px). Mismos gráficos que 31_.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch

LADO = 600
SIGMA_ACTIVIDAD = 25
SIGMA_DEGRADE = 2
BINS = 128
GRISES_INVERTIDOS = np.array([0.90, 0.70, 0.50, 0.30, 0.10])   # por población (0 = r4 ... 4 = r30)
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
RADIOS = {0: 4, 1: 8, 3: 15, 4: 30}


def burbujas(semilla=0):
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, radio in sorted(RADIOS.items(), key=lambda kv: -kv[1]):
        for _ in range(int(0.08 * LADO * LADO / (np.pi * radio ** 2))):
            fy, fx = rng.integers(0, LADO, 2)
            rr, cc = ski.draw.disk((fy, fx), radio, shape=etiquetas.shape)
            etiquetas[rr, cc] = p
    return etiquetas


def componer(capas):
    maximo = max(c.max() for c in capas) or 1
    rgb = np.ones(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        rgb -= (c / maximo)[..., None] * (1 - np.array(to_rgb(color)))
    return np.clip(rgb, 0, 1)


def componer_aditivo(capas):
    maximo = max(c.max() for c in capas) or 1
    rgb = np.zeros(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        rgb += (c / maximo)[..., None] * np.array(to_rgb(color))
    return np.clip(rgb, 0, 1)


etiquetas = burbujas()
nitida = GRISES_INVERTIDOS[etiquetas]
casos = {"burbujas INVERTIDAS, bordes nítidos": nitida,
         f"burbujas INVERTIDAS, con degradé (σ={SIGMA_DEGRADE}px)": ski.filters.gaussian(nitida, SIGMA_DEGRADE)}

fig, ejes = plt.subplots(2, 5, figsize=(40, 16.4))
for fila, (nombre, img) in enumerate(casos.items()):
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    capas = [np.log1p(np.histogram2d(actividad[etiquetas == p], img[etiquetas == p], bins=BINS,
                                     range=((0, 1), (0, 1)))[0]) for p in range(5)]
    fourier = [np.log1p(np.abs(np.fft.rfft(c, axis=1))) for c in capas]

    ejes[fila, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ejes[fila, 0].set_title(nombre, fontsize=20)
    ejes[fila, 1].imshow(etiquetas, cmap=ListedColormap(COLORES), vmin=-0.5, vmax=4.5, interpolation="nearest")
    ejes[fila, 1].set_title("población verdadera de cada píxel", fontsize=20)
    im = ejes[fila, 2].imshow(actividad, cmap="magma")
    ejes[fila, 2].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=20)
    fig.colorbar(im, ax=ejes[fila, 2], fraction=0.046)
    for ax in ejes[fila, :3]:
        ax.set_axis_off()
    ax = ejes[fila, 3]
    ax.imshow(componer(capas), origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("gris (0-1)", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("paisaje gris-actividad, coloreado por población", fontsize=20)
    ax = ejes[fila, 4]
    frec = np.fft.rfftfreq(BINS)
    ax.imshow(componer_aditivo(fourier), origin="lower", aspect="auto", extent=(frec[0], frec[-1], 0, 1))
    ax.set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("Fourier fila por fila, una capa por población", fontsize=20)

leyenda = [Patch(color=COLORES[p], label=f"gris {GRISES_INVERTIDOS[p]:.2f} · " +
                 (f"burbujas r={RADIOS[p]}px" if p in RADIOS else "fondo")) for p in range(5)]
fig.legend(handles=leyenda, loc="upper center", ncol=5, fontsize=20, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(Path("pruebas_sem/segmentacion/32_burbujas_invertidas_paisaje_fourier.png"), dpi=34)
print("ok")
