"""Mapas de derivadas en el plano de la imagen: gradiente y "actividad" a varias escalas.

Por ROI: suavizada | white_tophat (r=50) | |gradiente| (Sobel) | actividad = |gradiente|
promediado con gaussiana de sigma 10, 25 y 50 px.
Imágenes: cada mapa llevado a [0, 1] por percentiles y ecualizado global con límite (clip 0.01).
Histogramas (escala lineal): gris = mapa sin ecualizar, rojo = ecualizado.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
ESCALAS = (10, 25, 50)
CLIP = 0.01


def normalizar(valores):
    bajo, alto = np.percentile(valores, (0.5, 99.5))
    return np.clip((valores - bajo) / (alto - bajo), 0, 1)


def ecualizar_global(img01):
    return ski.exposure.equalize_adapthist(img01, kernel_size=img01.shape, clip_limit=CLIP)


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    gradiente = ski.filters.sobel(suave)
    mapas = {
        "suavizada": suave,
        "white_tophat (r=50)": ski.morphology.white_tophat(suave, ski.morphology.disk(50, decomposition="crosses")),
        "|gradiente| (Sobel)": gradiente,
    }
    for s in ESCALAS:
        mapas[f"actividad, σ={s}px\n(|gradiente| promediado)"] = ski.filters.gaussian(gradiente, s)

    fig, ejes = plt.subplots(2, len(mapas), figsize=(5.4 * len(mapas), 10.5),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for j, (titulo, mapa) in enumerate(mapas.items()):
        crudo = normalizar(mapa)
        ecualizado = ecualizar_global(crudo)
        ejes[0, j].imshow(ecualizado, cmap="gray", vmin=0, vmax=1)
        ejes[0, j].set_title(f"ROI {k} — {titulo}" if j == 0 else titulo, fontsize=13)
        ejes[0, j].set_axis_off()
        ax = ejes[1, j]
        ax.hist(crudo.ravel(), bins=128, range=(0, 1), histtype="step", lw=1.8, color="0.4", label="sin ecualizar")
        ax.hist(ecualizado.ravel(), bins=128, range=(0, 1), histtype="step", lw=1.8, color="C3", label="ecualizado")
        ax.set_xlabel("valor normalizado", fontsize=11)
        ax.legend(fontsize=10)
    fig.suptitle(f"ROI {k}: imágenes ecualizadas (global con límite {CLIP}); histogramas en escala lineal",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(SALIDA / f"11_mapas_actividad_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
