"""Corrección de iluminación con N automático (valles del histograma) y ventanas de 25 y 50 px.

Una figura por ROI: filas = ventana; columnas = antes | mapa | después | diferencia.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

BINS = 256
SIGMA_HISTOGRAMA = 2
ALTURA_MINIMA = 0.01
PROFUNDIDAD = 0.5
RADIO_FRANJA = 2
VENTANAS = (25, 50)


def umbrales_por_valles(imagen):
    conteos, bordes = np.histogram(imagen, bins=BINS, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ski.filters.gaussian(conteos.astype(float), SIGMA_HISTOGRAMA, preserve_range=True)
    relleno = np.concatenate(([-np.inf], h, [-np.inf]))
    es_maximo = (h >= relleno[:-2]) & (h > relleno[2:])
    picos = [i for i in np.flatnonzero(es_maximo) if h[i] >= ALTURA_MINIMA * h.max()]
    valles, actual = [], picos[0]
    for siguiente in picos[1:]:
        v = actual + int(np.argmin(h[actual:siguiente + 1]))
        if h[v] < PROFUNDIDAD * min(h[actual], h[siguiente]):
            valles.append(v)
            actual = siguiente
        elif h[siguiente] > h[actual]:
            actual = siguiente
    return centros[valles]


def mapa_iluminacion(imagen, umbrales, sigma_ventana):
    """Promedio local de la imagen / promedio local de la imagen 'esperada', con el interior de las fases."""
    clases = np.digitize(imagen, umbrales)
    n = len(umbrales) + 1
    grises = np.array([np.median(imagen[clases == k]) for k in range(n)])
    disco = ski.morphology.disk(RADIO_FRANJA)
    interior = np.zeros(imagen.shape, bool)
    for k in range(n):
        interior |= ski.morphology.erosion(clases == k, disco)
    peso = interior.astype(np.float32)
    numerador = ski.filters.gaussian(peso * imagen, sigma_ventana)
    denominador = ski.filters.gaussian(peso * grises[clases], sigma_ventana)
    return numerador / np.maximum(denominador, 1e-6)


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    antes = suavizar(roi)
    umbrales = umbrales_por_valles(antes)
    n = len(umbrales) + 1
    lo, hi = np.percentile(antes, (0.5, 99.5))

    fig, ejes = plt.subplots(len(VENTANAS), 4, figsize=(22, 5.6 * len(VENTANAS)))
    for fila, sigma_v in enumerate(VENTANAS):
        ilum = mapa_iluminacion(antes, umbrales, sigma_v)
        despues = antes / ilum
        diferencia = despues - antes
        lim = max(np.percentile(np.abs(diferencia), 99), 1e-4)
        print(f"ROI_{k}: N={n}  ventana σ={sigma_v}px  mapa (percentiles 1-99) = "
              f"{np.percentile(ilum, 1):.3f}-{np.percentile(ilum, 99):.3f}  "
              f"cambio (percentil 99) = ±{lim:.3f} (±{lim * 255:.1f} niveles de 8 bits)")

        ejes[fila, 0].imshow(antes, cmap="gray", vmin=lo, vmax=hi)
        ejes[fila, 0].set_title(f"ROI {k} — ANTES\n(N = {n} automático, ventana σ = {sigma_v} px)", fontsize=14)
        im = ejes[fila, 1].imshow(ilum, cmap="gray", vmin=0.92, vmax=1.08)
        ejes[fila, 1].set_title("MAPA de iluminación\noscuro = más oscuro de lo que le corresponde", fontsize=14)
        fig.colorbar(im, ax=ejes[fila, 1], fraction=0.046)
        ejes[fila, 2].imshow(despues, cmap="gray", vmin=lo, vmax=hi)
        ejes[fila, 2].set_title("DESPUÉS\n(misma escala de gris que ANTES)", fontsize=14)
        im = ejes[fila, 3].imshow(diferencia, cmap="gray", vmin=-lim, vmax=lim)
        ejes[fila, 3].set_title("DIFERENCIA (después − antes)\nclaro = se aclaró · oscuro = se oscureció", fontsize=14)
        fig.colorbar(im, ax=ejes[fila, 3], fraction=0.046)
        for ax in ejes[fila]:
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(Path(f"pruebas_sem/sombras/09_correccion_N_auto_ROI{k}.png"), dpi=55)
    plt.close(fig)
print("ok")
