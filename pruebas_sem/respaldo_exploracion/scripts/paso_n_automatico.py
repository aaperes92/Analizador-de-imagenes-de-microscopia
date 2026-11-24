"""Cantidad de poblaciones automática: el N más grande cuyos umbrales multi-Otsu caen en valles.

Un umbral está en un valle si el histograma (suavizado) en el umbral es menor que
PROFUNDIDAD x el pico más bajo de los dos lados (cada lado = entre umbrales vecinos).
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

N_MAXIMO = 5
PROFUNDIDAD = 0.5          # el valle debe estar por debajo de la mitad del pico más bajo vecino
BINS = 256
SIGMA_HISTOGRAMA = 2       # suavizado del histograma, en bins


def histograma_suave(imagen):
    conteos, bordes = np.histogram(imagen, bins=BINS, range=(0, 1))
    suave = ski.filters.gaussian(conteos.astype(float), SIGMA_HISTOGRAMA, preserve_range=True)
    centros = (bordes[:-1] + bordes[1:]) / 2
    return centros, conteos, suave


def evaluar_valles(centros, hist, umbrales):
    """Para cada umbral: (valor en el umbral) / (pico más bajo de los dos lados)."""
    cortes = np.concatenate(([0.0], umbrales, [1.0]))
    relaciones = []
    for i, t in enumerate(umbrales):
        izquierda = hist[(centros >= cortes[i]) & (centros < t)]
        derecha = hist[(centros >= t) & (centros < cortes[i + 2])]
        if izquierda.size == 0 or derecha.size == 0:
            relaciones.append(np.inf)
            continue
        en_umbral = hist[np.argmin(np.abs(centros - t))]
        relaciones.append(en_umbral / min(izquierda.max(), derecha.max()))
    return relaciones


casos = {f"ROI_{k}": np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy") for k in (1, 3, 4)}

fig, ejes = plt.subplots(1, len(casos), figsize=(8 * len(casos), 6.5))
for ax, (nombre, roi) in zip(ejes, casos.items()):
    suave = suavizar(roi)
    centros, conteos, hist = histograma_suave(suave)
    n_elegido, umbrales_elegidos = 1, np.array([])
    print(f"\n{nombre}")
    for n in range(2, N_MAXIMO + 1):
        umbrales = ski.filters.threshold_multiotsu(suave, classes=n)
        relaciones = evaluar_valles(centros, hist, umbrales)
        valido = all(r < PROFUNDIDAD for r in relaciones)
        print(f"  N={n}: umbrales={np.round(umbrales, 3)}  valle/pico={np.round(relaciones, 2)}  "
              f"{'OK' if valido else 'no'}")
        if valido:
            n_elegido, umbrales_elegidos = n, umbrales
    print(f"  -> N elegido = {n_elegido}")

    ax.bar(centros, conteos, width=1 / BINS, color="0.75", label="histograma")
    ax.plot(centros, hist, color="k", lw=1.5, label="histograma suavizado")
    for t in umbrales_elegidos:
        ax.axvline(t, color="C3", lw=2, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.8, None)
    ax.set_xlabel("intensidad", fontsize=13)
    ax.set_title(f"{nombre}: N elegido = {n_elegido}\n"
                 f"(líneas rojas = umbrales en valles)", fontsize=15)
    ax.legend(fontsize=11)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/sombras/07_N_automatico_por_valles.png"), dpi=60)
print("ok")
