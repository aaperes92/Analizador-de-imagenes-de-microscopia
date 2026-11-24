"""Cantidad de poblaciones buscando los valles directamente en el histograma suavizado.

Reglas (fijas para cualquier imagen):
- pico: máximo local con al menos ALTURA_MINIMA x la altura del pico mayor;
- valle entre dos picos vecinos: el mínimo entre ellos, válido si está por debajo de
  PROFUNDIDAD x el más bajo de los dos picos. Si no es válido, los picos se fusionan.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/sombras/.
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


def buscar_valles(imagen):
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
            actual = siguiente          # se fusionan: queda el pico más alto
    return centros, conteos, h, picos, valles


casos = {f"ROI_{k}": np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy") for k in (1, 3, 4)}

fig, ejes = plt.subplots(1, len(casos), figsize=(8 * len(casos), 6.5))
for ax, (nombre, roi) in zip(ejes, casos.items()):
    centros, conteos, h, picos, valles = buscar_valles(suavizar(roi))
    umbrales = centros[valles]
    clases = np.digitize(suavizar(roi) if False else roi, umbrales)
    print(f"{nombre}: picos en {np.round(centros[picos], 3)}  umbrales (valles) en {np.round(umbrales, 3)}"
          f"  ->  N = {len(valles) + 1}")
    ax.bar(centros, conteos, width=1 / BINS, color="0.75", label="histograma")
    ax.plot(centros, h, color="k", lw=1.5, label="histograma suavizado")
    ax.plot(centros[picos], h[picos], "o", color="C0", ms=9, label="picos considerados")
    for t in umbrales:
        ax.axvline(t, color="C3", lw=2, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.8, None)
    ax.set_xlabel("intensidad", fontsize=13)
    ax.set_title(f"{nombre}: N = {len(valles) + 1}\n(rojo = umbrales en valles)", fontsize=15)
    ax.legend(fontsize=11)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/sombras/08_N_automatico_valles_directos.png"), dpi=60)
print("ok")
