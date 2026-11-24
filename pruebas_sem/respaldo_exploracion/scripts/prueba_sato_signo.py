"""Fase oscura del patrón = respuesta POSITIVA del filtro sato (crestas oscuras, escalas 1-6 px).

Dentro de la región con patrón (como en 15_crestas): respuesta > 0 -> fase oscura; = 0 -> fase clara.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
REGION_PATRON = {1: 0, 3: 1}
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def escalar_min_max(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def cortes_por_valles(valores01):
    conteos, bordes = np.histogram(valores01, bins=BINS, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SIGMA_HISTOGRAMA)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])),
                                 prominence=PROMINENCIA_MINIMA * h.max(), distance=DISTANCIA_MINIMA)
    picos = [int(p) - 1 for p in picos]
    while len(picos) > 1:
        m = masas(h, picos)
        peor = int(np.argmin(m))
        if m[peor] < MASA_MINIMA:
            picos.pop(peor)
            continue
        valles = valles_entre(h, picos)
        rel = [h[v] / min(h[a], h[b]) for v, a, b in zip(valles, picos, picos[1:])]
        i = int(np.argmax(rel))
        if rel[i] <= VALLE_MAXIMO:
            break
        picos.pop(i if h[picos[i]] < h[picos[i + 1]] else i + 1)
    return centros[valles_entre(h, picos)]


for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))
    region = np.digitize(actividad, cortes_por_valles(actividad.ravel())) == REGION_PATRON[k]
    respuesta = ski.filters.sato(suave, sigmas=range(1, 7), black_ridges=True)

    oscura = region & (respuesta > 0)
    clara = region & (respuesta <= 0)
    frac_oscura = oscura.sum() / region.sum()
    print(f"ROI_{k}: región con patrón = {region.mean():.1%} de la ROI; respuesta > 0 en "
          f"{frac_oscura:.1%} de la región (fase oscura), = 0 en {1 - frac_oscura:.1%} (fase clara)")

    lo, hi = np.percentile(suave[region], (0.5, 99.5))
    fig, ejes = plt.subplots(1, 4, figsize=(30, 7.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.45]})
    ax = ejes[0]
    ax.set_facecolor("#ffd9e0")
    ax.imshow(np.where(region, suave, np.nan), cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(f"ROI {k} — región con patrón (rosa = fuera)", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

    mapa = np.full(suave.shape, -1)
    mapa[clara] = 0
    mapa[oscura] = 1
    ax = ejes[1]
    ax.set_facecolor("#ffd9e0")
    ax.imshow(np.ma.masked_where(mapa < 0, mapa), cmap=ListedColormap(["#f0e0a0", "#1f5fbf"]),
              vmin=-0.5, vmax=1.5, interpolation="nearest")
    ax.set_title("respuesta de sato: azul = positiva (fase oscura)\namarillo = nula (fase clara)", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = ejes[2]
    ax.set_facecolor("#ffd9e0")
    ax.imshow(np.where(region, suave, np.nan), cmap="gray", vmin=lo, vmax=hi)
    contorno = ski.segmentation.find_boundaries(oscura, mode="inner")
    ax.imshow(np.ma.masked_where(~contorno, contorno), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
    ax.set_title("contorno de la fase oscura sobre la imagen", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = ejes[3]
    ax.bar([0, 1], [100 * (1 - frac_oscura), 100 * frac_oscura], color=["#f0e0a0", "#1f5fbf"], edgecolor="k")
    ax.set_xticks([0, 1], ["fase clara", "fase oscura"], fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("área dentro de la región (%)", fontsize=14)
    for i, f in enumerate((1 - frac_oscura, frac_oscura)):
        ax.text(i, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=14)

    fig.tight_layout()
    fig.savefig(SALIDA / f"16_sato_respuesta_positiva_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
