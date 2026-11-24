"""En orden: 1) regiones de análisis por actividad; 2) dentro de cada región, zonas por gris.

Paso 1: actividad = gaussiana(sobel(suavizada), 25 px), escalada a [0, 1] por mínimo y máximo
(sin recortar); picos y valles de su histograma -> regiones.
Paso 2: para cada región, histograma del gris de la suavizada SOLO con sus píxeles (escalado a
[0, 1] con su propio mínimo y máximo); picos y valles -> zonas de gris dentro de la región.
Mismos criterios de picos que antes (prominencia 5 %, distancia 8 bins, masa 1 %, valle < 75 %).
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
COLORES_REGION = ["#4477aa", "#ee6677", "#228833", "#ccbb44"]
COLORES_ZONA = ["#1b1b8f", "#e07b00", "#2ca02c", "#d62728", "#9467bd"]
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def escalar_min_max(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def picos_y_valles(valores01):
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
    return centros, conteos, h, picos, centros[valles_entre(h, picos)]


def dibujar_histograma(ax, centros, conteos, h, picos, umbrales, xlabel, titulo):
    ax.bar(centros, conteos, width=centros[1] - centros[0], color="0.75")
    ax.plot(centros, h, color="k", lw=1.5)
    for p in picos:
        ax.axvline(centros[p], color="C0", lw=1, ls=":")
    for t in umbrales:
        ax.axvline(t, color="C3", lw=2.5, ls="--")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(titulo, fontsize=13)


for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))
    centros, conteos, h, picos, umbrales = picos_y_valles(actividad.ravel())
    regiones = np.digitize(actividad, umbrales)
    nr = len(picos)
    print(f"\nROI_{k}: paso 1 -> {nr} región(es) por actividad, áreas "
          f"{np.round([np.mean(regiones == r) for r in range(nr)], 3)}")

    fig, ejes = plt.subplots(1 + nr, 3, figsize=(24, 7 * (1 + nr)))
    lo, hi = np.percentile(suave, (0.5, 99.5))
    ejes[0, 0].imshow(actividad, cmap="gray")
    ejes[0, 0].set_title(f"ROI {k} — PASO 1: mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=15)
    ejes[0, 0].set_axis_off()
    dibujar_histograma(ejes[0, 1], centros, conteos, h, picos, umbrales, "actividad (0-1)",
                       "histograma de actividad (lineal)\nazul = picos, rojo = cortes")
    ejes[0, 2].imshow(regiones, cmap=ListedColormap(COLORES_REGION[:nr]), vmin=-0.5, vmax=nr - 0.5,
                      interpolation="nearest")
    ejes[0, 2].set_title(f"{nr} región(es) de análisis", fontsize=15)
    ejes[0, 2].set_axis_off()

    for r in range(nr):
        mascara = regiones == r
        gris = suave[mascara]
        gris01 = escalar_min_max(gris)
        c2, n2, h2, p2, u2 = picos_y_valles(gris01)
        nz = len(p2)
        cortes_gris = gris.min() + u2 * (gris.max() - gris.min())
        zonas = np.full(suave.shape, -1)
        zonas[mascara] = np.digitize(suave[mascara], cortes_gris)
        areas = [np.mean(zonas[mascara] == z) for z in range(nz)]
        print(f"   región {r} ({mascara.mean():.1%} de la ROI): {nz} zona(s) de gris, "
              f"cortes en gris {np.round(cortes_gris, 3)}, áreas dentro de la región {np.round(areas, 3)}")

        fila = 1 + r
        ax = ejes[fila, 0]
        ax.set_facecolor("#ffd9e0")
        ax.imshow(np.where(mascara, suave, np.nan), cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"PASO 2 — región {r} sola ({mascara.mean():.0%} de la ROI; rosa = fuera)", fontsize=15,
                     color=COLORES_REGION[r])
        ax.set_xticks([])
        ax.set_yticks([])
        dibujar_histograma(ejes[fila, 1], c2, n2, h2, p2, u2, "gris dentro de la región (escalado 0-1)",
                           f"histograma de GRIS de la región {r}: {nz} zona(s)")
        ax = ejes[fila, 2]
        ax.set_facecolor("#ffd9e0")
        ax.imshow(np.ma.masked_where(zonas < 0, zonas), cmap=ListedColormap(COLORES_ZONA[:nz]),
                  vmin=-0.5, vmax=nz - 0.5, interpolation="nearest")
        ax.set_title(f"zonas de gris dentro de la región {r}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(SALIDA / f"14_en_orden_actividad_y_gris_ROI{k}.png", dpi=42)
    plt.close(fig)
print("ok")
