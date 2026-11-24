"""Regiones automáticas a partir del mapa de actividad (|gradiente| promediado).

Mapa de actividad = gaussiana(sobel(suavizada), sigma), con sigma 10 y 25 px, llevado a [0, 1]
por percentiles (0.5-99.5). Sobre su histograma (sin ecualizar): picos y valles con los mismos
criterios de antes (prominencia 5 %, distancia 8 bins, masa 1 %, valle < 75 % del pico menor).
Cada píxel va a la región según el valle que le corresponde. No se fija la cantidad de regiones.
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
ESCALAS = (10, 25)
COLORES = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"]
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def normalizar(valores):
    bajo, alto = np.percentile(valores, (0.5, 99.5))
    return np.clip((valores - bajo) / (alto - bajo), 0, 1)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def picos_y_valles(valores):
    conteos, bordes = np.histogram(valores, bins=BINS, range=(0, 1))
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


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    gradiente = ski.filters.sobel(suave)
    lo, hi = np.percentile(suave, (0.5, 99.5))

    fig, ejes = plt.subplots(len(ESCALAS), 4, figsize=(28, 7 * len(ESCALAS)),
                             gridspec_kw={"width_ratios": [1.1, 1, 1, 0.6]})
    for fila, s in enumerate(ESCALAS):
        actividad = normalizar(ski.filters.gaussian(gradiente, s))
        centros, conteos, h, picos, umbrales = picos_y_valles(actividad.ravel())
        regiones = np.digitize(actividad, umbrales)
        n = len(picos)
        fracciones = [np.mean(regiones == r) for r in range(n)]
        print(f"ROI_{k}, σ={s}px: picos {np.round(centros[picos], 3)}, umbrales {np.round(umbrales, 3)} -> "
              f"{n} región(es), áreas {np.round(fracciones, 3)}")

        ax = ejes[fila, 0]
        ax.bar(centros, conteos, width=centros[1] - centros[0], color="0.75")
        ax.plot(centros, h, color="k", lw=1.5)
        for p in picos:
            ax.axvline(centros[p], color="C0", lw=1, ls=":")
        for t in umbrales:
            ax.axvline(t, color="C3", lw=2.5, ls="--")
        ax.set_xlabel("actividad (normalizada)", fontsize=13)
        ax.set_title(f"ROI {k} — histograma del mapa de actividad (σ={s}px), escala lineal\n"
                     "azul = picos que quedan, rojo = cortes en los valles", fontsize=14)

        ax = ejes[fila, 1]
        ax.imshow(regiones, cmap=ListedColormap(COLORES[:n]), vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
        ax.set_title(f"{n} región(es) encontradas (colores sólidos)", fontsize=15)
        ax.set_axis_off()

        ax = ejes[fila, 2]
        ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
        bordes = ski.segmentation.find_boundaries(regiones, mode="inner")
        ax.imshow(np.ma.masked_where(~bordes, bordes), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
        ax.set_title("bordes de las regiones sobre la ROI suavizada", fontsize=15)
        ax.set_axis_off()

        ax = ejes[fila, 3]
        ax.bar(range(n), [100 * f for f in fracciones], color=COLORES[:n], edgecolor="k")
        ax.set_xticks(range(n), [f"región {r}" for r in range(n)], fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_title("área (%)", fontsize=15)
        for r, f in enumerate(fracciones):
            ax.text(r, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=13)

    fig.tight_layout()
    fig.savefig(SALIDA / f"12_regiones_por_actividad_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
