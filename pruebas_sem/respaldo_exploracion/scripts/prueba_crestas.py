"""Filtros de crestas (segunda derivada) dentro de las regiones con patrón.

Región con patrón: la región de actividad media del paso anterior (actividad = gaussiana(sobel,
25 px), picos y valles). ROI 1 -> región 0 (única); ROI 3 -> región 1.
Filtros: skimage.filters.sato y meijering, crestas OSCURAS (black_ridges=True), escalas 1-6 px;
cada píxel se queda con la mejor respuesta entre escalas.
Sobre el histograma de la respuesta DENTRO de la región: picos y valles (sin fijar cantidades).
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
ESCALAS_CRESTA = range(1, 7)
REGION_PATRON = {1: 0, 3: 1}
COLORES_CLASE = ["#f2f2f2", "#1f5fbf", "#e07b00", "#2ca02c", "#d62728"]
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


for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))
    _, _, _, _, cortes = picos_y_valles(actividad.ravel())
    region = np.digitize(actividad, cortes) == REGION_PATRON[k]
    lo, hi = np.percentile(suave[region], (0.5, 99.5))

    filtros = {
        "sato": ski.filters.sato(suave, sigmas=ESCALAS_CRESTA, black_ridges=True),
        "meijering": ski.filters.meijering(suave, sigmas=ESCALAS_CRESTA, black_ridges=True),
    }
    fig, ejes = plt.subplots(len(filtros), 5, figsize=(36, 7.2 * len(filtros)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.5]})
    for fila, (nombre, respuesta) in enumerate(filtros.items()):
        vals = respuesta[region]
        vals01 = escalar_min_max(vals)
        centros, conteos, h, picos, umbrales = picos_y_valles(vals01)
        n = len(picos)
        clases = np.full(suave.shape, -1)
        clases[region] = np.digitize(vals01, umbrales)
        fracciones = [np.mean(clases[region] == c) for c in range(n)]
        print(f"ROI_{k}, {nombre}: región con patrón = {region.mean():.1%} de la ROI; picos en "
              f"{np.round(centros[picos], 3)}, cortes {np.round(umbrales, 3)} -> {n} clase(s), "
              f"áreas {np.round(fracciones, 3)}")

        ax = ejes[fila, 0]
        ax.set_facecolor("#ffd9e0")
        ax.imshow(np.where(region, suave, np.nan), cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"ROI {k} — región con patrón ({region.mean():.0%} de la ROI; rosa = fuera)", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = ejes[fila, 1]
        ax.set_facecolor("#ffd9e0")
        r_lo, r_hi = np.percentile(vals, (0.5, 99.5))
        ax.imshow(np.where(region, respuesta, np.nan), cmap="gray", vmin=r_lo, vmax=r_hi)
        ax.set_title(f"respuesta del filtro {nombre}\n(claro = sobre una franja oscura)", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = ejes[fila, 2]
        ax.bar(centros, conteos, width=centros[1] - centros[0], color="0.75")
        ax.plot(centros, h, color="k", lw=1.5)
        for p in picos:
            ax.axvline(centros[p], color="C0", lw=1, ls=":")
        for t in umbrales:
            ax.axvline(t, color="C3", lw=2.5, ls="--")
        ax.set_xlabel("respuesta (escalada 0-1)", fontsize=12)
        ax.set_title(f"histograma de la respuesta (lineal): {n} grupo(s)\nazul = picos, rojo = cortes", fontsize=14)

        ax = ejes[fila, 3]
        ax.set_facecolor("#ffd9e0")
        ax.imshow(np.ma.masked_where(clases < 0, clases), cmap=ListedColormap(COLORES_CLASE[:n]),
                  vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
        ax.set_title(f"clases (colores sólidos)\n{'gris claro = respuesta baja' if n > 1 else 'una sola clase'}",
                     fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = ejes[fila, 4]
        ax.bar(range(n), [100 * f for f in fracciones], color=COLORES_CLASE[:n], edgecolor="k")
        ax.set_xticks(range(n), [f"clase {c}" for c in range(n)], fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_title("área dentro de la región (%)", fontsize=13)
        for c, f in enumerate(fracciones):
            ax.text(c, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"15_crestas_region_patron_ROI{k}.png", dpi=42)
    plt.close(fig)
print("ok")
