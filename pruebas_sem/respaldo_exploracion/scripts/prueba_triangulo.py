"""Nivel 2 con el método del triángulo sobre el contraste local (ventana sigma = 4 px).

Nivel 1 igual que antes (zonas por picos y valles). En cada zona: contraste local con el
promedio de la propia zona; asimetría de su distribución; umbral del triángulo
(skimage.filters.threshold_triangle, que detecta solo de qué lado está la cola).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal, stats

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA = 4
AREA_MINIMA_ZONA = 0.01
COLORES_ZONA = ["#08306b", "#a05a00", "#2ca02c", "#7b3294"]
COLOR_SUBFASE = "#1f8fff"
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def normalizar(valores):
    bajo, alto = np.percentile(valores, (0.1, 99.9))
    return np.clip((valores - bajo) / (alto - bajo), 0, 1)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def zonas_por_valles(img):
    conteos, bordes = np.histogram(img, bins=BINS, range=(0, 1))
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
    return np.digitize(img, centros[valles_entre(h, picos)]), len(picos)


def contraste_local(img, mascara, sigma):
    peso = mascara.astype(float)
    promedio = ndimage.gaussian_filter(img * peso, sigma) / np.maximum(ndimage.gaussian_filter(peso, sigma), 1e-6)
    return img - promedio


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    img = normalizar(suavizar(roi))
    zonas, nz = zonas_por_valles(img)
    areas = [np.mean(zonas == z) for z in range(nz)]
    principal = int(np.argmax(areas))

    subfase = np.zeros(img.shape, bool)
    print(f"\nROI_{k}: {nz} zona(s), áreas {np.round(areas, 3)}")
    for z in range(nz):
        mascara = zonas == z
        if areas[z] < AREA_MINIMA_ZONA:
            continue
        c = contraste_local(img, mascara, SIGMA)
        vals = c[mascara]
        asimetria = float(stats.skew(vals))
        umbral = ski.filters.threshold_triangle(vals)
        # La subfase es la cola: del lado oscuro si la asimetría es negativa, del claro si es positiva
        cola = (c < umbral) if asimetria < 0 else (c > umbral)
        subfase |= mascara & cola
        print(f"  zona {z}: asimetría {asimetria:+.2f}, umbral triángulo {umbral:+.4f}, "
              f"cola ({'oscura' if asimetria < 0 else 'clara'}) = {np.mean(cola[mascara]):.1%} de la zona")
        if z == principal:
            c_principal, vals_principal, umbral_principal, asim_principal = \
                np.where(mascara, c, np.nan), vals, umbral, asimetria

    fig, ejes = plt.subplots(1, 4, figsize=(28, 7.2), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.3]})
    lim = np.nanpercentile(np.abs(c_principal), 99)
    ejes[0].imshow(c_principal, cmap="gray", vmin=-lim, vmax=lim)
    ejes[0].set_title(f"ROI {k} — contraste local, zona principal (σ={SIGMA}px)", fontsize=14)
    ejes[0].set_axis_off()
    for ax, escala in ((ejes[1], "linear"), (ejes[2], "log")):
        ax.hist(vals_principal, bins=256, color="0.5")
        ax.axvline(umbral_principal, color="C3", lw=2, ls="--")
        ax.set_yscale(escala)
        ax.set_xlabel("contraste local", fontsize=12)
        ax.set_title(f"histograma, escala {'LINEAL' if escala == 'linear' else 'LOG'}\n"
                     f"asimetría {asim_principal:+.2f}; rojo = umbral del triángulo", fontsize=13)
    ax = ejes[3]
    ax.imshow(img, cmap="gray")
    if nz > 1:
        capa_z = np.ma.masked_where(zonas == principal, zonas)
        ax.imshow(capa_z, cmap=ListedColormap(COLORES_ZONA[:nz]), vmin=-0.5, vmax=nz - 0.5, alpha=0.6,
                  interpolation="nearest")
    ax.imshow(np.ma.masked_where(~subfase, subfase), cmap=ListedColormap([COLOR_SUBFASE]), alpha=0.5,
              interpolation="nearest")
    ax.set_title(f"RESULTADO: zonas + cola de cada zona (celeste)\n"
                 f"celeste = {np.mean(subfase):.1%} del área", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(SALIDA / f"05_triangulo_contraste_local_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
