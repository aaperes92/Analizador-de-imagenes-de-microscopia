"""Segmentación en dos niveles.

Nivel 1: zonas de gris distinto por picos y valles del histograma (como 03_fases_valles_entre_picos).
Nivel 2: dentro de cada zona, contraste local = imagen - promedio local calculado SOLO con
píxeles de esa zona; sobre su histograma, la misma detección de picos y valles.
Tres ventanas para el promedio local (sigma 4, 8 y 16 px).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
VENTANAS = (4, 8, 16)
AREA_MINIMA_ZONA = 0.01
COLORES_ZONA = ["#08306b", "#a05a00", "#2ca02c", "#7b3294"]
COLOR_SUBFASE_OSCURA = "#1f8fff"

BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def normalizar(valores, bajo=None, alto=None):
    if bajo is None:
        bajo, alto = np.percentile(valores, (0.1, 99.9))
    return np.clip((valores - bajo) / (alto - bajo), 0, 1), (bajo, alto)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def picos_y_valles(valores):
    """valores en [0, 1] (1D). Devuelve (centros, conteos, h, picos finales, umbrales)."""
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


def contraste_local(img, mascara, sigma):
    """img - promedio local calculado solo con los píxeles de la máscara."""
    peso = mascara.astype(float)
    promedio = ndimage.gaussian_filter(img * peso, sigma) / np.maximum(ndimage.gaussian_filter(peso, sigma), 1e-6)
    return img - promedio


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    img, _ = normalizar(suavizar(roi))

    # Nivel 1: zonas
    _, _, _, picos1, umbrales1 = picos_y_valles(img.ravel())
    zonas = np.digitize(img, umbrales1)
    nz = len(picos1)
    areas = [np.mean(zonas == z) for z in range(nz)]
    principal = int(np.argmax(areas))
    print(f"\nROI_{k}: nivel 1 -> {nz} zona(s), áreas {np.round(areas, 3)}, principal = zona {principal}")

    fig, ejes = plt.subplots(1 + len(VENTANAS), 3, figsize=(21, 6.6 * (1 + len(VENTANAS))),
                             gridspec_kw={"width_ratios": [1.3, 1.2, 1.3]})
    ax = ejes[0, 0]
    ax.imshow(img, cmap="gray")
    capa = np.ma.masked_where(zonas == principal, zonas)
    ax.imshow(capa, cmap=ListedColormap(COLORES_ZONA[:nz]), vmin=-0.5, vmax=nz - 0.5, alpha=0.6,
              interpolation="nearest")
    ax.set_title(f"ROI {k} — NIVEL 1: {nz} zona(s) por gris\n(sin color = zona principal)", fontsize=14)
    ax.set_axis_off()
    ejes[0, 1].axis("off")
    ejes[0, 2].axis("off")
    ejes[0, 1].text(0, 0.5, "\n".join(f"zona {z}: {100 * a:.1f} % del área" for z, a in enumerate(areas)),
                    fontsize=15, va="center")

    for fila, sigma in enumerate(VENTANAS, start=1):
        resultado = np.zeros(img.shape, int)   # 0 = sin marcar; 1 = subfase oscura en alguna zona
        resumen = []
        for z in range(nz):
            mascara = zonas == z
            if areas[z] < AREA_MINIMA_ZONA:
                continue
            c = contraste_local(img, mascara, sigma)
            vals, (bajo, alto) = normalizar(c[mascara])
            centros, conteos, h, picos2, umbrales2 = picos_y_valles(vals)
            resumen.append(f"zona {z}: {len(picos2)} subfase(s)")
            if len(picos2) > 1:
                # Se marca la subfase más oscura (por debajo del primer umbral)
                corte = bajo + umbrales2[0] * (alto - bajo)
                resultado[mascara & (c < corte)] = 1
            if z == principal:
                c_principal = np.where(mascara, c, np.nan)
                hist_principal = (centros, conteos, h, picos2, umbrales2, len(picos2))
        print(f"  ventana σ={sigma}px: " + ", ".join(resumen) +
              f"; subfase oscura = {np.mean(resultado == 1):.1%} del área")

        ax = ejes[fila, 0]
        lim = np.nanpercentile(np.abs(c_principal), 99)
        ax.imshow(c_principal, cmap="gray", vmin=-lim, vmax=lim)
        ax.set_title(f"NIVEL 2, ventana σ={sigma}px\ncontraste local de la zona principal", fontsize=14)
        ax.set_axis_off()

        centros, conteos, h, picos2, umbrales2, n2 = hist_principal
        ax = ejes[fila, 1]
        ax.bar(centros, conteos, width=centros[1] - centros[0], color="0.75")
        ax.plot(centros, h, color="k", lw=1.5)
        for p in picos2:
            ax.axvline(centros[p], color="C0", lw=1, ls=":")
        for t in umbrales2:
            ax.axvline(t, color="C3", lw=2, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.8, None)
        ax.set_xlabel("contraste local (normalizado)", fontsize=12)
        ax.set_title(f"histograma del contraste local: {n2} subfase(s)\nazul = picos, rojo = umbrales", fontsize=13)

        ax = ejes[fila, 2]
        ax.imshow(img, cmap="gray")
        capa_z = np.ma.masked_where(zonas == principal, zonas)
        ax.imshow(capa_z, cmap=ListedColormap(COLORES_ZONA[:nz]), vmin=-0.5, vmax=nz - 0.5, alpha=0.6,
                  interpolation="nearest")
        capa_s = np.ma.masked_where(resultado == 0, resultado)
        ax.imshow(capa_s, cmap=ListedColormap([COLOR_SUBFASE_OSCURA]), alpha=0.5, interpolation="nearest")
        ax.set_title(f"RESULTADO σ={sigma}px: zonas + subfase oscura\n"
                     f"(celeste = subfase oscura, {np.mean(resultado == 1):.1%} del área)", fontsize=13)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(SALIDA / f"04_dos_niveles_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
