"""Nivel 2 ecualizado: contraste local de cada zona, ecualizado sin límite y con límite (ROI 3).

Nivel 1: zonas por picos y valles del histograma (igual que antes).
Nivel 2: contraste local (sigma 4 px, promedio solo con la propia zona), y luego:
- ecualización sin límite: CDF del histograma de la zona (skimage.exposure.equalize_hist con mask);
- ecualización con límite: el histograma de la zona se recorta a clip_limit x N píxeles por
  barra, lo recortado se reparte parejo entre todas las barras y se usa esa CDF (la cuenta de
  CLAHE con una sola región, restringida a los píxeles de la zona).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA = 4
CLIP_LIMIT = 0.01
NIVELES = 256
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75
NOMBRES_ZONA = ["zona 0 (negra)", "zona 1 (gris)", "zona 2 (clara, principal)"]


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


def ecualizar_con_limite(valores, clip_limit, niveles=NIVELES):
    """Ecualización global con límite de recorte sobre un conjunto de valores (1D) -> [0, 1]."""
    conteos, bordes = np.histogram(valores, bins=niveles)
    tope = max(1, int(clip_limit * valores.size))
    exceso = np.maximum(conteos - tope, 0).sum()
    recortado = np.minimum(conteos, tope) + exceso / niveles
    cdf = np.cumsum(recortado)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    indice = np.clip(np.digitize(valores, bordes[1:-1]), 0, niveles - 1)
    return cdf[indice]


roi = np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")
img = normalizar(suavizar(roi))
zonas, nz = zonas_por_valles(img)
print(f"ROI 3: {nz} zonas, áreas {np.round([np.mean(zonas == z) for z in range(nz)], 3)}")

armada_sin = np.zeros(img.shape)
armada_con = np.zeros(img.shape)
fig, ejes = plt.subplots(nz, 4, figsize=(27, 6.6 * nz), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
for z in range(nz):
    mascara = zonas == z
    c = contraste_local(img, mascara, SIGMA)
    sin_limite = ski.exposure.equalize_hist(c, nbins=NIVELES, mask=mascara)
    con_limite = np.zeros(img.shape)
    con_limite[mascara] = ecualizar_con_limite(c[mascara], CLIP_LIMIT)
    armada_sin[mascara] = sin_limite[mascara]
    armada_con[mascara] = con_limite[mascara]

    lim = np.percentile(np.abs(c[mascara]), 99)
    paneles = [
        (np.where(mascara, c, np.nan), -lim, lim, "contraste local"),
        (np.where(mascara, sin_limite, np.nan), 0, 1, "ecualizado SIN límite"),
        (np.where(mascara, con_limite, np.nan), 0, 1, f"ecualizado CON límite (clip {CLIP_LIMIT})"),
    ]
    for j, (panel, vmin, vmax, titulo) in enumerate(paneles):
        ax = ejes[z, j]
        ax.set_facecolor("#ffd9e0")    # rosa = fuera de la zona
        ax.imshow(panel, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{NOMBRES_ZONA[z]} — {titulo}" if j == 0 else titulo, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = ejes[z, 3]
    for valores, nombre, color in ((normalizar(c[mascara]), "contraste local (normalizado)", "0.4"),
                                   (sin_limite[mascara], "sin límite", "C0"),
                                   (con_limite[mascara], "con límite", "C3")):
        ax.hist(valores, bins=128, range=(0, 1), histtype="step", lw=1.8, color=color, label=nombre)
    ax.set_xlabel("valor (0-1)", fontsize=12)
    ax.set_title(f"histogramas de la zona (escala lineal), {mascara.sum()} píxeles", fontsize=13)
    ax.legend(fontsize=11)
fig.suptitle("ROI 3 — cada zona por separado (rosa = fuera de la zona)", fontsize=16)
fig.tight_layout()
fig.savefig(SALIDA / "06_nivel2_ecualizado_por_zona_ROI3.png", dpi=45)
plt.close(fig)

# ROI armada: cada zona con su contraste local ecualizado, y el contorno de las zonas
fig, ejes = plt.subplots(1, 3, figsize=(27, 9))
bordes_zonas = ski.segmentation.find_boundaries(zonas, mode="inner")
for ax, (imagen, titulo) in zip(ejes, ((img, "ROI 3 suavizada"),
                                        (armada_sin, "armada: cada zona ecualizada SIN límite"),
                                        (armada_con, f"armada: cada zona ecualizada CON límite ({CLIP_LIMIT})"))):
    ax.imshow(imagen, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~bordes_zonas, bordes_zonas), cmap="autumn", alpha=1, interpolation="nearest")
    ax.set_title(f"{titulo}\n(naranja = borde entre zonas)", fontsize=15)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "07_nivel2_ecualizado_armado_ROI3.png", dpi=50)
plt.close(fig)
print("ok")
