"""Paisaje gris-actividad por región: original y filtrado (solo mirar, sin clasificar).

1. Regiones por actividad: actividad = gaussiana(sobel, 25 px) a [0, 1]; picos y valles de su
   histograma (prominencia 5 %, distancia 8, masa 1 %, valle < 75 %).
2. Para cada región, con SOLO sus píxeles:
   - gris en la escala 0-1 de toda la imagen (comparable entre regiones);
   - actividad estirada al rango de la región;
   - paisaje = log(1 + histograma 2D 128 x 128);
   - filtrado = cada fila menos su versión suavizada (gaussiana 6 bins), negativos a 0.
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
BINS = 128
SIGMA_FILTRO = 6
COLORES_REGION = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee"]
BINS_H, SIGMA_H, PROMINENCIA, DISTANCIA, MASA_MINIMA, VALLE_MAXIMO = 256, 2.0, 0.05, 8, 0.01, 0.75


def escalar(v):
    return (v - v.min()) / (v.max() - v.min())


def imagen_sintetica(lado=650, semilla=3):
    rng = np.random.default_rng(semilla)
    y, x = np.mgrid[0:lado, 0:lado] / lado
    img = np.full((lado, lado), 0.70)
    estrias = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (2, 12))
    estrias = ndimage.rotate(estrias, 35, reshape=False, mode="reflect")
    img += 0.02 * estrias / estrias.std()
    zona_patron = (x + y) < 0.75
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), 6)
    img[zona_patron & (np.abs(campo / campo.std()) < 0.25)] = 0.58
    img[ski.data.binary_blobs(lado, blob_size_fraction=0.01, volume_fraction=0.02, rng=semilla)] = 0.85
    d = (x + y) - 1.55
    img[np.abs(d) < 0.03] = 0.0
    img[d >= 0.03] = 0.45
    img = ndimage.gaussian_filter(img, 1.5) + rng.normal(0, 0.005, img.shape)
    return np.clip(img, 0, 1)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def regiones_por_actividad(actividad01):
    conteos, bordes = np.histogram(actividad01, bins=BINS_H, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SIGMA_H)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])), prominence=PROMINENCIA * h.max(),
                                 distance=DISTANCIA)
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
    return np.digitize(actividad01, centros[valles_entre(h, picos)]), len(picos)


casos = {
    "ROI3": suavizar(np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")),
    "sintetica": imagen_sintetica(),
}
for nombre, img in casos.items():
    gris = escalar(img)
    actividad = escalar(ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD))
    regiones, nr = regiones_por_actividad(actividad)
    print(f"{nombre}: {nr} región(es), áreas {np.round([np.mean(regiones == r) for r in range(nr)], 3)}")

    fig, ejes = plt.subplots(1 + nr, 3, figsize=(25, 7.5 * (1 + nr)))
    ejes[0, 0].imshow(gris, cmap="gray")
    ejes[0, 0].set_title(f"{nombre}: imagen", fontsize=17)
    ejes[0, 1].imshow(actividad, cmap="gray")
    ejes[0, 1].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=17)
    ejes[0, 2].imshow(regiones, cmap=ListedColormap(COLORES_REGION[:nr]), vmin=-0.5, vmax=nr - 0.5,
                      interpolation="nearest")
    ejes[0, 2].set_title(f"{nr} región(es) por actividad", fontsize=17)
    for ax in ejes[0]:
        ax.set_axis_off()

    for r in range(nr):
        mascara = regiones == r
        g = gris[mascara]
        a = escalar(actividad[mascara])
        conteos, _, _ = np.histogram2d(a, g, bins=BINS, range=((0, 1), (0, 1)))
        paisaje = np.log1p(conteos)
        filtrado = np.maximum(paisaje - ndimage.gaussian_filter1d(paisaje, SIGMA_FILTRO, axis=1, mode="constant"), 0)

        fila = 1 + r
        ax = ejes[fila, 0]
        ax.set_facecolor("#ffd9e0")
        ax.imshow(np.where(mascara, gris, np.nan), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"región {r} ({mascara.mean():.0%} de la imagen; rosa = fuera)", fontsize=17,
                     color=COLORES_REGION[r])
        ax.set_xticks([])
        ax.set_yticks([])
        for ax, datos, titulo in ((ejes[fila, 1], paisaje, "paisaje ORIGINAL de la región"),
                                  (ejes[fila, 2], filtrado, "paisaje FILTRADO de la región\n(sin lo ancho de cada fila)")):
            im = ax.imshow(datos, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="Greys")
            ax.set_xlabel("gris (0-1, escala de toda la imagen)", fontsize=14)
            ax.set_ylabel("actividad (rango de la región)", fontsize=14)
            ax.set_title(titulo, fontsize=17)
            fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(SALIDA / f"24_paisaje_por_region_{nombre}.png", dpi=42)
    plt.close(fig)
print("ok")
