"""Picos y cuencas en el plano gris-actividad (histograma 2D tratado como un paisaje).

1. gris = suavizada; actividad = gaussiana(sobel(suavizada), 25 px); ambos a [0, 1] (mín-máx).
2. Histograma 2D de 128 x 128, suavizado (gaussiana de 2 bins) = "paisaje".
3. Picos: skimage.feature.peak_local_max (distancia mínima 6 bins).
4. Cuencas: skimage.segmentation.watershed sobre el paisaje invertido, una por pico.
5. Se descartan los picos cuya cuenca tiene menos del 1 % de los píxeles y se recalcula.
6. Cada píxel va a la cuenca de su par (gris, actividad).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
BINS = 128
SIGMA_PAISAJE = 2.0
DISTANCIA_MINIMA = 6
MASA_MINIMA = 0.01
COLORES = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#000000"]


def escalar_min_max(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


def cuencas(paisaje, picos):
    marcadores = np.zeros(paisaje.shape, int)
    for i, (f, c) in enumerate(picos, start=1):
        marcadores[f, c] = i
    return ski.segmentation.watershed(-paisaje, marcadores, mask=paisaje > 0)


for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    gris = escalar_min_max(suave)
    actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))

    # Índice de bin de cada píxel (filas = actividad, columnas = gris)
    fila = np.minimum((actividad * BINS).astype(int), BINS - 1)
    col = np.minimum((gris * BINS).astype(int), BINS - 1)
    conteos = np.zeros((BINS, BINS))
    np.add.at(conteos, (fila.ravel(), col.ravel()), 1)
    paisaje = ski.filters.gaussian(conteos, SIGMA_PAISAJE, preserve_range=True)

    picos = ski.feature.peak_local_max(paisaje, min_distance=DISTANCIA_MINIMA, exclude_border=False)
    n_inicial = len(picos)
    while True:
        etiquetas_plano = cuencas(paisaje, picos)
        masas = np.array([conteos[etiquetas_plano == i + 1].sum() for i in range(len(picos))]) / conteos.sum()
        if len(picos) == 1 or masas.min() >= MASA_MINIMA:
            break
        picos = np.delete(picos, int(np.argmin(masas)), axis=0)   # se descarta el más chico y se recalcula

    # Regiones ordenadas por gris del pico (de oscura a clara)
    orden = np.argsort(picos[:, 1])
    reetiqueta = np.zeros(len(picos) + 1, int) - 1
    reetiqueta[orden + 1] = np.arange(len(picos))
    regiones = reetiqueta[etiquetas_plano[fila, col]]
    n = len(picos)
    fracciones = [np.mean(regiones == r) for r in range(n)]
    print(f"ROI_{k}: {n_inicial} picos encontrados, {n} tras descartar cuencas < {MASA_MINIMA:.0%}; "
          f"áreas {np.round(fracciones, 3)}")
    for r, p in enumerate(picos[orden]):
        print(f"   región {r}: pico en gris {p[1] / BINS:.2f}, actividad {p[0] / BINS:.2f}")

    fig, ejes = plt.subplots(1, 4, figsize=(31, 7.6), gridspec_kw={"width_ratios": [1.1, 1, 1, 0.5]})
    ax = ejes[0]
    ax.imshow(np.log1p(paisaje), origin="lower", cmap="Greys", extent=(0, 1, 0, 1), aspect="auto")
    bordes_plano = ski.segmentation.find_boundaries(etiquetas_plano, mode="inner")
    ax.imshow(np.ma.masked_where(~bordes_plano, bordes_plano), origin="lower", extent=(0, 1, 0, 1),
              aspect="auto", cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
    for r, p in enumerate(picos[orden]):
        ax.plot((p[1] + 0.5) / BINS, (p[0] + 0.5) / BINS, "o", ms=13, color=COLORES[r], mec="k", mew=1.5)
    ax.set_xlabel("gris (0-1)", fontsize=13)
    ax.set_ylabel("actividad (0-1)", fontsize=13)
    ax.set_title(f"ROI {k} — paisaje gris-actividad (log)\npuntos = picos, rojo = bordes de las cuencas", fontsize=14)

    ax = ejes[1]
    ax.imshow(regiones, cmap=ListedColormap(COLORES[:n]), vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    ax.set_title(f"{n} región(es) (colores sólidos)", fontsize=15)
    ax.set_axis_off()

    ax = ejes[2]
    lo, hi = np.percentile(suave, (0.5, 99.5))
    ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
    b = ski.segmentation.find_boundaries(regiones, mode="inner")
    ax.imshow(np.ma.masked_where(~b, b), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
    ax.set_title("bordes sobre la ROI suavizada", fontsize=15)
    ax.set_axis_off()

    ax = ejes[3]
    ax.bar(range(n), [100 * f for f in fracciones], color=COLORES[:n], edgecolor="k")
    ax.set_xticks(range(n), [f"reg. {r}" for r in range(n)], fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("área (%)", fontsize=15)
    for r, f in enumerate(fracciones):
        ax.text(r, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"17_plano_2d_picos_y_cuencas_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
