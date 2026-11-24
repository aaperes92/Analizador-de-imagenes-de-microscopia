"""Multi-Otsu directo sobre la ROI suavizada (sin ecualizar), con 2 y 3 clases.

Una figura por ROI: filas = N; columnas = superposición | histograma con umbrales | fracción de área.
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
SALIDA.mkdir(parents=True, exist_ok=True)
POBLACIONES = (2, 3)
# Colores por clase (de la más oscura a la más clara); la más clara queda sin color.
COLORES = ["#1f5fbf", "#e8891a", "#2ca02c", "#d62728"]

for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    lo, hi = np.percentile(suave, (0.5, 99.5))

    fig, ejes = plt.subplots(len(POBLACIONES), 3, figsize=(22, 7.2 * len(POBLACIONES)),
                             gridspec_kw={"width_ratios": [1.5, 1.2, 0.7]})
    print(f"ROI_{k}")
    for fila, n in enumerate(POBLACIONES):
        umbrales = ski.filters.threshold_multiotsu(suave, classes=n)
        clases = np.digitize(suave, umbrales)
        fracciones = [np.mean(clases == c) for c in range(n)]
        print(f"  N={n}: umbrales={np.round(umbrales, 3)}  fracción de área={np.round(fracciones, 3)}")

        # Superposición: ROI en gris + clases coloreadas semitransparentes (salvo la más clara)
        ax = ejes[fila, 0]
        ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
        capa = np.ma.masked_where(clases == n - 1, clases)
        ax.imshow(capa, cmap=ListedColormap(COLORES[:n - 1]), vmin=-0.5, vmax=n - 1.5,
                  alpha=0.45, interpolation="nearest")
        ax.set_title(f"ROI {k} — multi-Otsu N={n} sobre la suavizada\n"
                     "(color = clases oscuras; sin color = la más clara)", fontsize=14)
        ax.set_axis_off()

        ax = ejes[fila, 1]
        ax.hist(suave.ravel(), bins=256, range=(0, 1), color="0.4")
        for t in umbrales:
            ax.axvline(t, color="k", ls="--", lw=2)
        ax.set_yscale("log")
        ax.set_xlim(max(0, lo - 0.05), min(1, hi + 0.05))
        ax.set_xlabel("intensidad (suavizada)", fontsize=12)
        ax.set_title("histograma y umbrales: " + ", ".join(f"{t:.3f}" for t in umbrales), fontsize=13)

        ax = ejes[fila, 2]
        ax.bar(range(n), [100 * f for f in fracciones], color=COLORES[:n - 1] + ["0.75"], edgecolor="k")
        ax.set_xticks(range(n), [f"clase {c}" for c in range(n)])
        ax.set_ylim(0, 100)
        ax.set_ylabel("área (%)", fontsize=12)
        ax.set_title("fracción de área", fontsize=13)
        for c, f in enumerate(fracciones):
            ax.text(c, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"01_multiotsu_sobre_suavizada_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
