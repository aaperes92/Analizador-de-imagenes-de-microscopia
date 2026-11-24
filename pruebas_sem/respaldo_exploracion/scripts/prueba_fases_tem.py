"""Detección automática de fases del módulo TEM aplicada a las ROIs SEM (sin modificar TEM).

Usa src/tem/segmentation.py tal como está: detect_n_phases + select_threshold_method.
Dos entradas por ROI: suavizada y suavizada + white_tophat (radio 50), ambas llevadas a
[0, 1] con percentiles 0.1-99.9 como hace el cargador de TEM.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap

sys.path.insert(0, "src")
from sem.suavizado import suavizar
from tem.segmentation import analyze_histogram, detect_n_phases, select_threshold_method

SALIDA = Path("pruebas_sem/segmentacion")
COLORES = ["#1f5fbf", "#e8891a", "#2ca02c", "#d62728", "#9467bd"]


def normalizar(imagen):
    bajo, alto = np.percentile(imagen, (0.1, 99.9))
    return np.clip((imagen - bajo) / (alto - bajo), 0, 1).astype(np.float32)


def segmentar(imagen):
    """Devuelve (n_fases, método, umbrales, info del histograma, registro impreso)."""
    registro = io.StringIO()
    with contextlib.redirect_stdout(registro):
        info = analyze_histogram(imagen)
        n = detect_n_phases(imagen)
        metodo, umbrales = select_threshold_method(imagen, n)
    return n, metodo, np.asarray(umbrales), info, registro.getvalue()


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    disco = ski.morphology.disk(50, decomposition="crosses")
    entradas = {
        "suavizada": normalizar(suave),
        "suavizada + white_tophat (r=50)": normalizar(ski.morphology.white_tophat(suave, disco)),
    }

    fig, ejes = plt.subplots(2, 3, figsize=(22, 14.5), gridspec_kw={"width_ratios": [1.5, 1.2, 0.7]})
    for fila, (nombre, img) in enumerate(entradas.items()):
        n, metodo, umbrales, info, registro = segmentar(img)
        clases = np.digitize(img, umbrales)
        fracciones = [np.mean(clases == c) for c in range(n)]
        print(f"===== ROI_{k} — {nombre} =====")
        print(registro.strip())
        print(f"-> {n} fases, método '{metodo}', fracciones de área {np.round(fracciones, 3)}\n")

        ax = ejes[fila, 0]
        ax.imshow(img, cmap="gray")
        capa = np.ma.masked_where(clases == n - 1, clases)
        ax.imshow(capa, cmap=ListedColormap(COLORES[:n - 1]), vmin=-0.5, vmax=n - 1.5,
                  alpha=0.45, interpolation="nearest")
        ax.set_title(f"ROI {k} — {nombre}\n{n} fases detectadas, método '{metodo}' "
                     "(sin color = la más clara)", fontsize=14)
        ax.set_axis_off()

        ax = ejes[fila, 1]
        centros = (info["bin_edges"][:-1] + info["bin_edges"][1:]) / 2
        ax.bar(centros, info["histogram"], width=centros[1] - centros[0], color="0.75")
        ax.plot(centros, info["histogram_smooth"], color="k", lw=1.5)
        for p in info["peaks_positions"]:
            ax.axvline(p, color="C0", lw=1, ls=":")
        for t in umbrales:
            ax.axvline(t, color="C3", lw=2, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.8, None)
        ax.set_xlabel("intensidad normalizada", fontsize=12)
        ax.set_title("histograma: azul punteado = picos encontrados,\nrojo = umbrales finales", fontsize=13)

        ax = ejes[fila, 2]
        ax.bar(range(n), [100 * f for f in fracciones], color=COLORES[:n - 1] + ["0.75"], edgecolor="k")
        ax.set_xticks(range(n), [f"fase {c}" for c in range(n)])
        ax.set_ylim(0, 100)
        ax.set_title("fracción de área (%)", fontsize=13)
        for c, f in enumerate(fracciones):
            ax.text(c, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"02_deteccion_fases_TEM_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
