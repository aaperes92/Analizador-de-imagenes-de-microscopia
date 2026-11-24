"""Fases por picos del histograma (criterios del módulo TEM) con umbrales en los valles.

Cambios respecto de tem.segmentation.detect_n_phases + select_threshold_method:
1. Los umbrales son los valles entre los picos que quedan (no multi-Otsu).
2. Los extremos del histograma cuentan (lo negro puede ser una fase).
3. Masa mínima 1 % (en TEM 5 %) y sin mínimo forzado de 2 fases.
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
COLORES = ["#1f5fbf", "#e8891a", "#2ca02c", "#d62728", "#9467bd"]

# Parámetros del módulo TEM (src/tem/segmentation.py), salvo la masa mínima
BINS = 256
SIGMA_HISTOGRAMA = 2.0
PROMINENCIA_MINIMA = 0.05
DISTANCIA_MINIMA = 8
MASA_MINIMA = 0.01          # TEM usa 0.05
VALLE_MAXIMO = 0.75


def normalizar(imagen):
    bajo, alto = np.percentile(imagen, (0.1, 99.9))
    return np.clip((imagen - bajo) / (alto - bajo), 0, 1).astype(np.float32)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    total = h.sum()
    return [h[bordes[i]:bordes[i + 1]].sum() / total for i in range(len(picos))]


def fases_por_valles(imagen):
    conteos, bordes = np.histogram(imagen, bins=BINS, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SIGMA_HISTOGRAMA)
    # Se agrega un cero a cada lado para que un pico en el extremo (lo negro) se pueda detectar
    relleno = np.concatenate(([0.0], h, [0.0]))
    picos, _ = signal.find_peaks(relleno, prominence=PROMINENCIA_MINIMA * h.max(),
                                 distance=DISTANCIA_MINIMA)
    picos = [int(p) - 1 for p in picos]
    encontrados = list(picos)
    mensajes = []
    # Fusión (misma lógica que _merge_peaks de TEM): primero masa, después valle poco profundo
    while len(picos) > 1:
        m = masas(h, picos)
        peor = int(np.argmin(m))
        if m[peor] < MASA_MINIMA:
            mensajes.append(f"pico {centros[picos[peor]]:.3f} fusionado: masa {m[peor]:.1%}")
            picos.pop(peor)
            continue
        valles = valles_entre(h, picos)
        relaciones = [h[v] / min(h[a], h[b]) for v, a, b in zip(valles, picos, picos[1:])]
        i = int(np.argmax(relaciones))
        if relaciones[i] <= VALLE_MAXIMO:
            break
        mas_bajo = i if h[picos[i]] < h[picos[i + 1]] else i + 1
        mensajes.append(f"pico {centros[picos[mas_bajo]]:.3f} fusionado: valle al {relaciones[i]:.0%}")
        picos.pop(mas_bajo)
    umbrales = centros[valles_entre(h, picos)]
    return centros, conteos, h, encontrados, picos, umbrales, mensajes


for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    img = normalizar(suavizar(roi))
    centros, conteos, h, encontrados, picos, umbrales, mensajes = fases_por_valles(img)
    n = len(picos)
    clases = np.digitize(img, umbrales)
    fracciones = [np.mean(clases == c) for c in range(n)]
    print(f"ROI_{k}: picos encontrados {np.round(centros[encontrados], 3)}")
    for m in mensajes:
        print(f"  {m}")
    print(f"  -> {n} fase(s), picos {np.round(centros[picos], 3)}, umbrales {np.round(umbrales, 3)}, "
          f"fracciones {np.round(fracciones, 3)}")

    fig, ejes = plt.subplots(1, 3, figsize=(22, 7.5), gridspec_kw={"width_ratios": [1.5, 1.2, 0.7]})
    ax = ejes[0]
    ax.imshow(img, cmap="gray")
    if n > 1:
        capa = np.ma.masked_where(clases == n - 1, clases)
        ax.imshow(capa, cmap=ListedColormap(COLORES[:n - 1]), vmin=-0.5, vmax=n - 1.5,
                  alpha=0.45, interpolation="nearest")
    ax.set_title(f"ROI {k} (suavizada) — {n} fase(s)\n(sin color = la más clara)", fontsize=14)
    ax.set_axis_off()

    ax = ejes[1]
    ax.bar(centros, conteos, width=centros[1] - centros[0], color="0.75")
    ax.plot(centros, h, color="k", lw=1.5)
    for p in encontrados:
        ax.axvline(centros[p], color="C0", lw=1, ls=":")
    for t in umbrales:
        ax.axvline(t, color="C3", lw=2, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.8, None)
    ax.set_xlabel("intensidad normalizada", fontsize=12)
    ax.set_title("azul punteado = picos encontrados\nrojo = umbrales (valles entre picos que quedan)", fontsize=13)

    ax = ejes[2]
    ax.bar(range(n), [100 * f for f in fracciones], color=(COLORES[:n - 1] + ["0.75"]) if n > 1 else ["0.75"],
           edgecolor="k")
    ax.set_xticks(range(n), [f"fase {c}" for c in range(n)])
    ax.set_ylim(0, 105)
    ax.set_title("fracción de área (%)", fontsize=13)
    for c, f in enumerate(fracciones):
        ax.text(c, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"03_fases_valles_entre_picos_ROI{k}.png", dpi=50)
    plt.close(fig)
print("ok")
