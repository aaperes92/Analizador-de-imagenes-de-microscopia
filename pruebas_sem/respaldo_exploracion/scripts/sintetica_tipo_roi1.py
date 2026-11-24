"""Patrón artificial con el aspecto de la ROI 1, con tres poblaciones.

Campo aleatorio suavizado de forma alargada (gaussiana de 3 x 6 px, rotada 60°), normalizado:
- red oscura (|campo| < 0.3): gris 0.58  -> los "canales" que separan las islas;
- islas tipo A (campo >= 0.3): gris 0.66;
- islas tipo B (campo <= -0.3): gris 0.74.
Los grises están en el rango estrecho de la ROI 1 (0.57-0.74). Versión nítida y con degradé
(gaussiana 1.5 px) + ruido residual (desvío 0.003). Arriba, la ROI 1 real para comparar.
Gráficos: imagen, población verdadera, mapa de actividad (gaussiana(sobel), 25 px) y paisaje
gris-actividad coloreado por población.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, "src")
from sem.suavizado import suavizar

ALTO, ANCHO = 520, 560
SIGMA_ACTIVIDAD = 25
BINS = 128
GRISES = np.array([0.58, 0.66, 0.74])
NOMBRES = ["red oscura", "islas A", "islas B"]
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])


def patron(semilla=7):
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * max(ALTO, ANCHO))
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 6))
    campo = ndimage.rotate(campo, 60, reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    campo = campo[y0:y0 + ALTO, x0:x0 + ANCHO]
    campo /= campo.std()
    etiquetas = np.where(np.abs(campo) < 0.3, 0, np.where(campo >= 0.3, 1, 2))
    return etiquetas, rng


def componer(capas):
    maximo = max(c.max() for c in capas) or 1
    rgb = np.ones(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        rgb -= (c / maximo)[..., None] * (1 - np.array(to_rgb(color)))
    return np.clip(rgb, 0, 1)


def actividad_de(img):
    a = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    return (a - a.min()) / (a.max() - a.min())


etiquetas, rng = patron()
nitida = GRISES[etiquetas]
degrade = ski.filters.gaussian(nitida, 1.5) + rng.normal(0, 0.003, nitida.shape)
print("fracción de cada población:", {n: round(float(np.mean(etiquetas == p)), 3) for p, n in enumerate(NOMBRES)})

real = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
lo, hi = np.percentile(real, (0.5, 99.5))

fig, ejes = plt.subplots(3, 4, figsize=(34, 25))
# Fila 0: ROI 1 real (referencia)
ejes[0, 0].imshow(real, cmap="gray", vmin=lo, vmax=hi)
ejes[0, 0].set_title("ROI 1 REAL (suavizada), para comparar", fontsize=19)
act_real = actividad_de(real)
im = ejes[0, 2].imshow(act_real, cmap="magma")
ejes[0, 2].set_title("actividad de la ROI 1 real", fontsize=19)
fig.colorbar(im, ax=ejes[0, 2], fraction=0.046)
conteos, _, _ = np.histogram2d(act_real.ravel(), real.ravel(), bins=BINS, range=((0, 1), (lo - 0.05, hi + 0.05)))
ejes[0, 3].imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE,
                  extent=(lo - 0.05, hi + 0.05, 0, 1))
ejes[0, 3].set_xlabel("gris", fontsize=15)
ejes[0, 3].set_ylabel("actividad (0-1)", fontsize=15)
ejes[0, 3].set_title("paisaje de la ROI 1 real (sin poblaciones conocidas)", fontsize=19)
ejes[0, 1].axis("off")
for ax in (ejes[0, 0], ejes[0, 2]):
    ax.set_axis_off()

for fila, (nombre, img) in enumerate((("artificial, bordes nítidos", nitida),
                                      ("artificial, con degradé (σ=1.5px) + ruido", degrade)), start=1):
    actividad = actividad_de(img)
    capas = [np.log1p(np.histogram2d(actividad[etiquetas == p], img[etiquetas == p], bins=BINS,
                                     range=((0, 1), (lo - 0.05, hi + 0.05)))[0]) for p in range(3)]
    ejes[fila, 0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ejes[fila, 0].set_title(nombre, fontsize=19)
    ejes[fila, 1].imshow(etiquetas, cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5, interpolation="nearest")
    ejes[fila, 1].set_title("población verdadera de cada píxel", fontsize=19)
    im = ejes[fila, 2].imshow(actividad, cmap="magma")
    ejes[fila, 2].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=19)
    fig.colorbar(im, ax=ejes[fila, 2], fraction=0.046)
    for ax in ejes[fila, :3]:
        ax.set_axis_off()
    ax = ejes[fila, 3]
    ax.imshow(componer(capas), origin="lower", aspect="auto", extent=(lo - 0.05, hi + 0.05, 0, 1))
    ax.set_xlabel("gris", fontsize=15)
    ax.set_ylabel("actividad (0-1)", fontsize=15)
    ax.set_title("paisaje gris-actividad, coloreado por población", fontsize=19)

fig.legend(handles=[Patch(color=COLORES[p], label=f"{NOMBRES[p]} (gris {GRISES[p]:.2f})") for p in range(3)],
           loc="upper center", ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(Path("pruebas_sem/segmentacion/33_sintetica_tipo_ROI1_tres_poblaciones.png"), dpi=34)
print("ok")
