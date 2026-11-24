"""Paso 1: ROI 1 — imagen, clasificación en fases y mapa de iluminación (solo mirar).

Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

N = 4
RADIO_FRANJA = 2

roi = np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy")
suave = suavizar(roi)

# Clasificación en N fases gruesas (multi-Otsu)
umbrales = ski.filters.threshold_multiotsu(suave, classes=N)
clases = np.digitize(suave, umbrales)
grises = np.array([np.median(suave[clases == k]) for k in range(N)])

# Ventana: 1.5 x grosor típico de la fase que ocupa más área (clasificación con 3 fases)
clases3 = np.digitize(suave, ski.filters.threshold_multiotsu(suave, classes=3))
dominante = np.argmax([np.sum(clases3 == k) for k in range(3)])
esqueleto, distancia = ski.morphology.medial_axis(clases3 == dominante, return_distance=True)
grosor = 2 * float(np.median(distancia[esqueleto]))
sigma_v = 1.5 * grosor

# Mapa de iluminación con el interior de las fases
disco = ski.morphology.disk(RADIO_FRANJA)
interior = np.zeros(suave.shape, bool)
for k in range(N):
    interior |= ski.morphology.erosion(clases == k, disco)
peso = interior.astype(np.float32)
ilum = ski.filters.gaussian(peso * suave, sigma_v) / np.maximum(
    ski.filters.gaussian(peso * grises[clases], sigma_v), 1e-6)

print(f"umbrales = {np.round(umbrales, 3)}   gris típico por fase = {np.round(grises, 3)}")
print(f"área de cada fase = {np.round([np.mean(clases == k) for k in range(N)], 3)}")
print(f"grosor típico de la fase dominante = {grosor:.1f} px  ->  ventana σ = {sigma_v:.1f} px")
print(f"mapa de iluminación: percentiles 1-99 = {np.percentile(ilum, 1):.3f} - {np.percentile(ilum, 99):.3f}")

fig, ejes = plt.subplots(1, 3, figsize=(21, 7.5))
lo, hi = np.percentile(suave, (0.5, 99.5))
ejes[0].imshow(suave, cmap="gray", vmin=lo, vmax=hi)
ejes[0].set_title("ROI 1 suavizada", fontsize=15)
ejes[1].imshow(clases, cmap="gray")
ejes[1].set_title(f"Fases gruesas (N={N})\nnegro = fase más oscura · blanco = más clara", fontsize=15)
im = ejes[2].imshow(ilum, cmap="gray", vmin=0.95, vmax=1.05)
ejes[2].set_title(f"Mapa de iluminación (ventana σ={sigma_v:.0f} px)\n"
                  "oscuro = más oscuro de lo que le corresponde a su fase", fontsize=15)
fig.colorbar(im, ax=ejes[2], fraction=0.046)
for ax in ejes:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(Path("pruebas_sem/sombras/06_paso1_ROI1_fases_y_mapa.png"), dpi=60)
print("ok")
