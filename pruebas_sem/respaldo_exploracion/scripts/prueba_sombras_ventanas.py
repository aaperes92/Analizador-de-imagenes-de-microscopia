"""Prueba 2: tamaño de estructura por espesor de fases y mapas de iluminación con varias ventanas.

Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.carga import cargar_imagen, leer_metadata, recortar_barra_datos
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/sombras")
N_POBLACIONES = 3


def clasificar(imagen, n):
    umbrales = ski.filters.threshold_multiotsu(imagen, classes=n)
    return np.digitize(imagen, umbrales)


def espesores(clases, n):
    """Espesor típico (px) de cada fase: 2 x mediana de la distancia al borde sobre su esqueleto."""
    resultado = []
    for k in range(n):
        mascara = clases == k
        esqueleto, distancia = ski.morphology.medial_axis(mascara, return_distance=True)
        valores = distancia[esqueleto]
        resultado.append(2 * float(np.median(valores)) if valores.size else np.nan)
    return resultado


def iluminacion_por_fases(imagen, clases, sigma_ventana, n):
    grises = np.array([np.median(imagen[clases == k]) for k in range(n)])
    return (ski.filters.gaussian(imagen, sigma_ventana)
            / ski.filters.gaussian(grises[clases], sigma_ventana))


casos = {
    "ROI_3": np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy"),
    "ROI_4": np.load("data/sem/roi/BSE_1/BSE_1_4/roi_BSE_1_4.npy"),
}
ruta = Path("imagenes/SEM/BSE/BSE_1.tif")
casos["BSE_1_completa"] = recortar_barra_datos(cargar_imagen(ruta), leer_metadata(ruta))

VENTANAS = (10, 25, 50, 100)
fig, ejes = plt.subplots(len(casos), 1 + len(VENTANAS), figsize=(5.2 * (1 + len(VENTANAS)), 4.6 * len(casos)))
for i, (nombre, imagen) in enumerate(casos.items()):
    suave = suavizar(imagen)
    clases = clasificar(suave, N_POBLACIONES)
    esp = espesores(clases, N_POBLACIONES)
    fracciones = [np.mean(clases == k) for k in range(N_POBLACIONES)]
    print(f"{nombre}: espesor típico por fase (px) = {np.round(esp, 1)}  "
          f"fracción de área = {np.round(fracciones, 3)}")
    ejes[i, 0].imshow(suave, cmap="gray", vmin=0, vmax=1)
    ejes[i, 0].set_title(f"{nombre} suavizada", fontsize=12)
    ejes[i, 0].set_axis_off()
    for j, sigma_v in enumerate(VENTANAS):
        ilum = iluminacion_por_fases(suave, clases, sigma_v, N_POBLACIONES)
        im = ejes[i, 1 + j].imshow(ilum, cmap="RdBu_r", vmin=0.92, vmax=1.08)
        ejes[i, 1 + j].set_title(
            f"iluminación por fases, σ={sigma_v}px\n"
            f"percentiles 1-99: {np.percentile(ilum, 1):.3f}-{np.percentile(ilum, 99):.3f}",
            fontsize=11)
        ejes[i, 1 + j].set_axis_off()
        fig.colorbar(im, ax=ejes[i, 1 + j], fraction=0.046)

fig.tight_layout()
fig.savefig(SALIDA / "02_mapas_iluminacion_varias_ventanas_BSE_1.png", dpi=55)
print("ok")
