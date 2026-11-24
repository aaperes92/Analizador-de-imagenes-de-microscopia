"""Prueba 4: qué cambia realmente la corrección por fases (N=4, interior de las fases).

Por caso: antes | mapa de iluminación usado | después | diferencia (después - antes).
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/sombras/.
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
N = 4
RADIO_FRANJA = 2


def clasificar(imagen, n):
    return np.digitize(imagen, ski.filters.threshold_multiotsu(imagen, classes=n))


def espesor_fase_dominante(clases, n):
    k = np.argmax([np.sum(clases == k) for k in range(n)])
    esqueleto, distancia = ski.morphology.medial_axis(clases == k, return_distance=True)
    return 2 * float(np.median(distancia[esqueleto]))


def corregir_por_fases(imagen, n, sigma_ventana):
    clases = clasificar(imagen, n)
    grises = np.array([np.median(imagen[clases == k]) for k in range(n)])
    disco = ski.morphology.disk(RADIO_FRANJA)
    interior = np.zeros(imagen.shape, bool)
    for k in range(n):
        interior |= ski.morphology.erosion(clases == k, disco)
    peso = interior.astype(np.float32)
    numerador = ski.filters.gaussian(peso * imagen, sigma_ventana)
    denominador = ski.filters.gaussian(peso * grises[clases], sigma_ventana)
    iluminacion = numerador / np.maximum(denominador, 1e-6)
    return imagen / iluminacion, iluminacion


casos = {f"ROI_{k}": np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy") for k in (1, 2, 3, 4)}
ruta = Path("imagenes/SEM/BSE/BSE_1.tif")
casos["BSE_1_completa"] = recortar_barra_datos(cargar_imagen(ruta), leer_metadata(ruta))

fig, ejes = plt.subplots(len(casos), 4, figsize=(22, 5.2 * len(casos)))
for i, (nombre, imagen) in enumerate(casos.items()):
    antes = suavizar(imagen)
    sigma_v = 1.5 * espesor_fase_dominante(clasificar(antes, 3), 3)
    despues, ilum = corregir_por_fases(antes, N, sigma_v)
    diferencia = despues - antes
    lim = np.percentile(np.abs(diferencia), 99)
    print(f"{nombre}: ventana σ={sigma_v:.0f}px  iluminación {ilum.min():.3f}-{ilum.max():.3f}  "
          f"cambio máximo (percentil 99) = ±{lim:.3f} ({lim * 255:.1f} niveles de 8 bits)")

    lo, hi = np.percentile(antes, (0.5, 99.5))
    ejes[i, 0].imshow(antes, cmap="gray", vmin=lo, vmax=hi)
    ejes[i, 0].set_title(f"{nombre}\nANTES (suavizada)", fontsize=13)
    im = ejes[i, 1].imshow(ilum, cmap="gray", vmin=0.92, vmax=1.08)
    ejes[i, 1].set_title(f"MAPA DE ILUMINACIÓN usado (N={N}, σ={sigma_v:.0f}px)\n"
                         "oscuro = zona más oscura de lo que le corresponde", fontsize=11)
    fig.colorbar(im, ax=ejes[i, 1], fraction=0.046)
    ejes[i, 2].imshow(despues, cmap="gray", vmin=lo, vmax=hi)
    ejes[i, 2].set_title("DESPUÉS (corregida)\nmisma escala de gris que ANTES", fontsize=13)
    im = ejes[i, 3].imshow(diferencia, cmap="gray", vmin=-lim, vmax=lim)
    ejes[i, 3].set_title("DIFERENCIA (después − antes), contraste estirado\n"
                         "claro = se aclaró · oscuro = se oscureció · gris medio = sin cambio", fontsize=11)
    fig.colorbar(im, ax=ejes[i, 3], fraction=0.046)
    for ax in ejes[i]:
        ax.set_axis_off()

fig.tight_layout()
fig.savefig(SALIDA / "05_antes_mapa_despues_diferencia_N4_BSE_1.png", dpi=50)
print("ok")
