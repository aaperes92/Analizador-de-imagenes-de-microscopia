"""Prueba de corrección de iluminación comparando cada fase con su propio gris.

Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/sombras/.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.carga import cargar_imagen, leer_metadata, recortar_barra_datos
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/sombras")
N_POBLACIONES = 3          # matriz, canales y zona negra (a ojo, para esta prueba)
FACTOR_VENTANA = 2.0       # sigma de la ventana = FACTOR_VENTANA x tamaño de estructura


def tamano_estructura(imagen):
    """Distancia (px) donde la autocorrelación radial cae a cero por primera vez."""
    centrada = imagen - imagen.mean()
    espectro = np.abs(np.fft.fft2(centrada)) ** 2
    autocorr = np.fft.fftshift(np.real(np.fft.ifft2(espectro)))
    autocorr /= autocorr.max()
    cy, cx = np.array(autocorr.shape) // 2
    y, x = np.indices(autocorr.shape)
    radio = np.hypot(y - cy, x - cx).astype(int)
    perfil = np.bincount(radio.ravel(), autocorr.ravel()) / np.bincount(radio.ravel())
    cero = np.argmax(perfil < 0)
    return float(cero), perfil


def iluminacion_por_fases(imagen, sigma_ventana, n_poblaciones):
    """Mapa de iluminación: promedio local de la imagen / promedio local del gris de cada fase."""
    umbrales = ski.filters.threshold_multiotsu(imagen, classes=n_poblaciones)
    clases = np.digitize(imagen, umbrales)
    grises = np.array([np.median(imagen[clases == k]) for k in range(n_poblaciones)])
    esperado = grises[clases]
    ilum = (ski.filters.gaussian(imagen, sigma_ventana)
            / ski.filters.gaussian(esperado, sigma_ventana))
    return ilum, clases, grises


def iluminacion_simple(imagen, sigma_ventana):
    """Promedio local sin tener en cuenta las fases (como el método anterior), normalizado."""
    fondo = ski.filters.gaussian(imagen, sigma_ventana)
    return fondo / fondo.mean()


casos = {
    "ROI_3": np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy"),
    "ROI_4": np.load("data/sem/roi/BSE_1/BSE_1_4/roi_BSE_1_4.npy"),
}
ruta = Path("imagenes/SEM/BSE/BSE_1.tif")
casos["BSE_1_completa"] = recortar_barra_datos(cargar_imagen(ruta), leer_metadata(ruta))

fig, ejes = plt.subplots(len(casos), 6, figsize=(27, 4.8 * len(casos)))
for i, (nombre, imagen) in enumerate(casos.items()):
    t0 = time.perf_counter()
    suave = suavizar(imagen)
    t_suave = time.perf_counter() - t0
    L, _ = tamano_estructura(suave)
    sigma_v = FACTOR_VENTANA * L
    ilum, clases, grises = iluminacion_por_fases(suave, sigma_v, N_POBLACIONES)
    ilum_simple = iluminacion_simple(suave, sigma_v)
    corregida = suave / ilum
    print(f"{nombre}: {imagen.shape[1]}x{imagen.shape[0]} px  suavizado {t_suave:.0f}s  "
          f"tamaño estructura L={L:.0f}px  sigma ventana={sigma_v:.0f}px  "
          f"grises por fase={np.round(grises, 3)}  "
          f"iluminación por fases: min={ilum.min():.3f} max={ilum.max():.3f} "
          f"(percentiles 1-99: {np.percentile(ilum, 1):.3f}-{np.percentile(ilum, 99):.3f})")

    ejes[i, 0].imshow(suave, cmap="gray", vmin=0, vmax=1)
    ejes[i, 0].set_title(f"{nombre} suavizada", fontsize=11)
    ejes[i, 1].imshow(clases, cmap="viridis")
    ejes[i, 1].set_title(f"fases gruesas (multi-Otsu, {N_POBLACIONES})", fontsize=11)
    for j, (mapa, titulo) in enumerate(((ilum, "iluminación POR FASES"),
                                        (ilum_simple, "promedio local SIMPLE (método anterior)"))):
        im = ejes[i, 2 + j].imshow(mapa, cmap="RdBu_r", vmin=0.85, vmax=1.15)
        ejes[i, 2 + j].set_title(f"{titulo}\nventana σ={sigma_v:.0f}px", fontsize=11)
        fig.colorbar(im, ax=ejes[i, 2 + j], fraction=0.046)
    ejes[i, 4].imshow(corregida, cmap="gray", vmin=0, vmax=1)
    ejes[i, 4].set_title("corregida (suavizada / iluminación por fases)", fontsize=11)
    for ax in ejes[i, :5]:
        ax.set_axis_off()
    ax = ejes[i, 5]
    ax.hist(suave.ravel(), bins=256, range=(0, 1), histtype="step", color="0.5", label="suavizada")
    ax.hist(corregida.ravel(), bins=256, range=(0, 1), histtype="step", color="C3", label="corregida")
    ax.set_yscale("log")
    ax.set_xlabel("intensidad")
    ax.set_title("histogramas", fontsize=11)
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(SALIDA / "01_iluminacion_por_fases_vs_simple_BSE_1.png", dpi=55)
print("ok")
