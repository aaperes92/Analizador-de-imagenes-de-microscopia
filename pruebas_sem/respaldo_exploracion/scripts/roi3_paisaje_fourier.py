"""ROI 3: transformada de Fourier del paisaje gris-actividad (solo para ver).

Entrada: paisaje = log(1 + histograma 2D 128 x 128), filas = actividad, columnas = gris.
1. Transformada 2D: |FFT2| centrada, en escala log.
2. Fila por fila: para cada actividad, |FFT| de su fila (distribución de grises) a lo largo del gris.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SIGMA_ACTIVIDAD = 25
BINS = 128


def escalar_min_max(valores):
    return (valores - valores.min()) / (valores.max() - valores.min())


roi = np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")
suave = suavizar(roi)
gris = escalar_min_max(suave)
actividad = escalar_min_max(ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD))
conteos, _, _ = np.histogram2d(actividad.ravel(), gris.ravel(), bins=BINS, range=((0, 1), (0, 1)))
paisaje = np.log1p(conteos)          # filas = actividad, columnas = gris

fft2 = np.fft.fftshift(np.abs(np.fft.fft2(paisaje)))
fft_filas = np.abs(np.fft.rfft(paisaje, axis=1))   # para cada actividad, espectro a lo largo del gris

fig, ejes = plt.subplots(1, 3, figsize=(26, 8))
im = ejes[0].imshow(paisaje, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="Greys")
ejes[0].set_xlabel("gris (0-1)  ['frecuencia']", fontsize=14)
ejes[0].set_ylabel("actividad (0-1)  ['tiempo']", fontsize=14)
ejes[0].set_title("paisaje gris-actividad: log(1 + cantidad de píxeles)", fontsize=15)
fig.colorbar(im, ax=ejes[0], fraction=0.046)

frec = np.fft.fftshift(np.fft.fftfreq(BINS))
im = ejes[1].imshow(np.log1p(fft2), origin="lower", aspect="auto", cmap="magma",
                    extent=(frec[0], frec[-1], frec[0], frec[-1]))
ejes[1].set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=13)
ejes[1].set_ylabel("frecuencia a lo largo de la actividad (ciclos por bin)", fontsize=13)
ejes[1].set_title("1) transformada 2D del paisaje: log(1 + |FFT2|), centrada", fontsize=15)
fig.colorbar(im, ax=ejes[1], fraction=0.046)

frec_filas = np.fft.rfftfreq(BINS)
im = ejes[2].imshow(np.log1p(fft_filas), origin="lower", aspect="auto", cmap="magma",
                    extent=(frec_filas[0], frec_filas[-1], 0, 1))
ejes[2].set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=13)
ejes[2].set_ylabel("actividad (0-1)  ['tiempo']", fontsize=14)
ejes[2].set_title("2) fila por fila: para cada actividad,\nlog(1 + |FFT|) de su distribución de grises", fontsize=15)
fig.colorbar(im, ax=ejes[2], fraction=0.046)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/19_ROI3_paisaje_fourier.png"), dpi=55)
print("ok")
