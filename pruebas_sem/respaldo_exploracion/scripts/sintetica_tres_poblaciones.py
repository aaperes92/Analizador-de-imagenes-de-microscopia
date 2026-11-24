"""Imagen sintética con tres poblaciones discretas y los mismos gráficos que para la ROI 3.

Imagen 600 x 600 sin ruido: fondo gris 0.8; manchas grandes 0.5; manchas chicas 0.2
(skimage.data.binary_blobs). Solo tres valores de gris.
Gráficos: imagen, mapa de actividad (gaussiana(sobel), 25 px), paisaje gris-actividad,
transformada 2D del paisaje y transformada fila por fila.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

SIGMA_ACTIVIDAD = 25
BINS = 128
LADO = 600

grandes = ski.data.binary_blobs(LADO, blob_size_fraction=0.12, volume_fraction=0.35, rng=1)
chicas = ski.data.binary_blobs(LADO, blob_size_fraction=0.03, volume_fraction=0.15, rng=2)
imagen = np.full((LADO, LADO), 0.8)
imagen[grandes] = 0.5
imagen[chicas & ~grandes] = 0.2
print("fracción de cada gris:", {g: round(float(np.mean(imagen == g)), 3) for g in (0.2, 0.5, 0.8)})

actividad = ski.filters.gaussian(ski.filters.sobel(imagen), SIGMA_ACTIVIDAD)
actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
conteos, _, _ = np.histogram2d(actividad.ravel(), imagen.ravel(), bins=BINS, range=((0, 1), (0, 1)))
paisaje = np.log1p(conteos)
fft2 = np.fft.fftshift(np.abs(np.fft.fft2(paisaje)))
fft_filas = np.abs(np.fft.rfft(paisaje, axis=1))

fig, ejes = plt.subplots(2, 3, figsize=(26, 16))
ejes[0, 0].imshow(imagen, cmap="gray", vmin=0, vmax=1)
ejes[0, 0].set_title("imagen sintética: 3 grises (0.2, 0.5, 0.8), sin ruido", fontsize=16)
ejes[0, 0].set_axis_off()

im = ejes[0, 1].imshow(actividad, cmap="gray")
ejes[0, 1].set_title(f"mapa de actividad (σ = {SIGMA_ACTIVIDAD} px)\nclaro = muchos bordes alrededor", fontsize=16)
ejes[0, 1].set_axis_off()
fig.colorbar(im, ax=ejes[0, 1], fraction=0.046, label="actividad (0-1)")

for ax in (ejes[0, 2], ejes[1, 0]):
    im = ax.imshow(paisaje, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="Greys")
    ax.set_xlabel("gris (0-1)  ['frecuencia']", fontsize=14)
    ax.set_ylabel("actividad (0-1)  ['tiempo']", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + cantidad de píxeles)")
ejes[0, 2].set_title("paisaje gris-actividad\n(más oscuro = más píxeles)", fontsize=16)
ejes[1, 0].set_title("paisaje gris-actividad (el mismo, para comparar)", fontsize=16)

frec = np.fft.fftshift(np.fft.fftfreq(BINS))
im = ejes[1, 1].imshow(np.log1p(fft2), origin="lower", aspect="auto", cmap="magma",
                       extent=(frec[0], frec[-1], frec[0], frec[-1]))
ejes[1, 1].set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=13)
ejes[1, 1].set_ylabel("frecuencia a lo largo de la actividad (ciclos por bin)", fontsize=13)
ejes[1, 1].set_title("1) transformada 2D del paisaje: log(1 + |FFT2|)", fontsize=16)
fig.colorbar(im, ax=ejes[1, 1], fraction=0.046, label="amplitud (log)")

frec_filas = np.fft.rfftfreq(BINS)
im = ejes[1, 2].imshow(np.log1p(fft_filas), origin="lower", aspect="auto", cmap="magma",
                       extent=(frec_filas[0], frec_filas[-1], 0, 1))
ejes[1, 2].set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=13)
ejes[1, 2].set_ylabel("actividad (0-1)  ['tiempo']", fontsize=14)
ejes[1, 2].set_title("2) fila por fila: log(1 + |FFT|)\nde la distribución de grises de cada actividad", fontsize=16)
fig.colorbar(im, ax=ejes[1, 2], fraction=0.046, label="amplitud (log)")

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/20_sintetica_tres_poblaciones_paisaje_fourier.png"), dpi=50)
print("ok")
