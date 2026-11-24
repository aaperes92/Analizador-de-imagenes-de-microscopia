"""Cuatro imágenes sintéticas de 5 poblaciones: burbujas y franjas, con bordes nítidos y con degradé.

Poblaciones (gris): 0.10, 0.30, 0.50, 0.70, 0.90, cada una con un color.
- Burbujas: fondo 0.50; burbujas de radio 4, 8, 15 y 30 px para los grises 0.10, 0.30, 0.70, 0.90.
- Franjas: franjas verticales de espesor 3, 6, 10, 18 y 30 px (una población cada una), en orden
  al azar, repetidas a lo ancho de la imagen.
- Degradé: gaussiana de 2 px sobre la imagen nítida. La población "verdadera" de cada píxel sigue
  siendo la de la imagen nítida.
Gráficos por imagen: imagen, verdad en colores, mapa de actividad (gaussiana(sobel), 25 px),
paisaje gris-actividad y Fourier fila por fila, ambos coloreados por población (cada población
pinta su parte con su color; donde se superponen, los colores se mezclan).
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch

LADO = 600
SIGMA_ACTIVIDAD = 25
SIGMA_DEGRADE = 2
BINS = 128
GRISES = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
RADIOS = {0: 4, 1: 8, 3: 15, 4: 30}           # población -> radio de sus burbujas (la 2 es el fondo)
ESPESORES_GUSANO = {0: 3, 1: 6, 3: 12, 4: 24}  # población -> espesor de sus gusanos (la 2 es el fondo)


def burbujas(semilla=0):
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    # Primero las grandes, después las chicas (las chicas quedan encima)
    for p, radio in sorted(RADIOS.items(), key=lambda kv: -kv[1]):
        cantidad = int(0.08 * LADO * LADO / (np.pi * radio ** 2))
        for _ in range(cantidad):
            fy, fx = rng.integers(0, LADO, 2)
            rr, cc = ski.draw.disk((fy, fx), radio, shape=etiquetas.shape)
            etiquetas[rr, cc] = p
    return etiquetas


def gusanos(semilla=0):
    """Fondo 0.50 con gusanos serpenteantes: cada población tiene su espesor (los gruesos primero)."""
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, espesor in sorted(ESPESORES_GUSANO.items(), key=lambda kv: -kv[1]):
        radio = espesor / 2
        pintado = 0
        while pintado < 0.08 * LADO * LADO:
            y, x = rng.uniform(0, LADO, 2)
            angulo = rng.uniform(0, 2 * np.pi)
            giro = 0.0
            pasos = int(rng.uniform(120, 260))
            for _ in range(pasos):
                # el giro cambia de a poco: curvas suaves, como un gusano
                giro = 0.85 * giro + rng.normal(0, 0.06)
                angulo += giro
                y, x = y + 2 * np.sin(angulo), x + 2 * np.cos(angulo)
                if not (0 <= y < LADO and 0 <= x < LADO):
                    break
                rr, cc = ski.draw.disk((y, x), max(radio, 1), shape=etiquetas.shape)
                etiquetas[rr, cc] = p
            pintado = np.sum(etiquetas == p)
    return etiquetas


def paisaje_por_poblacion(img, etiquetas, actividad):
    """Lista de log(1 + histograma 2D) de cada población (filas = actividad, columnas = gris)."""
    return [np.log1p(np.histogram2d(actividad[etiquetas == p], img[etiquetas == p], bins=BINS,
                                    range=((0, 1), (0, 1)))[0]) for p in range(5)]


def componer(capas):
    """Mezcla las capas (una por población) sobre blanco, cada una con su color."""
    maximo = max(c.max() for c in capas) or 1
    rgb = np.ones(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        peso = (c / maximo)[..., None]
        rgb -= peso * (1 - np.array(to_rgb(color)))
    return np.clip(rgb, 0, 1)


def componer_aditivo(capas):
    """Mezcla aditiva sobre negro: cada población suma su color; donde se superponen, se aclara."""
    maximo = max(c.max() for c in capas) or 1
    rgb = np.zeros(capas[0].shape + (3,))
    for c, color in zip(capas, COLORES):
        rgb += (c / maximo)[..., None] * np.array(to_rgb(color))
    return np.clip(rgb, 0, 1)


casos = {}
for geometria, etiquetas in (("burbujas", burbujas()), ("gusanos", gusanos())):
    nitida = GRISES[etiquetas]
    casos[f"{geometria}, bordes nítidos"] = (nitida, etiquetas)
    casos[f"{geometria}, con degradé (σ={SIGMA_DEGRADE}px)"] = (ski.filters.gaussian(nitida, SIGMA_DEGRADE), etiquetas)

fig, ejes = plt.subplots(len(casos), 5, figsize=(40, 8.2 * len(casos)))
for fila, (nombre, (img, etiquetas)) in enumerate(casos.items()):
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    capas = paisaje_por_poblacion(img, etiquetas, actividad)
    fourier = [np.log1p(np.abs(np.fft.rfft(c, axis=1))) for c in capas]

    ejes[fila, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ejes[fila, 0].set_title(f"{nombre}", fontsize=20)
    ejes[fila, 1].imshow(etiquetas, cmap=ListedColormap(COLORES), vmin=-0.5, vmax=4.5, interpolation="nearest")
    ejes[fila, 1].set_title("población verdadera de cada píxel", fontsize=20)
    im = ejes[fila, 2].imshow(actividad, cmap="magma")
    ejes[fila, 2].set_title(f"mapa de actividad (σ={SIGMA_ACTIVIDAD}px)", fontsize=20)
    fig.colorbar(im, ax=ejes[fila, 2], fraction=0.046)
    for ax in ejes[fila, :3]:
        ax.set_axis_off()

    ax = ejes[fila, 3]
    ax.imshow(componer(capas), origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("gris (0-1)", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("paisaje gris-actividad, coloreado por población", fontsize=20)

    ax = ejes[fila, 4]
    frec = np.fft.rfftfreq(BINS)
    ax.imshow(componer_aditivo(fourier), origin="lower", aspect="auto", extent=(frec[0], frec[-1], 0, 1))
    ax.set_xlabel("frecuencia a lo largo del gris (ciclos por bin)", fontsize=16)
    ax.set_ylabel("actividad (0-1)", fontsize=16)
    ax.set_title("Fourier fila por fila, una capa por población", fontsize=20)

leyenda = [Patch(color=COLORES[p], label=f"gris {GRISES[p]:.2f} · " +
                 (f"burbujas r={RADIOS[p]}px · gusanos {ESPESORES_GUSANO[p]}px" if p in RADIOS else "fondo"))
           for p in range(5)]
fig.legend(handles=leyenda, loc="upper center", ncol=5, fontsize=20, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(Path("pruebas_sem/segmentacion/31_sinteticas_5_poblaciones_paisaje_fourier.png"), dpi=34)
print("ok")
