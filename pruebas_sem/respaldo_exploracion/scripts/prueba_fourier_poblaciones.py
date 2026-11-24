"""Separar poblaciones filtrando el paisaje gris-actividad (quitar lo ancho, dejar lo angosto).

Imagen sintética parecida a la ROI 3 (650 x 650):
- zona lisa: matriz gris 0.70 con estrías diagonales suaves (variación continua, no es población);
- zona con patrón (arriba a la izquierda): red laberíntica de fase oscura 0.58 sobre matriz 0.70;
- partículas claras chicas 0.85 dispersas;
- franja negra diagonal 0.00 y, más allá, zona gris 0.45 (abajo a la derecha);
- degradé en las interfaces (gaussiana 1.5 px) y ruido residual (desvío 0.005).
Poblaciones reales: 0.00, 0.45, 0.58, 0.70, 0.85.

Prueba (igual para la sintética y la ROI 3 suavizada):
1. actividad = gaussiana(sobel, 25 px); paisaje = log(1 + histograma 2D 128 x 128).
2. Filtro en cada fila (a lo largo del gris): fila - fila suavizada (gaussiana 6 bins), negativos a 0.
   Equivale a quitar las frecuencias bajas de la transformada fila por fila, sin corte brusco.
3. Perfil = suma de las filas filtradas; picos (prominencia 5 % del máximo, distancia 3 bins).
4. Cada píxel va a la población de gris más cercano; si está a más de TOLERANCIA de todas, "interfaz".
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
BINS = 128
SIGMA_FILTRO = 6           # bins: lo más ancho que esto se considera "cortina" y se quita
TOLERANCIA = 0.03          # distancia de gris máxima a una población (si no, interfaz)
COLORES = ["#000000", "#8c564b", "#1f5fbf", "#e8d8a0", "#ff7f0e", "#2ca02c", "#9467bd"]
COLOR_INTERFAZ = "#ff2020"


def imagen_sintetica(lado=650, semilla=3):
    rng = np.random.default_rng(semilla)
    y, x = np.mgrid[0:lado, 0:lado] / lado
    img = np.full((lado, lado), 0.70)
    # Estrías diagonales suaves en toda la matriz (variación continua de +-0.02)
    ruido_estrias = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (2, 12))
    ruido_estrias = ndimage.rotate(ruido_estrias, 35, reshape=False, mode="reflect")
    img += 0.02 * ruido_estrias / ruido_estrias.std()
    # Zona con patrón arriba a la izquierda: red laberíntica (contornos de un campo aleatorio)
    zona_patron = (x + y) < 0.75
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), 6)
    campo /= campo.std()
    red = np.abs(campo) < 0.25
    verdad = np.full((lado, lado), 3)            # 0 negro, 1 gris, 2 fase oscura, 3 matriz, 4 partículas
    img[zona_patron & red] = 0.58
    verdad[zona_patron & red] = 2
    # Partículas claras chicas dispersas
    particulas = ski.data.binary_blobs(lado, blob_size_fraction=0.01, volume_fraction=0.02, rng=semilla)
    img[particulas] = 0.85
    verdad[particulas] = 4
    # Franja negra diagonal y zona gris más allá
    d = (x + y) - 1.55
    franja = np.abs(d) < 0.03
    gris = d >= 0.03
    img[franja] = 0.0
    verdad[franja] = 0
    img[gris] = 0.45
    verdad[gris] = 1
    # Degradé en las interfaces y ruido residual
    img = ndimage.gaussian_filter(img, 1.5) + rng.normal(0, 0.005, img.shape)
    return np.clip(img, 0, 1), verdad


def poblaciones_por_paisaje(img):
    """Devuelve (paisaje, paisaje filtrado, perfil, perfil original, grises de las poblaciones)."""
    gris = (img - img.min()) / (img.max() - img.min())
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), gris.ravel(), bins=BINS, range=((0, 1), (0, 1)))
    paisaje = np.log1p(conteos)
    filtrado = np.maximum(paisaje - ndimage.gaussian_filter1d(paisaje, SIGMA_FILTRO, axis=1, mode="constant"), 0)
    perfil = filtrado.sum(axis=0)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=0.05 * perfil.max(),
                                 distance=3)
    centros = (np.arange(BINS) + 0.5) / BINS
    grises_01 = centros[picos - 1]
    grises = img.min() + grises_01 * (img.max() - img.min())
    return paisaje, filtrado, perfil, paisaje.sum(axis=0), grises, gris


def clasificar(img, grises):
    distancias = np.abs(img[..., None] - grises[None, None, :])
    clase = np.argmin(distancias, axis=-1)
    clase[distancias.min(axis=-1) > TOLERANCIA] = -1
    return clase


sintetica, verdad = imagen_sintetica()
roi3 = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy"))

for nombre, img, ver in (("sintetica", sintetica, verdad), ("ROI3", roi3, None)):
    paisaje, filtrado, perfil, perfil_original, grises, _ = poblaciones_por_paisaje(img)
    clase = clasificar(img, grises)
    n = len(grises)
    fracciones = [np.mean(clase == c) for c in range(n)]
    print(f"{nombre}: {n} poblaciones en gris {np.round(grises, 3)}; áreas {np.round(fracciones, 3)}; "
          f"interfaz {np.mean(clase == -1):.1%}")
    if ver is not None:
        reales = np.array([0.0, 0.45, 0.58, 0.70, 0.85])
        print(f"   poblaciones reales: {reales}")

    paneles = 5 if ver is not None else 4
    fig, ejes = plt.subplots(1, paneles, figsize=(8 * paneles, 8))
    ejes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ejes[0].set_title(f"{'imagen sintética (como la ROI 3)' if ver is not None else 'ROI 3 suavizada'}",
                      fontsize=16)
    ejes[0].set_axis_off()

    im = ejes[1].imshow(filtrado, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="Greys")
    ejes[1].set_xlabel("gris (0-1)", fontsize=14)
    ejes[1].set_ylabel("actividad (0-1)", fontsize=14)
    ejes[1].set_title("paisaje FILTRADO\n(sin lo ancho de cada fila: queda lo angosto)", fontsize=16)
    fig.colorbar(im, ax=ejes[1], fraction=0.046)

    ax = ejes[2]
    centros = (np.arange(BINS) + 0.5) / BINS
    ax.plot(centros, perfil_original / perfil_original.max(), color="0.6", lw=2, label="paisaje original (suma)")
    ax.plot(centros, perfil / perfil.max(), color="k", lw=2, label="paisaje filtrado (suma)")
    rango = img.max() - img.min()
    for g in grises:
        ax.axvline((g - img.min()) / rango, color="C3", lw=2, ls="--")
    ax.set_xlabel("gris (0-1)", fontsize=14)
    ax.set_title(f"perfil sumando las filas (normalizado)\nrojo = {n} poblaciones encontradas", fontsize=16)
    ax.legend(fontsize=12)

    mapa_colores = ListedColormap([COLOR_INTERFAZ] + COLORES[:n])
    ax = ejes[3]
    ax.imshow(clase + 1, cmap=mapa_colores, vmin=-0.5, vmax=n + 0.5, interpolation="nearest")
    ax.set_title(f"clasificación: {n} poblaciones (sólidos)\nrojo = interfaz ({np.mean(clase == -1):.0%})",
                 fontsize=16)
    ax.set_axis_off()

    if ver is not None:
        ax = ejes[4]
        ax.imshow(ver, cmap=ListedColormap(COLORES[:5]), vmin=-0.5, vmax=4.5, interpolation="nearest")
        ax.set_title("VERDAD de la sintética\n(5 poblaciones: 0.00, 0.45, 0.58, 0.70, 0.85)", fontsize=16)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(SALIDA / f"22_poblaciones_por_paisaje_filtrado_{nombre}.png", dpi=45)
    plt.close(fig)
print("ok")
