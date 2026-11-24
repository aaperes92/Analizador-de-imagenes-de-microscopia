"""Búsqueda de poblaciones por PERSISTENCIA de picos en el paisaje gris-actividad (sintéticas).

Para cada imagen:
1. actividad = gaussiana(sobel, 25 px) a [0, 1]; paisaje = log(1 + histograma 2D), 128 x 128,
   eje del gris entre el mínimo y el máximo de la imagen.
2. En cada fila con píxeles suficientes (>= 0.5 % de la fila más poblada): máximos locales del gris
   con prominencia >= 10 % del máximo de esa fila.
3. Persistencia(gris) = fracción de filas válidas con un pico en ese gris (±1 bin).
4. Poblaciones = picos del perfil de persistencia (prominencia >= 15 % de su máximo).
Casos: burbujas, burbujas invertidas, gusanos y patrón tipo ROI 1; cada uno nítido y con degradé.
Se corre desde la raíz del proyecto. Guarda la figura en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage, signal

LADO = 600
SIGMA_ACTIVIDAD = 25
BINS = 128
FILA_MINIMA = 0.005
PROMINENCIA_FILA = 0.10
PROMINENCIA_PERFIL = 0.15
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])


def burbujas(grises, semilla=0):
    rng = np.random.default_rng(semilla)
    radios = {0: 4, 1: 8, 3: 15, 4: 30}
    etiquetas = np.full((LADO, LADO), 2)
    for p, radio in sorted(radios.items(), key=lambda kv: -kv[1]):
        for _ in range(int(0.08 * LADO * LADO / (np.pi * radio ** 2))):
            fy, fx = rng.integers(0, LADO, 2)
            rr, cc = ski.draw.disk((fy, fx), radio, shape=etiquetas.shape)
            etiquetas[rr, cc] = p
    return np.asarray(grises)[etiquetas], grises


def gusanos(semilla=0):
    grises = [0.10, 0.30, 0.50, 0.70, 0.90]
    espesores = {0: 3, 1: 6, 3: 12, 4: 24}
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, espesor in sorted(espesores.items(), key=lambda kv: -kv[1]):
        while np.sum(etiquetas == p) < 0.08 * LADO * LADO:
            y, x = rng.uniform(0, LADO, 2)
            angulo, giro = rng.uniform(0, 2 * np.pi), 0.0
            for _ in range(int(rng.uniform(120, 260))):
                giro = 0.85 * giro + rng.normal(0, 0.06)
                angulo += giro
                y, x = y + 2 * np.sin(angulo), x + 2 * np.cos(angulo)
                if not (0 <= y < LADO and 0 <= x < LADO):
                    break
                rr, cc = ski.draw.disk((y, x), max(espesor / 2, 1), shape=etiquetas.shape)
                etiquetas[rr, cc] = p
    return np.asarray(grises)[etiquetas], grises


def tipo_roi1(semilla=7):
    grises = [0.58, 0.66, 0.74]
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * LADO)
    campo = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 6)), 60,
                           reshape=False, mode="reflect")
    y0 = (lado - LADO) // 2
    campo = campo[y0:y0 + LADO, y0:y0 + LADO]
    campo /= campo.std()
    etiquetas = np.where(np.abs(campo) < 0.3, 0, np.where(campo >= 0.3, 1, 2))
    return np.asarray(grises)[etiquetas], grises


def persistencia(img):
    bajo, alto = img.min(), img.max()
    margen = 0.02 * (alto - bajo)
    rango = (bajo - margen, alto + margen)
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, bordes_g = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
    centros_g = (bordes_g[:-1] + bordes_g[1:]) / 2
    paisaje = np.log1p(conteos)
    filas_validas = np.flatnonzero(conteos.sum(axis=1) >= FILA_MINIMA * conteos.sum(axis=1).max())
    marcas = np.zeros_like(paisaje, bool)
    for f in filas_validas:
        fila = ndimage.gaussian_filter1d(paisaje[f], 1)
        picos, _ = signal.find_peaks(np.concatenate(([0.0], fila, [0.0])), prominence=PROMINENCIA_FILA * fila.max())
        marcas[f, picos - 1] = True
    # ±1 bin para tolerar que el pico se corra un bin de una fila a otra
    marcas_anchas = ndimage.binary_dilation(marcas, structure=np.ones((1, 3), bool))
    perfil = marcas_anchas[filas_validas].mean(axis=0)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=PROMINENCIA_PERFIL * perfil.max())
    return paisaje, marcas, perfil, centros_g, centros_g[picos - 1], rango


casos = {}
for nombre, (img, grises) in (("burbujas", burbujas([0.10, 0.30, 0.50, 0.70, 0.90])),
                              ("burbujas invertidas", burbujas([0.90, 0.70, 0.50, 0.30, 0.10])),
                              ("gusanos", gusanos()),
                              ("tipo ROI 1", tipo_roi1())):
    rng = np.random.default_rng(1)
    casos[f"{nombre}, nítido"] = (img, grises)
    degradada = ski.filters.gaussian(img, 2 if "ROI" not in nombre else 1.5) + rng.normal(0, 0.003, img.shape)
    casos[f"{nombre}, con degradé"] = (degradada, grises)

fig, ejes = plt.subplots(len(casos), 3, figsize=(26, 6.4 * len(casos)), gridspec_kw={"width_ratios": [0.8, 1.2, 1.2]})
for fila, (nombre, (img, grises)) in enumerate(casos.items()):
    paisaje, marcas, perfil, centros_g, encontrados, rango = persistencia(img)
    grises = np.sort(np.asarray(grises))
    aciertos = sum(np.any(np.abs(encontrados - g) <= 0.03 * (rango[1] - rango[0]) + 0.01) for g in grises)
    falsos = sum(not np.any(np.abs(grises - e) <= 0.03 * (rango[1] - rango[0]) + 0.01) for e in encontrados)
    print(f"{nombre:<32} verdaderas {grises} | encontradas {np.round(encontrados, 3)} | "
          f"acertadas {aciertos}/{len(grises)}, falsas {falsos}")

    ejes[fila, 0].imshow(img, cmap="gray")
    ejes[fila, 0].set_title(nombre, fontsize=19)
    ejes[fila, 0].set_axis_off()

    ax = ejes[fila, 1]
    ax.imshow(paisaje, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(rango[0], rango[1], 0, 1))
    f_idx, g_idx = np.nonzero(marcas)
    ax.plot(centros_g[g_idx], (f_idx + 0.5) / BINS, ".", color="k", ms=3)
    ax.set_xlabel("gris", fontsize=15)
    ax.set_ylabel("actividad (0-1)", fontsize=15)
    ax.set_title("paisaje; puntos negros = picos de cada fila", fontsize=18)

    ax = ejes[fila, 2]
    ax.plot(centros_g, perfil, color="k", lw=2)
    for g in grises:
        ax.axvline(g, color="#2ca02c", lw=4, alpha=0.5)
    for e in encontrados:
        ax.axvline(e, color="C3", lw=2, ls="--")
    ax.set_xlim(rango)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("gris", fontsize=15)
    ax.set_ylabel("persistencia (fracción de filas)", fontsize=15)
    ax.set_title(f"persistencia: verde = poblaciones reales, rojo = encontradas\n"
                 f"acertadas {aciertos}/{len(grises)}, falsas {falsos}", fontsize=18)

fig.tight_layout()
fig.savefig(Path("pruebas_sem/segmentacion/34_persistencia_sinteticas.png"), dpi=30)
print("ok")
