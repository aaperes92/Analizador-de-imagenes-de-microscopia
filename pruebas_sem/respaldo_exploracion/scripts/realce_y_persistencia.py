"""Realce de contraste antes del paisaje: sin realce, máscara de enfoque, deconvolución y CLAHE.

Solo las 4 sintéticas CON degradé (las nítidas ya daban 100 %).
- Ancho del degradé (sigma) estimado de la imagen: en los bordes fuertes se mide el perfil de gris
  a través del borde y su ancho 10-90 %; sigma = mediana(ancho) / 2.563 (relación para un borde
  gaussiano).
- Máscara de enfoque: skimage.filters.unsharp_mask(radio = sigma estimado, cantidad = 1).
- Deconvolución: skimage.restoration.richardson_lucy con una gaussiana de ese sigma, 30 vueltas.
- CLAHE: skimage.exposure.equalize_adapthist(clip_limit = 0.01).
Después, persistencia de picos igual que en 34_. Como el realce cambia los grises, el gris
"verdadero" de cada población es la mediana de sus píxeles en la imagen realzada.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy import ndimage, signal

LADO = 600
SIGMA_ACTIVIDAD = 25
BINS = 128
FILA_MINIMA, PROMINENCIA_FILA, PROMINENCIA_PERFIL = 0.005, 0.10, 0.15
TOLERANCIA = 0.04          # fracción del rango de gris para dar por acertada una población


def burbujas(grises, semilla=0):
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, radio in sorted({0: 4, 1: 8, 3: 15, 4: 30}.items(), key=lambda kv: -kv[1]):
        for _ in range(int(0.08 * LADO * LADO / (np.pi * radio ** 2))):
            fy, fx = rng.integers(0, LADO, 2)
            rr, cc = ski.draw.disk((fy, fx), radio, shape=etiquetas.shape)
            etiquetas[rr, cc] = p
    return etiquetas, np.asarray(grises)


def gusanos(semilla=0):
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, espesor in sorted({0: 3, 1: 6, 3: 12, 4: 24}.items(), key=lambda kv: -kv[1]):
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
    return etiquetas, np.array([0.10, 0.30, 0.50, 0.70, 0.90])


def tipo_roi1(semilla=7):
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * LADO)
    campo = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 6)), 60,
                           reshape=False, mode="reflect")
    y0 = (lado - LADO) // 2
    campo = campo[y0:y0 + LADO, y0:y0 + LADO]
    campo /= campo.std()
    return np.where(np.abs(campo) < 0.3, 0, np.where(campo >= 0.3, 1, 2)), np.array([0.58, 0.66, 0.74])


def estimar_sigma_borde(img, muestras=600, semilla=0):
    """Ancho del degradé a partir del perfil 10-90 % a través de los bordes más fuertes."""
    gy, gx = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    magnitud = np.hypot(gx, gy)
    maximos = ski.feature.peak_local_max(magnitud, min_distance=5, threshold_rel=0.3, exclude_border=10)
    rng = np.random.default_rng(semilla)
    if len(maximos) > muestras:
        maximos = maximos[rng.choice(len(maximos), muestras, replace=False)]
    t = np.linspace(-8, 8, 161)
    anchos = []
    for y, x in maximos:
        ny, nx = gy[y, x], gx[y, x]
        norma = np.hypot(ny, nx)
        perfil = ndimage.map_coordinates(img, [y + t * ny / norma, x + t * nx / norma], order=1)
        bajo, alto = perfil.min(), perfil.max()
        if alto - bajo < 1e-6:
            continue
        p = (perfil - bajo) / (alto - bajo)
        if p[0] > p[-1]:
            p = p[::-1]
        i10, i90 = np.argmax(p >= 0.1), np.argmax(p >= 0.9)
        if i90 > i10:
            anchos.append(t[i90] - t[i10])
    return float(np.median(anchos)) / 2.563


def persistencia(img):
    bajo, alto = img.min(), img.max()
    rango = (bajo - 0.02 * (alto - bajo), alto + 0.02 * (alto - bajo))
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, bordes_g = np.histogram2d(actividad.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
    centros_g = (bordes_g[:-1] + bordes_g[1:]) / 2
    paisaje = np.log1p(conteos)
    filas = np.flatnonzero(conteos.sum(axis=1) >= FILA_MINIMA * conteos.sum(axis=1).max())
    marcas = np.zeros_like(paisaje, bool)
    for f in filas:
        fila = ndimage.gaussian_filter1d(paisaje[f], 1)
        picos, _ = signal.find_peaks(np.concatenate(([0.0], fila, [0.0])), prominence=PROMINENCIA_FILA * fila.max())
        marcas[f, picos - 1] = True
    perfil = ndimage.binary_dilation(marcas, structure=np.ones((1, 3), bool))[filas].mean(axis=0)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=PROMINENCIA_PERFIL * perfil.max())
    return perfil, centros_g, centros_g[picos - 1], rango


def realces(img, sigma):
    psf_lado = 2 * int(np.ceil(3 * sigma)) + 1
    psf = np.zeros((psf_lado, psf_lado))
    psf[psf_lado // 2, psf_lado // 2] = 1
    psf = ndimage.gaussian_filter(psf, sigma)
    positiva = img - img.min() + 1e-3
    return {
        "sin realce": img,
        "máscara de enfoque": ski.filters.unsharp_mask(img, radius=sigma, amount=1.0, preserve_range=True),
        "deconvolución": ski.restoration.richardson_lucy(positiva, psf, num_iter=30, clip=False),
        "CLAHE": ski.exposure.equalize_adapthist((img - img.min()) / (img.max() - img.min()), clip_limit=0.01),
    }


rng = np.random.default_rng(1)
casos = {}
for nombre, (etiquetas, grises) in (("burbujas", burbujas([0.10, 0.30, 0.50, 0.70, 0.90])),
                                    ("burbujas invertidas", burbujas([0.90, 0.70, 0.50, 0.30, 0.10])),
                                    ("gusanos", gusanos()),
                                    ("tipo ROI 1", tipo_roi1())):
    sigma_real = 1.5 if "ROI" in nombre else 2
    img = ski.filters.gaussian(grises[etiquetas], sigma_real) + rng.normal(0, 0.003, etiquetas.shape)
    casos[nombre] = (img, etiquetas, len(grises), sigma_real)

METODOS = ["sin realce", "máscara de enfoque", "deconvolución", "CLAHE"]
fig_p, ejes_p = plt.subplots(len(casos), 4, figsize=(34, 6 * len(casos)))
fig_i, ejes_i = plt.subplots(len(casos), 4, figsize=(28, 7 * len(casos)))
resumen = []
for fila, (nombre, (img, etiquetas, n, sigma_real)) in enumerate(casos.items()):
    sigma = estimar_sigma_borde(img)
    print(f"\n{nombre}: sigma del degradé real {sigma_real}, estimado {sigma:.2f} px")
    for col, (metodo, procesada) in enumerate(realces(img, sigma).items()):
        verdad = np.array([np.median(procesada[etiquetas == p]) for p in range(n)])
        perfil, centros_g, encontradas, rango = persistencia(procesada)
        tol = TOLERANCIA * (rango[1] - rango[0])
        aciertos = sum(np.any(np.abs(encontradas - g) <= tol) for g in verdad)
        falsas = sum(not np.any(np.abs(verdad - e) <= tol) for e in encontradas)
        resumen.append((nombre, metodo, aciertos, n, falsas))
        print(f"   {metodo:<20} acertadas {aciertos}/{n}, falsas {falsas}")

        ax = ejes_p[fila, col]
        ax.plot(centros_g, perfil, color="k", lw=2)
        for g in verdad:
            ax.axvline(g, color="#2ca02c", lw=5, alpha=0.5)
        for e in encontradas:
            ax.axvline(e, color="C3", lw=2, ls="--")
        ax.set_xlim(rango)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{nombre} — {metodo}\nacertadas {aciertos}/{n}, falsas {falsas}", fontsize=17)
        ax.set_xlabel("gris", fontsize=13)
        if col == 0:
            ax.set_ylabel("persistencia", fontsize=14)

        ax = ejes_i[fila, col]
        recorte = procesada[150:350, 150:350]
        ax.imshow(recorte, cmap="gray", vmin=np.percentile(recorte, 0.5), vmax=np.percentile(recorte, 99.5))
        ax.set_title(f"{nombre} — {metodo}", fontsize=17)
        ax.set_axis_off()

fig_p.suptitle("Persistencia tras cada realce (verde = poblaciones reales, rojo = encontradas)", fontsize=22)
fig_p.tight_layout(rect=(0, 0, 1, 0.98))
fig_p.savefig(Path("pruebas_sem/segmentacion/35_realce_persistencia.png"), dpi=30)
fig_i.suptitle("Recortes de 200 x 200 px de cada imagen realzada", fontsize=22)
fig_i.tight_layout(rect=(0, 0, 1, 0.98))
fig_i.savefig(Path("pruebas_sem/segmentacion/36_realce_recortes.png"), dpi=30)
print("ok")
