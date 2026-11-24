"""Realce + persistencia (v2): 3 imágenes y 4 métodos, con mejor estimación del degradé.

Imágenes: burbujas nítidas, burbujas con degradé (2 px) y tipo ROI 1 con degradé (1.5 px), todas
con ruido residual de 0.003.
Estimación del degradé: perfil de gris a través de cada borde fuerte (±10 px). Solo se usan bordes
AISLADOS: el perfil tiene que ser parejo en los extremos (más allá de ±5 px) y casi monótono, es
decir, sin otro borde cerca. sigma = mediana(ancho 10-90 %) / 2.563.
Métodos: sin realce; máscara de enfoque (radio = sigma, cantidad 1); deconvolución Richardson-Lucy
(gaussiana de ese sigma, 30 vueltas); CLAHE (clip 0.01). Si sigma < 0.5 px (no hay degradé), la
máscara de enfoque y la deconvolución no hacen nada.
Persistencia igual que en 34_. Gris "verdadero" de cada población = mediana de sus píxeles en la
imagen procesada.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
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
FILA_MINIMA, PROMINENCIA_FILA, PROMINENCIA_PERFIL = 0.005, 0.10, 0.15
TOLERANCIA = 0.04
METODOS = ["sin realce", "máscara de enfoque", "deconvolución", "CLAHE"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])


def burbujas(semilla=0):
    rng = np.random.default_rng(semilla)
    etiquetas = np.full((LADO, LADO), 2)
    for p, radio in sorted({0: 4, 1: 8, 3: 15, 4: 30}.items(), key=lambda kv: -kv[1]):
        for _ in range(int(0.08 * LADO * LADO / (np.pi * radio ** 2))):
            fy, fx = rng.integers(0, LADO, 2)
            rr, cc = ski.draw.disk((fy, fx), radio, shape=etiquetas.shape)
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


def estimar_sigma_borde(img, muestras=1500, semilla=0):
    gy, gx = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    magnitud = np.hypot(gx, gy)
    maximos = ski.feature.peak_local_max(magnitud, min_distance=3, threshold_rel=0.2, exclude_border=12)
    rng = np.random.default_rng(semilla)
    if len(maximos) > muestras:
        maximos = maximos[rng.choice(len(maximos), muestras, replace=False)]
    t = np.linspace(-10, 10, 201)
    extremos = np.abs(t) > 5
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
        # Borde aislado: extremos parejos (cerca de 0 a la izquierda y de 1 a la derecha) y perfil casi monótono
        izquierda, derecha = p[(t < -5)], p[(t > 5)]
        if izquierda.max() > 0.12 or derecha.min() < 0.88 or np.min(np.diff(p)) < -0.03:
            continue
        i10, i90 = np.argmax(p >= 0.1), np.argmax(p >= 0.9)
        if i90 > i10:
            anchos.append(t[i90] - t[i10])
    if len(anchos) < 20:
        return 0.0, len(anchos)
    sigma = float(np.median(anchos)) / 2.563
    return sigma, len(anchos)


def realzar(img, metodo, sigma):
    if metodo == "sin realce":
        return img
    if metodo == "CLAHE":
        return ski.exposure.equalize_adapthist((img - img.min()) / (img.max() - img.min()), clip_limit=0.01)
    if sigma < 0.5:
        return img
    if metodo == "máscara de enfoque":
        return ski.filters.unsharp_mask(img, radius=sigma, amount=1.0, preserve_range=True)
    lado = 2 * int(np.ceil(3 * sigma)) + 1
    psf = np.zeros((lado, lado))
    psf[lado // 2, lado // 2] = 1
    psf = ndimage.gaussian_filter(psf, sigma)
    return ski.restoration.richardson_lucy(img - img.min() + 1e-3, psf, num_iter=30, clip=False)


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
    return paisaje, perfil, centros_g, centros_g[picos - 1], rango


rng = np.random.default_rng(1)
etq_b, gris_b = burbujas()
etq_r, gris_r = tipo_roi1()
casos = {
    "burbujas, nítidas": (gris_b[etq_b] + rng.normal(0, 0.003, etq_b.shape), etq_b, 5),
    "burbujas, con degradé (σ=2px)": (ski.filters.gaussian(gris_b[etq_b], 2) + rng.normal(0, 0.003, etq_b.shape), etq_b, 5),
    "tipo ROI 1, con degradé (σ=1.5px)": (ski.filters.gaussian(gris_r[etq_r], 1.5) + rng.normal(0, 0.003, etq_r.shape), etq_r, 3),
}

fig1, ejes1 = plt.subplots(len(casos), 4, figsize=(34, 7.4 * len(casos)))
fig2, ejes2 = plt.subplots(2 * len(casos), 4, figsize=(30, 6.3 * 2 * len(casos)))
for fila, (nombre, (img, etiquetas, n)) in enumerate(casos.items()):
    sigma, usados = estimar_sigma_borde(img)
    print(f"\n{nombre}: degradé estimado σ = {sigma:.2f} px (con {usados} bordes aislados)")
    for col, metodo in enumerate(METODOS):
        procesada = realzar(img, metodo, sigma)
        verdad = np.array([np.median(procesada[etiquetas == p]) for p in range(n)])
        paisaje, perfil, centros_g, encontradas, rango = persistencia(procesada)
        tol = TOLERANCIA * (rango[1] - rango[0])
        aciertos = sum(np.any(np.abs(encontradas - g) <= tol) for g in verdad)
        falsas = sum(not np.any(np.abs(verdad - e) <= tol) for e in encontradas)
        print(f"   {metodo:<20} acertadas {aciertos}/{n}, falsas {falsas}")
        titulo = f"{nombre}\n{metodo}: acertadas {aciertos}/{n}, falsas {falsas}"

        # Figura 1: paisaje con las poblaciones encontradas
        ax = ejes1[fila, col]
        ax.imshow(paisaje, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(rango[0], rango[1], 0, 1))
        for g in verdad:
            ax.axvline(g, color="#2ca02c", lw=6, alpha=0.35)
        for e in encontradas:
            ax.axvline(e, color="k", lw=2, ls="--")
        ax.set_xlabel("gris", fontsize=14)
        ax.set_ylabel("actividad (0-1)", fontsize=14)
        ax.set_title(titulo, fontsize=16)

        # Figura 2: imagen procesada (recorte) y perfil de persistencia
        ax = ejes2[2 * fila, col]
        recorte = procesada[150:350, 150:350]
        ax.imshow(recorte, cmap="gray", vmin=np.percentile(recorte, 0.5), vmax=np.percentile(recorte, 99.5))
        ax.set_title(f"{nombre}\n{metodo}", fontsize=16)
        ax.set_axis_off()
        ax = ejes2[2 * fila + 1, col]
        ax.plot(centros_g, perfil, color="k", lw=2)
        for g in verdad:
            ax.axvline(g, color="#2ca02c", lw=6, alpha=0.35)
        for e in encontradas:
            ax.axvline(e, color="C3", lw=2, ls="--")
        ax.set_xlim(rango)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("gris", fontsize=14)
        ax.set_ylabel("persistencia", fontsize=14)
        ax.set_title(f"persistencia: acertadas {aciertos}/{n}, falsas {falsas}", fontsize=16)

fig1.suptitle("Paisaje gris-actividad tras cada realce. Verde = poblaciones reales; línea negra = encontradas por persistencia",
              fontsize=21)
fig1.tight_layout(rect=(0, 0, 1, 0.97))
fig1.savefig(Path("pruebas_sem/segmentacion/37_realce_v2_paisajes.png"), dpi=32)
fig2.suptitle("Imagen tras cada realce (recorte de 200 x 200 px) y su perfil de persistencia "
              "(verde = reales, rojo = encontradas)", fontsize=21)
fig2.tight_layout(rect=(0, 0, 1, 0.98))
fig2.savefig(Path("pruebas_sem/segmentacion/38_realce_v2_imagenes_y_persistencia.png"), dpi=30)
print("ok")
