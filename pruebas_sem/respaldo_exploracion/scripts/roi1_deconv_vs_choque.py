"""[funciones copiadas de laberinto_deconv_vs_choque.py] ROI 1 real: deconvolución contra filtro de choque.

Imagen: laberinto conectado (0.58) sobre matriz (0.70), degradé gaussiano de 1.5 px, ruido 0.003.
Métodos:
- Deconvolución Richardson-Lucy (skimage.restoration.richardson_lucy), gaussiana con el sigma
  estimado de los bordes aislados de la imagen, 30 vueltas.
- Filtro de choque (Osher-Rudin): I <- I - dt * signo(D2) * |grad I|, con D2 = segunda derivada
  en la dirección del gradiente, calculada sobre la imagen suavizada 1 px. Cada píxel se empuja
  hacia el lado del borde al que pertenece. Se itera hasta que el cambio medio por paso es menor
  que 0.1 % del rango de la imagen (máximo 200 pasos).
Resultado final: dos clases, corte en el valle del histograma de gris entre sus dos picos más altos
(si no hay dos picos, Otsu). Se compara con la verdad.
Actividad = gaussiana(sobel, 3 px). Se corre desde la raíz del proyecto. Guarda en
pruebas_sem/segmentacion/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage, signal

ALTO, ANCHO = 520, 560
GRISES = np.array([0.58, 0.70])
COLORES = ["#1f77b4", "#ff7f0e"]
SIGMA_ACTIVIDAD = 3
BINS = 160
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
SALIDA = Path("pruebas_sem/segmentacion")
Y0, X0, LADO_RECORTE = 160, 180, 160


def laberinto(semilla=11):
    rng = np.random.default_rng(semilla)
    lado = int(1.5 * max(ALTO, ANCHO))
    campo = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 8)), 60,
                           reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    campo = campo[y0:y0 + ALTO, x0:x0 + ANCHO]
    semillas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=4)
    semillas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, semillas, watershed_line=True)
    paredes = ndimage.binary_dilation(islas == 0, structure=ski.morphology.disk(1))
    return np.where(paredes, 0, 1), rng


def estimar_sigma_borde(img, muestras=1500, semilla=0):
    gy, gx = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    maximos = ski.feature.peak_local_max(np.hypot(gx, gy), min_distance=3, threshold_rel=0.2, exclude_border=12)
    rng = np.random.default_rng(semilla)
    if len(maximos) > muestras:
        maximos = maximos[rng.choice(len(maximos), muestras, replace=False)]
    t = np.linspace(-10, 10, 201)
    anchos = []
    for y, x in maximos:
        n = np.hypot(gy[y, x], gx[y, x])
        perfil = ndimage.map_coordinates(img, [y + t * gy[y, x] / n, x + t * gx[y, x] / n], order=1)
        p = (perfil - perfil.min()) / max(perfil.max() - perfil.min(), 1e-9)
        if p[0] > p[-1]:
            p = p[::-1]
        if p[t < -5].max() > 0.12 or p[t > 5].min() < 0.88 or np.min(np.diff(p)) < -0.03:
            continue
        i10, i90 = np.argmax(p >= 0.1), np.argmax(p >= 0.9)
        if i90 > i10:
            anchos.append(t[i90] - t[i10])
    return (float(np.median(anchos)) / 2.563 if len(anchos) >= 10 else 1.0), len(anchos)


def deconvolucion(img, sigma, vueltas=30):
    lado = 2 * int(np.ceil(3 * sigma)) + 1
    psf = np.zeros((lado, lado))
    psf[lado // 2, lado // 2] = 1
    psf = ndimage.gaussian_filter(psf, sigma)
    return ski.restoration.richardson_lucy(img - img.min() + 1e-3, psf, num_iter=vueltas, clip=False) + img.min()


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def filtro_de_choque(img, dt=0.25, tolerancia=1e-3, max_pasos=200):
    """Osher-Rudin con derivadas minmod (esquema estable: no crea valores nuevos fuera del rango local)."""
    I = img.astype(float).copy()
    rango = np.ptp(img)
    cambios = []
    for paso in range(max_pasos):
        S = ndimage.gaussian_filter(I, 1.0)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)
        I += delta
        cambios.append(np.mean(np.abs(delta)) / rango)
        if cambios[-1] < tolerancia:
            break
    return I, paso + 1, cambios


def clasificar(img):
    conteos, bordes = np.histogram(img, bins=256)
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), 2)
    picos, prop = signal.find_peaks(np.concatenate(([0.0], h, [0.0])), prominence=0.03 * h.max())
    picos = picos - 1
    if len(picos) >= 2:
        dos = np.sort(picos[np.argsort(prop["prominences"])[-2:]])
        corte = centros[dos[0] + int(np.argmin(h[dos[0]:dos[1] + 1]))]
        criterio = "valle entre los dos picos"
    else:
        corte = ski.filters.threshold_otsu(img)
        criterio = "Otsu (un solo pico)"
    return (img > corte).astype(int), corte, criterio


# ---------------------------------------------------------------------------------------------
# ROI 1 real (suavizada). Sin verdad: se muestran imagen, histograma, paisaje y la clasificación
# en dos clases (corte en el valle entre los dos picos más altos, o Otsu si hay uno solo).
# ---------------------------------------------------------------------------------------------
import sys
sys.path.insert(0, "src")
from sem.suavizado import suavizar

suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
sigma, usados = estimar_sigma_borde(suave)
choque, pasos, _ = filtro_de_choque(suave)
print(f"ROI 1: degradé estimado σ = {sigma:.2f} px ({usados} bordes aislados; "
      f"{'estimación válida' if usados >= 10 else 'FALLÓ, se usa el respaldo de 1 px'}); filtro de choque: {pasos} pasos")
versiones = {
    "ROI 1 suavizada (antes)": suave,
    f"deconvolución (σ {sigma:.2f} px, 30 vueltas)": deconvolucion(suave, sigma),
    f"filtro de choque ({pasos} pasos)": choque,
}

fig, ejes = plt.subplots(len(versiones), 4, figsize=(40, 10.5 * len(versiones)),
                         gridspec_kw={"width_ratios": [1, 1, 1.15, 1]})
for fila, (nombre, v) in enumerate(versiones.items()):
    bajo, alto = np.percentile(v, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(v), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), v.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    clase, corte, criterio = clasificar(v)
    print(f"   {nombre:<42} corte {corte:.4f} ({criterio}); fase oscura = {np.mean(clase == 0):.1%} del área")

    ax = ejes[fila, 0]
    im = ax.imshow(v[Y0:Y0 + LADO_RECORTE, X0:X0 + LADO_RECORTE], cmap="gray", vmin=bajo, vmax=alto)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("gris", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(f"{nombre}\n(recorte {LADO_RECORTE}x{LADO_RECORTE} px)", fontsize=22)
    ax.set_axis_off()

    ax = ejes[fila, 1]
    ax.hist(v.ravel(), bins=BINS, range=(bajo, alto), color="0.4")
    ax.axvline(corte, color="C3", lw=3, ls="--", label=f"corte {corte:.3f}")
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel("cantidad de píxeles", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=17)
    ax.set_title(f"histograma de gris\n(corte: {criterio})", fontsize=22)

    ax = ejes[fila, 2]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    ax.axvline(corte, color="k", lw=2, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log(1 + cantidad de píxeles)", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel(f"actividad (σ={SIGMA_ACTIVIDAD}px, 0-1)", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title("paisaje gris-actividad (línea = corte)", fontsize=22)

    ax = ejes[fila, 3]
    ax.imshow(clase[Y0:Y0 + LADO_RECORTE, X0:X0 + LADO_RECORTE], cmap=ListedColormap(COLORES),
              vmin=-0.5, vmax=1.5, interpolation="nearest")
    ax.legend(handles=[Patch(color=COLORES[0], label=f"fase oscura ({np.mean(clase == 0):.1%})"),
                       Patch(color=COLORES[1], label=f"fase clara ({np.mean(clase == 1):.1%})")],
              loc="lower left", fontsize=17)
    ax.set_title("RESULTADO (mismo recorte)", fontsize=22)
    ax.set_axis_off()

fig.tight_layout()
fig.savefig(SALIDA / "50_ROI1_deconv_vs_choque.png", dpi=30)
print("ok")
