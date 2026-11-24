"""Laberinto + matriz con degradé: deconvolución contra filtro de choque.

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


etiquetas, rng = laberinto()
img = ski.filters.gaussian(GRISES[etiquetas], 1.5) + rng.normal(0, 0.003, etiquetas.shape)
sigma, usados = estimar_sigma_borde(img)
choque, pasos, cambios = filtro_de_choque(img)
print(f"degradé estimado σ = {sigma:.2f} px ({usados} bordes aislados); filtro de choque: {pasos} pasos")
versiones = {
    "con degradé (antes)": img,
    f"deconvolución (σ estimado {sigma:.2f} px, 30 vueltas)": deconvolucion(img, sigma),
    f"filtro de choque ({pasos} pasos)": choque,
}

# ---------------- Figura 1: imagen, histograma y paisajes antes y después
fig, ejes = plt.subplots(len(versiones), 4, figsize=(40, 10.5 * len(versiones)),
                         gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
resultados = {}
for fila, (nombre, v) in enumerate(versiones.items()):
    bajo, alto = np.percentile(v, (0.1, 99.9))
    actividad = ski.filters.gaussian(ski.filters.sobel(v), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), v.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    capas = [np.histogram2d(actividad[etiquetas == p], v[etiquetas == p], bins=BINS,
                            range=((0, 1), (bajo, alto)))[0] for p in range(2)]

    ax = ejes[fila, 0]
    im = ax.imshow(v[Y0:Y0 + LADO_RECORTE, X0:X0 + LADO_RECORTE], cmap="gray", vmin=bajo, vmax=alto)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("gris", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(f"{nombre}\n(recorte {LADO_RECORTE}x{LADO_RECORTE} px)", fontsize=22)
    ax.set_axis_off()

    ax = ejes[fila, 1]
    for p in range(2):
        ax.hist(v[etiquetas == p].ravel(), bins=BINS, range=(bajo, alto), color=COLORES[p], alpha=0.65,
                label=["laberinto", "matriz"][p])
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel("cantidad de píxeles", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=17)
    ax.set_title("histograma de gris, por población verdadera", fontsize=22)

    ax = ejes[fila, 2]
    im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log(1 + cantidad de píxeles)", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel(f"actividad (σ={SIGMA_ACTIVIDAD}px, 0-1)", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title("paisaje gris-actividad (densidad)", fontsize=22)

    ax = ejes[fila, 3]
    fraccion_lab = capas[0] / np.maximum(capas[0] + capas[1], 1)
    total = capas[0] + capas[1]
    rgb = np.ones(total.shape + (3,))
    color = (fraccion_lab[..., None] * np.array(to_rgb(COLORES[0]))
             + (1 - fraccion_lab[..., None]) * np.array(to_rgb(COLORES[1])))
    alfa = (np.log1p(total) / np.log1p(total).max())[..., None]
    rgb = rgb * (1 - alfa) + color * alfa
    ax.imshow(rgb, origin="lower", aspect="auto", extent=(bajo, alto, 0, 1))
    mapa = LinearSegmentedColormap.from_list("lab_mat", [COLORES[1], COLORES[0]])
    sm = plt.cm.ScalarMappable(cmap=mapa, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046)
    cb.set_label("fracción de píxeles del laberinto en la casilla", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel(f"actividad (σ={SIGMA_ACTIVIDAD}px, 0-1)", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title("paisaje coloreado por población verdadera\n(intensidad del color = cantidad de píxeles)",
                 fontsize=22)
    resultados[nombre] = v

fig.tight_layout()
fig.savefig(SALIDA / "48_laberinto_deconv_vs_choque_paisajes.png", dpi=30)
plt.close(fig)

# ---------------- Figura 2: resultados finales (clasificación en dos clases)
fig, ejes = plt.subplots(len(versiones), 3, figsize=(32, 10.5 * len(versiones)),
                         gridspec_kw={"width_ratios": [1, 1, 0.8]})
for fila, (nombre, v) in enumerate(resultados.items()):
    clase, corte, criterio = clasificar(v)
    acierto_lab = np.mean(clase[etiquetas == 0] == 0)
    acierto_mat = np.mean(clase[etiquetas == 1] == 1)
    area_lab = np.mean(clase == 0)
    print(f"{nombre:<45} corte {corte:.4f} ({criterio}); laberinto acertado {acierto_lab:.1%}, "
          f"matriz {acierto_mat:.1%}; área laberinto {area_lab:.1%} (real {np.mean(etiquetas == 0):.1%})")

    ax = ejes[fila, 0]
    ax.imshow(clase[Y0:Y0 + LADO_RECORTE, X0:X0 + LADO_RECORTE], cmap=ListedColormap(COLORES),
              vmin=-0.5, vmax=1.5, interpolation="nearest")
    ax.set_title(f"{nombre}\nRESULTADO (corte en gris {corte:.3f}, {criterio})", fontsize=21)
    ax.set_axis_off()

    ax = ejes[fila, 1]
    error = np.full(etiquetas.shape, 2)           # 2 = bien
    error[(etiquetas == 0) & (clase == 1)] = 0    # laberinto perdido
    error[(etiquetas == 1) & (clase == 0)] = 1    # matriz tomada como laberinto
    ax.imshow(error[Y0:Y0 + LADO_RECORTE, X0:X0 + LADO_RECORTE],
              cmap=ListedColormap(["#d62728", "#9467bd", "#f2f2f2"]), vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.legend(handles=[Patch(color="#d62728", label="laberinto perdido"),
                       Patch(color="#9467bd", label="matriz tomada como laberinto"),
                       Patch(color="#f2f2f2", label="bien", ec="k")], loc="lower left", fontsize=16)
    ax.set_title("errores contra la verdad (mismo recorte)", fontsize=21)
    ax.set_axis_off()

    ax = ejes[fila, 2]
    barras = ax.bar([0, 1, 2], [100 * acierto_lab, 100 * acierto_mat, 100 * area_lab],
                    color=[COLORES[0], COLORES[1], "0.6"], edgecolor="k")
    ax.axhline(100 * np.mean(etiquetas == 0), color="k", ls="--", lw=2)
    ax.text(2.45, 100 * np.mean(etiquetas == 0) + 2, "área real", fontsize=15, ha="right")
    ax.set_xticks([0, 1, 2], ["laberinto\nacertado", "matriz\nacertada", "área\nlaberinto"], fontsize=17)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%", fontsize=19)
    ax.tick_params(labelsize=15)
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=17)
    ax.set_title("aciertos", fontsize=21)

fig.tight_layout()
fig.savefig(SALIDA / "49_laberinto_deconv_vs_choque_resultados.png", dpi=30)
print("ok")
