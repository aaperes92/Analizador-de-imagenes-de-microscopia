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


# ---------------------------------------------------------------------------------------------
# Variantes del filtro de choque en la ROI 1 real.
# - Realce previo: máscara de enfoque (skimage.filters.unsharp_mask) con radio = degradé estimado.
# - Sigma interno del choque: suavizado usado para decidir el lado del borde (1 y 0.5 px).
# - Corte corregido: histograma (256 casillas) suavizado 5 casillas; los dos picos con MÁS
#   píxeles (masa entre valles); corte en el mínimo entre ellos.
# ---------------------------------------------------------------------------------------------
import sys
sys.path.insert(0, "src")
from sem.suavizado import suavizar

SUAVIZADO_HIST = 5
COLOR_OSCURA, COLOR_CLARA = COLORES


def choque(img, sigma_interno, dt=0.25, tolerancia=1e-3, max_pasos=200):
    I = img.astype(float).copy()
    rango = np.ptp(img)
    for paso in range(max_pasos):
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)
        I += delta
        if np.mean(np.abs(delta)) / rango < tolerancia:
            break
    return I, paso + 1


def corte_por_masa(img):
    conteos, bordes = np.histogram(img, bins=256, range=tuple(np.percentile(img, (0.1, 99.9))))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SUAVIZADO_HIST)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])))
    picos = list(picos - 1)
    if len(picos) < 2:
        return ski.filters.threshold_otsu(img), centros, conteos, h, picos, [], "Otsu (un solo pico)"
    valles = [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]
    limites = [0, *valles, len(h)]
    masas = [h[limites[i]:limites[i + 1]].sum() for i in range(len(picos))]
    dos = sorted(np.argsort(masas)[-2:])
    a, b = picos[dos[0]], picos[dos[1]]
    v = a + int(np.argmin(h[a:b + 1]))
    return centros[v], centros, conteos, h, picos, [a, b], "mínimo entre los dos picos con más píxeles"


suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
sigma_deg, usados = estimar_sigma_borde(suave)
realzada = ski.filters.unsharp_mask(suave, radius=sigma_deg, amount=1.0, preserve_range=True)
print(f"degradé estimado σ = {sigma_deg:.2f} px ({usados} bordes aislados)")

variantes = {}
for etiqueta, base, s_int in (("a) sin realce, σ interno 1 px", suave, 1.0),
                              ("b) sin realce, σ interno 0.5 px", suave, 0.5),
                              ("c) enfoque + choque, σ interno 1 px", realzada, 1.0),
                              ("d) enfoque + choque, σ interno 0.5 px", realzada, 0.5)):
    resultado, pasos = choque(base, s_int)
    corte, centros, conteos, h, picos, dos, criterio = corte_por_masa(resultado)
    clase = (resultado > corte).astype(int)
    variantes[etiqueta] = dict(img=resultado, pasos=pasos, corte=corte, centros=centros, conteos=conteos,
                               h=h, picos=picos, dos=dos, criterio=criterio, clase=clase)
    print(f"{etiqueta:<40} {pasos:3d} pasos; corte {corte:.4f} ({criterio}); fase oscura {np.mean(clase == 0):.1%}")

lo_o, hi_o = np.percentile(suave, (0.1, 99.9))
recorte = (slice(Y0, Y0 + LADO_RECORTE), slice(X0, X0 + LADO_RECORTE))

# ---------------- Figura 51: imagen procesada, histograma con el corte y paisaje
fig, ejes = plt.subplots(len(variantes) + 1, 3, figsize=(36, 10.5 * (len(variantes) + 1)),
                         gridspec_kw={"width_ratios": [1, 1.25, 1.2]})
filas = [("referencia: ROI 1 suavizada (sin choque)", dict(img=suave, corte=None))] + list(variantes.items())
for fila, (nombre, d) in enumerate(filas):
    v = d["img"]
    bajo, alto = np.percentile(v, (0.1, 99.9))
    ax = ejes[fila, 0]
    im = ax.imshow(v[recorte], cmap="gray", vmin=bajo, vmax=alto)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("gris", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(nombre + (f"\n({d['pasos']} pasos)" if "pasos" in d else ""), fontsize=23)
    ax.set_axis_off()

    ax = ejes[fila, 1]
    if d["corte"] is None:
        ax.hist(v.ravel(), bins=256, range=(bajo, alto), color="0.6")
        ax.set_title("histograma de gris", fontsize=23)
    else:
        c = d["centros"]
        ax.bar(c, d["conteos"], width=c[1] - c[0], color="0.75", label="histograma")
        ax.plot(c, d["h"], color="k", lw=2.5, label=f"suavizado ({SUAVIZADO_HIST} casillas)")
        ax.plot(c[d["picos"]], d["h"][d["picos"]], "o", color="0.4", ms=9, label="picos encontrados")
        ax.plot(c[d["dos"]], d["h"][d["dos"]], "o", color="C2", ms=15, mec="k", label="los 2 con más píxeles")
        ax.axvline(d["corte"], color="C3", lw=3.5, ls="--", label=f"corte {d['corte']:.3f}")
        ax.legend(fontsize=16)
        ax.set_title(f"histograma de gris — corte: {d['criterio']}", fontsize=21)
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel("cantidad de píxeles", fontsize=19)
    ax.tick_params(labelsize=15)

    ax = ejes[fila, 2]
    actividad = ski.filters.gaussian(ski.filters.sobel(v), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos2, _, _ = np.histogram2d(actividad.ravel(), v.ravel(), bins=BINS, range=((0, 1), (bajo, alto)))
    im = ax.imshow(np.log1p(conteos2), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(bajo, alto, 0, 1))
    if d["corte"] is not None:
        ax.axvline(d["corte"], color="k", lw=2.5, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log(1 + cantidad de píxeles)", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("gris", fontsize=19)
    ax.set_ylabel(f"actividad (σ={SIGMA_ACTIVIDAD}px, 0-1)", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title("paisaje gris-actividad (línea = corte)", fontsize=23)
fig.tight_layout()
fig.savefig(SALIDA / "51_ROI1_choque_variantes_histogramas_paisajes.png", dpi=28)
plt.close(fig)

# ---------------- Figura 52: resultados (clases, contorno sobre la original, ROI completa)
fig, ejes = plt.subplots(len(variantes), 3, figsize=(36, 11 * len(variantes)),
                         gridspec_kw={"width_ratios": [1, 1, 1.05]})
for fila, (nombre, d) in enumerate(variantes.items()):
    clase = d["clase"]
    ax = ejes[fila, 0]
    ax.imshow(clase[recorte], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=1.5, interpolation="nearest")
    ax.set_title(f"{nombre}\nclases (recorte)", fontsize=23)
    ax.legend(handles=[Patch(color=COLOR_OSCURA, label=f"fase oscura {np.mean(clase == 0):.1%}"),
                       Patch(color=COLOR_CLARA, label=f"fase clara {np.mean(clase == 1):.1%}")],
              loc="lower left", fontsize=18)
    ax.set_axis_off()

    ax = ejes[fila, 1]
    ax.imshow(suave[recorte], cmap="gray", vmin=lo_o, vmax=hi_o)
    borde = ski.segmentation.find_boundaries(clase[recorte] == 0, mode="inner")
    ax.imshow(np.ma.masked_where(~borde, borde), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
    ax.set_title("contorno de la fase oscura (rojo)\nsobre la ROI ORIGINAL suavizada", fontsize=23)
    ax.set_axis_off()

    ax = ejes[fila, 2]
    ax.imshow(suave, cmap="gray", vmin=lo_o, vmax=hi_o)
    capa = np.zeros(suave.shape + (4,))
    capa[clase == 0] = (*to_rgb(COLOR_OSCURA), 0.55)
    ax.imshow(capa)
    ax.add_patch(plt.Rectangle((X0, Y0), LADO_RECORTE, LADO_RECORTE, fill=False, ec="#ff2020", lw=3))
    ax.set_title("ROI completa: fase oscura en azul\n(cuadro rojo = recorte)", fontsize=23)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "52_ROI1_choque_variantes_resultados.png", dpi=28)
plt.close(fig)

# ---------------- Figura 53: resumen y diferencias con la variante a
fig, ejes = plt.subplots(1, 4, figsize=(40, 10), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
nombres = list(variantes)
letras = [n.split(")")[0] + ")" for n in nombres]
ax = ejes[0]
fr = [100 * np.mean(variantes[n]["clase"] == 0) for n in nombres]
barras = ax.bar(range(4), fr, color=["#7fa7d1", "#1f77b4", "#9ecf8a", "#2ca02c"], edgecolor="k")
for b, n in zip(barras, nombres):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
            f"{b.get_height():.1f}%\ncorte {variantes[n]['corte']:.3f}", ha="center", fontsize=17)
ax.set_xticks(range(4), letras, fontsize=20)
ax.set_ylabel("fase oscura (% del área)", fontsize=19)
ax.set_ylim(0, max(fr) * 1.3)
ax.tick_params(labelsize=15)
ax.set_title("fracción de fase oscura y corte de cada variante", fontsize=22)
base = variantes[nombres[0]]["clase"]
for col, (n, letra) in enumerate(zip(nombres[1:], letras[1:]), start=1):
    otra = variantes[n]["clase"]
    dif = np.full(base.shape, 2)
    dif[(base == 1) & (otra == 0)] = 0
    dif[(base == 0) & (otra == 1)] = 1
    ax = ejes[col]
    ax.imshow(dif[recorte], cmap=ListedColormap(["#1f77b4", "#ff7f0e", "#f2f2f2"]), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(f"{letra} contra a)\n(recorte)", fontsize=22)
    ax.legend(handles=[Patch(color="#1f77b4", label=f"pasa a oscura ({np.mean(dif == 0):.1%})"),
                       Patch(color="#ff7f0e", label=f"pasa a clara ({np.mean(dif == 1):.1%})"),
                       Patch(color="#f2f2f2", ec="k", label="igual")], loc="lower left", fontsize=16)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "53_ROI1_choque_variantes_resumen.png", dpi=30)
print("ok")
