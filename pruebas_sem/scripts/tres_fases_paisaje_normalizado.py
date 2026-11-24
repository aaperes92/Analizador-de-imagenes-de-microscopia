"""Simulada de TRES fases con presencia real: normalización del paisaje gris-actividad.

Imagen (520 x 560 px), grises en el rango de la ROI 1:
- matriz clara: 0.70;
- patrón A: laberinto conectado (paredes entre islas alargadas, ~4 px): 0.60;
- patrón B: gusanos más gruesos y abundantes (franja de contorno de otro campo aleatorio): 0.52
  (encima de A).
Versiones: bordes nítidos y con degradé (gaussiana 1.5 px); ruido residual 0.005. Además, la
versión INVERTIDA (grises espejados: matriz 0.52, A 0.62, B 0.70) para ver si el "rastro" del
paisaje depende del gris o de la geometría (espesor y masa de cada fase).

Procesamiento:
1. suavizar (src/sem/suavizado.py), con y sin variante c (máscara de enfoque + filtro de choque);
2. paisaje gris-actividad (actividad = gaussiana(sobel), 3 px) en tres versiones:
   - log(1 + cuentas) (la de siempre);
   - cada fila dividida por su máximo (cada nivel de actividad pesa lo mismo);
   - cada fila dividida por su suma (probabilidad del gris dado el nivel de actividad);
   las filas con menos del 0.5 % de los píxeles de la fila más poblada se dejan en cero;
3. perfil de gris de cada versión (suma de las filas) y paisaje coloreado por fase verdadera.
No se detectan regiones ni se asignan píxeles: solo se mira qué hace la normalización.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/tres_fases/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido, suavizar

SALIDA = Path("pruebas_sem/tres_fases")
SALIDA.mkdir(parents=True, exist_ok=True)
ALTO, ANCHO = 520, 560
GRISES = {"normal": np.array([0.52, 0.60, 0.70]),       # 0 = patrón B, 1 = patrón A, 2 = matriz
          "invertida": np.array([0.70, 0.62, 0.52])}
FASES = ["patrón B", "patrón A", "matriz"]
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e"]
SIGMA_ACT, BINS, FILA_MINIMA = 3, 128, 0.005
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 170, 190, 170


def campo_alargado(rng, escala, angulo):
    lado = int(1.5 * max(ALTO, ANCHO))
    c = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), escala), angulo,
                       reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    c = c[y0:y0 + ALTO, x0:x0 + ANCHO]
    return c / c.std()


def tres_fases_con_presencia(semilla=21):
    """Etiquetas de la simulada: 0 = patrón B, 1 = patrón A, 2 = matriz."""
    rng = np.random.default_rng(semilla)
    campo = campo_alargado(rng, (3.5, 9), 60)
    marcas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=5)
    marcas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, marcas, watershed_line=True)
    patron_a = ndimage.binary_dilation(islas == 0, structure=ski.morphology.disk(2))
    campo_b = campo_alargado(rng, (4, 10), 20)
    patron_b = np.abs(campo_b) < 0.25
    patron_b = ski.morphology.remove_small_objects(patron_b, 60)
    etiquetas = np.full(campo.shape, 2)
    etiquetas[patron_a] = 1
    etiquetas[patron_b] = 0
    return etiquetas, rng


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


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def choque(img, sigma_interno, tolerancia_abs, dt=0.25, max_pasos=200):
    I = img.astype(float).copy()
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
        if np.mean(np.abs(delta)) < tolerancia_abs:
            break
    return I, paso + 1


def paisajes_normalizados(img, etiquetas):
    """Paisaje gris-actividad en tres versiones y conteos por fase verdadera."""
    rango = tuple(np.percentile(img, (0.1, 99.9)))
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    act = (act - act.min()) / (act.max() - act.min())
    conteos, _, bordes = np.histogram2d(act.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
    centros = (bordes[:-1] + bordes[1:]) / 2
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= FILA_MINIMA * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        por_maximo = np.where(validas, conteos / conteos.max(axis=1, keepdims=True), 0.0)
        por_suma = np.where(validas, conteos / por_fila, 0.0)
    por_fase = np.stack([np.histogram2d(act[etiquetas == i], img[etiquetas == i], bins=BINS,
                                        range=((0, 1), rango))[0] for i in range(3)])
    versiones = {"log(1 + cuentas)": np.log1p(conteos), "fila / máximo de la fila": por_maximo,
                 "fila / suma de la fila": por_suma}
    return versiones, por_fase, centros, rango


def paisaje_por_fase(por_fase):
    """Imagen RGB: cada bin toma el color de la fase de sus píxeles, con brillo log(cuentas)."""
    total = por_fase.sum(axis=0)
    mezcla = np.einsum("fyx,fc->yxc", por_fase, np.array([to_rgb(c) for c in COLORES]))
    mezcla /= np.maximum(total, 1)[..., None]
    intensidad = np.log1p(total) / np.log1p(total).max()
    return 1 - intensidad[..., None] * (1 - mezcla)


etiquetas, rng = tres_fases_con_presencia()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(FASES)})
ruido = rng.normal(0, 0.005, etiquetas.shape)

for clave_grises, grises in GRISES.items():
    nombres = [f"{n} ({g:.2f})" for n, g in zip(FASES, grises)]
    casos = {"nitida": ("nítida", grises[etiquetas] + ruido),
             "degrade": ("con degradé (σ=1.5px)", ski.filters.gaussian(grises[etiquetas], 1.5) + ruido)}
    for clave, (nombre, img) in casos.items():
        suave = suavizar(img)
        sigma_deg, usados = estimar_sigma_borde(suave)
        realzada = ski.filters.unsharp_mask(suave, radius=max(sigma_deg, 0.5), amount=1.0, preserve_range=True)
        procesada, pasos = choque(realzada, 1.0, 0.1 * estimar_ruido(suave))
        print(f"{clave_grises}, {nombre}: degradé estimado {sigma_deg:.2f} px ({usados} bordes); choque {pasos} pasos")

        procesos = {"suavizada (sin variante c)": suave, "variante c (enfoque + choque)": procesada}
        fig, ejes = plt.subplots(2, 6, figsize=(62, 22), gridspec_kw={"width_ratios": [0.9, 1.1, 1.1, 1.1, 1, 1]})
        for fila, (etiqueta, v) in enumerate(procesos.items()):
            versiones, por_fase, centros, rango = paisajes_normalizados(v, etiquetas)
            verdad = np.array([np.median(v[etiquetas == i]) for i in range(3)])

            ax = ejes[fila, 0]
            im = ax.imshow(v[Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango[0], vmax=rango[1])
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.tick_params(labelsize=16)
            ax.set_title(f"{clave_grises}, {nombre}\n{etiqueta} (recorte)", fontsize=24)
            ax.set_axis_off()

            for col, (titulo, paisaje) in enumerate(versiones.items(), start=1):
                ax = ejes[fila, col]
                im = ax.imshow(paisaje, origin="lower", aspect="auto", cmap=CMAP_PAISAJE,
                               extent=(rango[0], rango[1], 0, 1))
                for g, c in zip(verdad, COLORES):
                    ax.axvline(g, color=c, lw=6, alpha=0.45)
                cb = fig.colorbar(im, ax=ax, fraction=0.046)
                cb.ax.tick_params(labelsize=16)
                ax.set_xlabel("gris", fontsize=21)
                ax.set_ylabel(f"actividad (σ={SIGMA_ACT}px, 0-1)", fontsize=21)
                ax.tick_params(labelsize=17)
                ax.set_title(f"paisaje: {titulo}\nfranjas = gris real de cada fase", fontsize=24)

            ax = ejes[fila, 4]
            for (titulo, paisaje), estilo in zip(versiones.items(), ("-", "--", ":")):
                perfil = paisaje.sum(axis=0)
                ax.plot(centros, perfil / perfil.max(), "k", ls=estilo, lw=3, label=titulo)
            for g, c in zip(verdad, COLORES):
                ax.axvline(g, color=c, lw=8, alpha=0.4)
            ax.set_xlim(rango)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("gris", fontsize=21)
            ax.set_ylabel("suma de las filas (normalizada)", fontsize=21)
            ax.tick_params(labelsize=17)
            ax.legend(fontsize=17, loc="upper left")
            ax.set_title("perfil de gris de cada versión", fontsize=24)

            ax = ejes[fila, 5]
            ax.imshow(paisaje_por_fase(por_fase), origin="lower", aspect="auto", extent=(rango[0], rango[1], 0, 1))
            ax.set_xlabel("gris", fontsize=21)
            ax.set_ylabel(f"actividad (σ={SIGMA_ACT}px, 0-1)", fontsize=21)
            ax.tick_params(labelsize=17)
            ax.set_title("paisaje coloreado por fase verdadera\n(brillo = log cuentas)", fontsize=24)

        fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORES, nombres)], loc="upper center", ncol=3,
                   fontsize=24, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(SALIDA / f"02_paisaje_normalizado_{clave_grises}_{clave}.png", dpi=24)
        plt.close(fig)
print("ok")
