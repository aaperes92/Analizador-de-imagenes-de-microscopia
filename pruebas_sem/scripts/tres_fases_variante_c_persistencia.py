"""Simulada de TRES fases mezcladas: variante c + persistencia en el paisaje.

Imagen (520 x 560 px), grises en el rango de la ROI 1:
- matriz clara: 0.70;
- patrón A: laberinto conectado (paredes entre islas alargadas, ~4 px): 0.60;
- patrón B: gusanos sueltos (contornos de otro campo aleatorio, ~4 px): 0.52 (encima de A).
Versiones: bordes nítidos y con degradé (gaussiana 1.5 px); ruido residual 0.005.

Procesamiento (igual que la ROI real):
1. suavizar (src/sem/suavizado.py);
2. variante c: máscara de enfoque (radio = degradé estimado en bordes aislados, cantidad 1) +
   filtro de choque (σ interno 1 px, frena con cambio medio < 0.1 x ruido);
3. paisaje gris-actividad (actividad = gaussiana(sobel), 3 px) y PERSISTENCIA: en cada fila, picos
   del gris (prominencia 10 % de la fila); persistencia(gris) = fracción de filas con un pico ahí;
   poblaciones = picos del perfil de persistencia (prominencia 15 %). No se fija cuántas hay;
4. cada píxel va a la población de gris más cercano; se compara con la verdad.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/tres_fases/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido, suavizar

SALIDA = Path("pruebas_sem/tres_fases")
SALIDA.mkdir(parents=True, exist_ok=True)
ALTO, ANCHO = 520, 560
GRISES = np.array([0.52, 0.60, 0.70])            # 0 = patrón B, 1 = patrón A, 2 = matriz
NOMBRES = ["patrón B (0.52)", "patrón A (0.60)", "matriz (0.70)"]
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e"]
SIGMA_ACT, BINS = 3, 128
FILA_MINIMA, PROMINENCIA_FILA, PROMINENCIA_PERFIL = 0.005, 0.10, 0.15
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 170, 190, 170


def campo_alargado(rng, escala, angulo):
    lado = int(1.5 * max(ALTO, ANCHO))
    c = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), escala), angulo,
                       reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    c = c[y0:y0 + ALTO, x0:x0 + ANCHO]
    return c / c.std()


def tres_fases(semilla=21):
    rng = np.random.default_rng(semilla)
    campo = campo_alargado(rng, (3.5, 9), 60)
    marcas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=5)
    marcas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, marcas, watershed_line=True)
    patron_a = ndimage.binary_dilation(islas == 0, structure=ski.morphology.disk(2))
    campo_b = campo_alargado(rng, (2.5, 7), 20)
    patron_b = (np.abs(campo_b - 1.2) < 0.18)
    patron_b = ski.morphology.remove_small_objects(patron_b, 30)
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


def persistencia(img):
    rango = tuple(np.percentile(img, (0.1, 99.9)))
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    act = (act - act.min()) / (act.max() - act.min())
    conteos, _, bordes = np.histogram2d(act.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
    centros = (bordes[:-1] + bordes[1:]) / 2
    paisaje = np.log1p(conteos)
    filas = np.flatnonzero(conteos.sum(axis=1) >= FILA_MINIMA * conteos.sum(axis=1).max())
    marcas = np.zeros_like(paisaje, bool)
    for f in filas:
        fila = ndimage.gaussian_filter1d(paisaje[f], 1)
        p, _ = signal.find_peaks(np.concatenate(([0.0], fila, [0.0])), prominence=PROMINENCIA_FILA * fila.max())
        marcas[f, p - 1] = True
    perfil = ndimage.binary_dilation(marcas, structure=np.ones((1, 3), bool))[filas].mean(axis=0)
    p, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=PROMINENCIA_PERFIL * perfil.max())
    return paisaje, marcas, perfil, centros, centros[p - 1], rango, act


etiquetas, rng = tres_fases()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(NOMBRES)})
casos = {
    "nítida": GRISES[etiquetas] + rng.normal(0, 0.005, etiquetas.shape),
    "con degradé (σ=1.5px)": ski.filters.gaussian(GRISES[etiquetas], 1.5) + rng.normal(0, 0.005, etiquetas.shape),
}

for clave, (nombre, img) in zip(("nitida", "degrade"), casos.items()):
    suave = suavizar(img)
    sigma_deg, usados = estimar_sigma_borde(suave)
    realzada = ski.filters.unsharp_mask(suave, radius=max(sigma_deg, 0.5), amount=1.0, preserve_range=True)
    procesada, pasos = choque(realzada, 1.0, 0.1 * estimar_ruido(suave))
    print(f"\n{nombre}: degradé estimado {sigma_deg:.2f} px ({usados} bordes aislados); choque {pasos} pasos")

    versiones = {"suavizada (sin variante c)": suave, "variante c (enfoque + choque)": procesada}
    fig, ejes = plt.subplots(2, 5, figsize=(52, 22), gridspec_kw={"width_ratios": [0.9, 1.15, 1, 0.9, 0.9]})
    for fila, (etiqueta, v) in enumerate(versiones.items()):
        paisaje, marcas, perfil, centros, encontradas, rango, act = persistencia(v)
        verdad = np.array([np.median(v[etiquetas == i]) for i in range(3)])
        tol = 0.04 * (rango[1] - rango[0])
        aciertos = [bool(np.any(np.abs(encontradas - g) <= tol)) for g in verdad]
        falsas = sum(not np.any(np.abs(verdad - e) <= tol) for e in encontradas)
        clase = np.argmin(np.abs(v[..., None] - encontradas[None, None, :]), axis=-1) if len(encontradas) else None
        # Cada población encontrada se identifica con la verdadera más cercana
        real_de = np.argmin(np.abs(encontradas[:, None] - verdad[None, :]), axis=1)
        prediccion = real_de[clase]
        acierto_por = [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]
        print(f"   {etiqueta:<32} encontradas {np.round(encontradas, 3)} (verdad {np.round(verdad, 3)}); "
              f"acertadas {sum(aciertos)}/3, falsas {falsas}; acierto por píxel "
              + ", ".join(f"{n.split(' (')[0]} {a:.0%}" for n, a in zip(NOMBRES, acierto_por)))

        ax = ejes[fila, 0]
        im = ax.imshow(v[Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango[0], vmax=rango[1])
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("gris", fontsize=20)
        cb.ax.tick_params(labelsize=16)
        ax.set_title(f"{nombre}\n{etiqueta} (recorte)", fontsize=25)
        ax.set_axis_off()

        ax = ejes[fila, 1]
        im = ax.imshow(paisaje, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(rango[0], rango[1], 0, 1))
        f_i, g_i = np.nonzero(marcas)
        ax.plot(centros[g_i], (f_i + 0.5) / BINS, ".", color="k", ms=4)
        for g, c in zip(verdad, COLORES):
            ax.axvline(g, color=c, lw=7, alpha=0.4)
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("log(1 + cantidad de píxeles)", fontsize=20)
        cb.ax.tick_params(labelsize=16)
        ax.set_xlabel("gris", fontsize=21)
        ax.set_ylabel(f"actividad (σ={SIGMA_ACT}px, 0-1)", fontsize=21)
        ax.tick_params(labelsize=17)
        ax.set_title("paisaje: puntos negros = picos de cada fila\nfranjas de color = gris real de cada fase", fontsize=24)

        ax = ejes[fila, 2]
        ax.plot(centros, perfil, color="k", lw=3)
        for g, c in zip(verdad, COLORES):
            ax.axvline(g, color=c, lw=8, alpha=0.4)
        for e in encontradas:
            ax.axvline(e, color="C3", lw=3, ls="--")
        ax.set_xlim(rango)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("gris", fontsize=21)
        ax.set_ylabel("persistencia (fracción de filas)", fontsize=21)
        ax.tick_params(labelsize=17)
        ax.set_title(f"persistencia: {len(encontradas)} poblaciones encontradas (rojo)\n"
                     f"acertadas {sum(aciertos)}/3, falsas {falsas}", fontsize=24)

        ax = ejes[fila, 3]
        ax.imshow(prediccion[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
                  interpolation="nearest")
        ax.set_title("RESULTADO (población más cercana)", fontsize=25)
        ax.legend(handles=[Patch(color=c, label=f"{n.split(' (')[0]}: {a:.0%} bien")
                           for c, n, a in zip(COLORES, NOMBRES, acierto_por)], loc="lower left", fontsize=18)
        ax.set_axis_off()

        ax = ejes[fila, 4]
        ax.imshow(etiquetas[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
                  interpolation="nearest")
        ax.set_title("VERDAD (mismo recorte)", fontsize=25)
        ax.set_axis_off()

    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORES, NOMBRES)], loc="upper center", ncol=3,
               fontsize=24, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SALIDA / f"01_tres_fases_{clave}.png", dpi=24)
    plt.close(fig)
print("ok")
